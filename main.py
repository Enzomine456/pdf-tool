#!/usr/bin/env python3
"""
main.py - Single-file Flask web app for Fly.io deployment
Features:
- Train a small Keras language model (Transformer-lite or GRU) on uploaded .txt corpus
- Generate text and render it into a PDF (ReportLab)
- Upload a PDF, extract text, improve it via the model, and return an enhanced PDF
- Web GUI (Bootstrap) for a neat interface
- All in one file for easy deployment to fly.io

Notes before running:
- Install dependencies: pip install flask tensorflow reportlab pypdf pillow markdown
- For OCR on scanned PDFs, install tesseract & pdf2image and pip install pytesseract pdf2image
- Training in a web request is blocking; for production use run training on worker dynos or use task queue.
- Provide model storage directory (./model_data) which will be created automatically.
"""

import os, io, tempfile, zipfile, json, datetime, threading, traceback
from pathlib import Path
from flask import Flask, request, render_template_string, send_file, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.units import cm
from pypdf import PdfReader, PdfWriter
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# Configuration
UPLOAD_FOLDER = "./uploads"
MODEL_DIR = "./model_data"
ALLOWED_TEXT_EXT = {"txt"}
ALLOWED_PDF_EXT = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-key"

DOWNLOAD_DIR = os.path.expanduser("~/Downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)



# ------------------------
# Simple tokenizer utilities
# ------------------------
SEQ_LEN = 128
DEFAULT_VOCAB = 8000

def build_tokenizer_from_texts(texts, vocab_size=DEFAULT_VOCAB):
    vect = layers.TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=SEQ_LEN, standardize="lower_and_strip_punctuation")
    ds = tf.data.Dataset.from_tensor_slices(texts).batch(16)
    vect.adapt(ds)
    return vect

def save_tokenizer(tokenizer, path):
    # save as a Keras model wrapper
    tmp = os.path.join(path, "tokenizer.keras")
    tokenizer.save(tmp)
    return tmp

def load_tokenizer(path):
    try:
        return keras.models.load_model(path)
    except Exception as e:
        print("Failed to load tokenizer:", e)
        return None


# ------------------------
# Models
# ------------------------
def make_gru_model(vocab_size, d_model=256, seq_len=SEQ_LEN):
    inputs = keras.Input(shape=(seq_len,), dtype="int32")
    x = layers.Embedding(vocab_size, d_model)(inputs)
    x = layers.GRU(d_model, return_sequences=True)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="gru_lm")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    return model

def make_transformer_lite(vocab_size, d_model=192, num_heads=3, ff=512, layers_n=3, seq_len=SEQ_LEN):
    inputs = keras.Input(shape=(seq_len,), dtype="int32")
    x = layers.Embedding(vocab_size, d_model)(inputs)
    pos = tf.range(0, seq_len)
    pos_emb = layers.Embedding(seq_len, d_model)(pos)
    x = x + pos_emb
    for _ in range(layers_n):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff_block = keras.Sequential([layers.Dense(ff, activation="relu"), layers.Dense(d_model)])(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_block)
    out = layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model(inputs, out, name="transformer_lite")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    return model

# ------------------------
# Training / dataset utils
# ------------------------
def texts_to_dataset(tokenizer, texts, seq_len=SEQ_LEN, batch_size=16):
    # Tokenize & create sliding windows
    all_ids = []
    for t in texts:
        toks = tokenizer(tf.constant([t])).numpy()[0]
        all_ids.extend(list(toks))
    # create sequences
    inputs = []
    targets = []
    for i in range(0, max(1, len(all_ids) - seq_len - 1), seq_len):
        x = all_ids[i:i+seq_len]
        y = all_ids[i+1:i+seq_len+1]
        if len(x) == seq_len and len(y) == seq_len:
            inputs.append(x)
            targets.append(y)
    if not inputs:
        raise ValueError("Not enough tokenized data to build dataset. Add more text or increase seq_len.")
    ds = tf.data.Dataset.from_tensor_slices((np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)))
    ds = ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def save_model_and_tokenizer(model, tokenizer, outdir):
    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, "model.keras")
    tokenizer_path = os.path.join(outdir, "tokenizer.keras")
    model.save(model_path)
    try:
        tokenizer.save(tokenizer_path)
    except Exception as e:
        print("Tokenizer save failed:", e)
    # write metadata
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"saved": datetime.datetime.utcnow().isoformat()}, f)
    return model_path, tokenizer_path

def load_model_and_tokenizer(outdir):
    model_path = os.path.join(outdir, "model.keras")
    tokenizer_path = os.path.join(outdir, "tokenizer.keras")
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        try:
            model = keras.models.load_model(model_path)
            tokenizer = keras.models.load_model(tokenizer_path)
            return model, tokenizer
        except Exception as e:
            print("Load failed:", e)
            return None, None
    return None, None

# ------------------------
# Sampling utils
# ------------------------
def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.0):
    logits = logits / max(temperature, 1e-5)
    probs = tf.nn.softmax(logits).numpy()
    if top_k and top_k > 0:
        idxs = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool); mask[idxs] = True
        probs = np.where(mask, probs, 0); probs = probs / probs.sum()
    if top_p and top_p > 0.0:
        sorted_idx = np.argsort(-probs); cum = np.cumsum(probs[sorted_idx])
        keep = cum <= top_p
        if not keep.any(): keep[0] = True
        mask = np.zeros_like(probs, dtype=bool); mask[sorted_idx[keep]] = True
        probs = np.where(mask, probs, 0); probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)

def generate_from_model(model, tokenizer, prompt, max_tokens=200, temperature=0.9, top_k=0, top_p=0.0):
    vocab = tokenizer.get_vocabulary()
    seq_len = model.input_shape[1] if hasattr(model.input_shape, "__len__") else SEQ_LEN
    x = tokenizer(tf.constant([prompt])).numpy()[0]
    out_ids = list(x[-seq_len:])
    for _ in range(max_tokens):
        inp = tf.constant([out_ids[-seq_len:]], dtype=tf.int32)
        preds = model.predict(inp, verbose=0)[0, -1]
        nxt = int(sample_from_logits(preds, temperature=temperature, top_k=top_k, top_p=top_p))
        out_ids.append(nxt)
    id2tok = {i:t for i,t in enumerate(vocab)}
    tokens = [id2tok.get(i, "") for i in out_ids]
    text = " ".join(tokens)
    # basic cleanup
    return text.replace("  ", " ").strip()

# ------------------------
# PDF utilities
# ------------------------
def text_to_pdf_bytes(text, title="Document", author=None):
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []
    if title:
        style_title = ParagraphStyle(name="Title", fontSize=20, alignment=TA_CENTER)
        story.append(Paragraph(title, style_title))
        story.append(Spacer(1, 12))
    body_style = ParagraphStyle(name="Body", fontSize=11, leading=16, alignment=TA_JUSTIFY)
    for para in text.split("\n\n"):
        story.append(Paragraph(para.replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 8))
    doc.build(story)
    bio.seek(0)
    return bio

def extract_text_from_pdf(file_stream, max_pages=None):
    # file_stream: file-like
    reader = PdfReader(file_stream)
    texts = []
    for i, page in enumerate(reader.pages):
        if max_pages and i >= max_pages: break
        txt = page.extract_text() or ""
        texts.append(txt)
    return "\n\n".join(texts)

def merge_pdfs_bytes(list_of_file_streams):
    writer = PdfWriter()
    for fs in list_of_file_streams:
        reader = PdfReader(fs)
        for p in reader.pages:
            writer.add_page(p)
    out = io.BytesIO()
    writer.write(out); out.seek(0)
    return out

# ------------------------
# Web pages / templates
# ------------------------
BASE_HTML = """
<!doctype html>
<html lang="pt-br">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>PDF AI Suite</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { background: linear-gradient(180deg,#0f172a 0%, #0b1220 100%); color: #e6eef8; }
      .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); }
      .accent { color: #60a5fa; }
      footer { opacity: 0.7; color:#9fb5d8; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace; }
    </style>
  </head>
  <body class="p-3">
    <div class="container">
      <div class="d-flex align-items-center mb-4">
        <img src="https://raw.githubusercontent.com/google/material-design-icons/master/png/action/description/materialicons/48dp/2x/baseline_description_black_48dp.png" width=48 height=48 alt="pdf">
        <div class="ms-3">
          <h2 class="mb-0 accent">PDF AI Suite</h2>
          <div class="text-muted">Gere e aprimore PDFs usando uma rede neural Keras — pronto para Fly.io</div>
        </div>
      </div>

      <div class="row g-3">
        <div class="col-md-6">
          <div class="card p-3">
            <h5>1) Treinar modelo</h5>
            <form method="post" action="/train" enctype="multipart/form-data">
              <div class="mb-2"><label class="form-label">Envie um ZIP com arquivos .txt (corpus)</label>
                <input required class="form-control" type="file" name="corpus_zip" accept=".zip"></div>
              <div class="mb-2"><label class="form-label">Modelo</label>
                <select name="model_type" class="form-select"><option value="gru">GRU (rápido)</option><option value="transformer">Transformer lite (mais lento)</option></select></div>
              <div class="mb-2 row">
                <div class="col"><input class="form-control" name="epochs" placeholder="epochs" value="2"></div>
                <div class="col"><input class="form-control" name="batch" placeholder="batch" value="16"></div>
              </div>
              <button class="btn btn-primary">Treinar</button>
            </form>
            <small class="text-muted">Treino é executado durante a requisição — para produção use workers.</small>
          </div>
        </div>

        <div class="col-md-6">
          <div class="card p-3">
            <h5>2) Gerar PDF a partir de prompt</h5>
            <form method="post" action="/generate_pdf">
              <div class="mb-2"><input class="form-control" name="prompt" placeholder="Escreva um prompt inicial (ex: Relatório anual)"></div>
              <div class="mb-2 row">
                <div class="col"><input class="form-control" name="max_tokens" placeholder="tokens" value="200"></div>
                <div class="col"><input class="form-control" name="title" placeholder="Título do PDF"></div>
              </div>
              <button class="btn btn-success">Gerar PDF</button>
            </form>
            <small class="text-muted">Use o modelo treinado em ./model_data</small>
          </div>
        </div>

        <div class="col-md-6">
          <div class="card p-3">
            <h5>3) Aprimorar PDF existente</h5>
            <form method="post" action="/enhance_pdf" enctype="multipart/form-data">
              <div class="mb-2"><input required class="form-control" type="file" name="pdf_file" accept=".pdf"></div>
              <div class="mb-2"><input class="form-control" name="style" placeholder="Estilo (formal, conciso, técnico)"></div>
              <div class="mb-2"><input class="form-control" name="title" placeholder="Título do PDF aprimorado"></div>
              <button class="btn btn-warning">Aprimorar e Baixar</button>
            </form>
            <small class="text-muted">Extrai o texto do PDF, reescreve com o modelo e retorna novo PDF.</small>
          </div>
        </div>

        <div class="col-md-6">
          <div class="card p-3">
            <h5>Utilitários</h5>
            <form method="post" action="/merge_pdfs" enctype="multipart/form-data">
              <div class="mb-2"><label class="form-label">Mesclar PDFs</label><input required multiple class="form-control" type="file" name="pdfs" accept=".pdf"></div>
              <button class="btn btn-outline-light">Mesclar</button>
            </form>
            <hr class="dropdown-divider">
            <form method="get" action="/status">
              <button class="btn btn-outline-info">Status do modelo</button>
            </form>
          </div>
        </div>

      </div>

      <footer class="mt-4">
        <div class="d-flex justify-content-between">
          <small>Deploy-ready: coloque em um app Fly.io com Dockerfile simples.</small>
          <small class="mono">./model_data</small>
        </div>
      </footer>
    </div>
  </body>
</html>
"""

# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(BASE_HTML)

@app.route("/status", methods=["GET"])
def status():
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)
    meta = {}
    if os.path.exists(os.path.join(MODEL_DIR, "meta.json")):
        try:
            meta = json.load(open(os.path.join(MODEL_DIR, "meta.json"), "r", encoding="utf-8"))
        except: meta = {}
    return jsonify({"model_loaded": bool(model is not None), "meta": meta})

@app.route("/train", methods=["POST"])
def train_route():
    try:
        f = request.files.get("corpus_zip")
        if not f:
            flash("Envie um arquivo zip contendo .txt files", "danger"); return redirect(url_for("index"))
        model_type = request.form.get("model_type", "gru")
        epochs = int(request.form.get("epochs", 2))
        batch = int(request.form.get("batch", 16))

        # Save zip to temp and extract texts
        tmpd = tempfile.mkdtemp(prefix="corpus_")
        zpath = os.path.join(tmpd, secure_filename(f.filename))
        f.save(zpath)
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(tmpd)
        texts = []
        for p in Path(tmpd).rglob("*.txt"):
            try:
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                print("read failed", p, e)
        if not texts:
            flash("Nenhum .txt encontrado no zip", "danger"); return redirect(url_for("index"))

        # build tokenizer & dataset
        tokenizer = build_tokenizer_from_texts(texts, vocab_size=DEFAULT_VOCAB)
        ds = texts_to_dataset(tokenizer, texts, batch_size=batch)

        # build model
        vocab_size = len(tokenizer.get_vocabulary())
        if model_type == "transformer":
            model = make_transformer_lite(vocab_size=vocab_size)
        else:
            model = make_gru_model(vocab_size=vocab_size)
        # train (blocking)
        model.fit(ds, epochs=epochs)

        # save
        save_model_and_tokenizer(model, tokenizer, MODEL_DIR)
        flash("Treino finalizado e modelo salvo em ./model_data", "success")
        return redirect(url_for("index"))
    except Exception as e:
        traceback.print_exc()
        flash(f"Erro no treino: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    try:
        prompt = request.form.get("prompt", "").strip() or "Relatório:"
        max_tokens = int(request.form.get("max_tokens", 200))
        title = request.form.get("title", "Documento Gerado")

        model, tokenizer = load_model_and_tokenizer(MODEL_DIR)
        if not model or not tokenizer:
            flash("Modelo não encontrado — treine primeiro", "danger"); return redirect(url_for("index"))

        text = generate_from_model(model, tokenizer, prompt, max_tokens=max_tokens)
        bio = text_to_pdf_bytes(text, title=title)

        # --- salvar também no diretório Downloads local ---
        filename = f"generated_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M')}.pdf"
        with open(os.path.join(DOWNLOAD_DIR, filename), "wb") as f:
            f.write(bio.getbuffer())
        bio.seek(0)
        return send_file(bio, mimetype="application/pdf", download_name=f"generated_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M')}.pdf", as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        flash(f"Erro na geração: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/enhance_pdf", methods=["POST"])
def enhance_pdf_route():
    try:
        pdf_file = request.files.get("pdf_file")
        if not pdf_file:
            flash("Envie um PDF", "danger"); return redirect(url_for("index"))
        style = request.form.get("style", "formal")
        title = request.form.get("title", "Documento Aprimorado")

        model, tokenizer = load_model_and_tokenizer(MODEL_DIR)
        if not model or not tokenizer:
            flash("Modelo não encontrado — treine primeiro", "danger"); return redirect(url_for("index"))

        # read pdf bytes
        stream = io.BytesIO(pdf_file.read())
        raw_text = extract_text_from_pdf(stream)
        if not raw_text.strip():
            flash("Nenhum texto extraído do PDF (talvez seja um scan).", "warning")
            # For scanned PDFs, OCR would be required (not enabled here).
            return redirect(url_for("index"))

        prompt = f"Reescreva o texto abaixo no estilo {style}, deixando mais claro e coerente:\n\n{raw_text}\n\nVersão aprimorada:"
        improved = generate_from_model(model, tokenizer, prompt, max_tokens=400)
        bio = text_to_pdf_bytes(improved, title=title)

        # --- salvar também no diretório Downloads local ---
        filename = f"enhanced_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M')}.pdf"
        with open(os.path.join(DOWNLOAD_DIR, filename), "wb") as f:
            f.write(bio.getbuffer())
        bio.seek(0)
        return send_file(bio, mimetype="application/pdf", download_name=f"enhanced_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M')}.pdf", as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        flash(f"Erro ao aprimorar: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/merge_pdfs", methods=["POST"])
def merge_pdfs():
    try:
        files = request.files.getlist("pdfs")
        streams = []
        for f in files:
            if f and f.filename.lower().endswith(".pdf"):
                streams.append(io.BytesIO(f.read()))
        if not streams:
            flash("Envie ao menos um PDF", "danger"); return redirect(url_for("index"))
        out = merge_pdfs_bytes(streams)
        return send_file(out, mimetype="application/pdf", download_name="merged.pdf", as_attachment=True)
    except Exception as e:
        traceback.print_exc(); flash("Erro ao mesclar: "+str(e),"danger"); return redirect(url_for("index"))

# ------------------------
# Run (for local testing)
# ------------------------
if __name__ == "__main__":
    # On Fly.io, use gunicorn with main:app; locally run with flask CLI or python main.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
