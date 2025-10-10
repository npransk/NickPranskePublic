# app.py
import os
import json
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import requests
from ebooklib import epub
from io import BytesIO
from bs4 import BeautifulSoup

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "data"
BOOK_FILE = os.path.join(DATA_DIR, "book.epub")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
EMBED_FILE = os.path.join(DATA_DIR, "embeddings.npy")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_GENERATION_MODEL = "google/flan-t5-large"
HF_MAX_NEW_TOKENS = 256
TOP_K = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ---------------------------
# Streamlit Setup
# ---------------------------
st.set_page_config(page_title="Book Club Chatbot", layout="wide")
st.title("Neumaier Book Club Chatbot")

# ---------------------------
# Helpers
# ---------------------------
def load_epub_text(path: str) -> str:
    book = epub.read_epub(path)
    text = ""
    for item in book.get_items():
        if item.get_type() == 9:  # DOCUMENT
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text += soup.get_text() + "\n"
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_texts(model, texts):
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def build_knn_index(embeddings):
    nn = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric="cosine")
    nn.fit(embeddings)
    return nn

def call_hf(model, token, prompt):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": HF_MAX_NEW_TOKENS, "return_full_text": False},
    }
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json=payload,
        timeout=60,
    )
    data = resp.json()
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    return str(data)

# ---------------------------
# Load embeddings or create them
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    os.makedirs(DATA_DIR, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    if os.path.exists(CHUNKS_FILE) and os.path.exists(EMBED_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(EMBED_FILE)
    else:
        st.info("Embedding book for the first time...")
        text = load_epub_text(BOOK_FILE)
        chunks = chunk_text(text)
        embeddings = embed_texts(model, chunks)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f)
        np.save(EMBED_FILE, embeddings)
        st.success("Embeddings created and cached.")

    nn = build_knn_index(embeddings)
    return model, chunks, embeddings, nn

model, chunks, embeddings, nn = load_resources()

# ---------------------------
# Hugging Face token input
# ---------------------------
hf_token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
if not hf_token:
    hf_token = st.sidebar.text_input("Hugging Face API token", type="password")

# ---------------------------
# Chat Interface
# ---------------------------
question = st.text_input("Ask a question about the book:")
if question and hf_token:
    q_emb = model.encode([question], convert_to_numpy=True)[0].reshape(1, -1)
    distances, indices = nn.kneighbors(q_emb, n_neighbors=TOP_K)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    prompt = f"""
You are a helpful assistant that answers questions based only on the following passages from a book.
If the answer isn't in the passages, say "I don't know."

Passages:
{chr(10).join(retrieved_chunks)}

Question:
{question}

Answer:
"""

    with st.spinner("Generating answer..."):
        answer = call_hf(HF_GENERATION_MODEL, hf_token, prompt)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Source passages"):
        for i, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"**Passage {i}:**\n\n{chunk}")

elif not hf_token:
    st.warning("Please add your Hugging Face API token in Streamlit secrets or sidebar.")
