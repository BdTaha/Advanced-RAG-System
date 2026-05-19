"""
Embedding service — 7 embedding models + FAISS vector store + PCA 2D projection.
"""

import os
import json
import pickle
import numpy as np
from typing import Optional
from pathlib import Path
from app.core.config import FAISS_INDEX_DIR, DATA_DIR


# ── Cache paths ────────────────────────────────────────────────────────────────
_CHUNKS_CACHE = DATA_DIR / "cached_chunks.json"
_EMBEDDINGS_CACHE = DATA_DIR / "cached_embeddings.npy"
_FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
_META_PATH = FAISS_INDEX_DIR / "meta.json"

# ── Global state ───────────────────────────────────────────────────────────────
_faiss_index = None
_chunks_store: list[str] = []
_embeddings_matrix: Optional[np.ndarray] = None


def _ensure_dir():
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Embedding Models
# ═══════════════════════════════════════════════════════════════════════════════

def embed_tfidf(chunks: list[str]) -> np.ndarray:
    """1 — TF-IDF embeddings (scikit-learn)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=384)
    matrix = vectorizer.fit_transform(chunks).toarray().astype(np.float32)
    return matrix


def embed_word2vec(chunks: list[str]) -> np.ndarray:
    """2 — Word2Vec mean-pooled embeddings (gensim)."""
    from gensim.models import Word2Vec
    tokenized = [c.lower().split() for c in chunks]
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5,
                     min_count=1, workers=2, epochs=10)
    embeddings = []
    for tokens in tokenized:
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        if vecs:
            embeddings.append(np.mean(vecs, axis=0))
        else:
            embeddings.append(np.zeros(100))
    return np.array(embeddings, dtype=np.float32)


def embed_glove(chunks: list[str]) -> np.ndarray:
    """3 — GloVe mean-pooled embeddings.
    Uses gensim to load pre-trained GloVe. Falls back to Word2Vec if not available.
    """
    try:
        import gensim.downloader as gloader
        model = gloader.load("glove-wiki-gigaword-100")
        embeddings = []
        for chunk in chunks:
            tokens = chunk.lower().split()
            vecs = [model[w] for w in tokens if w in model]
            if vecs:
                embeddings.append(np.mean(vecs, axis=0))
            else:
                embeddings.append(np.zeros(100))
        return np.array(embeddings, dtype=np.float32)
    except Exception:
        return embed_word2vec(chunks)


def embed_bert(chunks: list[str]) -> np.ndarray:
    """4 — BERT [CLS] embeddings (transformers)."""
    from transformers import AutoTokenizer, AutoModel
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt",
                               truncation=True, max_length=512, padding=True).to(device)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_emb)
    return np.array(embeddings, dtype=np.float32)


def embed_sbert(chunks: list[str]) -> np.ndarray:
    """5 — Sentence-BERT embeddings (sentence-transformers)."""
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(chunks, show_progress_bar=False,
                               batch_size=32, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def embed_openai(chunks: list[str]) -> np.ndarray:
    """6 — OpenAI Embeddings (requires OPENAI_API_KEY)."""
    try:
        from openai import OpenAI
        from app.core.config import OPENAI_API_KEY
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
    except Exception:
        # Fallback to SBERT
        return embed_sbert(chunks)


def embed_instructor(chunks: list[str]) -> np.ndarray:
    """7 — Instructor Embeddings."""
    try:
        from InstructorEmbedding import INSTRUCTOR
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = INSTRUCTOR("hkunlp/instructor-base", device=device)
        instruction = "Represent the scientific document passage for retrieval:"
        pairs = [[instruction, c] for c in chunks]
        embeddings = model.encode(pairs)
        return np.array(embeddings, dtype=np.float32)
    except Exception:
        return embed_sbert(chunks)


# ── Dispatcher ─────────────────────────────────────────────────────────────────
MODELS = {
    "tfidf":      embed_tfidf,
    "word2vec":   embed_word2vec,
    "glove":      embed_glove,
    "bert":       embed_bert,
    "sbert":      embed_sbert,
    "openai":     embed_openai,
    "instructor": embed_instructor,
}


def generate_embeddings(chunks: list[str], model_id: str = "sbert") -> dict:
    """Generate embeddings for chunks using the chosen model and build FAISS index."""
    global _chunks_store, _embeddings_matrix, _faiss_index

    fn = MODELS.get(model_id, embed_sbert)
    embeddings = fn(chunks)
    _embeddings_matrix = embeddings
    _chunks_store = chunks

    # Build FAISS index
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vecs)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = (embeddings / norms).astype(np.float32)
    index.add(normalized)
    _faiss_index = index

    # Save to disk
    _ensure_dir()
    faiss.write_index(index, str(_FAISS_INDEX_PATH))
    np.save(str(_EMBEDDINGS_CACHE), embeddings)
    with open(_CHUNKS_CACHE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    with open(_META_PATH, "w") as f:
        json.dump({"model": model_id, "dim": dim, "count": len(chunks)}, f)

    # PCA 2D projection
    pca_2d = compute_pca_2d(embeddings)

    return {
        "model": model_id,
        "count": len(chunks),
        "dimension": dim,
        "embeddings_2d": pca_2d,
        "index_size": index.ntotal,
    }


def compute_pca_2d(embeddings: np.ndarray) -> list[dict]:
    """Compute PCA 2D projection for visualization."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    n_samples = embeddings.shape[0]
    n_components = min(2, embeddings.shape[1], n_samples)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    # Cluster for coloring
    n_clusters = min(5, n_samples)
    if n_clusters >= 2:
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        labels = km.fit_predict(embeddings)
    else:
        labels = [0] * n_samples

    points = []
    for i in range(n_samples):
        points.append({
            "x": float(coords[i][0]) if n_components >= 1 else 0,
            "y": float(coords[i][1]) if n_components >= 2 else 0,
            "cluster": int(labels[i]),
            "index": i,
        })
    return points


def load_index():
    """Load persisted FAISS index + chunks from disk."""
    global _faiss_index, _chunks_store, _embeddings_matrix
    import faiss

    if _FAISS_INDEX_PATH.exists() and _CHUNKS_CACHE.exists():
        _faiss_index = faiss.read_index(str(_FAISS_INDEX_PATH))
        with open(_CHUNKS_CACHE, "r", encoding="utf-8") as f:
            _chunks_store = json.load(f)
        if _EMBEDDINGS_CACHE.exists():
            _embeddings_matrix = np.load(str(_EMBEDDINGS_CACHE))
        return True
    return False


def get_index():
    global _faiss_index
    if _faiss_index is None:
        load_index()
    return _faiss_index


def get_chunks():
    global _chunks_store
    if not _chunks_store:
        load_index()
    return _chunks_store


def get_embeddings():
    global _embeddings_matrix
    if _embeddings_matrix is None:
        load_index()
    return _embeddings_matrix

def get_active_model_id() -> str:
    """Return the active model ID from meta.json."""
    if _META_PATH.exists():
        try:
            with open(_META_PATH, "r") as f:
                meta = json.load(f)
                return meta.get("model", "sbert")
        except Exception:
            pass
    return "sbert"
