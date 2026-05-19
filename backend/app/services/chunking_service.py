"""
Chunking service — 7 methods for splitting text into chunks.
Uses the document at DOCUMENT_PATH as the default source.
"""

import re
import math
from typing import Optional
from app.core.config import DOCUMENT_PATH


def load_document(path=None) -> str:
    """Load the RAG source document."""
    p = path or DOCUMENT_PATH
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

# ═══════════════════════════════════════════════════════════════════════════════
#  Chunking Methods
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Fixed Size ──────────────────────────────────────────────────────────────
def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text by character count with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# ── 2. Sentence-based ─────────────────────────────────────────────────────────
def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> list[str]:
    """Split text by sentence boundaries."""
    # Simple sentence splitter for French text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks


# ── 3. Paragraph-based ────────────────────────────────────────────────────────
def chunk_by_paragraphs(text: str, paragraphs_per_chunk: int = 3) -> list[str]:
    """Split text by paragraph markers (double newlines)."""
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    chunks = []
    for i in range(0, len(paragraphs), paragraphs_per_chunk):
        chunk = "\n\n".join(paragraphs[i:i + paragraphs_per_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks


# ── 4. Semantic Similarity ────────────────────────────────────────────────────
def chunk_by_semantic_similarity(text: str, threshold: float = 0.5,
                                  max_chunk_size: int = 1000) -> list[str]:
    """Group sentences by cosine similarity threshold.
    Uses sentence-transformers for embedding comparison.
    Falls back to paragraph-based if model not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        if len(sentences) < 2:
            return [text]

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        for i in range(1, len(embeddings)):
            sim = float(np.dot(embeddings[i], embeddings[i - 1]) /
                        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i - 1]) + 1e-8))
            current_text = " ".join(current_chunk)
            if sim >= threshold and len(current_text) < max_chunk_size:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    except ImportError:
        return chunk_by_paragraphs(text)


# ── 5. Sliding Window ─────────────────────────────────────────────────────────
def chunk_sliding_window(text: str, window_size: int = 500,
                          step_size: int = 250) -> list[str]:
    """Overlapping windows of configurable size."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + window_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += step_size
    return chunks


# ── 6. Recursive Splitting ────────────────────────────────────────────────────
def chunk_recursive(text: str, chunk_size: int = 500,
                     chunk_overlap: int = 50) -> list[str]:
    """LangChain RecursiveCharacterTextSplitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs]
    except ImportError:
        return chunk_fixed_size(text, chunk_size, chunk_overlap)


# ── 7. Markdown / Structure ───────────────────────────────────────────────────
def chunk_by_structure(text: str) -> list[str]:
    """Split by document structure (headers, section markers)."""
    # Detect section-like patterns (Roman numerals, letters, numbers at start of lines)
    pattern = r'(?=\n(?:[IVX]+\.\s|[A-Z]\.\s|\d+\.\s|Chapitre|DÉFINITIONS))'
    sections = re.split(pattern, text)
    sections = [s.strip() for s in sections if s.strip() and len(s.strip()) > 30]

    # If too few sections, fall back to paragraph chunking
    if len(sections) < 3:
        return chunk_by_paragraphs(text)
    return sections


# ── Dispatcher ─────────────────────────────────────────────────────────────────
METHODS = {
    "fixed":     chunk_fixed_size,
    "sentence":  chunk_by_sentences,
    "paragraph": chunk_by_paragraphs,
    "semantic":  chunk_by_semantic_similarity,
    "sliding":   chunk_sliding_window,
    "recursive": chunk_recursive,
    "markdown":  chunk_by_structure,
}


def chunk_text(text: str, method: str = "recursive",
               chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    """Dispatch to the chosen chunking method and return results."""
    fn = METHODS.get(method, chunk_recursive)

    # Pass size params to methods that accept them
    if method in ("fixed",):
        chunks = fn(text, chunk_size=chunk_size, overlap=chunk_overlap)
    elif method in ("recursive",):
        chunks = fn(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif method in ("sliding",):
        chunks = fn(text, window_size=chunk_size // 2, step_size=chunk_size // 4)
    else:
        chunks = fn(text)

    # Build a simple tree structure for the frontend
    sub_chunks = []
    for i, c in enumerate(chunks):
        # Split each chunk into sub-chunks of ~half size
        mid = len(c) // 2
        if len(c) > 200:
            sub_chunks.append({
                "chunk_index": i,
                "sub_chunks": [c[:mid].strip(), c[mid:].strip()]
            })
        else:
            sub_chunks.append({
                "chunk_index": i,
                "sub_chunks": [c]
            })

    return {
        "method": method,
        "chunks": chunks,
        "count": len(chunks),
        "tree": {
            "document": f"Document ({len(text)} chars)",
            "chunks": [f"Chunk {i + 1} ({len(c)} chars)" for i, c in enumerate(chunks)],
            "sub_chunks": sub_chunks,
        },
        "avg_chunk_size": sum(len(c) for c in chunks) / max(len(chunks), 1),
    }
