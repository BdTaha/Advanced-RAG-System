"""
Vectorial RAG router — chunking, embeddings, vector store, retrieval.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────
class ChunkRequest(BaseModel):
    text: Optional[str] = None
    method: Optional[str] = "recursive"   # set to "auto" for profiler
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50


class EmbedRequest(BaseModel):
    chunks: Optional[list[str]] = None
    model: Optional[str] = "sbert"        # set to "auto" for profiler


class RetrieveRequest(BaseModel):
    query: str
    method: Optional[str] = "hybrid"
    top_k: Optional[int] = 5


# ── Endpoints ──────────────────────────────────────────────────────────────────
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF / TXT) for processing."""
    from app.core.config import DATA_DIR
    content = await file.read()
    save_path = DATA_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(content)
    return {"filename": file.filename, "status": "uploaded", "size": len(content)}


@router.post("/chunk")
async def chunk_text(req: ChunkRequest):
    """Split text using one of 7 chunking methods.
    If method is 'auto', the document profiler selects the best method."""
    try:
        from app.services.chunking_service import chunk_text as do_chunk, load_document

        # Use provided text or load document
        text = req.text
        if not text:
            text = load_document()

        method = req.method
        profiler_info = None

        # Auto-select via profiler
        if method is None or method == "auto":
            from app.services.document_profiler import profile_document
            profile = profile_document(text, req.chunk_size, req.chunk_overlap)
            method = profile["auto_config"]["chunking_method"]
            profiler_info = profile["chunking"]

        result = do_chunk(text, method=method,
                         chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)

        if profiler_info:
            result["auto_selected"] = True
            result["profiler"] = profiler_info

        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed")
async def embed_chunks(req: EmbedRequest):
    """Generate embeddings for chunks using one of 7 models.
    If model is 'auto', the document profiler selects the best model."""
    try:
        from app.services.embedding_service import generate_embeddings
        from app.services.chunking_service import chunk_text as do_chunk, load_document

        chunks = req.chunks
        if not chunks:
            # Auto-chunk the document first
            text = load_document()
            result = do_chunk(text, method="recursive", chunk_size=500, chunk_overlap=50)
            chunks = result["chunks"]

        model_id = req.model
        profiler_info = None

        # Auto-select via profiler
        if model_id is None or model_id == "auto":
            from app.services.document_profiler import evaluate_embedding_models
            embedding_eval = evaluate_embedding_models(chunks)
            model_id = embedding_eval["best_model"]
            profiler_info = embedding_eval

        embed_result = generate_embeddings(chunks, model_id=model_id)

        if profiler_info:
            embed_result["auto_selected"] = True
            embed_result["profiler"] = profiler_info

        return embed_result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store")
async def build_vector_store():
    """Build / rebuild FAISS vector store from current embeddings."""
    try:
        from app.services.embedding_service import get_index
        index = get_index()
        if index is None:
            raise HTTPException(status_code=400,
                                detail="No embeddings found. Run /embed first.")
        return {"status": "built", "index_size": index.ntotal}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    """Retrieve relevant documents from the vector store."""
    try:
        from app.services.retrieval_service import retrieve as do_retrieve
        result = do_retrieve(req.query, method=req.method, top_k=req.top_k)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods/chunking")
async def list_chunking_methods():
    """Return all available chunking methods with descriptions."""
    return {
        "methods": [
            {"id": "fixed",      "name": "Fixed Size",            "description": "Split by character count with overlap"},
            {"id": "sentence",   "name": "Sentence-based",        "description": "Split by sentence boundaries"},
            {"id": "paragraph",  "name": "Paragraph-based",       "description": "Split by paragraph markers"},
            {"id": "semantic",   "name": "Semantic Similarity",   "description": "Group by cosine similarity threshold"},
            {"id": "sliding",    "name": "Sliding Window",        "description": "Overlapping windows of configurable size"},
            {"id": "recursive",  "name": "Recursive Splitting",   "description": "LangChain RecursiveCharacterTextSplitter"},
            {"id": "markdown",   "name": "Markdown / Structure",  "description": "Split by document structure"},
        ]
    }


@router.get("/methods/embedding")
async def list_embedding_models():
    """Return all available embedding models."""
    return {
        "models": [
            {"id": "tfidf",       "name": "TF-IDF"},
            {"id": "word2vec",    "name": "Word2Vec"},
            {"id": "glove",       "name": "GloVe"},
            {"id": "bert",        "name": "BERT"},
            {"id": "sbert",       "name": "Sentence-BERT"},
            {"id": "openai",      "name": "OpenAI Embeddings"},
            {"id": "instructor",  "name": "Instructor Embeddings"},
        ]
    }


@router.get("/methods/retrieval")
async def list_retrieval_methods():
    """Return all available retrieval methods."""
    return {
        "methods": [
            {"id": "similarity", "name": "Brute-force Similarity"},
            {"id": "bm25",       "name": "BM25"},
            {"id": "dpr",        "name": "DPR"},
            {"id": "hybrid",     "name": "Hybrid (Lexical + Vector)"},
            {"id": "reranking",  "name": "Re-ranking"},
        ]
    }


@router.post("/profile")
async def profile_document_endpoint():
    """Run the full document profiler: evaluate all chunking × embedding
    combinations and return the auto-selected configuration."""
    try:
        from app.services.document_profiler import profile_document
        from app.services.chunking_service import load_document
        text = load_document()
        result = profile_document(text)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
