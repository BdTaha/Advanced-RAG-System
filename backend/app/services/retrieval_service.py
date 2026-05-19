"""
Retrieval service — 5 retrieval methods for querying the vector store.
Uses Gemini API for high-quality answer synthesis from retrieved chunks.
"""

import numpy as np
from typing import Optional
from app.services.embedding_service import (
    get_index, get_chunks, get_embeddings, get_active_model_id, MODELS
)

def _embed_query(query: str) -> np.ndarray:
    model_id = get_active_model_id()
    # Default to sbert if model is not found in MODELS (shouldn't happen)
    from app.services.embedding_service import embed_sbert
    fn = MODELS.get(model_id, embed_sbert)
    return fn([query])[0]



# ═══════════════════════════════════════════════════════════════════════════════
#  Retrieval Methods
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_similarity(query: str, top_k: int = 5) -> list[dict]:
    """1 — Brute-force cosine similarity search via FAISS."""
    import faiss
    index = get_index()
    chunks = get_chunks()

    if index is None or not chunks:
        return []

    # Encode query and reshape to 2D
    query_emb = _embed_query(query).reshape(1, -1)
    # Normalize
    norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    query_emb = (query_emb / norm).astype(np.float32)

    scores, indices = index.search(query_emb, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks) and idx >= 0:
            results.append({
                "index": int(idx),
                "text": chunks[idx][:300] + ("..." if len(chunks[idx]) > 300 else ""),
                "full_text": chunks[idx],
                "score": round(float(score), 4),
                "method": "similarity",
            })
    return results


def retrieve_bm25(query: str, top_k: int = 5) -> list[dict]:
    """2 — BM25 lexical retrieval (rank-bm25)."""
    from rank_bm25 import BM25Okapi
    chunks = get_chunks()

    if not chunks:
        return []

    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "index": int(idx),
            "text": chunks[idx][:300] + ("..." if len(chunks[idx]) > 300 else ""),
            "full_text": chunks[idx],
            "score": round(float(scores[idx]), 4),
            "method": "bm25",
        })
    return results


def retrieve_dpr(query: str, top_k: int = 5) -> list[dict]:
    """3 — Dense Passage Retrieval (using SBERT as a dense encoder)."""
    chunks = get_chunks()
    embeddings = get_embeddings()

    if embeddings is None or not chunks:
        return []

    query_emb = _embed_query(query)

    # Compute cosine similarities
    norms_docs = np.linalg.norm(embeddings, axis=1)
    norm_query = np.linalg.norm(query_emb)
    if norm_query == 0:
        return []

    sims = np.dot(embeddings, query_emb) / (norms_docs * norm_query + 1e-8)
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "index": int(idx),
            "text": chunks[idx][:300] + ("..." if len(chunks[idx]) > 300 else ""),
            "full_text": chunks[idx],
            "score": round(float(sims[idx]), 4),
            "method": "dpr",
        })
    return results


def retrieve_hybrid(query: str, top_k: int = 5,
                     vector_weight: float = 0.6) -> list[dict]:
    """4 — Hybrid retrieval (BM25 + vector similarity combined)."""
    chunks = get_chunks()
    if not chunks:
        return []

    # BM25 scores
    from rank_bm25 import BM25Okapi
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Normalize BM25 scores
    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_norm = bm25_scores / bm25_max

    # Vector scores
    embeddings = get_embeddings()
    if embeddings is None:
        # Fallback to BM25 only
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "text": chunks[idx][:300] + ("..." if len(chunks[idx]) > 300 else ""),
                "full_text": chunks[idx],
                "score": round(float(bm25_scores[idx]), 4),
                "method": "hybrid (bm25 only)",
            })
        return results

    query_emb = _embed_query(query)
    norms_docs = np.linalg.norm(embeddings, axis=1)
    norm_query = np.linalg.norm(query_emb)
    vec_scores = np.dot(embeddings, query_emb) / (norms_docs * norm_query + 1e-8)

    # Normalize vector scores
    vec_max = max(vec_scores) if max(vec_scores) > 0 else 1
    vec_norm = vec_scores / vec_max

    # Combine
    combined = (1 - vector_weight) * bm25_norm + vector_weight * vec_norm
    top_indices = np.argsort(combined)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "index": int(idx),
            "text": chunks[idx][:300] + ("..." if len(chunks[idx]) > 300 else ""),
            "full_text": chunks[idx],
            "score": round(float(combined[idx]), 4),
            "bm25_score": round(float(bm25_norm[idx]), 4),
            "vector_score": round(float(vec_norm[idx]), 4),
            "method": "hybrid",
        })
    return results


def retrieve_reranking(query: str, top_k: int = 5,
                        initial_k: int = 20) -> list[dict]:
    """5 — Re-ranking: retrieve more candidates then re-rank with cross-encoder."""
    # First get initial candidates using hybrid
    candidates = retrieve_hybrid(query, top_k=initial_k)

    if not candidates:
        return []

    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, c["full_text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs)

        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = round(float(score), 4)
            candidates[i]["original_score"] = candidates[i]["score"]
            candidates[i]["score"] = round(float(score), 4)
            candidates[i]["method"] = "reranking"

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    except Exception:
        # Fallback: just return hybrid results
        return candidates[:top_k]


def retrieve_auto(query: str, top_k: int = 5) -> list[dict]:
    """6 — Auto-sélection (Agentic): utilise l'agent Q-Learning pour choisir la meilleure méthode vectorielle."""
    try:
        from app.services.agent_service import extract_features, features_to_state, get_q_table, ACTIONS

        features = extract_features(query)
        state = features_to_state(features)
        q_table = get_q_table()

        # Filter actions to only vector-based ones
        vector_actions = [a for a in ACTIONS if a.startswith("use_vector_")]
        vector_indices = [ACTIONS.index(a) for a in vector_actions]

        # Exploit the Q-table to find the best vector method for this state
        best_idx = max(vector_indices, key=lambda idx: q_table[state][idx])
        best_action = ACTIONS[best_idx]

        selected_method = best_action.replace("use_vector_", "")
        if selected_method == "rerank":
            selected_method = "reranking"

        # Call the selected method
        fn = METHODS.get(selected_method, retrieve_hybrid)
        results = fn(query, top_k=top_k)

        for r in results:
            r["auto_selected_method"] = selected_method

        return results

    except Exception:
        return retrieve_hybrid(query, top_k=top_k)


# ── Dispatcher ─────────────────────────────────────────────────────────────────
METHODS = {
    "similarity": retrieve_similarity,
    "bm25":       retrieve_bm25,
    "dpr":        retrieve_dpr,
    "hybrid":     retrieve_hybrid,
    "reranking":  retrieve_reranking,
    "auto":       retrieve_auto,
}


def retrieve(query: str, method: str = "hybrid", top_k: int = 5) -> dict:
    """Dispatch to the chosen retrieval method."""
    fn = METHODS.get(method, retrieve_hybrid)
    results = fn(query, top_k=top_k)

    # Generate summary from top results — use more context for better answers
    top_texts = [r["full_text"] for r in results[:5]]
    summary = _generate_summary(query, top_texts)

    return {
        "query": query,
        "method": method,
        "results": results,
        "count": len(results),
        "summary": summary,
    }


def _generate_summary(query: str, contexts: list[str]) -> str:
    """Generate a high-quality retrieval summary using Gemini API."""
    if not contexts:
        return "Aucun document pertinent trouvé."

    try:
        from app.services.llm_service import generate_answer_gemini, generate_answer

        # Build rich context with numbered passages
        context_parts = []
        for i, ctx in enumerate(contexts[:5], 1):
            # Use more text per chunk for better context
            context_parts.append(f"[Passage {i}]\n{ctx[:800]}")
        context_text = "\n\n".join(context_parts)

        prompt = (
            "Tu es un assistant expert en réponse à des questions. "
            "Voici des passages pertinents extraits d'un document.\n\n"
            "INSTRUCTIONS:\n"
            "- Réponds à la question en te basant UNIQUEMENT sur les passages fournis.\n"
            "- Sois précis, détaillé et structuré dans ta réponse.\n"
            "- Cite les informations clés des passages.\n"
            "- Si les passages ne contiennent pas assez d'information, dis-le clairement.\n"
            "- Réponds dans la même langue que la question.\n\n"
            f"PASSAGES:\n{context_text}\n\n"
            f"QUESTION: {query}\n\n"
            f"RÉPONSE DÉTAILLÉE:"
        )

        # Try Gemini first for highest quality
        answer = generate_answer_gemini(prompt, max_tokens=1024)
        if answer:
            return answer

        # Fallback to the standard generate_answer (flan-t5)
        return generate_answer(prompt)

    except Exception as e:
        print(f"[Retrieval] Summary generation failed: {e}")
        if contexts:
            return contexts[0][:500]
        return "Aucun document pertinent trouvé."
