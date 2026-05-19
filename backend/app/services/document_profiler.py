"""
Document Profiler — Auto-RAG ingestion-time meta-engineering.

Exhaustively evaluates all 7 chunking methods × 7 embedding models using
similarity-based heuristics to auto-select the optimal configuration.

Metrics computed:
  Chunking  → chunk count, avg length, std dev, intra-group sim, inter-group sim
  Embedding → cosine similarity, intra-cluster sim, inter-cluster sim
  Retrieval → informational metrics (top-k accuracy, hybrid distribution, etc.)
"""

import time
import logging
import numpy as np
import json
import hashlib
from typing import Optional
from app.core.config import DATA_DIR

logger = logging.getLogger(__name__)

# ── Caching ────────────────────────────────────────────────────────────────────
PROFILER_CACHE_PATH = DATA_DIR / "profiler_cache.json"
EMBEDDING_EVAL_CACHE_PATH = DATA_DIR / "embedding_eval_cache.json"

def _load_json_cache(path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_json_cache(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)



# ═══════════════════════════════════════════════════════════════════════════════
#  Chunking Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_chunk_similarity_metrics(chunks: list[str]) -> dict:
    """Compute intra-group and inter-group similarity for a set of chunks
    using a lightweight SBERT encoding."""
    if len(chunks) < 2:
        return {"intra_similarity": 0.0, "inter_similarity": 0.0}

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks, show_progress_bar=False,
                                   batch_size=32, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
    except Exception:
        return {"intra_similarity": 0.0, "inter_similarity": 0.0}

    n = len(embeddings)

    # Intra-group similarity: average cosine similarity between consecutive chunks
    intra_sims = []
    for i in range(n - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        intra_sims.append(sim)
    intra_sim = float(np.mean(intra_sims)) if intra_sims else 0.0

    # Inter-group similarity: average cosine similarity between non-adjacent chunks
    # Sample to limit computation for large chunk sets
    inter_sims = []
    step = max(1, n // 20)  # sample up to ~20 pairs
    for i in range(0, n, step):
        for j in range(i + 2, n, step):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            inter_sims.append(sim)
    inter_sim = float(np.mean(inter_sims)) if inter_sims else 0.0

    return {
        "intra_similarity": round(intra_sim, 4),
        "inter_similarity": round(inter_sim, 4),
    }


def evaluate_chunking_methods(text: str, chunk_size: int = 500,
                               chunk_overlap: int = 50) -> dict:
    """Evaluate all 7 chunking methods and return metrics + best method."""
    from app.services.chunking_service import METHODS as CHUNKING_METHODS

    results = {}

    for method_id, fn in CHUNKING_METHODS.items():
        t0 = time.time()
        try:
            # Call with appropriate arguments
            if method_id == "fixed":
                chunks = fn(text, chunk_size=chunk_size, overlap=chunk_overlap)
            elif method_id == "recursive":
                chunks = fn(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            elif method_id == "sliding":
                chunks = fn(text, window_size=chunk_size // 2, step_size=chunk_size // 4)
            else:
                chunks = fn(text)
        except Exception as e:
            logger.warning(f"Chunking method '{method_id}' failed: {e}")
            results[method_id] = {"error": str(e)}
            continue

        elapsed = time.time() - t0

        if not chunks:
            results[method_id] = {"error": "No chunks produced"}
            continue

        lengths = [len(c) for c in chunks]
        avg_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        # Compute similarity metrics
        sim_metrics = _compute_chunk_similarity_metrics(chunks)

        results[method_id] = {
            "num_chunks": len(chunks),
            "avg_length": round(avg_len, 1),
            "std_dev": round(std_len, 1),
            "intra_similarity": sim_metrics["intra_similarity"],
            "inter_similarity": sim_metrics["inter_similarity"],
            "latency_s": round(elapsed, 3),
        }

    # ── Score each method ──────────────────────────────────────────────────────
    # Good chunking = high intra-similarity (coherent chunks),
    #                  low inter-similarity (distinct chunks),
    #                  low std dev (uniform sizes),
    #                  reasonable chunk count (not too few, not too many)
    best_method = None
    best_score = -float("inf")
    scored = {}

    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if valid_results:
        # Normalize metrics across methods for fair comparison
        all_intra = [v["intra_similarity"] for v in valid_results.values()]
        all_inter = [v["inter_similarity"] for v in valid_results.values()]
        all_std = [v["std_dev"] for v in valid_results.values()]

        intra_range = max(all_intra) - min(all_intra) if max(all_intra) != min(all_intra) else 1.0
        inter_range = max(all_inter) - min(all_inter) if max(all_inter) != min(all_inter) else 1.0
        std_range = max(all_std) - min(all_std) if max(all_std) != min(all_std) else 1.0

        for method_id, metrics in valid_results.items():
            # Normalize to [0, 1]
            norm_intra = (metrics["intra_similarity"] - min(all_intra)) / intra_range
            norm_inter = (metrics["inter_similarity"] - min(all_inter)) / inter_range
            norm_std = (metrics["std_dev"] - min(all_std)) / std_range

            # Weighted score: high intra, low inter, low std
            score = (
                0.40 * norm_intra +          # want high intra-similarity
                0.35 * (1 - norm_inter) +     # want low inter-similarity
                0.25 * (1 - norm_std)          # want uniform chunk sizes
            )

            scored[method_id] = round(score, 4)
            if score > best_score:
                best_score = score
                best_method = method_id

    return {
        "methods": results,
        "scores": scored,
        "best_method": best_method or "recursive",
        "best_score": round(best_score, 4) if best_score > -float("inf") else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Embedding Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_embedding_quality_metrics(embeddings: np.ndarray) -> dict:
    """Compute similarity-based quality metrics for an embedding matrix."""
    n = embeddings.shape[0]
    if n < 2:
        return {
            "avg_cosine_similarity": 0.0,
            "intra_similarity": 0.0,
            "inter_similarity": 0.0,
        }

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    # Average pairwise cosine similarity (sample for large sets)
    if n <= 100:
        sim_matrix = normed @ normed.T
        # Exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        avg_cosine = float(np.mean(sim_matrix[mask]))
    else:
        # Sample 500 pairs
        rng = np.random.default_rng(42)
        idx_a = rng.integers(0, n, size=500)
        idx_b = rng.integers(0, n, size=500)
        # Avoid same-index pairs
        valid = idx_a != idx_b
        sims = np.sum(normed[idx_a[valid]] * normed[idx_b[valid]], axis=1)
        avg_cosine = float(np.mean(sims))

    # Cluster-based intra/inter similarity using KMeans
    try:
        from sklearn.cluster import KMeans
        n_clusters = min(5, n)
        if n_clusters >= 2:
            km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
            labels = km.fit_predict(embeddings)

            intra_sims = []
            inter_sims = []
            for ci in range(n_clusters):
                cluster_mask = labels == ci
                cluster_vecs = normed[cluster_mask]
                if len(cluster_vecs) >= 2:
                    csim = cluster_vecs @ cluster_vecs.T
                    cmask = ~np.eye(len(cluster_vecs), dtype=bool)
                    intra_sims.append(float(np.mean(csim[cmask])))

            # Inter-cluster: compare centroids
            centroids = km.cluster_centers_
            c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            c_norms = np.where(c_norms == 0, 1, c_norms)
            c_normed = centroids / c_norms
            c_sim = c_normed @ c_normed.T
            c_mask = ~np.eye(n_clusters, dtype=bool)
            inter_sim = float(np.mean(c_sim[c_mask])) if n_clusters > 1 else 0.0

            intra_sim = float(np.mean(intra_sims)) if intra_sims else 0.0
        else:
            intra_sim = avg_cosine
            inter_sim = 0.0
    except Exception:
        intra_sim = avg_cosine
        inter_sim = 0.0

    return {
        "avg_cosine_similarity": round(avg_cosine, 4),
        "intra_similarity": round(intra_sim, 4),
        "inter_similarity": round(inter_sim, 4),
    }


def evaluate_embedding_models(chunks: list[str]) -> dict:
    """Evaluate all 7 embedding models on the given chunks."""
    cache = _load_json_cache(EMBEDDING_EVAL_CACHE_PATH)
    chunks_hash = hashlib.md5("".join(chunks).encode('utf-8')).hexdigest()
    
    if chunks_hash in cache:
        logger.info("Returning cached evaluate_embedding_models result.")
        return cache[chunks_hash]

    from app.services.embedding_service import MODELS as EMBEDDING_MODELS

    results = {}

    for model_id, fn in EMBEDDING_MODELS.items():
        t0 = time.time()
        try:
            embeddings = fn(chunks)
        except Exception as e:
            logger.warning(f"Embedding model '{model_id}' failed: {e}")
            results[model_id] = {"error": str(e)}
            continue

        elapsed = time.time() - t0

        if embeddings is None or len(embeddings) == 0:
            results[model_id] = {"error": "No embeddings produced"}
            continue

        quality = _compute_embedding_quality_metrics(embeddings)

        results[model_id] = {
            "dimension": int(embeddings.shape[1]),
            "count": int(embeddings.shape[0]),
            "avg_cosine_similarity": quality["avg_cosine_similarity"],
            "intra_similarity": quality["intra_similarity"],
            "inter_similarity": quality["inter_similarity"],
            "latency_s": round(elapsed, 3),
        }

    # ── Score each model ───────────────────────────────────────────────────────
    # Good embedding = high intra-cluster similarity (coherent clusters),
    #                   low inter-cluster similarity (well-separated),
    #                   moderate avg cosine (not all identical, not all random)
    best_model = None
    best_score = -float("inf")
    scored = {}

    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if valid_results:
        all_intra = [v["intra_similarity"] for v in valid_results.values()]
        all_inter = [v["inter_similarity"] for v in valid_results.values()]

        intra_range = max(all_intra) - min(all_intra) if max(all_intra) != min(all_intra) else 1.0
        inter_range = max(all_inter) - min(all_inter) if max(all_inter) != min(all_inter) else 1.0

        for model_id, metrics in valid_results.items():
            norm_intra = (metrics["intra_similarity"] - min(all_intra)) / intra_range
            norm_inter = (metrics["inter_similarity"] - min(all_inter)) / inter_range

            # Weighted: high intra (clusters are tight), low inter (clusters are separate)
            score = 0.55 * norm_intra + 0.45 * (1 - norm_inter)

            scored[model_id] = round(score, 4)
            if score > best_score:
                best_score = score
                best_model = model_id

    result = {
        "models": results,
        "scores": scored,
        "best_model": best_model or "sbert",
        "best_score": round(best_score, 4) if best_score > -float("inf") else 0.0,
    }
    
    cache[chunks_hash] = result
    _save_json_cache(EMBEDDING_EVAL_CACHE_PATH, cache)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Retrieval Evaluation (informational)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_retrieval_methods(chunks: list[str]) -> dict:
    """Compute informational retrieval metrics across all methods.
    These are reported but do not drive auto-selection at ingestion time."""
    from app.services.embedding_service import embed_sbert, get_index, get_chunks
    from app.services.retrieval_service import METHODS as RETRIEVAL_METHODS

    # Use a set of probe queries derived from chunk content
    probe_queries = _generate_probe_queries(chunks)

    results = {}
    for method_id, fn in RETRIEVAL_METHODS.items():
        t0 = time.time()
        try:
            all_scores = []
            hit_count = 0
            for probe in probe_queries:
                try:
                    docs = fn(probe, top_k=5)
                    scores = [d["score"] for d in docs]
                    all_scores.extend(scores)
                    if scores and max(scores) > 0.3:
                        hit_count += 1
                except Exception:
                    continue

            elapsed = time.time() - t0

            results[method_id] = {
                "top_k_accuracy": round(hit_count / max(len(probe_queries), 1), 4),
                "avg_score": round(float(np.mean(all_scores)), 4) if all_scores else 0.0,
                "score_std": round(float(np.std(all_scores)), 4) if all_scores else 0.0,
                "latency_s": round(elapsed, 3),
            }
        except Exception as e:
            results[method_id] = {"error": str(e)}

    # Compute graph-vs-vector split ratio and fallback rate from agent history
    try:
        from app.services.agent_service import _decision_log
        if _decision_log:
            graph_count = sum(1 for d in _decision_log if d.get("action") == "use_graph")
            vector_count = sum(1 for d in _decision_log
                               if d.get("action", "").startswith("use_vector"))
            fallback_count = sum(1 for d in _decision_log if d.get("action") == "fallback")
            total = len(_decision_log)
            split_info = {
                "graph_ratio": round(graph_count / total, 4),
                "vector_ratio": round(vector_count / total, 4),
                "fallback_rate": round(fallback_count / total, 4),
            }
        else:
            split_info = {"graph_ratio": 0.0, "vector_ratio": 0.0, "fallback_rate": 0.0}
    except Exception:
        split_info = {"graph_ratio": 0.0, "vector_ratio": 0.0, "fallback_rate": 0.0}

    return {
        "methods": results,
        "split_info": split_info,
    }


def _generate_probe_queries(chunks: list[str], n_probes: int = 5) -> list[str]:
    """Generate simple probe queries from chunk content for evaluation.
    Extracts key sentences from random chunks."""
    import re
    rng = np.random.default_rng(42)

    probes = []
    if not chunks:
        return probes

    indices = rng.choice(len(chunks), size=min(n_probes * 2, len(chunks)), replace=False)

    for idx in indices:
        sentences = re.split(r'(?<=[.!?])\s+', chunks[idx])
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if sentences:
            # Use the first substantial sentence as a probe
            probes.append(sentences[0])
        if len(probes) >= n_probes:
            break

    return probes


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Profiler Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def profile_document(text: str, chunk_size: int = 500,
                      chunk_overlap: int = 50) -> dict:
    """Run full document profiling: evaluate all chunking methods and
    embedding models, then return the auto-selected configuration.

    Returns:
        dict with keys:
          - chunking: evaluation results + best_method
          - embedding: evaluation results + best_model
          - retrieval: informational retrieval metrics
          - auto_config: {"chunking_method": ..., "embedding_model": ...}
    """
    cache = _load_json_cache(PROFILER_CACHE_PATH)
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    cache_key = f"{text_hash}_{chunk_size}_{chunk_overlap}"
    
    if cache_key in cache:
        logger.info("Returning cached profile_document result.")
        return cache[cache_key]

    logger.info("Starting document profiling (exhaustive evaluation)...")
    t_start = time.time()

    # Phase 1: Evaluate chunking
    logger.info("Phase 1: Evaluating chunking methods...")
    chunking_eval = evaluate_chunking_methods(text, chunk_size, chunk_overlap)
    best_chunk_method = chunking_eval["best_method"]

    # Phase 2: Chunk with the best method, then evaluate embeddings on ALL chunks
    logger.info(f"Phase 2: Chunking with best method '{best_chunk_method}', "
                "then evaluating embedding models...")
    from app.services.chunking_service import METHODS as CHUNKING_METHODS
    fn = CHUNKING_METHODS.get(best_chunk_method)
    if best_chunk_method == "fixed":
        best_chunks = fn(text, chunk_size=chunk_size, overlap=chunk_overlap)
    elif best_chunk_method == "recursive":
        best_chunks = fn(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif best_chunk_method == "sliding":
        best_chunks = fn(text, window_size=chunk_size // 2, step_size=chunk_size // 4)
    else:
        best_chunks = fn(text)

    embedding_eval = evaluate_embedding_models(best_chunks)

    # Phase 3: Retrieval evaluation (informational — requires embeddings to be built)
    retrieval_eval = {}
    try:
        from app.services.embedding_service import get_index
        if get_index() is not None:
            logger.info("Phase 3: Evaluating retrieval methods...")
            retrieval_eval = evaluate_retrieval_methods(best_chunks)
    except Exception as e:
        logger.warning(f"Retrieval evaluation skipped: {e}")

    total_time = time.time() - t_start
    logger.info(f"Document profiling complete in {total_time:.1f}s")

    result = {
        "chunking": chunking_eval,
        "embedding": embedding_eval,
        "retrieval": retrieval_eval,
        "auto_config": {
            "chunking_method": best_chunk_method,
            "embedding_model": embedding_eval["best_model"],
        },
        "profiling_time_s": round(total_time, 2),
    }
    
    cache[cache_key] = result
    _save_json_cache(PROFILER_CACHE_PATH, cache)
    return result
