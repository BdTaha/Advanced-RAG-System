"""
Q-Learning Agent service — routes queries between Graph RAG and Vector RAG.
Pre-seeded Q-table with intelligent priors for immediate quality answers.
Uses Gemini API for final answer synthesis.
"""

import json
import math
import random
import numpy as np
from typing import Optional
from pathlib import Path
from app.core.config import QL_ALPHA, QL_GAMMA, QL_EPSILON, QTABLE_PATH, DATA_DIR

# ── Actions ────────────────────────────────────────────────────────────────────
ACTIONS = [
    "use_graph",
    "use_vector_bm25",
    "use_vector_similarity",
    "use_vector_hybrid",
    "use_vector_rerank",
    "fallback",
]
NUM_ACTIONS = len(ACTIONS)  # 6

# ── State discretization ───────────────────────────────────────────────────────
# States are discretized query features:
#   semantic_score: low/high
#   systematic_score: low/high
#   query_length: short/long
# → 2×2×2 = 8 states

STATE_LABELS = [
    "Semantic_Low+Systematic_Low+Short",
    "Semantic_Low+Systematic_Low+Long",
    "Semantic_Low+Systematic_High+Short",
    "Semantic_Low+Systematic_High+Long",
    "Semantic_High+Systematic_Low+Short",
    "Semantic_High+Systematic_Low+Long",
    "Semantic_High+Systematic_High+Short",
    "Semantic_High+Systematic_High+Long",
]

# ── Pre-seeded Q-values ───────────────────────────────────────────────────────
# These encode domain knowledge about which retrieval strategy works best
# for each state, giving the agent a strong starting point.
#
# Columns: [graph, bm25, similarity, hybrid, rerank, fallback]
#
# Design rationale:
#   - Low semantic + Low systematic → keyword-heavy, BM25/hybrid best
#   - Low semantic + High systematic → relational, graph best
#   - High semantic + Low systematic → meaning-based, reranking/hybrid best
#   - High semantic + High systematic → complex, reranking best
#   - Short queries → BM25/similarity work well
#   - Long queries → hybrid/reranking capture more nuance
#   - Fallback always penalized (-0.5)

_PRIOR_Q_TABLE = np.array([
    # State 0: SemL+SysL+Short — simple keyword queries
    [0.1, 0.6, 0.5, 0.7, 0.8, -0.5],
    # State 1: SemL+SysL+Long  — longer keyword queries
    [0.1, 0.4, 0.4, 0.8, 0.9, -0.5],
    # State 2: SemL+SysH+Short — short relational queries
    [0.8, 0.3, 0.3, 0.5, 0.6, -0.5],
    # State 3: SemL+SysH+Long  — long relational queries
    [0.7, 0.3, 0.3, 0.6, 0.7, -0.5],
    # State 4: SemH+SysL+Short — short semantic queries
    [0.2, 0.3, 0.6, 0.7, 0.9, -0.5],
    # State 5: SemH+SysL+Long  — long semantic queries
    [0.2, 0.3, 0.5, 0.8, 0.9, -0.5],
    # State 6: SemH+SysH+Short — short complex queries
    [0.6, 0.3, 0.4, 0.7, 0.9, -0.5],
    # State 7: SemH+SysH+Long  — long complex queries
    [0.5, 0.3, 0.4, 0.8, 0.9, -0.5],
], dtype=np.float64)


# ── Q-Table ────────────────────────────────────────────────────────────────────
_q_table: Optional[np.ndarray] = None  # shape: (8, NUM_ACTIONS)
_reward_history: list[float] = []
_decision_log: list[dict] = []

# Force low epsilon for exploitation — the pre-seeded table is already good
_AGENT_EPSILON = 0.05


def _ensure_q_table():
    """Load or initialize Q-table. Uses pre-seeded priors for new tables."""
    global _q_table, _reward_history
    if _q_table is not None:
        return

    if QTABLE_PATH.exists():
        try:
            with open(QTABLE_PATH, "r") as f:
                data = json.load(f)
            loaded = np.array(data.get("q_table", []))

            if loaded.shape == (8, NUM_ACTIONS):
                # Check if the table is mostly zeros (unlearned) — if so, use priors
                nonzero_count = np.count_nonzero(loaded)
                if nonzero_count < 6:
                    # Table barely learned anything, reset to priors
                    print("[Agent] Q-table nearly empty, resetting to pre-seeded priors.")
                    _q_table = _PRIOR_Q_TABLE.copy()
                else:
                    _q_table = loaded
            else:
                # Wrong shape, use priors
                _q_table = _PRIOR_Q_TABLE.copy()

            _reward_history = data.get("rewards", [])
        except Exception:
            _q_table = _PRIOR_Q_TABLE.copy()
            _reward_history = []
    else:
        _q_table = _PRIOR_Q_TABLE.copy()
        _reward_history = []

    _save_q_table()


def _save_q_table():
    """Persist Q-table and reward history."""
    global _q_table, _reward_history
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(QTABLE_PATH, "w") as f:
        json.dump({
            "q_table": _q_table.tolist(),
            "rewards": _reward_history[-200:],  # Keep last 200
        }, f)

def get_q_table() -> np.ndarray:
    """Return the active Q-table, initializing it if necessary."""
    _ensure_q_table()
    return _q_table


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Extraction & State Mapping
# ═══════════════════════════════════════════════════════════════════════════════

# Keywords for classifying query type
SYSTEMATIC_KEYWORDS = [
    "relation", "lien", "entre", "structure", "connecte", "graphe", "réseau",
    "hiérarchie", "taxonomy", "espèce", "genre", "famille", "classe",
    "cause", "effet", "influence", "interaction", "comparaison",
    "relation", "link", "between", "structure", "connected", "graph", "network",
]

SEMANTIC_KEYWORDS = [
    "qu'est-ce", "comment", "pourquoi", "expliquer", "définir", "décrire",
    "quels sont", "quel est", "signification", "rôle", "fonction", "importance",
    "caractéristique", "propriété", "mécanisme", "processus",
    "what", "how", "why", "explain", "describe", "define",
]


def extract_features(query: str) -> dict:
    """Extract features from query for Q-Learning state determination."""
    query_lower = query.lower()
    words = query_lower.split()

    # Semantic score: proportion of semantic keywords found
    semantic_hits = sum(1 for kw in SEMANTIC_KEYWORDS if kw in query_lower)
    semantic_score = min(semantic_hits / 3.0, 1.0)

    # Systematic score: proportion of relational keywords found
    systematic_hits = sum(1 for kw in SYSTEMATIC_KEYWORDS if kw in query_lower)
    systematic_score = min(systematic_hits / 3.0, 1.0)

    # Query length
    is_long = len(words) > 8

    return {
        "semantic_score": round(semantic_score, 3),
        "systematic_score": round(systematic_score, 3),
        "query_length": "long" if is_long else "short",
        "word_count": len(words),
    }


def features_to_state(features: dict) -> int:
    """Convert features to a discrete state index (0-7)."""
    sem_high = 1 if features["semantic_score"] > 0.3 else 0
    sys_high = 1 if features["systematic_score"] > 0.3 else 0
    is_long = 1 if features["query_length"] == "long" else 0

    state = sem_high * 4 + sys_high * 2 + is_long
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  Q-Learning Core
# ═══════════════════════════════════════════════════════════════════════════════

def choose_action(state: int, epsilon: float = None) -> tuple[int, str]:
    """Epsilon-greedy action selection with very low exploration."""
    _ensure_q_table()
    eps = epsilon if epsilon is not None else _AGENT_EPSILON

    if random.random() < eps:
        # Explore — but avoid fallback during exploration
        action_idx = random.randint(0, NUM_ACTIONS - 2)  # Exclude fallback
    else:
        # Exploit — pick the best known action
        action_idx = int(np.argmax(_q_table[state]))

    return action_idx, ACTIONS[action_idx]


def update_q_table(state: int, action_idx: int, reward: float,
                    next_state: Optional[int] = None):
    """Update Q-value using the Q-Learning formula:
    Q(S,A) ← Q(S,A) + α[R + γ·max Q(S',A') - Q(S,A)]
    """
    _ensure_q_table()

    if next_state is not None:
        best_next = np.max(_q_table[next_state])
    else:
        best_next = 0.0

    old_value = _q_table[state][action_idx]
    _q_table[state][action_idx] = (
        old_value + QL_ALPHA * (reward + QL_GAMMA * best_next - old_value)
    )
    _reward_history.append(reward)
    _save_q_table()


# ═══════════════════════════════════════════════════════════════════════════════
#  Agent Query Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def agent_query(query: str) -> dict:
    """Full agent pipeline: analyze → route → execute → synthesize → evaluate → update."""
    global _decision_log

    # Step 1: Extract features
    features = extract_features(query)
    state = features_to_state(features)
    state_label = STATE_LABELS[state]

    # Step 2: Choose primary action
    action_idx, action = choose_action(state)

    # Step 3: Execute chosen branch
    answer = ""
    sources = []
    confidence = 0.0

    if action == "use_graph":
        answer, sources, confidence = _execute_graph_rag(query)
    elif action.startswith("use_vector_"):
        method = action.replace("use_vector_", "")
        if method == "rerank":
            method = "reranking"
        answer, sources, confidence = _execute_vector_rag(query, retrieval_method=method)
    else:  # fallback
        # Even on fallback, try hybrid as a safety net
        answer, sources, confidence = _execute_vector_rag(query, retrieval_method="hybrid")
        if not answer or confidence < 0.3:
            answer = "Je n'ai pas trouvé d'information pertinente dans les documents indexés pour cette question."
            confidence = 0.1

    # Step 4: If primary result is weak, try a second strategy
    if confidence < 0.4 and action != "use_vector_rerank":
        alt_answer, alt_sources, alt_confidence = _execute_vector_rag(
            query, retrieval_method="reranking"
        )
        if alt_confidence > confidence:
            answer = alt_answer
            sources = alt_sources
            confidence = alt_confidence
            action = "use_vector_rerank"
            action_idx = ACTIONS.index("use_vector_rerank")

    # Step 5: Evaluate reward
    reward = _compute_reward(answer, confidence, action)

    # Step 6: Update Q-table
    update_q_table(state, action_idx, reward)

    # Log decision
    decision = {
        "query": query,
        "features": features,
        "state": state,
        "state_label": state_label,
        "action": action,
        "action_index": action_idx,
        "confidence": round(confidence, 3),
        "reward": round(reward, 3),
        "answer_preview": answer[:200],
    }
    _decision_log.append(decision)
    if len(_decision_log) > 100:
        _decision_log = _decision_log[-100:]

    # Determine query type for frontend
    if features["systematic_score"] > features["semantic_score"]:
        query_type = "Systematic"
    elif features["semantic_score"] > 0.3:
        query_type = "Semantic"
    else:
        query_type = "Hybrid"

    # Determine routing label for frontend
    if action == "use_graph":
        routing_label = "Graph RAG"
    elif action.startswith("use_vector_"):
        method_label = action.replace("use_vector_", "").upper()
        routing_label = f"Vectoriel RAG ({method_label})"
    else:
        routing_label = "Fallback"

    return {
        "query": query,
        "type": query_type,
        "action": action,
        "routing": routing_label,
        "confidence": round(confidence, 3),
        "answer": answer,
        "sources": sources,
        "reward": round(reward, 3),
        "features": features,
        "state": state_label,
        "q_values": {
            ACTIONS[i]: round(float(_q_table[state][i]), 4)
            for i in range(NUM_ACTIONS)
        },
    }


def _execute_vector_rag(query: str,
                        retrieval_method: str = "hybrid") -> tuple[str, list, float]:
    """Execute vector retrieval branch with Gemini-powered answer synthesis."""
    try:
        from app.services.retrieval_service import retrieve
        result = retrieve(query, method=retrieval_method, top_k=5)
        sources = result.get("results", [])
        answer = result.get("summary", "")

        # Compute confidence from top retrieval scores
        scores = [r["score"] for r in sources if isinstance(r.get("score"), (int, float))]
        if scores:
            # Use weighted average of top scores
            top_scores = sorted(scores, reverse=True)[:3]
            confidence = sum(top_scores) / len(top_scores)
            confidence = min(max(confidence, 0.0), 1.0)
        else:
            confidence = 0.0

        return answer, sources, confidence
    except Exception as e:
        return f"Erreur Vector RAG: {str(e)[:100]}", [], 0.0


def _execute_graph_rag(query: str) -> tuple[str, list, float]:
    """Execute graph retrieval branch with Gemini-powered answer synthesis."""
    try:
        from app.services.graph_service import get_graph
        G = get_graph()
        if G is None:
            return "Le graphe n'est pas encore construit.", [], 0.1

        # Search for query keywords in graph nodes
        query_terms = [t for t in query.lower().split() if len(t) > 3]
        matching_nodes = []
        for node in G.nodes():
            node_lower = node.lower()
            for term in query_terms:
                if term in node_lower:
                    matching_nodes.append(node)
                    break

        if not matching_nodes:
            # Fallback to vector RAG with hybrid
            return _execute_vector_rag(query, retrieval_method="hybrid")

        # Get subgraph around matching nodes
        context_nodes = set(matching_nodes)
        for node in matching_nodes[:5]:
            neighbors = list(G.neighbors(node))
            context_nodes.update(neighbors[:10])

        # Build rich context text
        context_parts = []
        for node in list(context_nodes)[:20]:
            attrs = G.nodes[node]
            neighbors = list(G.neighbors(node))[:5]
            rels = []
            for n in neighbors:
                edge_data = G.edges[node, n] if G.has_edge(node, n) else {}
                rel_type = edge_data.get("type", "RELATED_TO")
                rels.append(f"{node} --[{rel_type}]--> {n}")
            node_info = f"Entity: {node} (type: {attrs.get('type', 'Concept')}, mentions: {attrs.get('count', 1)})"
            if rels:
                node_info += f"\n  Relations: {'; '.join(rels)}"
            context_parts.append(node_info)

        context = "\n".join(context_parts)
        sources = [
            {"index": i, "score": "N/A", "text": f"Graph: {node}", "method": "graph"}
            for i, node in enumerate(list(context_nodes)[:5])
        ]

        # Generate answer using Gemini
        try:
            from app.services.llm_service import generate_answer_gemini, generate_answer

            prompt = (
                "Tu es un assistant expert. Voici des informations extraites d'un graphe de connaissances.\n\n"
                "INSTRUCTIONS:\n"
                "- Réponds à la question en te basant sur les entités et relations fournies.\n"
                "- Sois précis et structuré.\n"
                "- Mentionne les entités et leurs relations quand c'est pertinent.\n"
                "- Réponds dans la même langue que la question.\n\n"
                f"GRAPHE DE CONNAISSANCES:\n{context[:2000]}\n\n"
                f"QUESTION: {query}\n\n"
                f"RÉPONSE:"
            )

            answer = generate_answer_gemini(prompt)
            if not answer:
                answer = generate_answer(prompt)
        except Exception:
            answer = context[:500]

        confidence = min(len(matching_nodes) * 0.15 + 0.3, 0.95)
        return answer, sources, confidence

    except Exception as e:
        return f"Erreur Graph RAG: {str(e)[:100]}", [], 0.0


def _compute_reward(answer: str, confidence: float, action: str) -> float:
    """Compute reward based on answer quality metrics."""
    if not answer:
        return -1.0

    answer_lower = answer.lower()

    # Strong negative signals
    if "erreur" in answer_lower or "error" in answer_lower:
        return -0.8
    if "je ne sais pas" in answer_lower or "je n'ai pas trouvé" in answer_lower:
        return -0.3
    if "information non trouvée" in answer_lower:
        return -0.2

    # Positive quality signals
    length = len(answer)

    # Length score: reward longer, more detailed answers
    if length < 50:
        length_score = 0.1
    elif length < 150:
        length_score = 0.4
    elif length < 500:
        length_score = 0.7
    else:
        length_score = 1.0

    # Confidence score
    conf_score = min(max(confidence, 0.0), 1.0)

    # Structure bonus: answers with structure indicators get a bonus
    structure_bonus = 0.0
    if any(marker in answer for marker in ["\n-", "\n•", "\n1.", "\n*", ":\n"]):
        structure_bonus = 0.1

    # Combine
    reward = conf_score * 0.5 + length_score * 0.4 + structure_bonus

    # Small penalty for fallback usage
    if action == "fallback":
        reward -= 0.2

    return round(max(-1.0, min(1.0, reward)), 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  API Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_policy() -> dict:
    """Return the current Q-table for visualization."""
    _ensure_q_table()
    policy = []
    for i, label in enumerate(STATE_LABELS):
        row = {
            "state": label,
            "state_index": i,
        }
        for j, action in enumerate(ACTIONS):
            row[action] = round(float(_q_table[i][j]), 4)
        row["best_action"] = ACTIONS[int(np.argmax(_q_table[i]))]
        policy.append(row)

    return {
        "q_table": policy,
        "states": STATE_LABELS,
        "actions": ACTIONS,
        "params": {
            "alpha": QL_ALPHA,
            "gamma": QL_GAMMA,
            "epsilon": _AGENT_EPSILON,
        },
    }


def get_rewards() -> dict:
    """Return reward history."""
    _ensure_q_table()
    return {
        "rewards": [round(r, 4) for r in _reward_history],
        "iterations": len(_reward_history),
        "avg_reward": round(np.mean(_reward_history), 4) if _reward_history else 0.0,
    }


def get_last_decision_path() -> dict:
    """Return the last decision for visualization."""
    if _decision_log:
        return {"steps": _decision_log[-1], "total_decisions": len(_decision_log)}
    return {"steps": {}, "total_decisions": 0}
