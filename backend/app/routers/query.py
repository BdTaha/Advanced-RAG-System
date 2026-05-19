"""
Query router — analyse, execute, history.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

router = APIRouter()

# ── In-memory query history ────────────────────────────────────────────────────
_query_history: list[dict] = []


class QueryAnalyzeRequest(BaseModel):
    query: str


class QueryExecuteRequest(BaseModel):
    query: str
    force_route: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────
@router.post("/analyze")
async def analyze_query(req: QueryAnalyzeRequest):
    """Classify query type and suggest routing."""
    try:
        from app.services.agent_service import extract_features, features_to_state
        from app.services.agent_service import choose_action, STATE_LABELS, ACTIONS, get_q_table

        features = extract_features(req.query)
        state = features_to_state(features)
        action_idx, action = choose_action(state, epsilon=0)  # No exploration for analysis

        q_table = get_q_table()
        q_values = {ACTIONS[i]: round(float(q_table[state][i]), 4) for i in range(len(ACTIONS))}
        confidence = max(q_values.values()) if max(q_values.values()) > 0 else 0.5

        if features["systematic_score"] > features["semantic_score"]:
            query_type = "Systematic"
        elif features["semantic_score"] > 0.3:
            query_type = "Semantic"
        else:
            query_type = "Hybrid"

        routing = "Graph RAG" if action == "use_graph" else (
            "Vectoriel RAG" if action == "use_vector" else "Fallback"
        )

        return {
            "query": req.query,
            "type": query_type,
            "routing": routing,
            "confidence": round(confidence, 3),
            "features": features,
            "state": STATE_LABELS[state],
            "q_values": q_values,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute_query(req: QueryExecuteRequest):
    """Execute query end-to-end through the agent pipeline."""
    global _query_history
    try:
        from app.services.agent_service import agent_query
        result = agent_query(req.query)

        # Add to history
        history_entry = {
            "query": req.query,
            "type": result.get("type", "Unknown"),
            "routing": result.get("routing", "Unknown"),
            "confidence": result.get("confidence", 0),
            "answer_preview": result.get("answer", "")[:150],
        }
        _query_history.insert(0, history_entry)
        _query_history = _query_history[:50]  # Keep last 50

        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_history():
    """Return recent query history."""
    return {"queries": _query_history}


@router.post("/reset")
async def reset():
    """Reset query history."""
    global _query_history
    _query_history = []
    return {"status": "reset"}
