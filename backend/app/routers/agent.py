"""
Agentic RAG router — Q-Learning agent, policy, rewards.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import traceback

router = APIRouter()


class AgentQueryRequest(BaseModel):
    query: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@router.post("/query")
async def agent_query_endpoint(req: AgentQueryRequest):
    """Route a query through the Q-Learning agent."""
    try:
        from app.services.agent_service import agent_query
        return agent_query(req.query)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/policy")
async def get_policy():
    """Return current Q-Learning policy (Q-table state→action mapping)."""
    try:
        from app.services.agent_service import get_policy
        return get_policy()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rewards")
async def get_rewards():
    """Return reward history over iterations."""
    try:
        from app.services.agent_service import get_rewards
        return get_rewards()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decision-path")
async def get_decision_path():
    """Return the last decision path for visualization."""
    try:
        from app.services.agent_service import get_last_decision_path
        return get_last_decision_path()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
