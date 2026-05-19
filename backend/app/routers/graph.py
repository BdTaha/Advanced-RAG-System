"""
Graph RAG router — knowledge graph, Neo4j Aura, communities, analytics, Cypher.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

router = APIRouter()


class GraphBuildRequest(BaseModel):
    use_llm: Optional[bool] = True


class CypherRequest(BaseModel):
    query: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@router.post("/build")
async def build_graph(req: GraphBuildRequest):
    """Build knowledge graph from ALL vector store chunks using LLM extraction."""
    try:
        from app.services.graph_service import build_knowledge_graph
        result = build_knowledge_graph(use_llm=req.use_llm)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize")
async def get_graph_data():
    """Return graph nodes and edges for visualization."""
    try:
        from app.services.graph_service import get_graph_visualization
        return get_graph_visualization()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community-graph")
async def get_community_graph_data():
    """Return graph data with community coloring for interactive force-graph."""
    try:
        from app.services.graph_service import get_community_graph
        return get_community_graph()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communities")
async def get_communities():
    """Run Louvain community detection and return clusters."""
    try:
        from app.services.graph_service import detect_communities
        return detect_communities()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_analytics():
    """Return advanced graph metrics."""
    try:
        from app.services.graph_service import get_graph_analytics
        return get_graph_analytics()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neo4j-status")
async def neo4j_status():
    """Check Neo4j Aura connection and return stats."""
    try:
        from app.services.graph_service import get_neo4j_status
        return get_neo4j_status()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cypher")
async def run_cypher(req: CypherRequest):
    """Execute a Cypher query on Neo4j Aura."""
    try:
        from app.services.graph_service import execute_cypher
        return execute_cypher(req.query)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
