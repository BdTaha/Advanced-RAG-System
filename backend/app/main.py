"""
FastAPI application entry-point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import CORS_ORIGINS
from app.routers import vectorial, graph, agent, query

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Platform API",
    description="Agentic-Vectorial-Graph RAG backend",
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(vectorial.router, prefix="/api/vectorial", tags=["Vectorial RAG"])
app.include_router(graph.router,     prefix="/api/graph",     tags=["Graph RAG"])
app.include_router(agent.router,     prefix="/api/agent",     tags=["Agentic RAG"])
app.include_router(query.router,     prefix="/api/query",     tags=["Query"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG Platform API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
