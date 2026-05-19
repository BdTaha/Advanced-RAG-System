"""
Application configuration — loaded from .env or environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent          # backend/
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
QTABLE_PATH = DATA_DIR / "q_table.json"

# Ensure dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "PLACEHOLDER_GOOGLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "PLACEHOLDER_OPENAI_KEY")

# ── Neo4j ──────────────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "your_neo4j_aura_uri_here")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "your_username_here")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "NEO4J_PASSWORD")

# ── Document source ────────────────────────────────────────────────────────────
DOCUMENT_PATH = DATA_DIR / "8_cleaned_for_RAG.txt"

# ── Model defaults ─────────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-base"          # via Hugging Face transformers
OPENAI_LLM_MODEL = "gpt-5.5"          # via OpenAI API
GEMINI_LLM_MODEL = "gemini-2.0-flash"          # via Google AI API


# ── Q-Learning hyper-parameters ────────────────────────────────────────────────
QL_ALPHA = float(os.getenv("QL_ALPHA", "0.1"))      # learning rate
QL_GAMMA = float(os.getenv("QL_GAMMA", "0.9"))      # discount factor
QL_EPSILON = float(os.getenv("QL_EPSILON", "0.2"))   # exploration rate

# ── CORS ───────────────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
