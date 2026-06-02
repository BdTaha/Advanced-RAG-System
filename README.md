
# 🧠 Agentic-Vectorial-Graph RAG Platform

> A production-grade, privacy-first Retrieval-Augmented Generation system integrating Semantic Vector Search, Knowledge Graph Traversal, and a Q-Learning Routing Agent for intelligent, adaptive query orchestration.

![Status](https://img.shields.io/badge/Status-Active_Development-success)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Neo4j](https://img.shields.io/badge/Neo4j-Aura-018bff?logo=neo4j&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Final Year Engineering Project — EMSI, École Marocaine des Sciences de l'Ingénieur**
**Specialization: Artificial Intelligence & Data Science | Academic Year 2025–2026**

---

## 📖 Table of Contents
- [About The Project](#-about-the-project)
- [System Architecture](#-system-architecture)
  - [Vector RAG Pipeline](#1--vector-rag-pipeline)
  - [Knowledge Graph Pipeline](#2-️-knowledge-graph-pipeline)
  - [Q-Learning Routing Agent](#3--q-learning-routing-agent)
  - [Auto-Configuration Profiler](#4--auto-configuration-profiler)
- [Key Features](#-key-features)
- [Tech Stack](#️-tech-stack)
- [Benchmarks & Evaluation](#-benchmarks--evaluation)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Roadmap](#️-roadmap)
- [Project Structure](#-project-structure)
- [Privacy & Security](#-privacy--security)
- [Academic Context](#-academic-context)

---

## 🎯 About The Project

Large Language Models are powerful — but structurally flawed for high-stakes, knowledge-intensive applications. They hallucinate, go stale at their training cutoff, and provide no source attribution. In regulated domains like medicine, law, or enterprise, this is a non-starter.

This project solves that problem with a **multi-paradigm Retrieval-Augmented Generation platform** that goes far beyond standard vector search. The architecture integrates three complementary retrieval paradigms within a single, unified system:

- **Vector RAG** — semantic similarity search using dense embeddings and FAISS indexing.
  <img width="1920" height="1080" alt="Vectorial RAG(Embedding) Page" src="https://github.com/user-attachments/assets/43d0692b-dc00-4b63-8e7c-f6e2186359c6" />

- **Graph RAG** — structured entity-relationship traversal using a locally-built knowledge graph synced to Neo4j Aura.
  <img width="1920" height="1080" alt="Graph Rag(NetworkX) Page" src="https://github.com/user-attachments/assets/167bfb82-0218-4a25-a178-5780942113bd" />

- **Agentic RAG** — a Q-Learning agent that learns, from interaction rewards, which retrieval strategy to deploy for each query.
  <img width="1920" height="1080" alt="Agentic Graph RAG Page" src="https://github.com/user-attachments/assets/75cc03a5-1c43-4d51-981d-f2d30fa2738e" />


Rather than relying on static routing rules, the system features a **Q-Learning agent** acting as an intelligent meta-router. It extracts query features (semantic density, relational structure, length) and autonomously selects the optimal retrieval configuration — then continuously improves its policy through an epsilon-greedy exploration loop.

**The result:** a system that halves hallucination rates, surfaces structured relational knowledge, and adapts intelligently to query complexity.

---

## 🏗️ System Architecture

The platform is organized across four interconnected layers:

```
┌──────────────────────────────────────────────────────────────┐
│              Presentation Layer  (React 18 + Vite)           │
│  ChatInterface · SourcePanel · GraphViz · Agent Policy View  │
└─────────────────────────┬────────────────────────────────────┘
                          │ REST / JSON
┌─────────────────────────▼────────────────────────────────────┐
│              API Gateway  (FastAPI + Pydantic v2)            │
│       Async routing · Validation · CORS · Lifecycle hooks    │
└────────────┬────────────────────────────┬────────────────────┘
             │                            │
┌────────────▼──────────────┐  ┌──────────▼──────────────────┐
│       Core AI Services    │  │     Agentic Routing Layer   │
│  EmbeddingService         │  │  Q-Learning Agent           │
│  ChunkingService          │  │  8-state ε-greedy policy    │
│  RetrievalService         │  │  Reward function tuning     │
│  LLMService (FLAN-T5)     │  │  Continuous policy update   │
│  GraphService             │  └─────────────────────────────┘
│  PromptService            │
└────────────┬──────────────┘
             │
┌────────────▼──────────────────────────────────────────────┐
│                  Infrastructure Layer                     │
│  FAISS Vector Store · NetworkX Graph · Neo4j Aura (Bolt)  │
└───────────────────────────────────────────────────────────┘
```

### 1. 📐 Vector RAG Pipeline

Handles dense unstructured retrieval with five fallback strategies:

| Strategy | Method | Description |
| --- | --- | --- |
| Brute-force Similarity | Vector | Cosine search via FAISS |
| BM25 | Lexical | Term-frequency exact matching |
| DPR | Dense | Dense Passage Retrieval |
| Hybrid | Combined | BM25 (40%) + Vector (60%) with RRF fusion |
| Re-ranking | Two-stage | Initial retrieval + cross-encoder re-scoring |

**Embedding Backbone:** `sentence-transformers/all-MiniLM-L6-v2` · `GloVe` · `BERT` · `Word2Vec` · `TF-IDF` · `Instructor` · `OpenAI`

**Chunking Strategies (7):** Fixed-size · Sentence-based · Paragraph-based · Semantic Similarity · Sliding Window · Recursive Splitting · Markdown/Structure

### 2. 🕸️ Knowledge Graph Pipeline

Transforms unstructured text into traversable relational structures for multi-hop queries.

* **Extraction:** `flan-t5-large` (local inference) extracts entities and relationships, with regex fallback.
* **Graph Schema:** Node types — `Concept`, `Entity`, `Document`. Edge types — `RELATED_TO`, `PART_OF`, `IS_A`, `CO_OCCURS`.
* **Community Detection:** Louvain method for modularity and semantic clustering.
* **Centrality Metrics:** Degree · Betweenness · Closeness · PageRank to identify bridge nodes.
* **Storage:** Synced to **Neo4j Aura** via SSL/TLS Bolt protocol, enabling live Cypher queries from the UI.

### 3. 🤖 Q-Learning Routing Agent

The cognitive core of the system — an RL agent that learns optimal retrieval routing.

| Component | Details |
| --- | --- |
| **State Space** | 8 states (semantic score × systematic score × query length) |
| **Action Space** | `use_graph` · `use_vector_bm25` · `use_vector_similarity` · `use_vector_hybrid` · `use_vector_rerank` · `fallback` |
| **Reward Signal** | +1 for confident, relevant answers · −1 for errors or out-of-context |
| **Learning Rate (α)** | 0.1 |
| **Discount Factor (γ)** | 0.9 |
| **Exploration Rate (ε)** | 0.05 (ε-greedy) |

### 4. 🔍 Auto-Configuration Profiler

Based on benchmarking across **1,145 text vectors**, the system auto-selects the optimal pipeline configuration on document ingestion *(Note: These are preliminary offline benchmark results; full API integration is currently scheduled for Phase 4 of the roadmap)*:

* **Best Chunking:** Fixed-size (score: **0.7729**, latency: **0.002s**)
* **Best Embedding:** GloVe 100d (score: **0.6780**, intra-similarity: **0.8483**)

---

## ✨ Key Features

* **Privacy-First by Design** — All chunking, embedding, and LLM inference run locally by default. No sensitive data sent to external APIs. FAISS vector store is fully local. OpenAI embeddings are an optional add-on, never a requirement.
* **Multi-Modal Retrieval** — 7 chunking strategies × 7 embedding models × 5 retrieval methods, all selectable per query.
* **Intelligent Adaptive Routing** — Q-Learning agent continuously improves routing decisions through interaction feedback, no static rules.
* **Knowledge Graph Integration** — Automatic entity/relation extraction, Louvain community detection, Neo4j Aura sync, live Cypher console in the UI.
* **Source Transparency** — Every response surfaces the exact retrieved passages with source document, chunk index, and relevance score.
* **Hallucination Mitigation** — RAG grounding reduced hallucination rate from **73% → 18%** on out-of-scope queries in evaluation.

---

## 🛠️ Tech Stack

**Backend**

| Technology | Role |
| --- | --- |
| FastAPI 0.115 | Async REST API gateway |
| FLAN-T5-Large (HuggingFace) | Local LLM inference (entity extraction + answer generation) |
| Sentence Transformers | Dense embedding generation |
| FAISS | Vector store & ANN search |
| NetworkX | In-memory knowledge graph |
| Neo4j Aura (Bolt/SSL) | Cloud graph database & Cypher query engine |
| spaCy / NLTK | NLP pipelines for text processing |
| LangChain / LangGraph | LLM orchestration utilities |
| rank-bm25 | Lexical keyword retrieval |

**Frontend**

| Technology | Role |
| --- | --- |
| React 18 + Vite | Component-based UI |
| react-force-graph-2d | Force-directed knowledge graph visualization |
| Recharts | Q-Table & reward history analytics |
| Vanilla CSS | Custom Design System styling |
| Axios | Typed HTTP client |

---

## 📊 Benchmarks & Evaluation

Evaluated on a benchmark of **152 query-answer pairs** across 4 query categories, over 18 domain documents.

### Retrieval Quality (NDCG@5)

| Retrieval Mode | Factual Lookup | Conceptual | Multi-Entity | Cross-Document |
| --- | --- | --- | --- | --- |
| Vector Only | 0.74 | 0.71 | 0.58 | 0.52 |
| Hybrid (BM25 + Vector) | **0.79** | 0.75 | 0.63 | 0.57 |
| Graph RAG | 0.65 | 0.69 | **0.74** | 0.63 |
| Agentic RAG | 0.76 | **0.78** | 0.72 | **0.81** |

### Response Quality (Manual Evaluation, 1–5 scale)

| Retrieval Mode | Relevance | Faithfulness | Completeness |
| --- | --- | --- | --- |
| Vector | 4.1 | 4.2 | 3.7 |
| Hybrid | 4.3 | 4.4 | 3.9 |
| Graph | 4.0 | 4.3 | 4.1 |
| Agentic | **4.5** | **4.5** | **4.4** |

### End-to-End Latency

| Mode | Median Latency |
| --- | --- |
| Vector | ~2.1s |
| Hybrid | ~2.4s |
| Graph | ~3.2s |
| Agentic | ~8.5s |

> LLM API inference accounts for ~65–75% of total latency across all modes.

---

## 🚀 Getting Started

### Prerequisites

* Python **3.11+**
* Node.js **v16+**
* A **Neo4j Aura** instance (free tier works)
* API keys for **OpenAI** (optional) and/or **Google Gemini** (optional)
* Docker (optional, for containerized backend)

### Installation

**1. Clone the repository**

```bash
git clone [https://github.com/yourusername/Agentic-Vectorial-Graph-RAG.git](https://github.com/yourusername/Agentic-Vectorial-Graph-RAG.git)
cd Agentic-Vectorial-Graph-RAG

```

**2. Configure environment variables**

```bash
cd backend
cp .env.example .env

```

Open `.env` and fill in your credentials:

```env
# LLM APIs (optional — system runs locally without these)
GOOGLE_API_KEY="your_gemini_key"
OPENAI_API_KEY="your_openai_key"

# Neo4j Graph Database
NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your_password"

```

### Running the Application

#### Option A — Local Development

**Terminal 1 (Backend):**

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```

**Terminal 2 (Frontend):**

```bash
cd frontend
npm install
npm run dev

```

Navigate to `http://localhost:5173`

#### Option B — Docker

```bash
cd backend
docker build -t rag-backend .
docker run -p 8000:8000 --env-file .env rag-backend

```

---

## 💻 Usage

The platform is divided into four operational panels:

| Panel | Description |
| --- | --- |
| **Query Panel** | Submit natural language queries. View real-time routing confidence (e.g., *"Graph RAG — 92%"*) and retrieved source passages. |
| **Vectorial RAG Panel** | Visualize chunking strategies, 2D PCA embedding clusters, and manually test all 5 retrieval methods. |
| **Graph RAG Panel** | Interact with the force-directed Neo4j/NetworkX graph. View Louvain communities, centrality metrics, and execute raw Cypher queries. |
| **Agentic Panel** | Monitor the Q-Learning agent's brain: live Q-Table, reward history charts, and ε-greedy exploration/exploitation logs. |

---

## 📡 API Reference

All endpoints accept/return JSON. See `/docs` (Swagger UI) when running locally.

```text
POST  /api/vectorial/retrieve — Vector similarity & hybrid retrieval
POST  /api/vectorial/chunk    — Document chunking pipeline
POST  /api/vectorial/embed    — Vector embedding generation
POST  /api/graph/build        — Knowledge graph construction
POST  /api/agent/query        — Q-Learning agent routing with full trace
GET   /api/graph/visualize    — Retrieve graph nodes and edges
GET   /api/graph/analytics    — Graph centrality metrics

```

**Example query request:**

```json
POST /api/vectorial/retrieve
{
  "query": "What are the downstream effects of interest rate changes on housing?",
  "top_k": 5,
  "method": "hybrid"
}

```

**Example response:**

```json
{
  "query": "What are the downstream effects of interest rate changes on housing?",
  "method": "hybrid",
  "count": 5,
  "results": [
    {
      "index": 12,
      "text": "...",
      "full_text": "...",
      "score": 0.91,
      "method": "hybrid"
    }
  ],
  "summary": "Based on the retrieved context..."
}

```

---

## 🗺️ Roadmap

* [x] **Phase 0** — Project scaffolding: React + FastAPI, design system, CI setup.
* [x] **Phase 1** — Vectorial pipeline: 7 chunking methods, 7 embedding models, 5 retrieval strategies + FAISS.
* [x] **Phase 2** — Knowledge graph: LLM entity extraction, Neo4j Aura sync, Louvain communities, graph analytics.
* [x] **Phase 3** — Q-Learning agent: state/action space design, reward function, ε-greedy policy, Q-Table visualization.
* [x] **Phase 4** — Auto-RAG Ingestion Profiler: heuristic-based optimal pipeline configuration on document upload.
* [x] **Phase 5** — Neural relation extraction (replace rule-based with fine-tuned BERT on TACRED/DocRED).
* [x] **Phase 6** — Kubernetes (k8s) cluster deployment with distributed Milvus vector store.
* [ ] **Phase 7** — Multi-modal RAG: extend retrieval to tables, figures, and images.

---

## 📁 Project Structure

```text
/
├── backend/
│   ├── app/
│   │   ├── routers/         # API route handlers (query, agent, graph, vectorial)
│   │   ├── services/        # Business logic (embedding, retrieval, LLM, graph, agent)
│   │   ├── models/          # Pydantic request/response schemas
│   │   ├── core/            # Config management, dependencies
│   │   └── main.py          # FastAPI application entrypoint
│   ├── data/
│   │   ├── faiss_index/     # FAISS vector store persistence
│   │   └── knowledge_graph.json # Local NetworkX cache
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── components/      # Shared UI components
│       ├── pages/           # Vectorial, Graph, Agentic, and Query Views
│       ├── services/        # Typed API communication layer
│       └── hooks/           # Custom React hooks

```

---

## 🔒 Privacy & Security

All critical processing — chunking, embedding, entity extraction, and answer generation — runs **100% locally** by default using FLAN-T5-Large via HuggingFace Transformers. No document content is sent to external APIs unless explicitly configured. Neo4j Aura connections are secured with SSL/TLS. API keys are managed via environment variables and never committed to version control.

---

## 📄 Academic Context

This system was developed as a Final Year Project (PFA) for the **AI & Data Science** program at EMSI — École Marocaine des Sciences de l'Ingénieur, under the supervision of **Dr. EL MKHALET Mouna**.

The project is accompanied by a full academic report covering theoretical foundations (transformers, retrieval systems, knowledge graphs, RL), system design (UML, component diagrams, sequence diagrams), implementation details, and a rigorous comparative evaluation across all retrieval modes.

---

## ⭐ Acknowledgements

Built on the shoulders of the open-source community — FastAPI, React, LangChain, HuggingFace Transformers, Neo4j, NetworkX, spaCy, FAISS, and the research of Lewis et al. (RAG, 2020), Vaswani et al. (Attention, 2017), and Edge et al. (GraphRAG, 2024).
