/**
 * API client — central axios instance for all backend calls.
 */
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
  timeout: 120000,
});

// Establish the HttpOnly workspace cookie once before parallel page requests.
let ready;
api.interceptors.request.use(async config => {
  if (!ready) ready = axios.get(`${API_BASE}/api/session`, { withCredentials: true }).catch(error => { ready = null; throw error; });
  await ready;
  config.withCredentials = true;
  return config;
});

export const workspaceAPI = {
  info: () => api.get('/api/session'),
  text: text => api.post('/api/vectorial/upload', { text }),
  delete: () => api.delete('/api/session'),
};

// ── Vectorial RAG ─────────────────────────────────────────────────────────────
export const vectorialAPI = {
  upload:      (file) => { const fd = new FormData(); fd.append('file', file); return api.post('/api/vectorial/upload', fd, { headers: { 'Content-Type': 'multipart/form-data' } }); },
  chunk:       (data) => api.post('/api/vectorial/chunk', data),
  embed:       (data) => api.post('/api/vectorial/embed', data),
  buildStore:  ()     => api.post('/api/vectorial/store'),
  retrieve:    (data) => api.post('/api/vectorial/retrieve', data),
  chunkingMethods:   () => api.get('/api/vectorial/methods/chunking'),
  embeddingModels:   () => api.get('/api/vectorial/methods/embedding'),
  retrievalMethods:  () => api.get('/api/vectorial/methods/retrieval'),
  profile:           () => api.post('/api/vectorial/profile'),
};

// ── Graph RAG ─────────────────────────────────────────────────────────────────
export const graphAPI = {
  build:          (data) => api.post('/api/graph/build', data),
  visualize:      ()     => api.get('/api/graph/visualize'),
  communityGraph: ()     => api.get('/api/graph/community-graph'),
  communities:    ()     => api.get('/api/graph/communities'),
  analytics:      ()     => api.get('/api/graph/analytics'),
  neo4jStatus:    ()     => api.get('/api/graph/neo4j-status'),
  cypher:         (data) => api.post('/api/graph/cypher', data),
};

// ── Agentic RAG ───────────────────────────────────────────────────────────────
export const agentAPI = {
  query:        (data) => api.post('/api/agent/query', data),
  policy:       ()     => api.get('/api/agent/policy'),
  rewards:      ()     => api.get('/api/agent/rewards'),
  decisionPath: ()     => api.get('/api/agent/decision-path'),
};

// ── Query ─────────────────────────────────────────────────────────────────────
export const queryAPI = {
  analyze: (data) => api.post('/api/query/analyze', data),
  execute: (data) => api.post('/api/query/execute', data),
  history: ()     => api.get('/api/query/history'),
};

export default api;
