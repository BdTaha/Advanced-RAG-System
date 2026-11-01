// Small, bounded retrieval algorithms for the Workers Free CPU allowance.
export const LIMITS = { document: 24000, chunks: 64, query: 1000, upload: 100000 };
export const DEMO = `Advanced RAG Platform — welcome guide

Retrieval augmented generation (RAG) answers questions using passages retrieved from a document. It combines retrieval with a language model and provides sources so readers can verify an answer.

Vector search represents document chunks as embeddings and ranks them by cosine similarity. TF-IDF represents words using term frequency and inverse document frequency. Cloudflare Workers AI can optionally generate neural embeddings using BGE Small. The same encoder must be used for both the document and the question.

BM25 is a lexical retrieval algorithm that scores matching words using document frequency and document length. Hybrid search combines BM25 scores with vector similarity. Lexical reranking adds query word coverage to the hybrid ranking.

A knowledge graph connects concepts that occur in the same passage. Co-occurrence is an observed textual relationship, not proof of a real-world causal or semantic relationship. Community detection groups strongly connected concepts. Degree counts a node's connections. PageRank measures network influence.

The routing agent uses eight states derived from semantic keywords, relationship keywords and query length. Its actions select graph search, BM25, vector similarity, hybrid search or lexical reranking. Q-learning updates the selected action using a retrieval-based reward. This reward is a heuristic, not human feedback or a guarantee of answer accuracy.

This public application uses Cloudflare Workers and D1. Each browser gets a separate anonymous workspace. Upload a UTF-8 TXT or Markdown document, or paste text, then chunk and index it. Workspaces expire after 30 days of inactivity. Clearing browser cookies loses access to the workspace. Use the delete workspace button to erase its stored data.

Workers AI generates an answer from the retrieved passages when available. If its free quota is unavailable, the application shows exact source excerpts with an explicit extractive-answer label. Free Cloudflare accounts have daily usage limits; this service cannot promise unlimited availability. No visitor API keys, local server, Neo4j account or paid plan are required.
`;
const STOP = new Set(('the a an and or to of in is are was were for with on at by from this that as it be can using uses use de la le les des du un une et ou à au aux en est sont pour avec dans sur par ce cette ces qui que d l il elle se son sa ses plus not no your you their its').split(' '));
export const round = n => Math.round((Number.isFinite(n) ? n : 0) * 10000) / 10000;
export function tokens(text) {
  return (text.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}_-]*/gu) || []).filter(w => w.length > 1 && !STOP.has(w));
}
export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}
export function requireText(value, max = LIMITS.query) {
  if (typeof value !== 'string' || !value.trim()) throw new ApiError(400, 'Please provide non-empty text.');
  if (value.length > max) throw new ApiError(413, `Text must be at most ${max.toLocaleString()} characters.`);
  return value.trim();
}
export const CHUNK_METHODS = ['fixed', 'sentence', 'paragraph', 'sliding', 'recursive', 'markdown'];
export function chunkDocument(text, method = 'recursive', size = 600, overlap = 50) {
  if (method === 'auto') method = 'recursive';
  if (!CHUNK_METHODS.includes(method)) throw new ApiError(400, 'Unsupported chunking method.');
  if (!Number.isInteger(size) || size < 100 || size > 2000 || !Number.isInteger(overlap) || overlap < 0 || overlap >= size) {
    throw new ApiError(400, 'Chunk size must be 100–2000; overlap must be smaller than chunk size.');
  }
  const chunks = [];
  const split = part => {
    for (let start = 0; start < part.length;) {
      let end = Math.min(start + size, part.length);
      if (method === 'recursive' && end < part.length) {
        const boundary = part.lastIndexOf(' ', end);
        if (boundary > start + size / 2) end = boundary;
      }
      const piece = part.slice(start, end).trim();
      if (piece) chunks.push(piece);
      if (end === part.length) break;
      start = Math.max(start + 1, end - overlap);
    }
  };
  if (method === 'fixed' || method === 'sliding' || method === 'recursive') split(text);
  else {
    const parts = text.split(method === 'sentence' ? /(?<=[.!?])\s+/u : method === 'markdown' ? /\n(?=#{1,6}\s)/u : /\n\s*\n/u);
    let current = '';
    for (const part of parts) {
      if (current.length + part.length > size && current) { split(current); current = ''; }
      if (part.length > size) split(part); else current += (current ? '\n' : '') + part;
    }
    if (current) split(current);
  }
  if (chunks.length > LIMITS.chunks) throw new ApiError(400, `This creates more than ${LIMITS.chunks} chunks. Increase chunk size or shorten the document.`);
  return { method, chunks, count: chunks.length, avg_chunk_size: text.length / Math.max(chunks.length, 1),
    tree: { document: `Document (${text.length} chars)`, chunks: chunks.map((c, i) => `Chunk ${i + 1} (${c.length} chars)`),
      sub_chunks: chunks.map((c, i) => ({ chunk_index: i, sub_chunks: [c] })) } };
}
export function corpus(chunks) {
  const terms = chunks.map(c => tokens(c.text ?? c));
  const df = new Map();
  for (const row of terms) for (const t of new Set(row)) df.set(t, (df.get(t) || 0) + 1);
  const idf = t => Math.log(1 + (terms.length + 1) / ((df.get(t) || 0) + 1));
  const encode = text => {
    const counts = new Map();
    for (const t of tokens(text)) counts.set(t, (counts.get(t) || 0) + 1);
    let norm = 0;
    for (const [t, count] of counts) { const v = (1 + Math.log(count)) * idf(t); counts.set(t, v); norm += v * v; }
    norm = Math.sqrt(norm) || 1;
    for (const [t, count] of counts) counts.set(t, count / norm);
    return counts;
  };
  return { terms, df, encode, vectors: chunks.map(c => encode(c.text ?? c)) };
}
export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return dot / (Math.sqrt(na * nb) || 1);
}
export function rank(query, chunks, method = 'hybrid', topK = 5, queryVector = null, graph = null) {
  if (!['similarity', 'bm25', 'hybrid', 'reranking', 'graph'].includes(method)) throw new ApiError(400, 'Unsupported retrieval method.');
  const c = corpus(chunks), q = c.encode(query), qt = [...new Set(tokens(query))];
  if (!qt.length) return [];
  const avg = c.terms.reduce((s, t) => s + t.length, 0) / (chunks.length || 1);
  const linked = new Set();
  if (graph) {
    const seeds = new Set(graph.nodes.filter(n => qt.includes(n.id)).map(n => n.id));
    for (const edge of graph.links) {
      if (seeds.has(edge.source)) linked.add(edge.target);
      if (seeds.has(edge.target)) linked.add(edge.source);
    }
  }
  const rows = chunks.map((chunk, i) => {
    let bm25 = 0, vector = 0;
    const counts = new Map();
    for (const t of c.terms[i]) counts.set(t, (counts.get(t) || 0) + 1);
    for (const t of qt) {
      const tf = counts.get(t) || 0, df = c.df.get(t) || 0;
      bm25 += Math.log(1 + (chunks.length - df + 0.5) / (df + 0.5)) * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * c.terms[i].length / (avg || 1)));
    }
    if (queryVector && chunk.vector) vector = Math.max(0, cosine(queryVector, chunk.vector));
    else for (const [t, v] of q) vector += v * (c.vectors[i].get(t) || 0);
    const coverage = qt.filter(t => counts.has(t)).length / qt.length;
    const relation = [...linked].filter(t => counts.has(t)).length / (linked.size || 1);
    return { index: i, text: chunk.text.slice(0, 300), full_text: chunk.text, bm25, vector, coverage, relation, method };
  });
  const maxBM = Math.max(...rows.map(r => r.bm25), 1e-9);
  for (const r of rows) {
    const lexical = r.bm25 / maxBM, hybrid = 0.6 * r.vector + 0.4 * lexical;
    r.score = round(method === 'bm25' ? lexical : method === 'similarity' ? r.vector : method === 'reranking' ? 0.7 * hybrid + 0.3 * r.coverage : method === 'graph' ? 0.65 * hybrid + 0.35 * r.relation : hybrid);
    r.bm25_score = round(lexical); r.vector_score = round(r.vector);
  }
  return rows.filter(r => r.score > 0.025).sort((a, b) => b.score - a.score).slice(0, topK);
}
const COLORS = ['#818cf8', '#34d399', '#fbbf24', '#fb7185', '#22d3ee', '#a78bfa'];
export function buildGraph(chunks) {
  const c = corpus(chunks);
  const vocabulary = [...c.df].sort((a, b) => b[1] - a[1]).slice(0, 40).map(([t]) => t);
  const keep = new Set(vocabulary), edges = new Map();
  // Edges explicitly represent co-occurrence in a passage.
  for (const terms of c.terms) {
    const row = [...new Set(terms.filter(t => keep.has(t)))].slice(0, 8).sort();
    for (let i = 0; i < row.length; i++) for (let j = i + 1; j < row.length; j++) {
      const key = `${row[i]}\t${row[j]}`; edges.set(key, (edges.get(key) || 0) + 1);
    }
  }
  const links = [...edges].map(([key, weight]) => { const [source, target] = key.split('\t'); return { source, target, weight, type: 'CO_OCCURS' }; });
  const adj = new Map(vocabulary.map(t => [t, []]));
  for (const e of links) { adj.get(e.source).push([e.target, e.weight]); adj.get(e.target).push([e.source, e.weight]); }
  const labels = new Map(vocabulary.map((t, i) => [t, i]));
  // Deterministic label propagation; no claim of Louvain or LLM extraction.
  for (let pass = 0; pass < 6; pass++) for (const t of vocabulary) {
    const votes = new Map();
    for (const [v, w] of adj.get(t)) votes.set(labels.get(v), (votes.get(labels.get(v)) || 0) + w);
    const winner = [...votes].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0];
    if (winner) labels.set(t, winner[0]);
  }
  const ids = [...new Set(labels.values())];
  const nodes = vocabulary.map(t => ({ id: t, label: t, name: t, type: 'Concept', count: c.df.get(t), degree: adj.get(t).length,
    community: ids.indexOf(labels.get(t)), color: COLORS[ids.indexOf(labels.get(t)) % COLORS.length], val: 4 + Math.sqrt(adj.get(t).length), size: 4 + Math.sqrt(adj.get(t).length) }));
  const byId = new Map(nodes.map(n => [n.id, n]));
  for (const e of links) e.interCommunity = byId.get(e.source).community !== byId.get(e.target).community;
  const communities = ids.map((_, id) => {
    const members = nodes.filter(n => n.community === id).sort((a, b) => b.degree - a.degree);
    const intra = links.filter(e => byId.get(e.source).community === id && !e.interCommunity).length;
    return { id, name: `Community ${id + 1}`, color: COLORS[id % COLORS.length], members: members.map(n => n.id), member_count: members.length,
      count: members.length, size: members.length, key_members: members.slice(0, 5).map(n => n.id), top_nodes: members.slice(0, 5).map(n => n.id),
      intra_edges: intra, intra_density: round(2 * intra / (members.length * (members.length - 1) || 1)) };
  });
  const n = nodes.length, m = links.length;
  const modularity = communities.reduce((s, comm) => {
    const degree = nodes.filter(n => n.community === comm.id).reduce((a, n) => a + n.degree, 0);
    return s + comm.intra_edges / (m || 1) - (degree / (2 * m || 1)) ** 2;
  }, 0);
  return { nodes, links, edges: links, communities, modularity: round(modularity), algorithm: 'weighted label propagation',
    stats: { nodeCount: n, edgeCount: m, communityCount: communities.length, density: round(2 * m / (n * (n - 1) || 1)), avgDegree: round(2 * m / (n || 1)) } };
}
export function graphAnalytics(graph) {
  const adj = new Map(graph.nodes.map(n => [n.id, []]));
  for (const e of graph.links) { adj.get(e.source).push(e.target); adj.get(e.target).push(e.source); }
  const n = graph.nodes.length, between = new Map(graph.nodes.map(v => [v.id, 0])), closeness = new Map();
  let distances = 0, pairs = 0, diameter = 0, components = 0;
  const seen = new Set();
  for (const src of adj.keys()) {
    const stack = [], queue = [src], pred = new Map([...adj.keys()].map(k => [k, []]));
    const sigma = new Map([[src, 1]]), dist = new Map([[src, 0]]);
    for (let i = 0; i < queue.length; i++) {
      const v = queue[i]; stack.push(v);
      for (const w of adj.get(v)) {
        if (!dist.has(w)) { dist.set(w, dist.get(v) + 1); queue.push(w); }
        if (dist.get(w) === dist.get(v) + 1) { sigma.set(w, (sigma.get(w) || 0) + sigma.get(v)); pred.get(w).push(v); }
      }
    }
    if (!seen.has(src)) { components++; for (const v of queue) seen.add(v); }
    const sum = [...dist.values()].reduce((s, d) => s + d, 0);
    closeness.set(src, (queue.length - 1) ** 2 / ((n - 1) * sum || 1));
    distances += sum; pairs += queue.length - 1; diameter = Math.max(diameter, ...dist.values());
    const delta = new Map();
    for (const w of stack.reverse()) {
      for (const v of pred.get(w)) delta.set(v, (delta.get(v) || 0) + sigma.get(v) / sigma.get(w) * (1 + (delta.get(w) || 0)));
      if (w !== src) between.set(w, between.get(w) + (delta.get(w) || 0));
    }
  }
  let pr = new Map([...adj.keys()].map(k => [k, 1 / (n || 1)]));
  for (let i = 0; i < 20; i++) {
    const dangling = [...adj.keys()].filter(k => !adj.get(k).length).reduce((s, k) => s + pr.get(k), 0);
    pr = new Map([...adj.keys()].map(k => [k, (0.15 + 0.85 * dangling) / (n || 1) + 0.85 * adj.get(k).reduce((s, v) => s + pr.get(v) / adj.get(v).length, 0)]));
  }
  const top = metric => graph.nodes.map(v => ({ node: v.id, score: round(metric(v)) })).sort((a, b) => b.score - a.score).slice(0, 10);
  const clustering = graph.nodes.reduce((s, v) => {
    const neighbors = adj.get(v.id); let count = 0;
    for (let i = 0; i < neighbors.length; i++) for (let j = i + 1; j < neighbors.length; j++) if (adj.get(neighbors[i]).includes(neighbors[j])) count++;
    return s + 2 * count / (neighbors.length * (neighbors.length - 1) || 1);
  }, 0) / (n || 1);
  return { density: graph.stats.density, node_count: n, edge_count: graph.links.length, avg_degree: graph.stats.avgDegree,
    avg_clustering: round(clustering), connected_components: components, diameter, avg_path_length: round(distances / (pairs || 1)),
    degree_centrality: top(v => v.degree / (n - 1 || 1)), betweenness_centrality: top(v => between.get(v.id) / ((n - 1) * (n - 2) || 1)),
    closeness_centrality: top(v => closeness.get(v.id)), pagerank: top(v => pr.get(v.id)) };
}
export const ACTIONS = ['use_graph', 'use_vector_bm25', 'use_vector_similarity', 'use_vector_hybrid', 'use_vector_rerank', 'fallback'];
export const STATES = Array.from({ length: 8 }, (_, i) => `Semantic_${i & 4 ? 'High' : 'Low'}+Systematic_${i & 2 ? 'High' : 'Low'}+${i & 1 ? 'Long' : 'Short'}`);
export function newAgent() { return { q: STATES.map((_, i) => [i & 2 ? 0.8 : 0.1, 0.5, 0.5, 0.7, 0.65, -0.5]), rewards: [], decisions: [] }; }
export function analyze(query, agent) {
  const lower = query.toLowerCase();
  const score = words => Math.min(1, words.filter(w => lower.includes(w)).length / 3);
  const semantic = score(['what', 'why', 'how', 'explain', 'define', 'comment', 'pourquoi', 'définir', 'quel']);
  const systematic = score(['relation', 'between', 'connect', 'graph', 'network', 'entre', 'lien', 'réseau']);
  const words = query.split(/\s+/).length;
  const state = (semantic > 0.3 ? 4 : 0) + (systematic > 0.3 ? 2 : 0) + (words > 8 ? 1 : 0);
  const q = agent.q[state], index = q.indexOf(Math.max(...q)), action = ACTIONS[index];
  return { query, type: systematic > semantic ? 'Systematic' : semantic > 0.3 ? 'Semantic' : 'Hybrid', state: STATES[state], state_index: state,
    action, routing: action === 'use_graph' ? 'Graph RAG' : `Vector RAG (${action.replace('use_vector_', '')})`, confidence: 0,
    features: { semantic_score: round(semantic), systematic_score: round(systematic), query_length: words > 8 ? 'long' : 'short', word_count: words },
    q_values: Object.fromEntries(ACTIONS.map((a, i) => [a, round(q[i])])) };
}
