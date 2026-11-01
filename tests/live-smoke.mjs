// Manual production smoke test. Uses and deletes only its own anonymous workspace.
import assert from 'node:assert/strict';
const base = process.argv[2];
if (!base?.startsWith('https://')) throw new Error('Pass the HTTPS deployment URL.');
let cookie;
async function call(path, body, method = body ? 'POST' : 'GET') {
  const response = await fetch(base + path, { method, headers: { ...(cookie ? { Cookie: cookie } : {}), ...(body ? { 'Content-Type': 'application/json' } : {}) }, ...(body ? { body: JSON.stringify(body) } : {}) });
  const data = await response.json();
  assert.equal(response.status, 200, `${path}: ${JSON.stringify(data)}`);
  return { response, data };
}
try {
  assert.equal((await call('/health')).data.status, 'healthy');
  const start = await call('/api/session'); cookie = start.response.headers.get('Set-Cookie').split(';')[0];
  await call('/api/vectorial/upload', { text: 'Saffron grows in Taliouine in Morocco. The purple flowers are harvested in autumn. Farmers collect saffron threads by hand.' });
  await call('/api/vectorial/chunk', { method: 'recursive' });
  await call('/api/vectorial/embed', { model: 'tfidf' });
  const result = (await call('/api/query/execute', { query: 'Where does saffron grow?' })).data;
  assert.ok(result.sources.some(s => s.full_text.includes('Taliouine')));
  console.log('Answer mode:', result.generation_mode, '| sources:', result.sources.length);
  await call('/api/graph/build', {});
  assert.ok((await call('/api/graph/analytics')).data.node_count > 0);
  assert.equal((await call('/api/agent/rewards')).data.iterations, 1);
  assert.equal((await call('/api/query/history')).data.queries.length, 1);
  const embedded = (await call('/api/vectorial/embed', { model: 'cloudflare' })).data;
  assert.equal(embedded.dimension, 384);
  const neural = (await call('/api/vectorial/retrieve', { query: 'Where is saffron cultivated?', method: 'similarity' })).data;
  assert.ok(neural.results.some(row => row.full_text.includes('Taliouine')));
  assert.equal(neural.warning, null);
  console.log('PASS: live BGE Small embeddings and neural query encoding');
  const second = await fetch(base + '/api/session');
  assert.equal((await second.json()).filename, 'Welcome guide');
  const maxDocument = Array.from({ length: 1000 }, (_, i) => `Concept${i % 80} connects to Concept${(i + 1) % 80} in the source document. `).join('').slice(0, 24000);
  await call('/api/vectorial/upload', { text: maxDocument });
  await call('/api/vectorial/chunk', { method: 'recursive', chunk_size: 600, chunk_overlap: 50 });
  await call('/api/vectorial/embed', { model: 'tfidf' });
  await call('/api/graph/build', {});
  assert.ok((await call('/api/graph/analytics')).data.node_count > 0);
  console.log('PASS: maximum-size document indexing and graph analytics on production runtime');
  console.log('PASS: live health, ingestion, indexing, answer, graph, history and visitor isolation');
} finally {
  if (cookie) await call('/api/session', undefined, 'DELETE');
}
