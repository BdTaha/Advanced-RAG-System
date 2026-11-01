import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { Miniflare } from 'miniflare';

let mf, db;
before(async () => {
  mf = new Miniflare({ modules: true, modulesRules: [{ type: 'ESModule', include: ['**/*.js'] }], scriptPath: resolve('worker/index.js'), compatibilityDate: '2026-08-06', d1Databases: ['DB'] });
  db = await mf.getD1Database('DB');
  const schema = await readFile('migrations/0001_init.sql', 'utf8');
  for (const statement of schema.split(';').filter(s => s.trim())) await db.prepare(statement).run();
});
after(async () => { await mf?.dispose(); });
const call = async (path, { cookie, body, method = body ? 'POST' : 'GET', origin, headers = {} } = {}) => {
  const response = await mf.dispatchFetch(`https://rag.test${path}`, { method, headers: { ...headers,
    ...(cookie ? { Cookie: cookie } : {}), ...(origin ? { Origin: origin } : {}), ...(body ? { 'Content-Type': 'application/json' } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}) });
  return { response, data: await response.json() };
};
const multipart = async (form, cookie) => {
  const request = new Request('https://rag.test/api/vectorial/upload', { method: 'POST', body: form });
  return mf.dispatchFetch(request.url, { method: 'POST', headers: { Cookie: cookie, 'Content-Type': request.headers.get('Content-Type') }, body: await request.arrayBuffer() });
};
test('isolated visitor upload → chunk → embed → query → graph → persistence → delete', async () => {
  const a = await call('/api/session'), b = await call('/api/session');
  const ca = a.response.headers.get('Set-Cookie').split(';')[0], cb = b.response.headers.get('Set-Cookie').split(';')[0];
  assert.notEqual(ca, cb);
  assert.match(a.response.headers.get('Set-Cookie'), /HttpOnly.*SameSite=Strict.*Secure/);
  const query = await call('/api/query/execute', { cookie: ca, body: { query: 'What is RAG?' } });
  assert.equal(query.response.status, 200); assert.equal(query.data.generation_mode, 'extractive');
  assert.ok(query.data.sources.length); assert.match(query.data.answer, /Source excerpts/);
  assert.equal((await call('/api/query/history', { cookie: cb })).data.queries.length, 0);
  const text = 'Saffron is cultivated in Taliouine, Morocco. Saffron flowers are harvested in autumn. The harvest is performed by hand.';
  assert.equal((await call('/api/vectorial/upload', { cookie: ca, body: { text } })).data.status, 'uploaded');
  assert.equal((await call('/api/query/execute', { cookie: ca, body: { query: 'Where is saffron cultivated?' } })).response.status, 400);
  assert.equal((await call('/api/vectorial/chunk', { cookie: ca, body: { method: 'recursive' } })).response.status, 200);
  assert.equal((await call('/api/vectorial/embed', { cookie: ca, body: { model: 'tfidf' } })).data.model, 'tfidf');
  const answer = await call('/api/query/execute', { cookie: ca, body: { query: 'Where is saffron cultivated?' } });
  assert.match(answer.data.sources[0].full_text, /Taliouine/);
  assert.equal((await call('/api/session', { cookie: cb })).data.filename, 'Welcome guide');
  assert.equal((await call('/api/query/history', { cookie: ca })).data.queries.length, 1);
  assert.equal((await call('/api/agent/rewards', { cookie: ca })).data.iterations, 1);
  assert.equal((await call('/api/graph/build', { cookie: ca, body: {} })).response.status, 200);
  assert.ok((await call('/api/graph/community-graph', { cookie: ca })).data.nodes.length);
  assert.ok((await call('/api/graph/analytics', { cookie: ca })).data.centrality.degree.length);
  const preset = 'MATCH (n:Concept) RETURN n.name, n.community, n.degree ORDER BY n.degree DESC LIMIT 20';
  assert.equal((await call('/api/graph/cypher', { cookie: ca, body: { query: preset } })).data.success, true);
  assert.equal((await call('/api/graph/cypher', { cookie: ca, body: { query: 'MATCH (n) DETACH DELETE n' } })).data.success, false);
  assert.equal((await call('/api/vectorial/embed', { cookie: ca, body: { model: 'cloudflare' } })).response.status, 503);
  assert.equal((await call('/api/vectorial/retrieve', { cookie: ca, body: { query: 'intergalactic spaceships', method: 'bm25' } })).data.generation_mode, 'no_match');
  assert.equal((await call('/api/session', { cookie: ca, method: 'DELETE' })).data.status, 'deleted');
  const oldId = ca.split('=')[1];
  assert.equal(await db.prepare('SELECT id FROM sessions WHERE id=?').bind(oldId).first(), null);
  assert.equal((await call('/api/session', { cookie: ca })).data.filename, 'Welcome guide');
});
test('validation rejects cross-origin writes, oversized documents, invalid chunks and bad JSON', async () => {
  const start = await call('/api/session'), cookie = start.response.headers.get('Set-Cookie').split(';')[0];
  assert.equal((await call('/api/vectorial/upload', { cookie, origin: 'https://evil.test', body: { text: 'hello' } })).response.status, 403);
  assert.equal((await call('/api/vectorial/upload', { cookie, body: { text: 'x'.repeat(24001) } })).response.status, 413);
  assert.equal((await call('/api/vectorial/chunk', { cookie, body: { method: 'fixed', chunk_size: 100, chunk_overlap: 100 } })).response.status, 400);
  assert.equal((await call('/api/query/execute', { cookie, body: { query: ' ' } })).response.status, 400);
  assert.equal((await call('/api/vectorial/retrieve', { cookie, body: { query: 'RAG', top_k: -1 } })).response.status, 400);
  const bad = await mf.dispatchFetch('https://rag.test/api/query/execute', { method: 'POST', body: '{', headers: { Cookie: cookie, 'Content-Type': 'application/json' } });
  assert.equal(bad.status, 400);
});
test('UTF-8 multipart upload validates filename and size without writing paths', async () => {
  const start = await call('/api/session'), cookie = start.response.headers.get('Set-Cookie').split(';')[0];
  const form = new FormData(); form.append('file', new File(['Les abeilles produisent du miel.'], '../../api.txt'));
  const upload = await multipart(form, cookie);
  assert.equal(upload.status, 200, await upload.text());
  const pdf = new FormData(); pdf.append('file', new File(['binary'], 'document.pdf'));
  assert.equal((await multipart(pdf, cookie)).status, 415);
});
