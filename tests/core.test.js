import test from 'node:test';
import assert from 'node:assert/strict';
import { chunkDocument, corpus, rank, buildGraph, graphAnalytics, newAgent, analyze, DEMO } from '../worker/core.js';

test('chunking rejects non-progressing overlaps and preserves trailing text', () => {
  assert.throws(() => chunkDocument('abc', 'fixed', 100, 100), /overlap/);
  assert.throws(() => chunkDocument('abc', 'fixed', -1, 0));
  const text = 'Cloudflare stores documents. '.repeat(30) + 'FINAL WORD';
  for (const method of ['fixed', 'recursive', 'sliding', 'sentence', 'paragraph', 'markdown']) {
    const result = chunkDocument(text, method, 200, 30);
    assert.ok(result.chunks.at(-1).endsWith('FINAL WORD'), method);
    assert.ok(result.chunks.every(c => c.length <= 200));
  }
});
test('TF-IDF uses one corpus vocabulary for queries and documents', () => {
  const chunks = [{ text: 'Oranges grow in Morocco.' }, { text: 'Whales live in the ocean.' }];
  assert.equal(rank('Morocco oranges', chunks, 'similarity')[0].index, 0);
  assert.equal(rank('ocean whales', chunks, 'hybrid')[0].index, 1);
  assert.deepEqual(rank('spaceship quantum', chunks, 'bm25'), []);
  assert.equal(corpus(chunks).vectors.length, 2);
});
test('co-occurrence graph and analytics handle empty, isolated and connected graphs', () => {
  assert.equal(graphAnalytics(buildGraph([])).node_count, 0);
  const g = buildGraph([{ text: 'Alice knows Bob and Carol.' }, { text: 'Bob works with David.' }]);
  assert.ok(g.links.every(e => e.type === 'CO_OCCURS'));
  const a = graphAnalytics(g);
  assert.ok(a.density > 0 && a.density <= 1);
  assert.ok(Math.abs(a.pagerank.reduce((sum, n) => sum + n.score, 0) - 1) < 0.001);
  assert.ok(g.nodes.every(n => Number.isInteger(n.community)));
});
test('routing labels match actual vector and graph actions', () => {
  assert.match(analyze('what is RAG', newAgent()).routing, /Vector RAG/);
  assert.equal(analyze('relationship between connected concepts', newAgent()).routing, 'Graph RAG');
  assert.ok(chunkDocument(DEMO).count > 1);
});
