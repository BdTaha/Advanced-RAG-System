import { useState, useEffect } from 'react';
import { HiOutlineScissors, HiOutlineChip, HiOutlineSearch, HiOutlinePlay, HiOutlineRefresh } from 'react-icons/hi';
import { vectorialAPI } from '../services/api';
import { useLanguage } from '../LanguageContext';
import './VectorialPage.css';

const CHUNKING_METHODS = [
  { id: 'auto',      name: 'Auto-Sélection (Profiler)', icon: '✨' },
  { id: 'fixed',     name: 'Taille Fixe',          icon: '📐' },
  { id: 'sentence',  name: 'Par Phrases',           icon: '📝' },
  { id: 'paragraph', name: 'Par Paragraphes',       icon: '📄' },
  { id: 'semantic',  name: 'Similarité Sémantique', icon: '🧠' },
  { id: 'sliding',   name: 'Fenêtre Glissante',     icon: '🪟' },
  { id: 'recursive', name: 'Récursif',              icon: '🔄' },
  { id: 'markdown',  name: 'Structurel',            icon: '📑' },
];

const EMBEDDING_MODELS = [
  { id: 'auto',       name: 'Auto-Sélection (Profiler)', color: 'slate' },
  { id: 'tfidf',      name: 'TF-IDF',         color: 'indigo' },
  { id: 'word2vec',   name: 'Word2Vec',        color: 'blue' },
  { id: 'glove',      name: 'GloVe',           color: 'cyan' },
  { id: 'bert',       name: 'BERT',            color: 'emerald' },
  { id: 'sbert',      name: 'Sentence-BERT',   color: 'amber' },
  { id: 'openai',     name: 'OpenAI',          color: 'rose' },
  { id: 'instructor', name: 'Instructor',      color: 'purple' },
];

const RETRIEVAL_METHODS = [
  { id: 'auto',       name: 'Auto-Sélection (Agentic)', desc: 'Routage intelligent' },
  { id: 'similarity', name: 'Similarité Brute',  desc: 'Cosine similarity directe' },
  { id: 'bm25',       name: 'BM25',              desc: 'Recherche lexicale' },
  { id: 'dpr',        name: 'DPR',               desc: 'Dense Passage Retrieval' },
  { id: 'hybrid',     name: 'Hybride',           desc: 'Lexical + Vectoriel' },
  { id: 'reranking',  name: 'Re-ranking',        desc: 'Cross-encoder re-ranking' },
];

const CLUSTER_COLORS = ['var(--accent-indigo)', 'var(--accent-emerald)', 'var(--accent-amber)', 'var(--accent-rose)', 'var(--accent-cyan)'];

// Module level cache to persist state across unmounts
let cachedActiveSection = 'chunking';
let cachedSelectedChunking = 'auto';
let cachedSelectedEmbedding = 'auto';
let cachedSelectedRetrieval = 'auto';
let cachedChunkResult = null;
let cachedEmbedResult = null;
let cachedRetrieveResult = null;
let cachedRetrieveQuery = '';

export default function VectorialPage() {
  const { t } = useLanguage();
  const [activeSection, setActiveSection] = useState(cachedActiveSection);
  const [selectedChunking, setSelectedChunking] = useState(cachedSelectedChunking);
  const [selectedEmbedding, setSelectedEmbedding] = useState(cachedSelectedEmbedding);
  const [selectedRetrieval, setSelectedRetrieval] = useState(cachedSelectedRetrieval);

  // Data states
  const [chunkResult, setChunkResult] = useState(cachedChunkResult);
  const [embedResult, setEmbedResult] = useState(cachedEmbedResult);
  const [retrieveResult, setRetrieveResult] = useState(cachedRetrieveResult);
  const [retrieveQuery, setRetrieveQuery] = useState(cachedRetrieveQuery);

  // Sync state to cache
  useEffect(() => { cachedActiveSection = activeSection; }, [activeSection]);
  useEffect(() => { cachedSelectedChunking = selectedChunking; }, [selectedChunking]);
  useEffect(() => { cachedSelectedEmbedding = selectedEmbedding; }, [selectedEmbedding]);
  useEffect(() => { cachedSelectedRetrieval = selectedRetrieval; }, [selectedRetrieval]);
  useEffect(() => { cachedChunkResult = chunkResult; }, [chunkResult]);
  useEffect(() => { cachedEmbedResult = embedResult; }, [embedResult]);
  useEffect(() => { cachedRetrieveResult = retrieveResult; }, [retrieveResult]);
  useEffect(() => { cachedRetrieveQuery = retrieveQuery; }, [retrieveQuery]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  // ── Actions ──────────────────────────────────────────────
  const runChunking = async () => {
    setLoading(true);
    setStatus('Chunking en cours...');
    try {
      const res = await vectorialAPI.chunk({ method: selectedChunking, chunk_size: 500, chunk_overlap: 50 });
      setChunkResult(res.data);
      setStatus(`✅ ${res.data.count} chunks créés (méthode: ${res.data.method})`);
    } catch (e) {
      setStatus(`❌ Erreur: ${e.response?.data?.detail || e.message}`);
    }
    setLoading(false);
  };

  const runEmbedding = async () => {
    setLoading(true);
    setStatus('Génération des embeddings... (peut prendre ~30s)');
    try {
      const res = await vectorialAPI.embed({ model: selectedEmbedding });
      setEmbedResult(res.data);
      setStatus(`✅ ${res.data.count} embeddings (dim: ${res.data.dimension}, model: ${res.data.model})`);
    } catch (e) {
      setStatus(`❌ Erreur: ${e.response?.data?.detail || e.message}`);
    }
    setLoading(false);
  };

  const runRetrieval = async () => {
    if (!retrieveQuery.trim()) return;
    setLoading(true);
    setStatus('Recherche en cours...');
    try {
      const res = await vectorialAPI.retrieve({ query: retrieveQuery, method: selectedRetrieval, top_k: 5 });
      setRetrieveResult(res.data);
      setStatus(`✅ ${res.data.count} résultats trouvés`);
    } catch (e) {
      setStatus(`❌ Erreur: ${e.response?.data?.detail || e.message}`);
    }
    setLoading(false);
  };

  return (
    <div className="vectorial-page">
      <div className="page-title">
        <h1>{t('vec.title')}</h1>
        <span className="subtitle">{t('vec.subtitle')}</span>
        {status && <span className="badge badge-indigo" style={{ marginLeft: 'auto' }}>{status}</span>}
      </div>

      <div className="sub-tabs">
        <button className={`sub-tab ${activeSection === 'chunking' ? 'active' : ''}`}
          onClick={() => setActiveSection('chunking')}>
          <HiOutlineScissors style={{ marginRight: 4 }} /> {t('vec.tab_chunk')}
        </button>
        <button className={`sub-tab ${activeSection === 'embeddings' ? 'active' : ''}`}
          onClick={() => setActiveSection('embeddings')}>
          <HiOutlineChip style={{ marginRight: 4 }} /> {t('vec.tab_embed')}
        </button>
        <button className={`sub-tab ${activeSection === 'retrieval' ? 'active' : ''}`}
          onClick={() => setActiveSection('retrieval')}>
          <HiOutlineSearch style={{ marginRight: 4 }} /> {t('vec.tab_retrieval')}
        </button>
      </div>

      {/* ── Chunking Panel ──────────────────────────────────── */}
      {activeSection === 'chunking' && (
        <div className="panel-grid page-enter">
          <div className="glass-card">
            <div className="glass-card-header">
              <div className="card-icon indigo"><HiOutlineScissors /></div>
              <h3>{t('vec.chunk_methods')}</h3>
            </div>
            <div className="method-list">
              {CHUNKING_METHODS.map((m) => (
                <button key={m.id}
                  className={`method-item ${selectedChunking === m.id ? 'selected' : ''}`}
                  onClick={() => setSelectedChunking(m.id)}>
                  <span className="method-icon">{m.icon}</span>
                  <span className="method-name">{m.name}</span>
                  {selectedChunking === m.id && <span className="badge badge-indigo">{t('vec.selected')}</span>}
                </button>
              ))}
            </div>
            {selectedChunking === 'auto' ? (
              <button className="btn btn-primary" onClick={runChunking} disabled={loading}
                style={{ marginTop: 'var(--space-md)', width: '100%', background: 'linear-gradient(45deg, #8b5cf6, #3b82f6)', border: 'none' }}>
                {loading ? <span className="spinner" /> : <HiOutlinePlay />} {t('vec.run_auto')}
              </button>
            ) : (
              <button className="btn btn-primary" onClick={runChunking} disabled={loading}
                style={{ marginTop: 'var(--space-md)', width: '100%' }}>
                {loading ? <span className="spinner" /> : <HiOutlinePlay />} {t('vec.run_chunk')}
              </button>
            )}
          </div>

          <div className="glass-card span-2">
            <div className="glass-card-header">
              <div className="card-icon cyan">📊</div>
              <h3>{t('vec.chunk_viz')}</h3>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
                {chunkResult && (
                  <span className="badge badge-emerald">
                    {chunkResult.count} chunks
                  </span>
                )}
                <button 
                  className="btn-icon" 
                  onClick={runChunking} 
                  disabled={loading} 
                  title={t('vec.reload')}
                  style={{ padding: '6px', background: 'transparent', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', cursor: 'pointer', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s' }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-primary)'; e.currentTarget.style.borderColor = 'var(--accent-cyan)'; e.currentTarget.style.background = 'rgba(34, 211, 238, 0.1)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.borderColor = 'var(--border-subtle)'; e.currentTarget.style.background = 'transparent'; }}
                >
                  <HiOutlineRefresh size={18} />
                </button>
              </div>
            </div>
            {chunkResult ? (
              <>
                <div className="chunk-tree">
                  <div className="chunk-node root">
                    <span>📄 {chunkResult.tree?.document || 'Document'}</span>
                  </div>
                  <div className="chunk-connector" />
                  <div className="chunk-level">
                    {chunkResult.chunks.slice(0, 6).map((c, i) => (
                      <div key={i} className="chunk-node chunk" title={c.slice(0, 200)}>
                        Chunk {i + 1} ({c.length} chars)
                      </div>
                    ))}
                    {chunkResult.count > 6 && (
                      <div className="chunk-node chunk" style={{ opacity: 0.5 }}>
                        +{chunkResult.count - 6} more...
                      </div>
                    )}
                  </div>
                </div>
                {chunkResult.auto_selected && chunkResult.profiler && (
                  <div className="auto-select-note" style={{ marginTop: 'var(--space-sm)', background: 'rgba(139, 92, 246, 0.1)', border: '1px solid var(--accent-purple)' }}>
                    <span className="badge badge-purple">✨ {t('vec.profiler_active')}</span>
                    {t('vec.winning_method')} <strong>{chunkResult.method}</strong> (Score: {chunkResult.profiler.best_score})
                  </div>
                )}
                <div className="auto-select-note">
                  <span className="badge badge-emerald">✓ {t('vec.stats')}</span>
                  {t('vec.avg')} {Math.round(chunkResult.avg_chunk_size)} {t('vec.chars')} — {t('vec.method')} {chunkResult.method}
                </div>
              </>
            ) : (
              <div className="viz-placeholder">
                <span className="viz-placeholder-icon">📄</span>
                <p>{t('vec.select_method_msg')}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Embeddings Panel ────────────────────────────────── */}
      {activeSection === 'embeddings' && (
        <div className="panel-grid page-enter">
          <div className="glass-card">
            <div className="glass-card-header">
              <div className="card-icon emerald"><HiOutlineChip /></div>
              <h3>{t('vec.embed_models')}</h3>
            </div>
            <div className="model-grid">
              {EMBEDDING_MODELS.map((m) => (
                <button key={m.id}
                  className={`model-chip ${selectedEmbedding === m.id ? 'selected' : ''}`}
                  onClick={() => setSelectedEmbedding(m.id)}>
                  <span className={`model-dot ${m.color}`} />
                  {m.name}
                </button>
              ))}
            </div>
            {selectedEmbedding === 'auto' ? (
              <button className="btn btn-primary" onClick={runEmbedding} disabled={loading}
                style={{ marginTop: 'var(--space-md)', width: '100%', background: 'linear-gradient(45deg, #8b5cf6, #3b82f6)', border: 'none' }}>
                {loading ? <span className="spinner" /> : <HiOutlinePlay />} {t('vec.run_auto')}
              </button>
            ) : (
              <button className="btn btn-primary" onClick={runEmbedding} disabled={loading}
                style={{ marginTop: 'var(--space-md)', width: '100%' }}>
                {loading ? <span className="spinner" /> : <HiOutlinePlay />} {t('vec.run_embed')}
              </button>
            )}
          </div>

          <div className="glass-card span-2">
            <div className="glass-card-header">
              <div className="card-icon blue">📈</div>
              <h3>{t('vec.pca')}</h3>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
                {embedResult && (
                  <span className="badge badge-cyan">
                    {embedResult.count} {t('vec.points_dim')} {embedResult.dimension}
                  </span>
                )}
                <button 
                  className="btn-icon" 
                  onClick={runEmbedding} 
                  disabled={loading} 
                  title={t('vec.reload')}
                  style={{ padding: '6px', background: 'transparent', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', cursor: 'pointer', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s' }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-primary)'; e.currentTarget.style.borderColor = 'var(--accent-blue)'; e.currentTarget.style.background = 'rgba(59, 130, 246, 0.1)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.borderColor = 'var(--border-subtle)'; e.currentTarget.style.background = 'transparent'; }}
                >
                  <HiOutlineRefresh size={18} />
                </button>
              </div>
            </div>
            <div className="pca-container">
              <div className="pca-plot">
                {embedResult && embedResult.embeddings_2d ? (
                  (() => {
                    const pts = embedResult.embeddings_2d;
                    const xs = pts.map(p => p.x);
                    const ys = pts.map(p => p.y);
                    const xMin = Math.min(...xs), xMax = Math.max(...xs);
                    const yMin = Math.min(...ys), yMax = Math.max(...ys);
                    const xRange = xMax - xMin || 1;
                    const yRange = yMax - yMin || 1;

                    return pts.slice(0, 300).map((p, i) => (
                      <div key={i} className="pca-point"
                        style={{
                          left: `${5 + ((p.x - xMin) / xRange) * 90}%`,
                          top: `${5 + ((p.y - yMin) / yRange) * 90}%`,
                          background: CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length],
                          animationDelay: `${i * 5}ms`,
                        }}
                        title={`Chunk ${p.index}, Cluster ${p.cluster}`}
                      />
                    ));
                  })()
                ) : (
                  Array.from({ length: 60 }, (_, i) => {
                    const cluster = i % 3;
                    const cx = cluster === 0 ? 30 : cluster === 1 ? 60 : 45;
                    const cy = cluster === 0 ? 40 : cluster === 1 ? 35 : 70;
                    return (
                      <div key={i} className="pca-point"
                        style={{
                          left: `${cx + (Math.random() - 0.5) * 25}%`,
                          top: `${cy + (Math.random() - 0.5) * 20}%`,
                          background: CLUSTER_COLORS[cluster],
                          animationDelay: `${i * 20}ms`,
                        }}
                      />
                    );
                  })
                )}
                <div className="pca-axis-x">PC1</div>
                <div className="pca-axis-y">PC2</div>
              </div>
              <div className="pca-legend">
                {CLUSTER_COLORS.map((c, i) => (
                  <span key={i}><span className="legend-dot" style={{ background: c }} /> Cluster {i + 1}</span>
                ))}
              </div>
              {embedResult?.auto_selected && embedResult.profiler && (
                <div className="auto-select-note" style={{ marginTop: 'var(--space-md)', background: 'rgba(139, 92, 246, 0.1)', border: '1px solid var(--accent-purple)' }}>
                  <span className="badge badge-purple">✨ {t('vec.profiler_active')}</span>
                  {t('vec.winning_model')} <strong>{embedResult.model}</strong> (Score: {embedResult.profiler.best_score})
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Retrieval Panel ─────────────────────────────────── */}
      {activeSection === 'retrieval' && (
        <div className="panel-grid page-enter">
          <div className="glass-card">
            <div className="glass-card-header">
              <div className="card-icon purple"><HiOutlineSearch /></div>
              <h3>{t('vec.retrieval_methods')}</h3>
            </div>
            <div className="method-list">
              {RETRIEVAL_METHODS.map((m) => (
                <button key={m.id}
                  className={`method-item ${selectedRetrieval === m.id ? 'selected' : ''}`}
                  onClick={() => setSelectedRetrieval(m.id)}>
                  <div>
                    <span className="method-name">{m.name}</span>
                    <span className="method-desc">{m.desc}</span>
                  </div>
                  {selectedRetrieval === m.id && <span className="badge badge-purple">{t('vec.active')}</span>}
                </button>
              ))}
            </div>
            <div style={{ marginTop: 'var(--space-md)' }}>
              <input className="input" placeholder={t('vec.test_query_ph')}
                value={retrieveQuery} onChange={(e) => setRetrieveQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && runRetrieval()} />
              <button className="btn btn-primary" onClick={runRetrieval} disabled={loading || !retrieveQuery.trim()}
                style={{ marginTop: 'var(--space-sm)', width: '100%' }}>
                {loading ? <span className="spinner" /> : <HiOutlineSearch />} {t('vec.search')}
              </button>
            </div>
          </div>

          <div className="glass-card">
            <div className="glass-card-header">
              <div className="card-icon amber">📋</div>
              <h3>{t('vec.relevant_docs')}</h3>
              {retrieveResult && (
                <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
                  {retrieveResult.method === 'auto' && retrieveResult.results?.[0]?.auto_selected_method && (
                    <span className="badge badge-purple">
                      ✨ Agentic: {retrieveResult.results[0].auto_selected_method}
                    </span>
                  )}
                  <span className="badge badge-cyan">
                    {retrieveResult.method}
                  </span>
                </div>
              )}
            </div>
            <div className="result-list">
              {(retrieveResult?.results || []).map((doc, i) => (
                <div key={i} className="doc-item" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
                  <div style={{ display: 'flex', width: '100%', justifyContent: 'space-between' }}>
                    <span className="doc-item-name" style={{ fontSize: '0.75rem' }}>
                      {t('query.document')} {doc.index + 1}
                    </span>
                    <span className="doc-item-score">{doc.score}</span>
                  </div>
                  <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                    {doc.text}
                  </p>
                </div>
              ))}
              {!retrieveResult && (
                <div className="empty-state" style={{ padding: 'var(--space-lg)' }}>
                  <p className="empty-state-title">{t('query.no_results')}</p>
                  <p>{t('vec.run_search_msg')}</p>
                </div>
              )}
            </div>
          </div>

          <div className="glass-card">
            <div className="glass-card-header">
              <div className="card-icon emerald">📝</div>
              <h3>{t('query.retrieved_summary')}</h3>
            </div>
            <div className="summary-box">
              <p>{retrieveResult?.summary || t('vec.summary_placeholder')}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
