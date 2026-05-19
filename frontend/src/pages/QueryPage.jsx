import { useState } from 'react';
import { HiOutlinePaperAirplane, HiOutlineRefresh, HiOutlineClock } from 'react-icons/hi';
import { queryAPI } from '../services/api';
import { useLanguage } from '../LanguageContext';
import './QueryPage.css';

export default function QueryPage() {
  const { t } = useLanguage();
  const [query, setQuery] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [fullResult, setFullResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setFullResult(null);

    try {
      // Step 1: Analyze
      const analyzeRes = await queryAPI.analyze({ query });
      setAnalysis(analyzeRes.data);

      // Step 2: Execute
      const execRes = await queryAPI.execute({ query });
      setFullResult(execRes.data);

      // Update history
      setHistory(prev => [{
        query,
        type: execRes.data.type || analyzeRes.data.type,
        routing: execRes.data.routing || analyzeRes.data.routing,
        confidence: Math.round((execRes.data.confidence || analyzeRes.data.confidence) * 100),
      }, ...prev].slice(0, 10));
    } catch (err) {
      console.error('Query failed:', err);
      setAnalysis({
        query,
        type: 'Error',
        routing: 'N/A',
        confidence: 0,
      });
    }
    setLoading(false);
  };

  const handleReset = () => {
    setQuery('');
    setAnalysis(null);
    setFullResult(null);
  };

  const confPercent = analysis
    ? Math.round((analysis.confidence || 0) * 100)
    : 0;

  return (
    <div className="query-page">
      <div className="page-title">
        <h1>{t('query.title')}</h1>
        <span className="subtitle">{t('query.subtitle')}</span>
      </div>

      <div className="query-layout">
        {/* ── Left: Input + Analysis ───────────────────────────── */}
        <div className="query-main">
          <div className="glass-card query-input-card">
            <form onSubmit={handleSubmit} className="query-form">
              <div className="query-input-wrap">
                <input
                  id="query-input"
                  type="text"
                  className="input query-field"
                  placeholder={t('query.placeholder')}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  autoFocus
                />
                <button type="submit" className="btn btn-primary query-submit"
                  disabled={loading || !query.trim()}>
                  {loading ? <span className="spinner" /> : <HiOutlinePaperAirplane />}
                  {loading ? t('query.analyzing') : t('query.send')}
                </button>
              </div>
            </form>
          </div>

          {analysis && (
            <div className="glass-card analysis-card">
              <div className="analysis-grid">
                <div className="analysis-item">
                  <span className="analysis-label">{t('query.analysis')}</span>
                  <span className={`analysis-value type-${(analysis.type || '').toLowerCase()}`}>
                    {analysis.type}
                  </span>
                </div>
                <div className="analysis-item">
                  <span className="analysis-label">{t('query.routing')}</span>
                  <span className="analysis-value routing">{analysis.routing}</span>
                </div>
                <div className="analysis-item">
                  <span className="analysis-label">{t('query.confidence')}</span>
                  <span className="analysis-value confidence">{confPercent}%</span>
                </div>
              </div>

              <div className="routing-preview">
                <div className="routing-arrow">
                  <span className="routing-from">Query</span>
                  <div className="routing-line" />
                  <span className={`routing-to ${(analysis.routing || '').includes('Graph') ? 'graph' : 'vector'}`}>
                    {t('query.to')} {analysis.routing}
                  </span>
                </div>
              </div>

              {/* Answer */}
              {fullResult && fullResult.answer && (
                <>
                  <div className="glass-card" style={{ marginTop: 'var(--space-md)' }}>
                    <div className="glass-card-header">
                      <div className="card-icon amber">📋</div>
                      <h3>{t('query.top_doc')}</h3>
                    </div>
                    <div className="result-list">
                      {(fullResult.sources || []).slice(0, 1).map((doc, i) => (
                        <div key={i} className="doc-item" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
                          <div style={{ display: 'flex', width: '100%', justifyContent: 'space-between' }}>
                            <span className="doc-item-name" style={{ fontSize: '0.75rem' }}>
                              {t('query.document')} {typeof doc.index !== 'undefined' ? doc.index + 1 : i + 1}
                            </span>
                            {doc.score !== undefined && (
                              <span className="doc-item-score">{doc.score}</span>
                            )}
                          </div>
                          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                            {doc.text || (typeof doc === 'string' ? doc : JSON.stringify(doc))}
                          </p>
                        </div>
                      ))}
                      {(!fullResult.sources || fullResult.sources.length === 0) && (
                        <div className="empty-state" style={{ padding: 'var(--space-lg)' }}>
                          <p className="empty-state-title">{t('query.no_results')}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="glass-card" style={{ marginTop: 'var(--space-md)' }}>
                    <div className="glass-card-header">
                      <div className="card-icon emerald">📝</div>
                      <h3>{t('query.retrieved_summary')}</h3>
                    </div>
                    <div className="summary-box">
                      <p>{fullResult.answer}</p>
                    </div>
                  </div>
                </>
              )}

              <div className="analysis-actions">
                <button className="btn btn-danger btn-sm" onClick={handleReset}>
                  <HiOutlineRefresh /> {t('query.reset')}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ── Right: History ───────────────────────────────────── */}
        <div className="glass-card history-card">
          <div className="glass-card-header">
            <div className="card-icon amber"><HiOutlineClock /></div>
            <h3>{t('query.history')}</h3>
          </div>
          <div className="history-list">
            {history.length === 0 ? (
              <div className="empty-state">
                <p className="empty-state-title">{t('query.no_queries')}</p>
                <p>{t('query.ask_first')}</p>
              </div>
            ) : (
              history.map((item, i) => (
                <div key={i} className="history-item" onClick={() => setQuery(item.query)}>
                  <p className="history-query">{item.query}</p>
                  <div className="history-meta">
                    <span className={`badge ${item.type === 'Semantic' ? 'badge-indigo' : 'badge-emerald'}`}>
                      {item.type}
                    </span>
                    <span className="history-confidence">{item.confidence}%</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
