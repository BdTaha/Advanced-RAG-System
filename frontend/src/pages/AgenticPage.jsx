import { useState, useEffect, useMemo } from 'react';
import { HiOutlineEye, HiOutlineTrendingUp, HiOutlineMap, HiOutlineDatabase, HiOutlineRefresh, HiOutlineAnnotation } from 'react-icons/hi';
import { agentAPI } from '../services/api';
import { useLanguage } from '../LanguageContext';
import './AgenticPage.css';

const ACTIONS = ['use_graph', 'use_vector_bm25', 'use_vector_similarity', 'use_vector_hybrid', 'use_vector_rerank', 'fallback'];

export default function AgenticPage() {
  const { t } = useLanguage();
  const [policy, setPolicy] = useState(null);
  const [rewards, setRewards] = useState(null);
  const [decisionPath, setDecisionPath] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [policyRes, rewardsRes, decisionRes] = await Promise.all([
        agentAPI.policy(),
        agentAPI.rewards(),
        agentAPI.decisionPath(),
      ]);
      setPolicy(policyRes.data);
      setRewards(rewardsRes.data);
      setDecisionPath(decisionRes.data);
    } catch (e) {
      console.error('Failed to load agentic data', e);
    }
    setLoading(false);
  };

  const maxReward = useMemo(() => {
    if (!rewards?.rewards?.length) return 1;
    return Math.max(...rewards.rewards.map(r => Math.abs(r)), 0.1);
  }, [rewards]);

  const qTable = policy?.q_table || [];
  const lastDecision = decisionPath?.steps || {};

  const decisionSteps = lastDecision.features ? [
    { label: 'User Query', icon: '💬', color: 'var(--accent-amber)', detail: lastDecision.query },
    { label: 'Semantic Analysis', icon: '🧠', color: 'var(--accent-indigo)', detail: `Score: ${lastDecision.features?.semantic_score}` },
    { label: 'Systematic Analysis', icon: '🔗', color: 'var(--accent-emerald)', detail: `Score: ${lastDecision.features?.systematic_score}` },
    { label: lastDecision.action === 'use_graph' ? 'Graph RAG' : lastDecision.action?.startsWith('use_vector_') ? 'Vectorial RAG' : 'Fallback',
      icon: lastDecision.action === 'use_graph' ? '🕸️' : '📊',
      color: lastDecision.action === 'use_graph' ? 'var(--accent-emerald)' : 'var(--accent-cyan)',
      detail: `Confiance: ${Math.round((lastDecision.confidence || 0) * 100)}%` },
  ] : [
    { label: 'User Query', icon: '💬', color: 'var(--accent-amber)' },
    { label: 'Semantic Analysis', icon: '🧠', color: 'var(--accent-indigo)' },
    { label: 'Systematic Analysis', icon: '🔗', color: 'var(--accent-emerald)' },
    { label: 'Vectorial RAG', icon: '📊', color: 'var(--accent-cyan)' },
  ];

  return (
    <div className="agentic-page">
      <div className="page-title">
        <h1>{t('agent.title')}</h1>
        <span className="subtitle">{t('agent.subtitle')}</span>
        <button className="btn btn-secondary btn-sm" onClick={loadData} style={{ marginLeft: 'auto' }}>
          <HiOutlineRefresh /> {t('agent.refresh')}
        </button>
      </div>

      {/* Top Row: Policy + Rewards */}
      <div className="agentic-grid-top">
        <div className="glass-card">
          <div className="glass-card-header">
            <div className="card-icon purple"><HiOutlineEye /></div>
            <h3>{t('agent.policy_viz')}</h3>
          </div>
          <p className="section-subtitle">{t('agent.policy_sub')}</p>
          <div className="policy-table-wrap">
            <table className="policy-table">
              <thead>
                <tr>
                  <th>{t('agent.state')}</th>
                  <th>{t('agent.graph')}</th>
                  <th>BM25</th><th>Similarity</th><th>Hybrid</th><th>Rerank</th>
                  <th>{t('agent.fallback')}</th>
                  <th>{t('agent.best')}</th>
                </tr>
              </thead>
              <tbody>
                {qTable.map((row, i) => (
                  <tr key={i}>
                    <td className="state-cell">{row.state?.replace(/\+/g, ' | ')?.slice(0, 30)}</td>
                    <td><span className={`q-val ${row.use_graph > 0.3 ? 'high' : ''}`}>{row.use_graph?.toFixed(3)}</span></td>
                    {ACTIONS.slice(1, 5).map(action => <td key={action}><span className={`q-val ${row[action] > 0.3 ? 'high' : ''}`}>{row[action]?.toFixed(3)}</span></td>)}
                    <td><span className={`q-val ${row.fallback > 0.3 ? 'high' : ''}`}>{row.fallback?.toFixed(3)}</span></td>
                    <td><span className="badge badge-emerald">{row.best_action}</span></td>
                  </tr>
                ))}
                {qTable.length === 0 && (
                  <tr><td colSpan="8" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                    {t('agent.empty_q')}
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
          {policy?.params && (
            <div style={{ display: 'flex', gap: 'var(--space-lg)', marginTop: 'var(--space-md)', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              <span>α = {policy.params.alpha}</span>
              <span>γ = {policy.params.gamma}</span>
              <span>ε = {policy.params.epsilon}</span>
            </div>
          )}
        </div>

        <div className="glass-card">
          <div className="glass-card-header">
            <div className="card-icon amber"><HiOutlineTrendingUp /></div>
            <h3>{t('agent.reward_monitor')}</h3>
            {rewards && <span className="badge badge-amber" style={{ marginLeft: 'auto' }}>
              {rewards.iterations} {t('agent.iter_avg')} {rewards.avg_reward}
            </span>}
          </div>
          <p className="section-subtitle">{t('agent.reward_sub')}</p>
          <div className="reward-chart">
            {(rewards?.rewards || []).slice(-50).map((r, i) => (
              <div key={i} className="reward-bar-wrap" title={`Reward: ${r}`}>
                <div className="reward-bar"
                  style={{
                    height: `${Math.max(5, (Math.abs(r) / maxReward) * 100)}%`,
                    background: r >= 0 ? 'var(--gradient-primary)' : 'var(--accent-rose)',
                  }}
                />
              </div>
            ))}
            {(!rewards?.rewards || rewards.rewards.length === 0) && (
              <div className="empty-state" style={{ width: '100%', height: '100%' }}>
                <p style={{ fontSize: '0.75rem' }}>{t('agent.no_reward')}</p>
              </div>
            )}
          </div>
          <div className="reward-axis">
            <span>0</span><span>Iterations</span><span>{rewards?.iterations || 0}</span>
          </div>
        </div>
      </div>

      {/* Decision Path */}
      <div className="glass-card decision-path-card">
        <div className="glass-card-header">
          <div className="card-icon cyan"><HiOutlineMap /></div>
          <h3>{t('agent.decision_path')}</h3>
          {lastDecision.state_label && (
            <span className="badge badge-purple" style={{ marginLeft: 'auto' }}>
              State: {lastDecision.state_label}
            </span>
          )}
        </div>
        <div className="decision-flow">
          {decisionSteps.map((step, i) => (
            <div key={i} className="decision-step-wrap">
              <div className="decision-step" style={{ borderColor: step.color }}>
                <span className="decision-icon">{step.icon}</span>
                <span className="decision-label">{step.label}</span>
                {step.detail && <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>{step.detail}</span>}
              </div>
              {i < decisionSteps.length - 1 && <div className="decision-arrow">→</div>}
            </div>
          ))}
        </div>
      </div>

      {/* Bottom Row */}
      <div className="agentic-grid-bottom">
        <div className="glass-card">
          <div className="glass-card-header">
            <div className="card-icon indigo"><HiOutlineDatabase /></div>
            <h3>{t('agent.graph_query')}</h3>
          </div>
          <div className="cypher-box">
            <code>Query concepts → co-occurring concepts → matching document passages</code>
          </div>
          <div className="mini-graph" aria-label="Illustrative routing diagram, not live graph data">
            <svg viewBox="0 0 200 120">
              {[[50,60,100,30],[100,30,150,70],[50,60,150,70],[100,30,170,100],[50,60,30,100]].map(([x1,y1,x2,y2], i) => (
                <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(129,140,248,0.3)" strokeWidth="1"/>
              ))}
              {[[50,60,'#fb7185'],[100,30,'#818cf8'],[150,70,'#34d399'],[170,100,'#fbbf24'],[30,100,'#22d3ee']].map(([x,y,c], i) => (
                <circle key={i} cx={x} cy={y} r="8" fill={c} opacity="0.8"/>
              ))}
            </svg>
          </div>
        </div>

        <div className="glass-card">
          <div className="glass-card-header">
            <div className="card-icon emerald"><HiOutlineRefresh /></div>
            <h3>{t('agent.hybrid_loop')}</h3>
          </div>
          <div className="feedback-info">
            <div className="feedback-item">
              <span className="feedback-label">Q-Table Update</span>
              <code className="feedback-code">Q(S,A) ← Q(S,A) + α[R − Q(S,A)]</code>
            </div>
            <div className="feedback-item">
              <span className="feedback-label">Learning Rate (α)</span>
              <span className="metric-value" style={{ fontSize: '1.1rem' }}>{policy?.params?.alpha || 0.1}</span>
            </div>
            <div className="feedback-item">
              <span className="feedback-label">Discount (γ)</span>
              <span className="metric-value" style={{ fontSize: '1.1rem' }}>{policy?.params?.gamma || 0.9}</span>
            </div>
            <div className="feedback-item">
              <span className="feedback-label">Exploration (ε)</span>
              <span className="metric-value" style={{ fontSize: '1.1rem' }}>{policy?.params?.epsilon ?? 0}</span>
            </div>
          </div>
        </div>

        <div className="glass-card">
          <div className="glass-card-header">
            <div className="card-icon rose"><HiOutlineAnnotation /></div>
            <h3>{t('agent.final_answer')}</h3>
          </div>
          <div className="final-answer">
            <div className="answer-text">
              <p>{lastDecision.answer_preview || t('agent.run_query_msg')}</p>
            </div>
            <div className="answer-meta">
              <div className="answer-meta-item">
                <span className="metric-label">Confiance</span>
                <span className="metric-value" style={{ fontSize: '1.5rem' }}>
                  {lastDecision.confidence ? Math.round(lastDecision.confidence * 100) : '—'}%
                </span>
              </div>
              <div className="answer-meta-item">
                <span className="metric-label">Reward</span>
                <span className="metric-value" style={{ fontSize: '1.5rem' }}>
                  {lastDecision.reward ?? '—'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
