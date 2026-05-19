import { useState, useRef, useCallback, useEffect } from 'react';
import { HiOutlineGlobe, HiOutlineUserGroup, HiOutlineChartBar, HiOutlinePlay, HiOutlineDatabase, HiOutlineCode } from 'react-icons/hi';
import ForceGraph2D from 'react-force-graph-2d';
import { graphAPI } from '../services/api';
import { useLanguage } from '../LanguageContext';
import './GraphPage.css';

export default function GraphPage() {
  const { t } = useLanguage();
  const [activeSection, setActiveSection] = useState('network');
  const [graphData, setGraphData] = useState(null);
  const [communityData, setCommunityData] = useState(null);
  const [communities, setCommunities] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [neo4jStatus, setNeo4jStatus] = useState(null);
  const [cypherQuery, setCypherQuery] = useState('MATCH (n:Concept) RETURN n.name, n.community, n.degree ORDER BY n.degree DESC LIMIT 20');
  const [cypherResult, setCypherResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedComm, setSelectedComm] = useState(null);
  const fgRef = useRef();

  const buildGraph = async () => {
    setLoading(true);
    setStatus('Construction du graphe avec flan-t5-base sur TOUS les chunks...');
    try {
      const res = await graphAPI.build({ use_llm: true });
      setStatus(`✅ ${res.data.nodes} nœuds, ${res.data.edges} arêtes — ${res.data.communities_detected} communautés — Neo4j: ${res.data.neo4j}`);
      await loadCommunityGraph();
    } catch (e) {
      setStatus(`❌ ${e.response?.data?.detail || e.message}`);
    }
    setLoading(false);
  };

  const loadCommunityGraph = async () => {
    try {
      const res = await graphAPI.communityGraph();
      setCommunityData(res.data);
    } catch (e) { console.error('community graph load failed', e); }
  };

  const loadGraphData = async () => {
    try {
      const res = await graphAPI.visualize();
      setGraphData(res.data);
    } catch (e) { console.error('graph data load failed', e); }
  };

  const loadCommunities = async () => {
    setLoading(true);
    try {
      const res = await graphAPI.communities();
      setCommunities(res.data);
    } catch (e) { setStatus(`❌ ${e.response?.data?.detail || e.message}`); }
    setLoading(false);
  };

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const res = await graphAPI.analytics();
      setAnalytics(res.data);
    } catch (e) { setStatus(`❌ ${e.response?.data?.detail || e.message}`); }
    setLoading(false);
  };

  const loadNeo4j = async () => {
    try {
      const res = await graphAPI.neo4jStatus();
      setNeo4jStatus(res.data);
    } catch (e) { console.error('neo4j status failed', e); }
  };

  const runCypher = async () => {
    setLoading(true);
    try {
      const res = await graphAPI.cypher({ query: cypherQuery });
      setCypherResult(res.data);
    } catch (e) { setCypherResult({ success: false, error: e.message }); }
    setLoading(false);
  };

  const nodeCanvasObject = useCallback((node, ctx, globalScale) => {
    const label = node.label || node.id;
    const size = node.val || 6;
    const fontSize = Math.max(10 / globalScale, 2.5);
    const isHovered = hoveredNode === node.id;
    const dimmed = selectedComm !== null && node.community !== selectedComm;

    ctx.beginPath();
    ctx.arc(node.x, node.y, size + (isHovered ? 3 : 0), 0, 2 * Math.PI);
    ctx.fillStyle = dimmed ? 'rgba(100,116,139,0.2)' : (node.color || '#818cf8');
    ctx.globalAlpha = dimmed ? 0.3 : (isHovered ? 1 : 0.85);
    ctx.fill();

    if (isHovered) {
      ctx.strokeStyle = node.color || '#818cf8';
      ctx.lineWidth = 2 / globalScale;
      ctx.shadowColor = node.color || '#818cf8';
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    if (globalScale > 1.2 || isHovered || size > 10) {
      ctx.globalAlpha = dimmed ? 0.2 : 0.9;
      ctx.font = `${isHovered ? 'bold ' : ''}${fontSize}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = '#e2e8f0';
      const displayLabel = label.length > 18 ? label.slice(0, 16) + '…' : label;
      ctx.fillText(displayLabel, node.x, node.y + size + 3);
    }
    ctx.globalAlpha = 1;
  }, [hoveredNode, selectedComm]);

  const linkCanvasObject = useCallback((link, ctx) => {
    const dimmed = selectedComm !== null &&
      (link.source?.community ?? link.source) !== selectedComm &&
      (link.target?.community ?? link.target) !== selectedComm;
    ctx.beginPath();
    ctx.moveTo(link.source.x, link.source.y);
    ctx.lineTo(link.target.x, link.target.y);
    ctx.strokeStyle = dimmed ? 'rgba(100,116,139,0.03)' :
      link.interCommunity ? 'rgba(100,116,139,0.08)' : 'rgba(129,140,248,0.12)';
    ctx.lineWidth = dimmed ? 0.3 : (link.weight > 1 ? 1.5 : 0.8);
    ctx.stroke();
  }, [selectedComm]);

  const sections = [
    { id: 'network', label: t('graph.tab_network'), icon: <HiOutlineGlobe /> },
    { id: 'communities', label: t('graph.tab_comm'), icon: <HiOutlineUserGroup /> },
    { id: 'neo4j', label: t('graph.tab_neo4j'), icon: <HiOutlineDatabase /> },
    { id: 'analytics', label: t('graph.tab_analytics'), icon: <HiOutlineChartBar /> },
  ];

  return (
    <div className="graph-page">
      <div className="page-title">
        <h1>{t('graph.title')}</h1>
        <span className="subtitle">{t('graph.subtitle')}</span>
        {status && <span className="badge badge-indigo gp-status-badge">{status}</span>}
      </div>

      <div className="sub-tabs">
        {sections.map(s => (
          <button key={s.id} className={`sub-tab ${activeSection === s.id ? 'active' : ''}`}
            onClick={() => {
              setActiveSection(s.id);
              if (s.id === 'network' && !communityData) loadCommunityGraph();
              if (s.id === 'communities' && !communities) loadCommunities();
              if (s.id === 'neo4j' && !neo4jStatus) loadNeo4j();
              if (s.id === 'analytics' && !analytics) loadAnalytics();
            }}>
            {s.icon} {s.label}
          </button>
        ))}
      </div>

      {/* ── Network Graph ── */}
      {activeSection === 'network' && (
        <div className="gp-network-layout page-enter">
          <div className="glass-card gp-graph-main">
            <div className="glass-card-header">
              <div className="card-icon cyan"><HiOutlineGlobe /></div>
              <h3>{t('graph.kg_title')}</h3>
              {communityData?.stats && (
                <span className="badge badge-cyan" style={{ marginLeft: 'auto' }}>
                  {communityData.stats.nodeCount} {t('graph.nodes')} · {communityData.stats.edgeCount} {t('graph.edges')} · {communityData.stats.communityCount} {t('graph.communities')}
                </span>
              )}
            </div>
            <div className="gp-force-container">
              {communityData?.nodes?.length > 0 ? (
                <ForceGraph2D
                  ref={fgRef}
                  graphData={{ nodes: communityData.nodes, links: communityData.links }}
                  nodeCanvasObject={nodeCanvasObject}
                  linkCanvasObjectMode={() => 'replace'}
                  linkCanvasObject={linkCanvasObject}
                  nodeId="id"
                  width={900}
                  height={500}
                  backgroundColor="rgba(0,0,0,0)"
                  onNodeHover={node => setHoveredNode(node?.id || null)}
                  onNodeClick={node => {
                    if (selectedComm === node.community) setSelectedComm(null);
                    else setSelectedComm(node.community);
                  }}
                  cooldownTicks={80}
                  d3AlphaDecay={0.03}
                  d3VelocityDecay={0.35}
                  linkDirectionalParticles={0}
                />
              ) : (
                <div className="empty-state" style={{ height: 500 }}>
                  <p className="empty-state-title">{t('graph.no_graph')}</p>
                  <p>{t('graph.build_msg')}</p>
                </div>
              )}
            </div>
            <div className="gp-controls">
              <button className="btn btn-primary" onClick={buildGraph} disabled={loading}>
                {loading ? <span className="spinner" /> : <HiOutlinePlay />}
                {t('graph.build_btn')}
              </button>
              <button className="btn btn-secondary btn-sm" onClick={loadCommunityGraph}>🔄 {t('graph.reload')}</button>
              <button className="btn btn-secondary btn-sm" onClick={() => fgRef.current?.zoomToFit(400, 40)}>⊞ {t('graph.zoom_fit')}</button>
              {selectedComm !== null && (
                <button className="btn btn-danger btn-sm" onClick={() => setSelectedComm(null)}>✕ {t('graph.reset_filter')}</button>
              )}
            </div>
          </div>
          {/* Community legend sidebar */}
          {communityData?.communities?.length > 0 && (
            <div className="glass-card gp-legend-sidebar">
              <div className="glass-card-header">
                <div className="card-icon emerald"><HiOutlineUserGroup /></div>
                <h3>{t('graph.communities')}</h3>
              </div>
              <div className="gp-legend-list">
                {communityData.communities.map(c => (
                  <div key={c.id}
                    className={`gp-legend-item ${selectedComm === c.id ? 'active' : ''}`}
                    onClick={() => setSelectedComm(selectedComm === c.id ? null : c.id)}>
                    <span className="gp-legend-dot" style={{ background: c.color }} />
                    <div className="gp-legend-info">
                      <span className="gp-legend-name">{c.name}</span>
                      <span className="gp-legend-count">{c.count} {t('graph.nodes')}</span>
                    </div>
                  </div>
                ))}
              </div>
              {communityData.stats && (
                <div className="gp-stats-mini">
                  <div className="gp-stat-row"><span>{t('graph.density')}</span><span>{communityData.stats.density}</span></div>
                  <div className="gp-stat-row"><span>{t('graph.avg_degree')}</span><span>{communityData.stats.avgDegree}</span></div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Louvain Communities ── */}
      {activeSection === 'communities' && (
        <div className="gp-comm-layout page-enter">
          <div className="glass-card">
            <div className="glass-card-header">
              <div className="card-icon emerald"><HiOutlineUserGroup /></div>
              <h3>{t('graph.louvain_title')}</h3>
            </div>
            {communities ? (
              <>
                <div className="gp-mod-display">
                  <div className="metric"><span className="metric-label">{t('graph.modularity')}</span><span className="metric-value">{communities.modularity}</span></div>
                  <div className="metric"><span className="metric-label">{t('graph.clusters')}</span><span className="metric-value">{communities.cluster_count}</span></div>
                  <div className="metric"><span className="metric-label">{t('graph.partitioned_nodes')}</span><span className="metric-value">{communities.total_nodes_partitioned}</span></div>
                </div>
                <div className="gp-cluster-grid">
                  {communities.clusters?.map(c => (
                    <div key={c.id} className="gp-cluster-card" style={{ borderLeftColor: c.color }}>
                      <div className="gp-cluster-header">
                        <span className="gp-legend-dot" style={{ background: c.color }} />
                        <strong>{c.name}</strong>
                        <span className="badge badge-indigo">{c.member_count}</span>
                      </div>
                      <div className="gp-cluster-meta">
                        <span>{t('graph.intra_density')} {c.intra_density}</span>
                        <span>{t('graph.edges_label')} {c.intra_edges}</span>
                      </div>
                      <div className="gp-cluster-members">
                        {c.key_members?.map((m, j) => (
                          <span key={j} className="badge badge-purple" style={{ fontSize: '0.6rem' }}>{m}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p className="empty-state-title">{t('graph.loading')}</p>
                <button className="btn btn-primary btn-sm" onClick={loadCommunities}>{t('graph.detect_btn')}</button>
              </div>
            )}
          </div>
          {communities?.inter_community_edges?.length > 0 && (
            <div className="glass-card">
              <div className="glass-card-header">
                <div className="card-icon amber">🔗</div>
                <h3>{t('graph.inter_comm')}</h3>
              </div>
              <div className="gp-inter-list">
                {communities.inter_community_edges.map((e, i) => (
                  <div key={i} className="gp-inter-item">
                    <span className="badge badge-cyan">{e.from}</span>
                    <span className="gp-inter-arrow">⟷</span>
                    <span className="badge badge-emerald">{e.to}</span>
                    <span className="gp-inter-count">{e.edges} {t('graph.edges')}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Neo4j Aura ── */}
      {activeSection === 'neo4j' && (
        <div className="gp-neo4j-layout page-enter">
          <div className="glass-card">
            <div className="glass-card-header">
              <div className="card-icon cyan"><HiOutlineDatabase /></div>
              <h3>{t('graph.neo4j_status')}</h3>
              {neo4jStatus && (
                <span className={`badge ${neo4jStatus.connected ? 'badge-emerald' : 'badge-rose'}`} style={{ marginLeft: 'auto' }}>
                  {neo4jStatus.connected ? `● ${t('graph.connected')}` : `● ${t('graph.disconnected')}`}
                </span>
              )}
            </div>
            {neo4jStatus ? (
              neo4jStatus.connected ? (
                <div className="gp-neo4j-stats">
                  <div className="metric"><span className="metric-label">URI</span><span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{neo4jStatus.uri}</span></div>
                  <div className="gp-neo4j-metrics">
                    <div className="metric"><span className="metric-label">{t('graph.nodes_label')}</span><span className="metric-value">{neo4jStatus.node_count}</span></div>
                    <div className="metric"><span className="metric-label">{t('graph.relations_label')}</span><span className="metric-value">{neo4jStatus.edge_count}</span></div>
                  </div>
                  {neo4jStatus.sample_triples?.length > 0 && (
                    <div className="gp-triples">
                      <h4 style={{ fontSize: '0.8rem', marginBottom: 8, color: 'var(--text-secondary)' }}>Sample Triples (Neo4j)</h4>
                      {neo4jStatus.sample_triples.slice(0, 6).map((t, i) => (
                        <div key={i} className="gp-triple-row">
                          <span className="badge badge-indigo">{t.source}</span>
                          <span className="gp-triple-rel">—[{t.relation}]→</span>
                          <span className="badge badge-emerald">{t.target}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="empty-state"><p>❌ {neo4jStatus.error}</p></div>
              )
            ) : (
              <div className="empty-state">
                <button className="btn btn-primary btn-sm" onClick={loadNeo4j}>{t('graph.check_conn')}</button>
              </div>
            )}
          </div>

          <div className="glass-card">
            <div className="glass-card-header">
              <div className="card-icon purple"><HiOutlineCode /></div>
              <h3>{t('graph.cypher_console')}</h3>
            </div>
            <textarea className="input textarea gp-cypher-input"
              value={cypherQuery} onChange={e => setCypherQuery(e.target.value)}
              rows={3} placeholder="MATCH (n) RETURN n LIMIT 10" />
            <div className="gp-cypher-presets">
              <button className="btn btn-secondary btn-sm" onClick={() => setCypherQuery('MATCH (n:Concept) RETURN n.name, n.community, n.degree ORDER BY n.degree DESC LIMIT 20')}>{t('graph.top_nodes')}</button>
              <button className="btn btn-secondary btn-sm" onClick={() => setCypherQuery('MATCH (a:Concept)-[r]->(b:Concept) RETURN a.name, type(r), b.name LIMIT 20')}>{t('graph.relations_btn')}</button>
              <button className="btn btn-secondary btn-sm" onClick={() => setCypherQuery('MATCH (n:Concept) RETURN n.community AS community, count(n) AS size ORDER BY size DESC')}>{t('graph.comm_btn')}</button>
              <button className="btn btn-secondary btn-sm" onClick={() => setCypherQuery('MATCH (a:Concept)-[r]->(b:Concept) WHERE a.community <> b.community RETURN a.name, a.community, type(r), b.name, b.community LIMIT 15')}>{t('graph.inter_btn')}</button>
            </div>
            <button className="btn btn-primary" onClick={runCypher} disabled={loading} style={{ marginTop: 12 }}>
              {loading ? <span className="spinner" /> : <HiOutlinePlay />} {t('graph.execute')}
            </button>
            {cypherResult && (
              <div className="gp-cypher-result">
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                  <span className={`badge ${cypherResult.success ? 'badge-emerald' : 'badge-rose'}`}>
                    {cypherResult.success ? `${cypherResult.result_count} ${t('graph.results')}` : t('graph.error')}
                  </span>
                </div>
                {cypherResult.success ? (
                  <div className="gp-cypher-table-wrap">
                    {cypherResult.results?.length > 0 && (
                      <table className="gp-cypher-table">
                        <thead>
                          <tr>{Object.keys(cypherResult.results[0]).map(k => <th key={k}>{k}</th>)}</tr>
                        </thead>
                        <tbody>
                          {cypherResult.results.slice(0, 20).map((row, i) => (
                            <tr key={i}>{Object.values(row).map((v, j) => <td key={j}>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</td>)}</tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                ) : (
                  <pre className="gp-cypher-error">{cypherResult.error}</pre>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Analytics ── */}
      {activeSection === 'analytics' && (
        <div className="gp-analytics-layout page-enter">
          {analytics ? (
            <>
              <div className="glass-card">
                <div className="glass-card-header"><div className="card-icon purple">📊</div><h3>{t('graph.global_metrics')}</h3></div>
                <div className="gp-metrics-grid">
                  <div className="metric"><span className="metric-label">{t('graph.density')}</span><span className="metric-value">{analytics.density}</span></div>
                  <div className="metric"><span className="metric-label">{t('graph.nodes_label')}</span><span className="metric-value">{analytics.node_count}</span></div>
                  <div className="metric"><span className="metric-label">{t('graph.edges_label')}</span><span className="metric-value">{analytics.edge_count}</span></div>
                  <div className="metric"><span className="metric-label">{t('graph.avg_degree')}</span><span className="metric-value">{analytics.avg_degree}</span></div>
                  <div className="metric"><span className="metric-label">{t('graph.max_degree')}</span><span className="metric-value">{analytics.max_degree}</span></div>
                  <div className="metric"><span className="metric-label">{t('graph.components')}</span><span className="metric-value">{analytics.components}</span></div>
                </div>
              </div>
              <div className="glass-card">
                <div className="glass-card-header"><div className="card-icon cyan">🎯</div><h3>{t('graph.degree_centrality')}</h3></div>
                <div className="gp-bar-chart">{analytics.centrality?.degree?.map((item, i) => (
                  <div key={i} className="gp-bar-row">
                    <span className="gp-bar-label">{item.node.length > 20 ? item.node.slice(0, 18) + '…' : item.node}</span>
                    <div className="gp-bar-track"><div className="gp-bar-fill" style={{ width: `${(item.score / (analytics.centrality.degree[0]?.score || 1)) * 100}%`, background: 'var(--accent-indigo)' }} /></div>
                    <span className="gp-bar-val">{item.score}</span>
                  </div>
                ))}</div>
              </div>
              <div className="glass-card">
                <div className="glass-card-header"><div className="card-icon amber">🔀</div><h3>{t('graph.betweenness')}</h3></div>
                <div className="gp-bar-chart">{analytics.centrality?.betweenness?.slice(0, 8).map((item, i) => (
                  <div key={i} className="gp-bar-row">
                    <span className="gp-bar-label">{item.node.length > 20 ? item.node.slice(0, 18) + '…' : item.node}</span>
                    <div className="gp-bar-track"><div className="gp-bar-fill" style={{ width: `${(item.score / (analytics.centrality.betweenness[0]?.score || 1)) * 100}%`, background: 'var(--accent-amber)' }} /></div>
                    <span className="gp-bar-val">{item.score}</span>
                  </div>
                ))}</div>
              </div>
              <div className="glass-card">
                <div className="glass-card-header"><div className="card-icon emerald">🧭</div><h3>{t('graph.semantic_paths')}</h3></div>
                <ul className="analytics-list">
                  {analytics.semantic_paths?.map((p, i) => (<li key={i}>{p.path?.join(' → ')}</li>))}
                  {(!analytics.semantic_paths || analytics.semantic_paths.length === 0) && (<li style={{ color: 'var(--text-muted)' }}>{t('graph.no_path')}</li>)}
                </ul>
              </div>
              <div className="glass-card" style={{ gridColumn: 'span 2' }}>
                <div className="glass-card-header"><div className="card-icon rose">💡</div><h3>{t('graph.recommendations')}</h3></div>
                <ul className="analytics-list">
                  {analytics.recommendations?.map((r, i) => (<li key={i}>{r}</li>))}
                </ul>
              </div>
            </>
          ) : (
            <div className="glass-card" style={{ gridColumn: 'span 2' }}>
              <div className="empty-state">
                <p className="empty-state-title">{t('graph.load_analytics')}</p>
                <button className="btn btn-primary btn-sm" onClick={loadAnalytics}>{t('graph.load_btn')}</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
