import { useState } from 'react';
import './App.css';
import QueryPage from './pages/QueryPage';
import VectorialPage from './pages/VectorialPage';
import GraphPage from './pages/GraphPage';
import AgenticPage from './pages/AgenticPage';
import { HiOutlineSearch, HiOutlineCube, HiOutlineShare, HiOutlineLightningBolt, HiTranslate } from 'react-icons/hi';
import { useLanguage } from './LanguageContext';

export default function App() {
  const [activeTab, setActiveTab] = useState('query');
  const { language, toggleLanguage, t } = useLanguage();

  const TABS = [
    { id: 'query',     label: t('app.query'),           icon: <HiOutlineSearch />,        badge: null },
    { id: 'vectorial', label: t('app.vectorial'),    icon: <HiOutlineCube />,          badge: '7+5' },
    { id: 'graph',     label: t('app.graph'),        icon: <HiOutlineShare />,         badge: 'Neo4j' },
    { id: 'agentic',   label: t('app.agentic'), icon: <HiOutlineLightningBolt />, badge: 'RL' },
  ];

  const renderPage = () => {
    switch (activeTab) {
      case 'query':     return <QueryPage />;
      case 'vectorial': return <VectorialPage />;
      case 'graph':     return <GraphPage />;
      case 'agentic':   return <AgenticPage />;
      default:          return <QueryPage />;
    }
  };

  return (
    <div className="app-layout">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">⚡</div>
            RAG Platform
          </div>

          <nav className="tab-nav" role="tablist">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                id={`tab-${tab.id}`}
                role="tab"
                aria-selected={activeTab === tab.id}
                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <span className="tab-icon">{tab.icon}</span>
                {tab.label}
                {tab.badge && <span className="tab-badge">{tab.badge}</span>}
              </button>
            ))}
          </nav>

          <div className="header-status" style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <button className="btn-icon" onClick={toggleLanguage} title="Toggle Language" style={{ padding: '6px', background: 'transparent', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', cursor: 'pointer', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <HiTranslate size={18} style={{ marginRight: '4px' }} /> {language.toUpperCase()}
            </button>
            <span className="badge badge-emerald">● {t('app.online')}</span>
          </div>
        </div>
      </header>

      {/* ── Page Content ────────────────────────────────────────────────── */}
      <main className="main-content">
        <div className="page-enter" key={activeTab}>
          {renderPage()}
        </div>
      </main>
    </div>
  );
}
