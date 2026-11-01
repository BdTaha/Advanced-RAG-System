import { useEffect, useState } from 'react';
import { workspaceAPI, vectorialAPI } from './services/api';
import { useLanguage } from './LanguageContext';

export default function Workspace({ onChange, onStatus }) {
  const { language } = useLanguage();
  const fr = language === 'fr';
  const [info, setInfo] = useState(null);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [expanded, setExpanded] = useState(false);
  const refresh = async () => {
    try { const res = await workspaceAPI.info(); setInfo(res.data); onStatus(true); }
    catch (error) { setMessage(error.response?.data?.detail || error.message); onStatus(false); }
  };
  useEffect(() => { refresh(); }, []);
  const ingest = async action => {
    setBusy(true); setMessage(fr ? 'Traitement du document…' : 'Processing document…');
    try {
      await action();
      await vectorialAPI.chunk({ method: 'recursive', chunk_size: 600, chunk_overlap: 50 });
      await vectorialAPI.embed({ model: 'tfidf' });
      await refresh(); onChange(); setExpanded(false); setText('');
      setMessage(fr ? 'Document indexé. Posez votre question.' : 'Document indexed. Ask a question.');
    } catch (error) { setMessage(error.response?.data?.detail || error.message); }
    setBusy(false);
  };
  return <section className="glass-card workspace" aria-label={fr ? 'Espace documentaire' : 'Document workspace'}>
    <div className="workspace-row">
      <div><strong>{fr ? 'Votre document' : 'Your document'}</strong><div className="workspace-detail">{info ? `${info.filename} · ${info.characters.toLocaleString()} ${fr ? 'caractères' : 'characters'} · ${info.chunks} chunks` : (fr ? 'Connexion…' : 'Connecting…')}</div></div>
      <label className={`btn btn-secondary btn-sm ${busy ? 'disabled' : ''}`}>
        {fr ? 'Importer TXT / MD' : 'Upload TXT / MD'}
        <input type="file" accept=".txt,.md,text/plain,text/markdown" disabled={busy || !info} hidden onChange={event => {
          const file = event.target.files?.[0]; event.target.value = '';
          if (file) ingest(() => vectorialAPI.upload(file));
        }} />
      </label>
      <button className="btn btn-secondary btn-sm" onClick={() => setExpanded(!expanded)} disabled={busy || !info}>{fr ? 'Coller du texte' : 'Paste text'}</button>
      <button className="btn btn-secondary btn-sm" onClick={refresh} disabled={busy}>{fr ? 'Actualiser' : 'Refresh'}</button>
      <button className="btn btn-danger btn-sm" disabled={busy || !info} onClick={async () => {
        if (!window.confirm(fr ? 'Effacer votre document, graphe et historique ?' : 'Delete your document, graph, and history?')) return;
        setBusy(true);
        try { await workspaceAPI.delete(); window.location.reload(); } catch (error) { setMessage(error.response?.data?.detail || error.message); setBusy(false); }
      }}>{fr ? 'Effacer mon espace' : 'Delete my workspace'}</button>
    </div>
    <p className="workspace-detail">{fr ? 'Guide de démonstration prêt à utiliser. TXT / Markdown UTF-8, 24 000 caractères maximum. Espace propre à ce navigateur, conservé 30 jours. IA selon quota gratuit ; extraits sourcés sinon. Documents stockés sur Cloudflare et passages transmis à Workers AI pour les réponses.' : 'Demo guide ready to use. UTF-8 TXT / Markdown, up to 24,000 characters. Workspace belongs to this browser and expires after 30 days. AI uses a shared free quota, with source excerpts as fallback. Documents are stored on Cloudflare; retrieved passages are sent to Workers AI for answers.'}</p>
    {expanded && <div className="workspace-editor">
      <textarea className="input textarea" aria-label={fr ? 'Texte du document' : 'Document text'} maxLength={24000} rows={5} value={text} onChange={e => setText(e.target.value)} placeholder={fr ? 'Collez votre document…' : 'Paste your document…'} />
      <button className="btn btn-primary btn-sm" disabled={busy || !text.trim()} onClick={() => ingest(() => workspaceAPI.text(text))}>{busy ? '…' : (fr ? 'Utiliser et indexer' : 'Use and index document')}</button>
    </div>}
    {message && <p role="status" className="workspace-detail">{message}</p>}
  </section>;
}
