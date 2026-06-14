import { useState } from 'react';
import { api, fmtBytes } from './api';
import { usePoll } from './usePoll';
import GraphView from './views/GraphView';
import RetrievalView from './views/RetrievalView';
import ConsolidationView from './views/ConsolidationView';
import MetricsView from './views/MetricsView';

const VIEWS = [
  { key: 'graph', label: 'Memory Graph' },
  { key: 'retrieval', label: 'Retrieval Inspector' },
  { key: 'consolidation', label: 'Consolidation' },
  { key: 'metrics', label: 'Metrics' },
] as const;

type ViewKey = (typeof VIEWS)[number]['key'];

export default function App() {
  const [view, setView] = useState<ViewKey>('graph');
  const { data: cfg } = usePoll(() => api.uiconfig(), 60_000);

  return (
    <>
      <nav className="sidebar">
        <h1>
          ForgetfulDB <span>observe</span>
        </h1>
        {VIEWS.map((v) => (
          <button
            key={v.key}
            className={`nav-btn ${view === v.key ? 'active' : ''}`}
            onClick={() => setView(v.key)}
          >
            {v.label}
          </button>
        ))}
        <div className="foot">
          {cfg ? (
            <>
              store “{cfg.name}”<br />
              {fmtBytes(cfg.db_size_bytes)} · read-only window
            </>
          ) : (
            'connecting…'
          )}
        </div>
      </nav>
      <main className="main">
        {/* Views stay mounted so the graph layout survives tab switches. */}
        <div style={{ display: view === 'graph' ? 'block' : 'none', height: '100%' }}>
          <GraphView cfg={cfg} active={view === 'graph'} />
        </div>
        {view === 'retrieval' && <RetrievalView cfg={cfg} />}
        {view === 'consolidation' && <ConsolidationView />}
        {view === 'metrics' && <MetricsView cfg={cfg} />}
      </main>
    </>
  );
}
