import { useEffect, useState } from 'react';
import {
  api,
  MEMORY_TYPES,
  MemoryTypeName,
  RetrieveResponse,
  RetrievedMemory,
  TYPE_COLORS,
  UiConfig,
} from '../api';

const COMPONENT_COLORS: [keyof UiConfig['retrieval_weights'], string, string][] = [
  ['semantic', '#58a6ff', 'similarity'],
  ['importance', '#3fb950', 'importance'],
  ['recurrence', '#d29922', 'recurrence'],
  ['recency', '#bc8cff', 'recency'],
  ['pinned_boost', '#e3b341', 'pinned'],
];

const TIME_RANGES: { label: string; days: number | null }[] = [
  { label: 'any time', days: null },
  { label: 'last 7d', days: 7 },
  { label: 'last 30d', days: 30 },
  { label: 'last 90d', days: 90 },
];

export default function RetrievalView({ cfg }: { cfg: UiConfig | null }) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(6);
  const [includeStale, setIncludeStale] = useState(false);
  const [types, setTypes] = useState<Set<MemoryTypeName>>(new Set());
  const [rangeDays, setRangeDays] = useState<number | null>(null);
  const [minScore, setMinScore] = useState<number | null>(null);
  const [result, setResult] = useState<RetrieveResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (cfg) {
      setTopK(cfg.chat.top_k);
      setMinScore((m) => m ?? cfg.chat.min_retrieval_score);
    }
  }, [cfg]);

  const run = async () => {
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.retrieve({
          query,
          top_k: topK,
          include_stale: includeStale,
          min_score: minScore ?? undefined,
          memory_types: types.size ? [...types] : undefined,
          since: rangeDays ? Math.floor(Date.now() / 1000) - rangeDays * 86_400 : undefined,
        }),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const weights = cfg?.retrieval_weights;

  return (
    <div className="view">
      <h2>Retrieval Inspector</h2>
      <p className="dim">
        Runs the exact chat-path retrieval (gate {cfg?.chat.min_retrieval_score ?? '…'}, damping{' '}
        {cfg?.chat.conversational_damping ?? '…'}) and shows what would be injected — and what almost was.
      </p>

      <div className="panel">
        <div className="row">
          <input
            type="text"
            className="grow"
            placeholder="what would the assistant remember for…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && run()}
          />
          <button className="act" onClick={run} disabled={busy || !query.trim()}>
            {busy ? 'retrieving…' : 'retrieve'}
          </button>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <label className="ctl">
            top_k
            <input type="number" min={1} max={50} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
          </label>
          <label className="ctl">
            min_score
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={minScore ?? 0}
              onChange={(e) => setMinScore(Number(e.target.value))}
            />
          </label>
          <label className="ctl">
            <input type="checkbox" checked={includeStale} onChange={(e) => setIncludeStale(e.target.checked)} />
            include stale
          </label>
          <select
            value={rangeDays ?? 'all'}
            onChange={(e) => setRangeDays(e.target.value === 'all' ? null : Number(e.target.value))}
          >
            {TIME_RANGES.map((r) => (
              <option key={r.label} value={r.days ?? 'all'}>
                {r.label}
              </option>
            ))}
          </select>
          {MEMORY_TYPES.filter((t) => t !== 'archive').map((t) => (
            <label key={t} className="ctl">
              <input
                type="checkbox"
                checked={types.has(t)}
                onChange={() =>
                  setTypes((prev) => {
                    const next = new Set(prev);
                    if (next.has(t)) next.delete(t);
                    else next.add(t);
                    return next;
                  })
                }
              />
              {t}
            </label>
          ))}
        </div>
      </div>

      {error && <div className="err">{error}</div>}

      {result && weights && (
        <div className="panel">
          <div className="row dim" style={{ marginBottom: 8 }}>
            <span>
              {result.memories.length} injected · {result.near_misses?.length ?? 0} near-misses ·{' '}
              <span className="mono">{result.retrieve_ms}ms</span>
            </span>
            <span className="grow" />
            {COMPONENT_COLORS.map(([k, c, label]) => (
              <span key={k} style={{ fontSize: 11 }}>
                <span className="legend-swatch" style={{ background: c, borderRadius: 2 }} />
                {label}
              </span>
            ))}
            <span style={{ fontSize: 11 }}>
              <span className="legend-swatch" style={{ background: 'var(--red)', borderRadius: 2 }} />
              stale penalty
            </span>
          </div>
          <table className="results">
            <thead>
              <tr>
                <th style={{ width: '42%' }}>memory</th>
                <th>score components</th>
                <th>total</th>
                <th>flags</th>
              </tr>
            </thead>
            <tbody>
              {result.memories.map((m) => (
                <ResultRow key={m.id} m={m} weights={weights} miss={false} />
              ))}
              {(result.near_misses?.length ?? 0) > 0 && (
                <tr className="cutline">
                  <td colSpan={4}>— below the gate (min_score {result.min_score.toFixed(2)}) — never injected —</td>
                </tr>
              )}
              {result.near_misses?.map((m) => (
                <ResultRow key={m.id} m={m} weights={weights} miss />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ResultRow({
  m,
  weights,
  miss,
}: {
  m: RetrievedMemory;
  weights: UiConfig['retrieval_weights'];
  miss: boolean;
}) {
  const s = m.score;
  const damped = s.conversational_damping < 1;
  // Weighted contributions, post-damping, so the bar length IS the score.
  const parts = COMPONENT_COLORS.map(([key, color]) => ({
    color,
    value:
      weights[key] *
      (key === 'semantic'
        ? s.semantic_similarity
        : key === 'importance'
          ? s.importance
          : key === 'recurrence'
            ? s.recurrence
            : key === 'recency'
              ? s.recency
              : s.pinned_boost) *
      s.conversational_damping,
  }));
  const penalty = weights.staleness_penalty * s.staleness_penalty * s.conversational_damping;

  return (
    <tr className={miss ? 'miss' : ''}>
      <td>
        <span className="chip" style={{ color: TYPE_COLORS[m.memory_type], marginRight: 6 }}>
          {m.memory_type}
        </span>
        {m.content.slice(0, 140)}
        {m.content.length > 140 ? '…' : ''}
      </td>
      <td>
        <div className="scorebar" title={`sim ${s.semantic_similarity.toFixed(2)} · imp ${s.importance.toFixed(2)} · rec ${s.recurrence.toFixed(2)} · recency ${s.recency.toFixed(2)}`}>
          {parts.map((p, i) => (
            <div key={i} style={{ width: `${p.value * 100}%`, background: p.color }} />
          ))}
          {penalty > 0 && <div style={{ width: `${penalty * 100}%`, background: 'var(--red)' }} />}
        </div>
      </td>
      <td className="mono">{s.total.toFixed(3)}</td>
      <td>
        {damped && (
          <span className="chip" title="verbatim chat turn — score damped">
            ×{s.conversational_damping}
          </span>
        )}
        {m.pinned && <span className="chip">📌</span>}
        {m.stale && (
          <span className="chip" style={{ color: 'var(--red)' }}>
            stale
          </span>
        )}
      </td>
    </tr>
  );
}
