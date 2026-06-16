import { useMemo } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api, fmtBytes, fmtTs, MEMORY_TYPES, TYPE_COLORS, UiConfig } from '../api';
import { usePoll } from '../usePoll';

const AXIS = { stroke: '#8b949e', fontSize: 11 };
const GRID = '#21262d';

export default function MetricsView({ cfg }: { cfg: UiConfig | null }) {
  const { data: turnsData } = usePoll(() => api.turns(300), 5000);
  const { data: stats } = usePoll(() => api.stats(), 5000);
  const { data: metrics } = usePoll(() => api.metrics(), 5000);
  const { data: epochsData } = usePoll(() => api.epochs(), 10_000);
  const { data: graph } = usePoll(() => api.graph({ since: 0 }), 10_000);

  const epochs = epochsData?.epochs ?? [];

  const fmt1 = (v: number | null | undefined) => (v == null ? undefined : v.toFixed(1));
  const pct = (v: number | null | undefined) => (v == null ? undefined : `${(v * 100).toFixed(1)}%`);

  const turns = useMemo(
    () =>
      (turnsData?.turns ?? []).map((t, i) => ({
        i,
        time: new Date(t.created_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
        prompt: t.prompt_tokens,
        completion: t.completion_tokens,
        retrieve_ms: t.retrieve_duration_ms,
        llm_ms: t.llm_duration_ms,
        injected: t.context_memory_count,
        // context share ≈ (context_chars/4 tokens) / prompt tokens
        context_share: t.prompt_tokens ? Math.min(100, (t.context_chars / 4 / t.prompt_tokens) * 100) : null,
      })),
    [turnsData],
  );

  const byType = useMemo(
    () =>
      MEMORY_TYPES.map((t) => ({
        type: t,
        count: stats?.by_type?.[t] ?? 0,
        fill: TYPE_COLORS[t],
      })),
    [stats],
  );

  const decayHist = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({
      bucket: `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`,
      count: 0,
    }));
    for (const n of graph?.nodes ?? []) {
      const b = Math.min(9, Math.floor(n.decay_score * 10));
      bins[b].count += 1;
    }
    return bins;
  }, [graph]);

  return (
    <div className="view">
      <h2>Metrics</h2>

      <div className="tiles">
        <Tile k="memories" v={stats?.total_memories} />
        <Tile k="pinned" v={stats?.pinned} />
        <Tile k="stale" v={stats?.stale} />
        <Tile k="edges" v={stats?.links} />
        <Tile k="chat turns" v={turnsData?.turns.length} />
        <Tile k="eras" v={stats?.epochs} />
        <Tile k="db size" v={cfg ? fmtBytes(cfg.db_size_bytes) : undefined} />
      </div>

      {epochs.length > 0 && (
        <>
          <h3 className="section">
            epochs{' '}
            <span className="hint">drift-segmented eras — the engine's exact sense of "when" (the model has no clock)</span>
          </h3>
          <div className="eras">
            {epochs.map((e) => (
              <div className="era-card" key={e.id}>
                <div className="era-head">
                  <span className="era-name">{e.label ?? `era ${e.ordinal + 1}`}</span>
                  <span className="era-count">{e.member_count} mem</span>
                </div>
                <div className="era-span">
                  {fmtTs(e.started_at)} → {e.ended_at ? fmtTs(e.ended_at) : 'now'}
                </div>
                {e.summary && <div className="era-summary">{e.summary}</div>}
                {e.ordinal > 0 && <div className="era-drift">drift in: {(e.drift_in * 100).toFixed(0)}%</div>}
              </div>
            ))}
          </div>
        </>
      )}

      <h3 className="section">
        retention efficiency{' '}
        <span className="hint">accuracy ÷ injected tokens — this is the cost denominator (lower is cheaper)</span>
      </h3>
      <div className="tiles">
        <Tile k="injected tok / turn" v={fmt1(metrics?.injected_tokens_per_turn)} />
        <Tile k="memory share of prompt" v={pct(metrics?.injected_token_share)} />
        <Tile k="tok / injected memory" v={fmt1(metrics?.tokens_per_injected_memory)} />
        <Tile k="total injected tok" v={metrics ? Math.round(metrics.injected_tokens) : undefined} />
      </div>

      <div className="charts">
        <div className="panel">
          <h3>tokens per turn</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={turns}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="time" {...AXIS} minTickGap={40} />
              <YAxis {...AXIS} />
              <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} />
              <Legend />
              <Line dataKey="prompt" stroke="#58a6ff" dot={false} name="prompt tok" />
              <Line dataKey="completion" stroke="#3fb950" dot={false} name="completion tok" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h3>latency per turn (ms)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={turns}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="time" {...AXIS} minTickGap={40} />
              <YAxis yAxisId="llm" {...AXIS} />
              <YAxis yAxisId="ret" orientation="right" {...AXIS} />
              <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} />
              <Legend />
              <Line yAxisId="llm" dataKey="llm_ms" stroke="#d29922" dot={false} name="LLM" />
              <Line yAxisId="ret" dataKey="retrieve_ms" stroke="#bc8cff" dot={false} name="retrieve" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h3>context share of prompt (%)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={turns}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="time" {...AXIS} minTickGap={40} />
              <YAxis {...AXIS} unit="%" />
              <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} />
              <Line dataKey="context_share" stroke="#58a6ff" dot={false} name="context share" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h3>injected memories per turn</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={turns}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="time" {...AXIS} minTickGap={40} />
              <YAxis {...AXIS} allowDecimals={false} />
              <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} />
              <Line dataKey="injected" stroke="#3fb950" dot={false} name="memories" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h3>memories by type</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={byType}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="type" {...AXIS} />
              <YAxis {...AXIS} allowDecimals={false} />
              <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} />
              <Bar dataKey="count">
                {byType.map((entry) => (
                  <Cell key={entry.type} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h3>decay distribution (live, all time)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={decayHist}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="bucket" {...AXIS} />
              <YAxis {...AXIS} allowDecimals={false} />
              <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} />
              <Bar dataKey="count" fill="#58a6ff" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function Tile({ k, v }: { k: string; v: number | string | undefined }) {
  return (
    <div className="tile">
      <div className="v">{v ?? '…'}</div>
      <div className="k">{k}</div>
    </div>
  );
}
