import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import {
  api,
  decayAt,
  fmtTs,
  EDGE_COLORS,
  GraphNode,
  GraphResponse,
  MemoryDetail,
  MemoryTypeName,
  MEMORY_TYPES,
  TYPE_COLORS,
  UiConfig,
} from '../api';
import { usePoll } from '../usePoll';

const PRESETS: { label: string; days: number | null }[] = [
  { label: '7d', days: 7 },
  { label: '30d', days: 30 },
  { label: '90d', days: 90 },
  { label: 'all', days: null },
];

// force-graph mutates link endpoints into node object references.
interface FgLink {
  source: string | GraphNode;
  target: string | GraphNode;
  edge_type: string;
  weight: number;
}

const nodeId = (e: string | GraphNode) => (typeof e === 'string' ? e : e.id);

export default function GraphView({ cfg, active }: { cfg: UiConfig | null; active: boolean }) {
  const [days, setDays] = useState<number | null>(30);
  const [types, setTypes] = useState<Set<MemoryTypeName>>(new Set(MEMORY_TYPES));
  const [tagFilter, setTagFilter] = useState('');
  const [minDecay, setMinDecay] = useState(0);
  const [scrubT, setScrubT] = useState<number | null>(null); // null = live (now)
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<MemoryDetail | null>(null);
  const [size, setSize] = useState({ w: 800, h: 520 });
  const wrapRef = useRef<HTMLDivElement>(null);

  const since = days === null ? 0 : Math.floor(Date.now() / 1000) - days * 86_400;
  const { data, error } = usePoll<GraphResponse>(
    () => api.graph({ since, types: [...types] }),
    4000,
    [since, [...types].sort().join(',')],
  );

  // Stable node/link identities across polls so the layout never jumps:
  // existing nodes are mutated in place, new ones appended.
  const graphRef = useRef<{ nodes: GraphNode[]; links: FgLink[] }>({ nodes: [], links: [] });
  const graphData = useMemo(() => {
    if (!data) return graphRef.current;
    const prev = new Map(graphRef.current.nodes.map((n) => [n.id, n]));
    const nodes = data.nodes.map((n) => {
      const old = prev.get(n.id);
      if (old) {
        Object.assign(old, n, { x: old.x, y: old.y });
        return old;
      }
      return { ...n };
    });
    const links: FgLink[] = data.edges.map((e) => ({
      source: e.src_id,
      target: e.dst_id,
      edge_type: e.edge_type,
      weight: e.weight,
    }));
    graphRef.current = { nodes, links };
    return graphRef.current;
  }, [data]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() =>
      setSize({ w: el.clientWidth, h: Math.max(420, window.innerHeight - 250) }),
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    let alive = true;
    api.memory(selected).then((d) => alive && setDetail(d)).catch(() => alive && setDetail(null));
    return () => {
      alive = false;
    };
  }, [selected]);

  const now = Math.floor(Date.now() / 1000);
  const at = scrubT ?? now;
  const oldest = useMemo(
    () => graphData.nodes.reduce((m, n) => Math.min(m, n.created_at), now),
    [graphData, now],
  );

  /** Decay of a node as of the scrubber time (the centerpiece math). */
  const liveDecay = useCallback(
    (n: GraphNode) =>
      cfg ? decayAt(n.importance_score, cfg.decay_lambdas[n.memory_type], n.created_at, at, n.pinned) : n.decay_score,
    [cfg, at],
  );

  const visible = useCallback(
    (n: GraphNode) => {
      if (n.created_at > at) return false; // not born yet at scrub time
      if (!types.has(n.memory_type)) return false;
      if (tagFilter && !n.tags.some((t) => t.includes(tagFilter)) && !(n.topic ?? '').includes(tagFilter))
        return false;
      return liveDecay(n) >= minDecay;
    },
    [at, types, tagFilter, minDecay, liveDecay],
  );

  const paintNode = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, scale: number) => {
      const d = liveDecay(node);
      const r = 2.5 + node.importance_score * 6;
      const alpha = node.stale ? 0.25 : Math.max(0.08, Math.min(1, d * 1.6));
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI);
      ctx.fillStyle = node.stale ? '#57606a' : TYPE_COLORS[node.memory_type];
      ctx.fill();
      if (node.pinned) {
        ctx.globalAlpha = 0.95;
        ctx.lineWidth = 1.2 / scale;
        ctx.strokeStyle = '#e3b341';
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, r + 2.2 / scale, 0, 2 * Math.PI);
        ctx.stroke();
      }
      if (node.id === selected) {
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1.6 / scale;
        ctx.strokeStyle = '#58a6ff';
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, r + 4 / scale, 0, 2 * Math.PI);
        ctx.stroke();
      }
      if (scale > 2.2) {
        ctx.globalAlpha = Math.min(1, alpha + 0.3);
        ctx.font = `${3.6}px sans-serif`;
        ctx.fillStyle = '#c9d1d9';
        const label = node.content_preview.slice(0, 34);
        ctx.fillText(label, node.x! + r + 1.5, node.y! + 1.2);
      }
      ctx.globalAlpha = 1;
    },
    [liveDecay, selected],
  );

  const toggleType = (t: MemoryTypeName) =>
    setTypes((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });

  const refreshDetail = async () => {
    if (selected) setDetail(await api.memory(selected));
  };

  return (
    <div className="view">
      <div className="panel row">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            className="ghost"
            style={days === p.days ? { borderColor: 'var(--accent)', color: 'var(--accent)' } : {}}
            onClick={() => setDays(p.days)}
          >
            {p.label}
          </button>
        ))}
        <span style={{ width: 8 }} />
        {MEMORY_TYPES.map((t) => (
          <label key={t} className="ctl">
            <input type="checkbox" checked={types.has(t)} onChange={() => toggleType(t)} />
            <span className="legend-swatch" style={{ background: TYPE_COLORS[t] }} />
            {t}
          </label>
        ))}
        <input
          type="text"
          placeholder="filter tag / topic…"
          value={tagFilter}
          onChange={(e) => setTagFilter(e.target.value)}
          style={{ width: 150 }}
        />
        <label className="ctl">
          min decay
          <input
            type="range"
            min={0}
            max={0.8}
            step={0.01}
            value={minDecay}
            onChange={(e) => setMinDecay(Number(e.target.value))}
          />
          {minDecay.toFixed(2)}
        </label>
        <span className="dim mono">
          {data ? `${graphData.nodes.filter(visible).length}/${data.total_count} memories` : '…'}
        </span>
      </div>

      <div className="panel scrubber">
        <span className="dim">⏪ past</span>
        <input
          type="range"
          min={oldest}
          max={now}
          step={3600}
          value={at}
          onChange={(e) => {
            const v = Number(e.target.value);
            setScrubT(v >= now - 3600 ? null : v);
          }}
        />
        <span className="dim">now ⏩</span>
        <span className="ts mono">{scrubT ? fmtTs(scrubT) : 'live'}</span>
      </div>

      {error && <div className="err">{error}</div>}

      <div className="graph-wrap" ref={wrapRef}>
        {active && (
          <ForceGraph2D
            graphData={graphData}
            width={size.w}
            height={size.h}
            backgroundColor="#0a0d12"
            nodeId="id"
            nodeVal={(n: GraphNode) => 1 + n.importance_score * 5}
            nodeVisibility={visible}
            nodeCanvasObject={paintNode}
            nodeLabel={(n: GraphNode) =>
              `${n.memory_type}${n.pinned ? ' 📌' : ''}${n.stale ? ' (stale)' : ''}\n${n.content_preview}`
            }
            linkVisibility={(l: FgLink) =>
              visible(l.source as GraphNode) && visible(l.target as GraphNode)
            }
            linkColor={(l: FgLink) => EDGE_COLORS[l.edge_type] ?? '#30363d'}
            linkWidth={(l: FgLink) => 0.5 + l.weight}
            linkDirectionalArrowLength={3}
            linkDirectionalArrowRelPos={0.9}
            onNodeClick={(n: GraphNode) => setSelected(n.id === selected ? null : n.id)}
            onBackgroundClick={() => setSelected(null)}
            cooldownTicks={120}
          />
        )}

        {detail && (
          <DetailPanel
            detail={detail}
            nodes={graphData.nodes}
            onNavigate={(id) => setSelected(id)}
            onClose={() => setSelected(null)}
            onChanged={refreshDetail}
          />
        )}
      </div>

      <div className="row dim" style={{ marginTop: 8, fontSize: 11 }}>
        edge types:
        {Object.entries(EDGE_COLORS).map(([k, c]) => (
          <span key={k}>
            <span className="legend-swatch" style={{ background: c }} />
            {k}
          </span>
        ))}
        <span className="grow" />
        size = importance · opacity = decay at scrub time · amber ring = pinned
      </div>
    </div>
  );
}

function DetailPanel({
  detail,
  nodes,
  onNavigate,
  onClose,
  onChanged,
}: {
  detail: MemoryDetail;
  nodes: GraphNode[];
  onNavigate: (id: string) => void;
  onClose: () => void;
  onChanged: () => void;
}) {
  const m = detail.memory;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const preview = (id: string) => byId.get(id)?.content_preview ?? id;
  const out = detail.links.filter((l) => l.source_id === m.id);
  const inn = detail.links.filter((l) => l.target_id === m.id);
  const provenance = out.filter((l) => l.relation === 'derived_from');
  const [busy, setBusy] = useState(false);

  const act = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside className="detail">
      <div className="row">
        <span className="chip" style={{ color: TYPE_COLORS[m.memory_type] }}>
          {m.memory_type}
        </span>
        {m.pinned && <span className="chip">📌 pinned</span>}
        {m.stale && <span className="chip" style={{ color: 'var(--red)' }}>stale</span>}
        <span className="grow" />
        <button className="ghost" onClick={onClose}>
          ✕
        </button>
      </div>
      <div className="content">{m.content}</div>
      <dl>
        <dt>importance</dt>
        <dd className="mono">{m.importance_score.toFixed(3)}</dd>
        <dt>decay (stored)</dt>
        <dd className="mono">{m.decay_score.toFixed(3)}</dd>
        <dt>recurrence</dt>
        <dd className="mono">{m.recurrence_score.toFixed(3)}</dd>
        <dt>accesses</dt>
        <dd className="mono">{m.access_count}</dd>
        <dt>created</dt>
        <dd>{fmtTs(m.created_at)}</dd>
        <dt>last accessed</dt>
        <dd>{fmtTs(m.last_accessed_at)}</dd>
        <dt>topic</dt>
        <dd>{m.topic ?? '—'}</dd>
        <dt>tags</dt>
        <dd>{m.tags.join(', ') || '—'}</dd>
      </dl>

      {provenance.length > 0 && (
        <>
          <h3>summary of {provenance.length} memories</h3>
          {provenance.map((l) => (
            <div key={l.target_id} className="link-row" onClick={() => onNavigate(l.target_id)}>
              ↳ {preview(l.target_id)}
            </div>
          ))}
        </>
      )}
      {out.filter((l) => l.relation !== 'derived_from').length > 0 && <h3>edges out</h3>}
      {out
        .filter((l) => l.relation !== 'derived_from')
        .map((l, i) => (
          <div key={i} className="link-row" onClick={() => onNavigate(l.target_id)}>
            <span style={{ color: EDGE_COLORS[l.relation] }}>{l.relation}</span> → {preview(l.target_id)}
          </div>
        ))}
      {inn.length > 0 && <h3>edges in</h3>}
      {inn.map((l, i) => (
        <div key={i} className="link-row" onClick={() => onNavigate(l.source_id)}>
          <span style={{ color: EDGE_COLORS[l.relation] }}>{l.relation}</span> ← {preview(l.source_id)}
        </div>
      ))}

      <div className="row" style={{ marginTop: 12 }}>
        <button className="ghost" disabled={busy} onClick={() => act(() => api.setPinned(m.id, !m.pinned))}>
          {m.pinned ? 'unpin' : 'pin'}
        </button>
        <button
          className="ghost danger"
          disabled={busy || m.memory_type === 'archive'}
          onClick={() => act(() => api.archive(m.id))}
        >
          archive
        </button>
        <span className="grow" />
        <span className="dim mono" style={{ fontSize: 10 }}>{m.id.slice(0, 14)}…</span>
      </div>
    </aside>
  );
}
