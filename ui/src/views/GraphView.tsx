import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { forceX, forceY } from 'd3-force';
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

// Edge types with a real direction (source → target). These get animated
// flow particles + an arrowhead; co_occurred is undirected (a mutual
// association) so it flows but carries no arrow.
const DIRECTED = new Set(['sequence', 'derived_from', 'updates', 'contradicts', 'supports', 'belongs_to_project']);

const now_ms = () => performance.now();
// Stable per-node phase so glows don't pulse in unison.
const phaseOf = (id: string) => {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return (h % 1000) / 1000 * Math.PI * 2;
};
const LABEL_CAP = 5; // how many node labels are visible at once
const LABEL_FADE = 600; // ms fade in/out
const LABEL_ROTATE = 1700; // ms between rotations
const FRESH_MS = 5000; // how long a just-formed node/edge stays highlighted

const edgeKey = (l: FgLink) => `${nodeId(l.source)}|${nodeId(l.target)}|${l.edge_type}`;
const freshness = (m: Map<string, number>, key: string, t: number) => {
  const stamp = m.get(key);
  if (stamp === undefined) return 0;
  const f = 1 - (t - stamp) / FRESH_MS;
  return f > 0 ? f : 0;
};

function withAlpha(hex: string, a: number) {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

export default function GraphView({ cfg, active }: { cfg: UiConfig | null; active: boolean }) {
  const [days, setDays] = useState<number | null>(30);
  const [types, setTypes] = useState<Set<MemoryTypeName>>(new Set(MEMORY_TYPES));
  const [tagFilter, setTagFilter] = useState('');
  const [minDecay, setMinDecay] = useState(0);
  const [showCoOccurred, setShowCoOccurred] = useState(true);
  const [hideIsolated, setHideIsolated] = useState(true);
  const [animate, setAnimate] = useState(true);
  const [scrubT, setScrubT] = useState<number | null>(null); // null = live (now)
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<MemoryDetail | null>(null);
  const [size, setSize] = useState({ w: 800, h: 520 });
  const wrapRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);
  const hasFitRef = useRef(false);
  // Rotating labels: only LABEL_CAP node captions show at once and cycle
  // over time, so the graph reads as a mind surfacing memories rather than
  // a wall of text. Driven by a ref (mutated on a timer, read each frame)
  // so it animates without re-rendering React.
  const labelRef = useRef<Map<string, { in: number; out: number | null }>>(new Map());
  const visibleIdsRef = useRef<string[]>([]);
  // Freshness: when a poll reveals a new node or a new/strengthened edge,
  // we stamp it so it pulses for FRESH_MS — the visible "a connection just
  // formed" moment while you chat.
  const freshNodeRef = useRef<Map<string, number>>(new Map());
  const freshEdgeRef = useRef<Map<string, number>>(new Map());
  const prevNodeRef = useRef<Set<string>>(new Set());
  const prevEdgeRef = useRef<Map<string, number>>(new Map());

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

  // Ids touched by at least one currently-shown edge. Disconnected
  // memories are what fly off into empty space, so this lets us hide them.
  const connectedIds = useMemo(() => {
    const ids = new Set<string>();
    for (const l of graphData.links) {
      if (!showCoOccurred && l.edge_type === 'co_occurred') continue;
      ids.add(nodeId(l.source));
      ids.add(nodeId(l.target));
    }
    return ids;
  }, [graphData, showCoOccurred]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() =>
      setSize({ w: el.clientWidth, h: Math.max(420, window.innerHeight - 250) }),
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Tame the layout: cap how far the many-body repulsion reaches (so
  // disconnected nodes don't shoot off forever) and add a gentle pull
  // toward the centre, so the whole graph stays compact and framed.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || !active) return;
    fg.d3Force('charge')?.strength(-26).distanceMax(180);
    fg.d3Force('x', forceX(0).strength(0.07));
    fg.d3Force('y', forceY(0).strength(0.07));
    fg.d3ReheatSimulation?.();
  }, [active, graphData]);

  // Keep the pool of label-eligible (currently visible) ids fresh.
  useEffect(() => {
    visibleIdsRef.current = graphData.nodes.filter(visible).map((n) => n.id);
  });

  // Rotate which node captions are shown: fade a couple out, fade a couple
  // fresh ones in, holding ~LABEL_CAP at a time.
  useEffect(() => {
    if (!active || !animate) return;
    const tick = () => {
      const t = now_ms();
      const ref = labelRef.current;
      for (const [id, v] of ref) if (v.out !== null && t - v.out > LABEL_FADE) ref.delete(id);
      const living = [...ref].filter(([, v]) => v.out === null).sort((a, b) => a[1].in - b[1].in);
      const fadeCount = living.length >= LABEL_CAP ? 2 : 0;
      for (let i = 0; i < fadeCount; i++) living[i][1].out = t;
      const present = new Set(ref.keys());
      const pool = visibleIdsRef.current.filter((id) => !present.has(id));
      const need = LABEL_CAP - (living.length - fadeCount);
      for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
      pool.slice(0, Math.max(0, need)).forEach((id) => ref.set(id, { in: t, out: null }));
    };
    tick();
    const h = setInterval(tick, LABEL_ROTATE);
    return () => clearInterval(h);
  }, [active, animate]);

  // Diff each poll against the last: brand-new nodes and new/strengthened
  // edges get a freshness stamp so they pulse. The very first load only
  // records the baseline (no "everything just formed" flash on open).
  useEffect(() => {
    if (!data) return;
    const t = now_ms();
    const firstLoad = prevNodeRef.current.size === 0;

    const curNodes = new Set(data.nodes.map((n) => n.id));
    if (!firstLoad) {
      for (const id of curNodes) if (!prevNodeRef.current.has(id)) freshNodeRef.current.set(id, t);
    }
    prevNodeRef.current = curNodes;

    const curEdges = new Map<string, number>();
    for (const e of data.edges) {
      const k = `${e.src_id}|${e.dst_id}|${e.edge_type}`;
      curEdges.set(k, e.weight);
      if (!firstLoad) {
        const prev = prevEdgeRef.current.get(k);
        if (prev === undefined || e.weight > prev + 1e-6) freshEdgeRef.current.set(k, t);
      }
    }
    prevEdgeRef.current = curEdges;
  }, [data]);

  // Alpha for a node's label given the fade timeline; 0 means hidden.
  const labelAlpha = useCallback((id: string, t: number) => {
    const v = labelRef.current.get(id);
    if (!v) return 0;
    return v.out !== null
      ? Math.max(0, 1 - (t - v.out) / LABEL_FADE)
      : Math.min(1, (t - v.in) / LABEL_FADE);
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
      if (hideIsolated && !connectedIds.has(n.id)) return false;
      return liveDecay(n) >= minDecay;
    },
    [at, types, tagFilter, minDecay, liveDecay, hideIsolated, connectedIds],
  );

  const paintNode = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, scale: number) => {
      const d = liveDecay(node);
      const r = 2.5 + node.importance_score * 6;
      const color = node.stale ? '#57606a' : TYPE_COLORS[node.memory_type];
      const alpha = node.stale ? 0.3 : Math.max(0.14, Math.min(1, d * 1.7));
      const t = now_ms();
      // Gentle breathing glow; brighter for stronger (less-decayed) memories.
      const pulse = animate ? 0.5 + 0.5 * Math.sin(t / 700 + phaseOf(node.id)) : 0.6;
      // Just appeared? Flare brightly, fading over FRESH_MS.
      const fresh = freshness(freshNodeRef.current, node.id, t);

      ctx.save();
      // A spreading flash ring for a memory that just formed.
      if (fresh > 0) {
        const fr = r + (3 + 14 * (1 - fresh)) / scale;
        ctx.globalAlpha = fresh * 0.7;
        ctx.lineWidth = (1.5 * fresh) / scale;
        ctx.strokeStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, fr, 0, 2 * Math.PI);
        ctx.stroke();
      }
      // Coloured glow halo (extra bloom while fresh).
      ctx.globalAlpha = Math.min(1, alpha + 0.4 * fresh);
      ctx.shadowColor = color;
      ctx.shadowBlur = (2 + 5 * pulse) * (0.6 + d) + 8 * fresh;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI);
      ctx.fill();
      // Bright inner spark, so each note reads as a glowing bead.
      ctx.shadowBlur = 0;
      ctx.globalAlpha = Math.min(1, alpha + 0.35);
      ctx.fillStyle = node.stale ? '#9aa4ae' : 'rgba(255,255,255,0.9)';
      ctx.beginPath();
      ctx.arc(node.x!, node.y!, Math.max(0.7, r * 0.32), 0, 2 * Math.PI);
      ctx.fill();
      ctx.restore();

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

      // Rotating captions (or all, when zoomed in / animation off).
      const la = animate ? labelAlpha(node.id, t) : scale > 2.2 ? 1 : 0;
      const show = node.id === selected ? 1 : la;
      if (show > 0) {
        const fs = 10 / scale;
        ctx.globalAlpha = show * (node.stale ? 0.6 : 1);
        ctx.font = `${fs}px -apple-system, system-ui, sans-serif`;
        ctx.shadowColor = '#05080d';
        ctx.shadowBlur = 4 / scale;
        ctx.fillStyle = withAlpha(color, 0.55);
        ctx.fillRect(node.x! + r + 2 / scale, node.y! - fs * 0.7, 0.4 / scale, fs * 1.4); // a tiny tick
        ctx.fillStyle = '#e6edf3';
        ctx.fillText(node.content_preview.slice(0, 40), node.x! + r + 4 / scale, node.y! + fs * 0.34);
        ctx.shadowBlur = 0;
      }
      ctx.globalAlpha = 1;
    },
    [liveDecay, selected, animate, labelAlpha],
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
        <label className="ctl">
          <input type="checkbox" checked={showCoOccurred} onChange={(e) => setShowCoOccurred(e.target.checked)} />
          <span className="legend-swatch" style={{ background: EDGE_COLORS.co_occurred }} />
          co-occurrence edges
        </label>
        <label className="ctl">
          <input type="checkbox" checked={hideIsolated} onChange={(e) => setHideIsolated(e.target.checked)} />
          hide unconnected
        </label>
        <label className="ctl">
          <input type="checkbox" checked={animate} onChange={(e) => setAnimate(e.target.checked)} />
          ✨ living
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
            ref={fgRef}
            graphData={graphData}
            width={size.w}
            height={size.h}
            backgroundColor="#0a0d12"
            onEngineStop={() => {
              // Frame the whole graph once it settles — but only the first
              // time, so background polls don't fight the user's zoom/pan.
              if (!hasFitRef.current) {
                hasFitRef.current = true;
                fgRef.current?.zoomToFit(400, 60);
              }
            }}
            nodeId="id"
            nodeVal={(n: GraphNode) => 1 + n.importance_score * 5}
            nodeVisibility={visible}
            nodeCanvasObject={paintNode}
            nodeLabel={(n: GraphNode) =>
              `${n.memory_type}${n.pinned ? ' 📌' : ''}${n.stale ? ' (stale)' : ''}\n${n.content_preview}`
            }
            linkVisibility={(l: FgLink) =>
              (showCoOccurred || l.edge_type !== 'co_occurred') &&
              visible(l.source as GraphNode) &&
              visible(l.target as GraphNode)
            }
            linkColor={(l: FgLink) => {
              const base = EDGE_COLORS[l.edge_type] ?? '#30363d';
              const f = freshness(freshEdgeRef.current, edgeKey(l), now_ms());
              const a = (l.edge_type === 'co_occurred' ? 0.35 : 0.7) + 0.6 * f;
              return withAlpha(base, Math.min(1, a));
            }}
            linkWidth={(l: FgLink) =>
              0.4 + Math.min(2.5, l.weight) + 2.5 * freshness(freshEdgeRef.current, edgeKey(l), now_ms())
            }
            linkDirectionalArrowLength={(l: FgLink) => (DIRECTED.has(l.edge_type) ? 3 : 0)}
            linkDirectionalArrowRelPos={0.9}
            linkDirectionalArrowColor={(l: FgLink) => EDGE_COLORS[l.edge_type] ?? '#888'}
            linkDirectionalParticles={(l: FgLink) => {
              if (!animate) return 0;
              const f = freshness(freshEdgeRef.current, edgeKey(l), now_ms());
              if (f > 0) return 5; // a burst along a connection that just formed
              return DIRECTED.has(l.edge_type) ? 2 : 1;
            }}
            linkDirectionalParticleSpeed={(l: FgLink) => 0.004 + Math.min(0.012, l.weight * 0.004)}
            linkDirectionalParticleWidth={(l: FgLink) =>
              (DIRECTED.has(l.edge_type) ? 2 : 1.3) + 2 * freshness(freshEdgeRef.current, edgeKey(l), now_ms())
            }
            linkDirectionalParticleColor={(l: FgLink) => EDGE_COLORS[l.edge_type] ?? '#888'}
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
        <dt>salience</dt>
        <dd className="mono" style={m.salience >= 0.6 ? { color: 'var(--green)' } : {}}>
          {m.salience.toFixed(3)}
          {m.salience >= 0.6 ? ' ✦ kept' : ''}
        </dd>
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
