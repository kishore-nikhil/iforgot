import { useState } from 'react';
import { api, ConsolidationRun, fmtTs, SummaryProvenance } from '../api';
import { usePoll } from '../usePoll';

export default function ConsolidationView() {
  const { data, error } = usePoll(() => api.consolidations(20), 5000);

  return (
    <div className="view">
      <h2>Consolidation — the sleep cycle</h2>
      <p className="dim">
        Each run merges duplicates, summarizes topic clusters, promotes rehearsed memories, marks
        contradictions stale, and archives/prunes the decayed. Newest first.
      </p>
      {error && <div className="err">{error}</div>}
      {data?.runs.length === 0 && (
        <div className="panel dim">
          No consolidation runs logged yet — run <code>forgetfuldb consolidate</code> (or{' '}
          <code>/consolidate</code> in chat) and this timeline fills in.
        </div>
      )}
      {data?.runs.map((run) => <RunCard key={run.id} run={run} />)}
    </div>
  );
}

function RunCard({ run }: { run: ConsolidationRun }) {
  const memoriesIn =
    run.duplicates_merged + run.summaries.reduce((acc, s) => acc + s.source_ids.length, 0);
  const memoriesOut = run.summaries.length;

  return (
    <details className="run panel">
      <summary>
        {fmtTs(run.ran_at)}
        <span className="counts">
          {' '}
          — {run.duplicates_merged} merged · {run.clusters_summarized} summarized · {run.promoted}{' '}
          promoted · {run.marked_stale} stale · {run.archived} archived · {run.pruned} pruned
        </span>
      </summary>
      <div style={{ marginTop: 10 }}>
        {memoriesIn > 0 && (
          <div className="dim" style={{ marginBottom: 8 }}>
            compression: {memoriesIn} memories in → {memoriesOut || run.duplicates_merged} out
          </div>
        )}
        {run.summaries.length === 0 ? (
          <div className="dim">No cluster summaries this run.</div>
        ) : (
          run.summaries.map((s) => <SummaryDiff key={s.summary_id} prov={s} />)
        )}
      </div>
    </details>
  );
}

function SummaryDiff({ prov }: { prov: SummaryProvenance }) {
  const [summary, setSummary] = useState<string | null>(null);
  const [sources, setSources] = useState<{ id: string; text: string }[] | null>(null);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    if (sources || loading) return;
    setLoading(true);
    try {
      const sum = await api.memory(prov.summary_id).catch(() => null);
      setSummary(sum?.memory.content ?? '(summary memory no longer exists)');
      const srcs = await Promise.all(
        prov.source_ids.map(async (id) => {
          const d = await api.memory(id).catch(() => null);
          return { id, text: d?.memory.content ?? '(pruned)' };
        }),
      );
      setSources(srcs);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="diff-group">
      <details onToggle={(e) => (e.target as HTMLDetailsElement).open && load()}>
        <summary>
          {prov.source_ids.length} memories → 1 summary{' '}
          <span className="dim mono">{prov.summary_id.slice(0, 14)}…</span>
        </summary>
        {loading && <div className="dim">loading…</div>}
        {sources?.map((s) => (
          <div key={s.id} className="diff-src">
            {s.text.slice(0, 110)}
            {s.text.length > 110 ? '…' : ''}
          </div>
        ))}
        {summary && <div className="diff-sum">⤷ {summary}</div>}
      </details>
    </div>
  );
}
