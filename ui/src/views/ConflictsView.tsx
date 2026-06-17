import { useState } from 'react';
import { api, fmtTs } from '../api';
import { usePoll } from '../usePoll';

/** Active supersessions: one memory marked an older one out of date. The newer
 *  is the source of truth; the user can revive a loser if the call was wrong. */
export default function ConflictsView() {
  const { data } = usePoll(() => api.conflicts(), 5000);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<string | null>(null);

  const conflicts = (data?.conflicts ?? []).filter((c) => !hidden.has(c.loser.id));

  const revive = async (id: string) => {
    setBusy(id);
    try {
      await api.revive(id);
      setHidden((h) => new Set(h).add(id)); // optimistic; SSE confirms
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="view">
      <h2>Conflicts</h2>
      <p className="conflict-hint">
        One memory superseded another (the newer is treated as the source of truth and the older is hidden from
        retrieval). Revive a memory if the supersession was wrong.
      </p>
      {conflicts.length === 0 ? (
        <div className="conflict-empty">No active supersessions.</div>
      ) : (
        <div className="conflict-list">
          {conflicts.map((c) => (
            <div className="conflict-card" key={c.loser.id}>
              <div className="conflict-row">
                <span className="conflict-badge keep">current</span>
                <span className="conflict-content">{c.winner.content}</span>
                <span className="conflict-date">{fmtTs(c.winner.created_at)}</span>
              </div>
              <div className="conflict-row superseded">
                <span className="conflict-badge stale">superseded</span>
                <span className="conflict-content">{c.loser.content}</span>
                <span className="conflict-date">{fmtTs(c.loser.created_at)}</span>
                <button className="conflict-revive" disabled={busy === c.loser.id} onClick={() => revive(c.loser.id)}>
                  {busy === c.loser.id ? '…' : 'Revive'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
