import { useEffect, useRef, useState } from 'react';

// One shared SSE connection feeds every poller, so the whole UI refreshes
// the instant the store changes (a chat turn, a consolidation, a pin) —
// the interval becomes a slow fallback, not the primary mechanism.
const liveListeners = new Set<() => void>();
let liveSource: EventSource | null = null;

function ensureLive() {
  if (liveSource || typeof EventSource === 'undefined') return;
  try {
    liveSource = new EventSource('/events');
    liveSource.onmessage = () => liveListeners.forEach((cb) => cb());
    // EventSource auto-reconnects on error; nothing else to do.
  } catch {
    liveSource = null; // poll-only fallback
  }
}

/** Poll an async fetcher on an interval (a fallback) AND refetch instantly
 *  on a server `change` push. Pauses when the tab is hidden. `deps`
 *  re-triggers an immediate fetch. */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    ensureLive();
    let alive = true;
    const tick = async () => {
      if (document.hidden) return;
      try {
        const d = await fetcherRef.current();
        if (alive) {
          setData(d);
          setError(null);
        }
      } catch (e) {
        if (alive) setError(String(e));
      }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    // Refetch immediately whenever the store changes (SSE push).
    const onLive = () => tick();
    liveListeners.add(onLive);
    return () => {
      alive = false;
      clearInterval(id);
      liveListeners.delete(onLive);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...deps]);

  return { data, error };
}
