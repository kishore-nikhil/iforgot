// Typed client for the forgetfuldb-server read-only API. The app is
// served from the same origin at /ui, so paths are root-relative.

export type MemoryTypeName =
  | 'raw_event'
  | 'episodic'
  | 'semantic'
  | 'procedural'
  | 'preference'
  | 'archive';

export const MEMORY_TYPES: MemoryTypeName[] = [
  'raw_event',
  'episodic',
  'semantic',
  'procedural',
  'preference',
  'archive',
];

export const TYPE_COLORS: Record<MemoryTypeName, string> = {
  raw_event: '#8b949e',
  episodic: '#58a6ff',
  semantic: '#3fb950',
  procedural: '#d29922',
  preference: '#bc8cff',
  archive: '#484f58',
};

export const EDGE_COLORS: Record<string, string> = {
  co_occurred: '#2dd4bf', // recalled together (Hebbian)
  semantic_similar: '#a371f7', // close in meaning (cosine kNN)
  sequence: '#f778ba', // discussed one after another (causal, directed)
  derived_from: '#3fb950',
  updates: '#d29922',
  contradicts: '#f85149',
  supports: '#58a6ff',
  duplicates: '#8b949e',
  belongs_to_project: '#bc8cff',
};

export interface GraphNode {
  id: string;
  content_preview: string;
  memory_type: MemoryTypeName;
  importance_score: number;
  decay_score: number;
  recurrence_score: number;
  pinned: boolean;
  stale: boolean;
  salience: number;
  created_at: number;
  last_accessed_at: number | null;
  tags: string[];
  topic: string | null;
  // added by force-graph at runtime
  x?: number;
  y?: number;
}

export interface GraphEdge {
  src_id: string;
  dst_id: string;
  edge_type: string;
  weight: number;
  co_count?: number;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_count: number;
  window: { since: number; until: number | null };
  generated_at: number;
}

export interface ScoreBreakdown {
  semantic_similarity: number;
  importance: number;
  recurrence: number;
  recency: number;
  pinned_boost: number;
  staleness_penalty: number;
  conversational_damping: number;
  association_boost?: number;
  salience?: number;
  total: number;
}

export interface RetrievedMemory {
  id: string;
  content: string;
  memory_type: MemoryTypeName;
  topic: string | null;
  tags: string[];
  pinned: boolean;
  stale: boolean;
  created_at: number;
  score: ScoreBreakdown;
}

export interface RetrieveResponse {
  query: string;
  generated_at: number;
  memories: RetrievedMemory[];
  near_misses?: RetrievedMemory[];
  min_score: number;
  retrieve_ms: number;
}

export interface UiConfig {
  name: string;
  db_path: string;
  db_size_bytes: number;
  decay_lambdas: Record<MemoryTypeName, number>;
  retrieval_weights: {
    semantic: number;
    importance: number;
    recurrence: number;
    recency: number;
    pinned_boost: number;
    staleness_penalty: number;
  };
  chat: { top_k: number; min_retrieval_score: number; conversational_damping: number };
  embedding?: { backend: string; model: string; dim: number };
  salience?: { resist: number; keep_threshold: number; spreading_activation: boolean };
}

export interface ChatTurn {
  id: string;
  session_id: string | null;
  created_at: number;
  user_text: string;
  assistant_text: string;
  model: string;
  backend: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_duration_ms: number | null;
  llm_duration_ms: number | null;
  retrieve_duration_ms: number;
  context_memory_count: number;
  context_chars: number;
  memory_ids: string[];
}

export interface SummaryProvenance {
  summary_id: string;
  source_ids: string[];
}

export interface ConsolidationRun {
  id: string;
  ran_at: number;
  duplicates_merged: number;
  recurrence_updated: number;
  clusters_summarized: number;
  promoted: number;
  marked_stale: number;
  archived: number;
  pruned: number;
  summaries: SummaryProvenance[];
}

export interface MemoryDetail {
  memory: RetrievedMemory & {
    summary: string | null;
    source: string | null;
    entities: string[];
    salience: number;
    access_count: number;
    importance_score: number;
    recurrence_score: number;
    recency_score: number;
    decay_score: number;
    confidence: number;
    updated_at: number;
    last_accessed_at: number | null;
  };
  links: { source_id: string; target_id: string; relation: string }[];
}

export interface StoreStats {
  total_memories: number;
  by_type: Record<string, number>;
  stale: number;
  pinned: number;
  raw_events: number;
  links: number;
  sessions: number;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
  return res.json();
}

export const api = {
  graph: (params: { since?: number; until?: number; types?: string[] }) => {
    const q = new URLSearchParams();
    if (params.since !== undefined) q.set('since', String(params.since));
    if (params.until !== undefined) q.set('until', String(params.until));
    if (params.types?.length) q.set('types', params.types.join(','));
    return getJson<GraphResponse>(`/graph?${q}`);
  },
  uiconfig: () => getJson<UiConfig>('/uiconfig'),
  stats: () => getJson<StoreStats>('/stats'),
  turns: (limit = 300) => getJson<{ turns: ChatTurn[] }>(`/turns?limit=${limit}`),
  consolidations: (limit = 20) => getJson<{ runs: ConsolidationRun[] }>(`/consolidations?limit=${limit}`),
  memory: (id: string) => getJson<MemoryDetail>(`/memory/${id}`),
  retrieve: async (body: {
    query: string;
    top_k?: number;
    include_stale?: boolean;
    min_score?: number;
    memory_types?: string[];
    since?: number;
    until?: number;
  }): Promise<RetrieveResponse> => {
    const res = await fetch('/retrieve', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ...body, debug: true }),
    });
    if (!res.ok) throw new Error(`retrieve: ${res.status} ${await res.text()}`);
    return res.json();
  },
  setPinned: async (id: string, pinned: boolean) => {
    const res = await fetch(`/memory/${id}/pin`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ pinned }),
    });
    if (!res.ok) throw new Error(await res.text());
  },
  archive: async (id: string) => {
    const res = await fetch(`/memory/${id}/archive`, { method: 'POST' });
    if (!res.ok) throw new Error(await res.text());
  },
};

/** Client-side decay, matching forgetfuldb-core::decay::decay_score. */
export function decayAt(
  importance: number,
  lambda: number,
  createdAt: number,
  atUnix: number,
  pinned: boolean,
): number {
  if (pinned) return importance;
  const ageDays = Math.max(0, (atUnix - createdAt) / 86_400);
  return importance * Math.exp(-lambda * ageDays);
}

export function fmtTs(unix: number | null | undefined): string {
  if (!unix) return '—';
  return new Date(unix * 1000).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function fmtBytes(n: number): string {
  if (n > 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MiB`;
  if (n > 1 << 10) return `${(n / (1 << 10)).toFixed(1)} KiB`;
  return `${n} B`;
}
