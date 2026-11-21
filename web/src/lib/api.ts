// Typed client for the FastAPI server (api/server.py). Shapes mirror api/serialize.py.

export type Mode = "auto" | "bm25" | "tfidf" | "dense" | "hybrid" | "boolean" | "bm25_champion" | "route";
export type FeatureFlag = "qpp_signals" | "voc_policy" | "risk_controlled_tau" | "qpp_fusion";

export interface Collection {
  id: string;
  label: string;
  short: string;
  description: string;
  examples: string[];
  ready: boolean;
  passages: number | null;
  domains: string[];
  defaults: { tau: number; budget: number; source: string };
  expansion_default: boolean;
  costs: Record<string, number>;
  max_budget: number;
  features: Record<FeatureFlag, boolean>;
  has_results: boolean;
  judging: boolean;
}

export interface CollectionsResponse {
  collections: Collection[];
  modes: Mode[];
  default: string;
}

export interface Segment extends Array<string | boolean> {
  0: string;
  1: boolean;
}

export interface TermContribution {
  term: string;
  tf: number | null;
  idf: number | null;
  weight: number | null;
  score: number | null;
  share: number | null;
}

export interface PassageRef {
  passage_id: string;
  doc_id: string;
  title: string;
  page: number | null;
  domain: string;
  source: string;
}

export interface Result extends PassageRef {
  rank: number;
  score: number;
  method: string;
  snippet: { segments: [string, boolean][]; before: boolean; after: boolean; hits: number };
  why: {
    alpha: number | null;
    lexical_method: string | null;
    lexical_norm: number | null;
    semantic_norm: number | null;
    cosine: number | null;
    in_lexical: boolean | null;
    in_semantic: boolean | null;
    terms: TermContribution[];
    eliminated: string[];
  };
}

export interface TraceStep {
  action: "route" | "retrieve" | "confidence" | "escalate" | "stop" | string;
  method: string | null;
  cost: number;
  budget_left: number;
  confidence: number | null;
  signals: Record<string, number | boolean | null>;
  latency_ms: number;
  note: string;
}

export interface MethodList {
  label: string;
  kind: "lexical" | "semantic";
  stats: Record<string, number>;
  eliminated_terms: string[];
  top: (PassageRef & { rank: number; score: number; terms: TermContribution[] })[];
}

export interface SearchResponse {
  query: { raw: string; terms: { term: string; df: number; idf: number; bm25_idf: number; known: boolean }[] };
  mode: Mode;
  total_cost: number;
  latency_ms: number;
  budget: number | null;
  tau: number | null;
  tau_source: string | null;
  stop_reason: string;
  methods_run: string[];
  route: { method: string | null; note: string; features: Record<string, number | boolean | null> } | null;
  trace: TraceStep[];
  results: Result[];
  lists: Record<string, MethodList>;
  agreement: { lexical: string; overlap: number; of: number } | null;
  expansion: { added_terms: Record<string, number | string>[]; feedback: string[] } | null;
}

export interface SearchRequest {
  collection: string;
  query: string;
  mode: Mode;
  budget?: number;
  tau?: number;
  expansion?: boolean;
  features?: Partial<Record<FeatureFlag, boolean>>;
  domains?: string[];
}

export interface Passage extends PassageRef {
  text: string;
  n_tokens: number;
}

export interface SystemRow {
  id: string;
  label: string;
  metrics: Record<"P@5" | "Recall@10" | "MAP" | "NDCG@10" | "MRR", number | null> | null;
  mean_cost: number | null;
  escalation_rate: number | null;
  latency_median_ms: number | null;
  latency_p95_ms: number | null;
  by_type: Record<string, { "NDCG@10": number | null; n: number; mean_cost: number | null }>;
  tau: number | null;
  budget: number | null;
}

export interface EvaluationResult {
  collection: string;
  split: string;
  n_queries: number;
  timestamp_utc: string;
  labelled: boolean;
  hardware: { cpu: string | null; cpu_count: number | null; platform: string | null };
  systems: SystemRow[];
  ablations: SystemRow[];
  sweep: { tau: number; budget: number; cost: number; ndcg: number | null }[];
  risk_curve: {
    delta: number;
    tau: number | null;
    certified_on_val: boolean;
    test_failure_rate: number | null;
    test_ndcg: number | null;
    test_cost: number | null;
  }[];
  significance: { a: string; b: string; mean_diff: number; p_holm: number; wilcoxon_p: number; ci95: number[] }[];
}

export interface EvaluationResponse {
  available: string[];
  result: EvaluationResult | null;
}

export interface JudgeQuery {
  qid: string;
  text: string;
  type: string;
  domain: string;
  judged: number;
  total: number;
}

export interface JudgeState {
  pool_ready: boolean;
  done: number;
  total: number;
  queries: JudgeQuery[];
}

export interface JudgeItem {
  qid: string;
  query: string;
  type: string;
  domain: string;
  passage: Passage;
  grade: number | null;
  position: number;
  of: number;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") msg = body.detail;
      else if (Array.isArray(body.detail)) msg = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  collections: () => call<CollectionsResponse>("/api/collections"),
  search: (req: SearchRequest, signal?: AbortSignal) =>
    call<SearchResponse>("/api/search", { method: "POST", body: JSON.stringify(req), signal }),
  passage: (c: string, pid: string) =>
    call<Passage>(`/api/collections/${encodeURIComponent(c)}/passages/${encodeURIComponent(pid)}`),
  evaluation: (c: string, split?: string) =>
    call<EvaluationResponse>(`/api/evaluation/${encodeURIComponent(c)}${split ? `?split=${encodeURIComponent(split)}` : ""}`),
  judge: (c: string) => call<JudgeState>(`/api/judge/${encodeURIComponent(c)}`),
  judgeItem: (c: string, qid?: string, pid?: string) => {
    const p = new URLSearchParams();
    if (qid) p.set("qid", qid);
    if (pid) p.set("passage_id", pid);
    const qs = p.toString();
    return call<{ item: JudgeItem | null } & JudgeState>(`/api/judge/${encodeURIComponent(c)}/item${qs ? `?${qs}` : ""}`);
  },
  judgeSave: (c: string, body: { qid: string; passage_id: string; grade: number; annotator: string }) =>
    call<JudgeState>(`/api/judge/${encodeURIComponent(c)}`, { method: "POST", body: JSON.stringify(body) }),
  buildPool: (c: string) => call<JudgeState>(`/api/judge/${encodeURIComponent(c)}/pool`, { method: "POST" }),
};
