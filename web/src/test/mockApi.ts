import { vi } from "vitest";
import collections from "./fixtures/collections.json";
import evaluationCs from "./fixtures/evaluation_cs.json";
import evaluationScifact from "./fixtures/evaluation_scifact.json";
import searchAuto from "./fixtures/search_auto.json";
import searchBm25 from "./fixtures/search_bm25.json";

export const fixtures = { collections, searchAuto, searchBm25, evaluationCs, evaluationScifact };

export interface Call {
  url: string;
  method: string;
  body: unknown;
}

type Handler = (url: string, method: string, body: unknown) => unknown | Response | undefined;

/** Stub window.fetch. Default routes serve the fixtures; `extra` handlers run first. Returns the call log. */
export function mockApi(extra?: Handler) {
  const calls: Call[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url, method, body });
    const custom = extra?.(url, method, body);
    if (custom instanceof Response) return custom;
    if (custom !== undefined) return json(custom);
    if (url === "/api/collections") return json(collections);
    if (url === "/api/search") return json((body as { mode: string }).mode === "bm25" ? searchBm25 : searchAuto);
    if (url.startsWith("/api/evaluation/cs")) return json(evaluationCs);
    if (url.startsWith("/api/evaluation/")) return json(evaluationScifact);
    if (url.startsWith("/api/collections/cs/passages/"))
      return json({ ...searchAuto.results[0], text: "Full passage text about semaphores.", n_tokens: 5 });
    return new Response(JSON.stringify({ detail: `no mock for ${method} ${url}` }), { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

export const json = (x: unknown, status = 200) =>
  new Response(JSON.stringify(x), { status, headers: { "Content-Type": "application/json" } });
