import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { Collection } from "../lib/api";
import { fixtures, mockApi } from "../test/mockApi";
import { JudgePage } from "./JudgePage";

const cs = { ...fixtures.collections.collections.find((c) => c.id === "cs")!, ready: true } as unknown as Collection;
const passage = { ...fixtures.searchAuto.results[0], text: "Semaphores let processes wait.", n_tokens: 4 };
const state = (done: number) => ({
  pool_ready: true,
  done,
  total: 2,
  queries: [{ qid: "q1", text: "what is a semaphore", type: "semantic", domain: "OS", judged: done, total: 2 }],
});
const item = { qid: "q1", query: "what is a semaphore", type: "semantic", domain: "OS", passage, grade: null, position: 1, of: 2 };

beforeEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("Judge page", () => {
  it("is only for the cs collection", () => {
    mockApi();
    const sci = fixtures.collections.collections.find((c) => c.id === "scifact") as unknown as Collection;
    render(<JudgePage collection={sci} />);
    expect(screen.getByText(/already judged/)).toBeInTheDocument();
  });

  it("grades with a click or the keyboard and moves to the next passage", async () => {
    let done = 0;
    const calls = mockApi((url, method) => {
      if (url === "/api/judge/cs" && method === "GET") return state(done);
      if (url.startsWith("/api/judge/cs/item")) return { item: done < 2 ? { ...item, position: done + 1 } : null, ...state(done) };
      if (url === "/api/judge/cs" && method === "POST") {
        done += 1;
        return state(done);
      }
      return undefined;
    });
    const user = userEvent.setup();
    render(<JudgePage collection={cs} />);
    expect(await screen.findByRole("heading", { name: "what is a semaphore" })).toBeInTheDocument();
    expect(screen.getByText("Semaphores let processes wait.")).toBeInTheDocument();

    await user.type(screen.getByLabelText("Your name"), "Ada L");
    await user.click(screen.getByRole("button", { name: /Highly relevant/ }));
    await waitFor(() => expect(screen.getByText(/passage 2 of 2/)).toBeInTheDocument());
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.body).toEqual({ qid: "q1", passage_id: passage.passage_id, grade: 2, annotator: "Ada L" });
    expect(localStorage.getItem("cw.annotator")).toBe("Ada L");

    (document.activeElement as HTMLElement).blur();
    await user.keyboard("0");
    expect(await screen.findByText(/Everything is judged/)).toBeInTheDocument();
    expect(calls.filter((c) => c.method === "POST").at(-1)!.body).toMatchObject({ grade: 0 });
  });

  it("offers to build the pool when there is none", async () => {
    mockApi((url, method) =>
      url === "/api/judge/cs" && method === "GET" ? { pool_ready: false, done: 0, total: 0, queries: [] } : undefined,
    );
    render(<JudgePage collection={cs} />);
    expect(await screen.findByRole("button", { name: /Build the pool/ })).toBeInTheDocument();
  });
});
