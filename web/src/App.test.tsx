import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { fixtures, json, mockApi } from "./test/mockApi";

beforeEach(() => {
  window.history.replaceState(null, "", "/");
  vi.unstubAllGlobals();
});

const searches = (calls: ReturnType<typeof mockApi>) => calls.filter((c) => c.url === "/api/search");

describe("App shell", () => {
  it("shows the collections, disabling the ones that are not built", async () => {
    mockApi();
    render(<App />);
    const group = await screen.findByRole("radiogroup", { name: "Collection" });
    expect(within(group).getByRole("radio", { name: "CS" })).toBeChecked();
    expect(within(group).getByRole("radio", { name: "NFCorpus" })).toBeDisabled();
    expect(screen.getByRole("heading", { level: 1, name: "ContextWeave" })).toBeInTheDocument();
  });

  it("explains how to start the API when it is unreachable", async () => {
    mockApi((url) => (url === "/api/collections" ? new Response("", { status: 502, statusText: "Bad Gateway" }) : undefined));
    render(<App />);
    expect(await screen.findByText(/Cannot reach the ContextWeave API/)).toBeInTheDocument();
    expect(screen.getByText("python main.py serve")).toBeInTheDocument();
  });

  it("navigates between pages and keeps the collection in the URL", async () => {
    mockApi();
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("radio", { name: /Evaluation/ }));
    expect(window.location.pathname).toBe("/evaluation");
    expect(new URLSearchParams(window.location.search).get("c")).toBe("cs");
    expect(await screen.findByText(/No relevance labels yet/)).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: /Search/ }));
    expect(window.location.pathname).toBe("/");
    expect(screen.getByRole("textbox", { name: "Query" })).toBeInTheDocument();
  });
});

describe("Search page", () => {
  it("runs a query with the collection's tuned defaults and shows how it decided", async () => {
    const calls = mockApi();
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Query"), "semaphore lock{Enter}");

    expect(await screen.findByRole("heading", { name: "Results" })).toBeInTheDocument();
    const [req] = searches(calls);
    const cs = fixtures.collections.collections.find((c) => c.id === "cs")!;
    expect(req.body).toMatchObject({ collection: "cs", query: "semaphore lock", mode: "auto", tau: cs.defaults.tau, budget: cs.defaults.budget });
    expect(new URLSearchParams(window.location.search).get("q")).toBe("semaphore lock");

    const results = within(screen.getByRole("region", { name: "Results" })).getAllByRole("listitem");
    expect(results).toHaveLength(fixtures.searchAuto.results.length);
    expect(results[0]).toHaveTextContent(fixtures.searchAuto.results[0].title);
    expect(results[0].querySelector("mark")).not.toBeNull(); // query terms highlighted
    expect(screen.getByRole("list", { name: "Decision path" })).toBeInTheDocument();
    expect(screen.getByText(/confidence stayed below τ/)).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "What each method retrieved" })).not.toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("button", { name: "Show the full analysis" }));
    expect(screen.getByRole("region", { name: "How the query was read" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "What each method retrieved" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Query expansion" })).toBeInTheDocument();
  });

  it("explains a result and opens the full passage", async () => {
    mockApi();
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Query"), "semaphore{Enter}");
    const first = (await screen.findAllByRole("button", { name: /Why this result/ }))[0];
    await user.click(first);
    expect(first).toHaveAttribute("aria-expanded", "true");
    expect(screen.getAllByText(/Query terms in this passage/)[0]).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: fixtures.searchAuto.results[0].title }));
    const dialog = await screen.findByRole("dialog", { name: "Full passage" });
    expect(await within(dialog).findByText("Full passage text about semaphores.")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("re-runs the query when the method or a setting changes", async () => {
    const calls = mockApi();
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: fixtures.collections.collections[0].examples[0] }));
    await screen.findByRole("heading", { name: "Results" });

    await user.selectOptions(screen.getByLabelText("Method"), "bm25");
    await waitFor(() => expect(searches(calls).at(-1)!.body).toMatchObject({ mode: "bm25" }));
    expect(new URLSearchParams(window.location.search).get("mode")).toBe("bm25");
    expect(await screen.findByText(/Fixed method: no confidence check/)).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Method"), "tfidf");
    await waitFor(() => expect(searches(calls).at(-1)!.body).toMatchObject({ mode: "tfidf" }));

    await user.click(screen.getByRole("button", { name: /^Settings/ }));
    await user.click(screen.getByRole("switch", { name: /Query expansion in the ladder/ }));
    await waitFor(() => expect(searches(calls).at(-1)!.body).toMatchObject({ expansion: false }));
    expect(screen.getByRole("button", { name: /Reset to tuned defaults/ })).toBeEnabled();
  });

  it("disables extensions that have no fitted model and filters by domain", async () => {
    const calls = mockApi();
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: /^Settings/ }));
    const cs = fixtures.collections.collections.find((c) => c.id === "cs")!;
    const voc = screen.getByRole("switch", { name: /Value-of-computation/ });
    if (cs.features.voc_policy) expect(voc).toBeEnabled();
    else expect(voc).toBeDisabled();
    await user.type(screen.getByLabelText("Query"), "deadlock{Enter}");
    await screen.findByRole("heading", { name: "Results" });
    if (cs.domains.length) {
      await user.click(screen.getByRole("button", { name: cs.domains[0] }));
      await waitFor(() => expect(searches(calls).at(-1)!.body).toMatchObject({ domains: [cs.domains[0]] }));
    }
  });

  it("shows the server's message when a search fails", async () => {
    mockApi((url) => (url === "/api/search" ? json({ detail: "collection 'cs' is not built yet" }, 409) : undefined));
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Query"), "x{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("collection 'cs' is not built yet");
  });

  it("runs the query in a shared link straight away", async () => {
    window.history.replaceState(null, "", "/?c=cs&q=tf-idf%20weight&mode=bm25");
    const calls = mockApi();
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Results" })).toBeInTheDocument();
    expect(searches(calls)[0].body).toMatchObject({ query: "tf-idf weight", mode: "bm25" });
    expect(screen.getByLabelText("Query")).toHaveValue("tf-idf weight");
  });
});
