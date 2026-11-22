import { expect, test, type Page } from "@playwright/test";

// Real server, real indexes, real MiniLM model. Screenshots go to web/test-results/screens/ for review.
const shot = (page: Page, name: string) => page.screenshot({ path: `test-results/screens/${name}.png`, fullPage: true });

async function search(page: Page, q: string) {
  const box = page.getByLabel("Query");
  await box.fill(q);
  await box.press("Enter");
  await settle(page);
}

/** Wait until the current search finished: results shown, or an error alert (whose text then fails the test). */
async function settle(page: Page) {
  const results = page.getByRole("heading", { name: "Results" });
  const alert = page.getByRole("alert");
  await expect(results.or(alert).first()).toBeVisible();
  if (await alert.count()) throw new Error(`search failed: ${await alert.first().innerText()}`);
  await expect(page.locator("[aria-busy=true]")).toHaveCount(0);
}

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (e) => {
    throw e;
  });
});

test("collections: built ones are selectable, cs explains how to build it", async ({ page }) => {
  await page.goto("/");
  const group = page.getByRole("radiogroup", { name: "Collection" });
  await expect(group.getByRole("radio", { name: "SciFact" })).toBeEnabled();
  await expect(group.getByRole("radio", { name: "NFCorpus" })).toBeEnabled();
  const api = await (await page.request.get("/api/collections")).json();
  const cs = api.collections.find((c: { id: string }) => c.id === "cs");
  if (!cs.ready) await expect(group.getByRole("radio", { name: "CS" })).toBeDisabled();
});

test("a short exact query stops after one cheap lexical pass", async ({ page }) => {
  await page.goto("/?c=nfcorpus");
  await search(page, "statins");
  const path = page.getByRole("list", { name: "Decision path" });
  await expect(path).toContainText("Route");
  await expect(path).toContainText("Stop");
  await expect(page.getByText("Stopped early: the results already looked good enough.")).toBeVisible();
  await expect(page.getByRole("region", { name: "Results" }).getByRole("listitem")).toHaveCount(10);
  await expect(page.getByRole("region", { name: "Results" }).locator("mark").first()).toBeVisible();
  await shot(page, "01-search-exact");
});

test("a question is routed to semantic search and fused", async ({ page }) => {
  await page.goto("/?c=nfcorpus");
  await search(page, "Why Deep Fried Foods May Cause Cancer");
  const path = page.getByRole("list", { name: "Decision path" });
  await expect(path).toContainText("semantic");
  await expect(path).toContainText("Dense");
  await page.getByRole("button", { name: /Why this result/ }).first().click();
  await expect(page.getByText(/score = α·lexical/).first()).toBeVisible();
  await shot(page, "02-search-semantic");
});

test("switching collections searches the right index every time (regression)", async ({ page }) => {
  await page.goto("/?c=scifact");
  await search(page, "statins heart");
  const titles = async () => page.getByRole("region", { name: "Results" }).locator("h3").allInnerTexts();
  const sci1 = await titles();
  const group = page.getByRole("radiogroup", { name: "Collection" });
  await group.getByRole("radio", { name: "NFCorpus" }).click();
  await expect(page.getByRole("heading", { name: "Results" })).toBeVisible();
  await expect.poll(titles).not.toEqual(sci1);
  await group.getByRole("radio", { name: "SciFact" }).click();
  await expect.poll(titles).toEqual(sci1);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("every method runs, and the full passage opens", async ({ page }) => {
  await page.goto("/?c=scifact");
  await search(page, "ALDH1 breast cancer prognosis");
  for (const m of ["bm25", "dense", "hybrid", "auto"]) {
    await page.getByLabel("Method").selectOption(m);
    await expect(page.locator("[aria-busy=true]")).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Results" }).getByRole("listitem").first()).toBeVisible();
  }
  for (const m of ["tfidf", "boolean", "bm25_champion", "route"]) {
    await page.getByLabel("Method").selectOption(m);
    await expect(page.locator("[aria-busy=true]")).toHaveCount(0);
    await expect(page.getByRole("alert")).toHaveCount(0);
  }
  await page.getByRole("region", { name: "Results" }).locator("h3 button").first().click();
  const dialog = page.getByRole("dialog", { name: "Full passage" });
  await expect(dialog.getByText(/Source:/)).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
});

test("controller settings change the run", async ({ page }) => {
  await page.goto("/?c=nfcorpus");
  await search(page, "Why Deep Fried Foods May Cause Cancer");
  await page.getByRole("button", { name: /^Settings/ }).click();
  const tau = page.getByLabel("Confidence threshold");
  await tau.focus();
  for (let i = 0; i < 20; i++) await tau.press("ArrowLeft"); // τ = 0: always confident after the first step
  await expect(page.getByRole("list", { name: "Decision path" })).toContainText("≥ τ 0.00");
  await expect(page.getByText("Methods run").locator("..")).toContainText("1");
  await shot(page, "03-settings");
  await page.getByRole("button", { name: /Reset to tuned defaults/ }).click();
  await expect(page.getByRole("button", { name: /Reset to tuned defaults/ })).toBeDisabled();
});

test("evaluation page reads the saved results", async ({ page }) => {
  await page.goto("/evaluation?c=scifact");
  await expect(page.getByText("Controller quality")).toBeVisible();
  await expect(page.getByRole("figure", { name: "Quality versus cost" })).toBeVisible();
  const saved = await (await page.request.get("/api/evaluation/scifact")).json();
  await expect(page.getByRole("region", { name: "All systems" }).getByRole("row")).toHaveCount(saved.result.systems.length + 1);
  if (saved.available.includes("val")) {  // the split switch only appears when both splits were saved
    await page.getByRole("radio", { name: "Validation" }).click();
    await expect(page.getByText(/val queries/).first()).toBeVisible();
    await page.getByRole("radio", { name: "Test" }).click();
  }
  await expect(page.getByText(/test queries/).first()).toBeVisible();
  await shot(page, "04-evaluation");
});

test("judge page explains it is for the cs benchmark", async ({ page }) => {
  await page.goto("/judge?c=scifact");
  await expect(page.getByText(/already judged/)).toBeVisible();
  await shot(page, "05-judge");
});

test("phone width: no horizontal scroll", async ({ browser }) => {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await page.goto("/?c=nfcorpus&q=Why%20Deep%20Fried%20Foods%20May%20Cause%20Cancer");
  await expect(page.getByRole("heading", { name: "Results" })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await shot(page, "06-mobile");
  await page.goto("/evaluation?c=scifact");
  await expect(page.getByRole("figure", { name: "Quality versus cost" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
  await shot(page, "07-mobile-evaluation");
  await page.close();
});
