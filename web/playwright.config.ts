import { defineConfig, devices } from "@playwright/test";

// End-to-end tests against the real server, real indexes and the real embedding model.
// Uses the Microsoft Edge already installed on Windows (no browser download). On macOS/Linux set
// PW_CHANNEL=chrome, or run `npx playwright install chromium` and PW_CHANNEL=chromium.
// Needs: the UI built (npm run build) and at least the scifact + nfcorpus indexes.
const channel = process.env.PW_CHANNEL ?? "msedge";
const port = Number(process.env.PW_PORT ?? 8765);

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    channel: channel === "chromium" ? undefined : channel,
    trace: "retain-on-failure",
  },
  projects: [{ name: "desktop", use: { ...devices["Desktop Edge"], channel: channel === "chromium" ? undefined : channel, viewport: { width: 1280, height: 900 } } }],
  webServer: {
    command: `python main.py serve --port ${port}`,
    cwd: "..",
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: true,
    timeout: 120_000,
    stdout: "ignore",
    stderr: "pipe",
  },
});
