import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

const baseUrl = process.env.SHOTSEEK_E2E_URL || "http://127.0.0.1:8765";
const chromium = process.env.CHROMIUM_PATH || "/snap/bin/chromium";
const screenshot = resolve(process.cwd(), "../../runs/ui/workbench-e2e.png");
await mkdir(resolve(process.cwd(), "../../runs/ui"), { recursive: true });

const browser = await puppeteer.launch({
  executablePath: chromium,
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
});

const pageErrors = [];
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.waitForSelector(".search-box input:not([disabled])", { timeout: 20_000 });

  await page.type(".search-box input", "Memory override in progress");
  await page.click(".search-submit");
  await page.waitForSelector(".result-card", { timeout: 20_000 });
  await page.click(".result-card");
  await page.waitForSelector(".evidence-drawer", { timeout: 10_000 });

  const resultCount = await page.$$eval(".result-card", (items) => items.length);
  const resultTitle = await page.$eval(
    ".result-card .result-heading strong",
    (node) => node.textContent?.trim() || "",
  );
  const evidenceText = await page.$eval(
    ".evidence-block p",
    (node) => node.textContent?.trim() || "",
  );

  await page.click(".drawer-tabs button:nth-child(3)");
  await page.waitForSelector(".boundary-panel");
  const boundaryText = await page.$eval(
    ".boundary-panel",
    (node) => node.textContent?.trim() || "",
  );
  if (!boundaryText.includes("shot_first")) {
    throw new Error("shot-first boundary evidence is not visible");
  }
  if (!resultCount || !resultTitle || !evidenceText) {
    throw new Error("search result or evidence drawer is empty");
  }
  await page.screenshot({ path: screenshot, fullPage: false });
  if (pageErrors.length) {
    throw new Error(`browser page errors: ${pageErrors.join("; ")}`);
  }
  process.stdout.write(
    `${JSON.stringify(
      {
        status: "pass",
        resultCount,
        resultTitle,
        boundaryVisible: true,
        screenshot,
      },
      null,
      2,
    )}\n`,
  );
} finally {
  await browser.close();
}
