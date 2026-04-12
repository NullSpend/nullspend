/**
 * Proxy latency benchmark script.
 *
 * Sends N requests through the proxy and measures overhead via the
 * x-nullspend-overhead-ms response header. Reports p50, p95, p99.
 *
 * Uses Bifrost's methodology: real proxy, real auth, measures the
 * overhead the proxy adds on top of the upstream provider call.
 *
 * Requires:
 *   - Proxy running (pnpm proxy:dev or deployed)
 *   - OPENAI_API_KEY and NULLSPEND_API_KEY env vars
 *
 * Usage:
 *   npx tsx bench.ts [--requests 100] [--concurrency 5] [--streaming]
 */

const BASE = process.env.PROXY_URL ?? `http://127.0.0.1:${process.env.PROXY_PORT ?? "8787"}`;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const NULLSPEND_API_KEY = process.env.NULLSPEND_API_KEY;

const args = process.argv.slice(2);
function getArg(name: string, defaultVal: string): string {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : defaultVal;
}

const TOTAL_REQUESTS = parseInt(getArg("requests", "50"), 10);
const CONCURRENCY = parseInt(getArg("concurrency", "5"), 10);
const STREAMING = args.includes("--streaming");

async function sendRequest(): Promise<{ overheadMs: number; totalMs: number; status: number }> {
  const start = performance.now();
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "x-nullspend-key": NULLSPEND_API_KEY!,
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "Say ok" }],
      max_tokens: 3,
      stream: STREAMING,
    }),
  });

  // Consume body
  await res.text();
  const totalMs = Math.round(performance.now() - start);

  const overheadHeader = res.headers.get("x-nullspend-overhead-ms");
  const overheadMs = overheadHeader ? parseInt(overheadHeader, 10) : -1;

  return { overheadMs, totalMs, status: res.status };
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

async function run() {
  if (!OPENAI_API_KEY || !NULLSPEND_API_KEY) {
    console.error("OPENAI_API_KEY and NULLSPEND_API_KEY are required.");
    process.exit(1);
  }

  // Verify proxy is up
  try {
    const health = await fetch(`${BASE}/health`);
    if (!health.ok) throw new Error(`Health check failed: ${health.status}`);
  } catch (_err) {
    console.error(`Proxy not reachable at ${BASE}. Start with pnpm proxy:dev.`);
    process.exit(1);
  }

  console.log(`\nNullSpend Proxy Benchmark`);
  console.log(`─────────────────────────`);
  console.log(`Target:      ${BASE}`);
  console.log(`Requests:    ${TOTAL_REQUESTS}`);
  console.log(`Concurrency: ${CONCURRENCY}`);
  console.log(`Streaming:   ${STREAMING}`);
  console.log();

  const results: { overheadMs: number; totalMs: number; status: number }[] = [];
  let completed = 0;
  let errors = 0;

  // Run in batches of CONCURRENCY
  for (let i = 0; i < TOTAL_REQUESTS; i += CONCURRENCY) {
    const batch = Math.min(CONCURRENCY, TOTAL_REQUESTS - i);
    const promises = Array.from({ length: batch }, () =>
      sendRequest()
        .then((r) => {
          results.push(r);
          completed++;
          if (r.status !== 200) errors++;
        })
        .catch(() => {
          errors++;
          completed++;
        }),
    );
    await Promise.all(promises);
    process.stdout.write(`\r  Progress: ${completed}/${TOTAL_REQUESTS}`);
  }

  console.log("\n");

  const successful = results.filter((r) => r.status === 200 && r.overheadMs >= 0);
  if (successful.length === 0) {
    console.error("No successful requests. Check proxy logs.");
    process.exit(1);
  }

  const overheads = successful.map((r) => r.overheadMs).sort((a, b) => a - b);
  const totals = successful.map((r) => r.totalMs).sort((a, b) => a - b);

  console.log(`Results (${successful.length} successful, ${errors} errors)`);
  console.log(`─────────────────────────────────────`);
  console.log(`Proxy Overhead:`);
  console.log(`  p50:  ${percentile(overheads, 50)}ms`);
  console.log(`  p95:  ${percentile(overheads, 95)}ms`);
  console.log(`  p99:  ${percentile(overheads, 99)}ms`);
  console.log(`  min:  ${overheads[0]}ms`);
  console.log(`  max:  ${overheads[overheads.length - 1]}ms`);
  console.log(`  mean: ${Math.round(overheads.reduce((a, b) => a + b, 0) / overheads.length)}ms`);
  console.log();
  console.log(`Total Request Time:`);
  console.log(`  p50:  ${percentile(totals, 50)}ms`);
  console.log(`  p95:  ${percentile(totals, 95)}ms`);
  console.log(`  p99:  ${percentile(totals, 99)}ms`);
  console.log();
}

run();
