/**
 * Surfaced-names ledger — the hardened "already in context" gate for the
 * gut-feeling tap.
 *
 * Problem: the tap's dedup used to be in-memory (dies with the process) and
 * tracked only the last notified name per thread inside a 5-minute window.
 * Nothing at all blocked memories the thread had already *seen* — boot-injected
 * ones, ones returned by `memory_query`, ones tapped an hour earlier. Once a
 * memory has been surfaced in a thread it lives in that thread's context (and
 * in mastracode's observational memory), so tapping about it again is pure
 * noise.
 *
 * Mechanism: one small JSON ledger on disk (NOT per-thread sqlite files —
 * gut-feeling state is machine-local because threads don't roam, and one file
 * is easier to inspect and prune) mapping threadId → memoryName → surfacedAt.
 * Every channel that shows a memory name to a thread records it:
 *
 *   - `memory_query` results (tools.ts) — strongest signal, the thread holds
 *     the actual content
 *   - gut-feeling tap starters (provider.ts)
 *   - boot-curation picks (provider.ts)
 *
 * Boot injection is the special case: it happens before any threadId exists,
 * so boot-injected names live in a process-global set instead — one session
 * per process, so process scope IS session scope.
 *
 * Escape hatch: a surfaced entry stops gating when the memory's `updatedAt`
 * is newer than the ledger's `surfacedAt` — the content changed since the
 * thread saw it, so it is news again.
 *
 * Fail-soft contract: every function here swallows I/O errors. A broken
 * ledger degrades to the old (chattier) behavior; it never breaks a tool call
 * or a poll.
 */

import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

/** Ledger shape on disk. */
type Ledger = {
  version: 1;
  /** threadId → memoryName → ISO timestamp of first surfacing. */
  threads: Record<string, Record<string, string>>;
};

/** Keep at most this many threads; oldest (by most recent entry) pruned first. */
const MAX_THREADS = 200;
/** Keep at most this many names per thread; oldest surfacedAt pruned first. */
const MAX_NAMES_PER_THREAD = 500;

/**
 * Ledger file path. `MEMOREASE_SURFACED_PATH` overrides for tests; default
 * sits next to the libsql store in the user's data dir. Computed lazily so
 * tests that swap the env var between cases are honored.
 */
function ledgerPath(): string {
  const override = process.env.MEMOREASE_SURFACED_PATH;
  if (override) return override;
  return join(homedir(), ".local", "share", "memorease", "surfaced.json");
}

async function readLedger(): Promise<Ledger> {
  try {
    const raw = await readFile(ledgerPath(), "utf8");
    const parsed = JSON.parse(raw) as Ledger;
    if (parsed && parsed.version === 1 && parsed.threads) return parsed;
  } catch {
    // Missing file, bad JSON, permission — all degrade to an empty ledger.
  }
  return { version: 1, threads: {} };
}

/** Newest surfacedAt in a thread's map — proxy for thread recency. */
function threadRecency(entries: Record<string, string>): number {
  let max = 0;
  for (const iso of Object.values(entries)) {
    const t = Date.parse(iso);
    if (!Number.isNaN(t) && t > max) max = t;
  }
  return max;
}

function prune(ledger: Ledger): void {
  const threadIds = Object.keys(ledger.threads);
  if (threadIds.length > MAX_THREADS) {
    threadIds
      .sort((a, b) => threadRecency(ledger.threads[a]) - threadRecency(ledger.threads[b]))
      .slice(0, threadIds.length - MAX_THREADS)
      .forEach((id) => delete ledger.threads[id]);
  }
  for (const id of Object.keys(ledger.threads)) {
    const entries = ledger.threads[id];
    const names = Object.keys(entries);
    if (names.length > MAX_NAMES_PER_THREAD) {
      names
        .sort((a, b) => Date.parse(entries[a]) - Date.parse(entries[b]))
        .slice(0, names.length - MAX_NAMES_PER_THREAD)
        .forEach((n) => delete entries[n]);
    }
  }
}

// Serialize writers within this process; cross-process races are last-write-
// wins on a low-write-rate file — acceptable for v1.
let writeChain: Promise<void> = Promise.resolve();

/**
 * Record that `names` were surfaced to `threadId`. Read-merge-write so
 * concurrent sessions on the same machine mostly compose. First surfacing
 * wins — recording an already-present name does not refresh its timestamp
 * (the gate should measure "when did this thread first see it").
 * Never throws.
 */
export function recordSurfaced(
  threadId: string,
  names: string[],
  now: Date = new Date(),
): Promise<void> {
  writeChain = writeChain.then(async () => {
    try {
      if (!threadId || names.length === 0) return;
      const ledger = await readLedger();
      const entries = (ledger.threads[threadId] ??= {});
      const iso = now.toISOString();
      for (const name of names) {
        if (name && !entries[name]) entries[name] = iso;
      }
      prune(ledger);
      const path = ledgerPath();
      await mkdir(dirname(path), { recursive: true });
      // Write-then-rename so a crash mid-write cannot truncate the ledger.
      const tmp = `${path}.tmp`;
      await writeFile(tmp, JSON.stringify(ledger), "utf8");
      await rename(tmp, path);
    } catch {
      // Fail-soft: a broken ledger must never break a tool call or a poll.
    }
  });
  return writeChain;
}

/**
 * All surfaced names for a thread (name → surfacedAt ISO). Returns an empty
 * record on any failure. Never throws.
 */
export async function getSurfaced(
  threadId: string,
): Promise<Record<string, string>> {
  const ledger = await readLedger();
  return ledger.threads[threadId] ?? {};
}

/**
 * Gate predicate: should a hit with this name/updatedAt be suppressed for
 * this thread's surfaced map? Suppressed when surfaced before, UNLESS the
 * memory was updated after it was surfaced (changed content is news again).
 */
export function isSuppressed(
  surfaced: Record<string, string>,
  name: string,
  updatedAt?: string,
): boolean {
  const at = surfaced[name];
  if (!at) return false;
  if (updatedAt) {
    const updated = Date.parse(updatedAt);
    const seen = Date.parse(at);
    if (!Number.isNaN(updated) && !Number.isNaN(seen) && updated > seen) {
      return false;
    }
  }
  return true;
}

/**
 * Boot injection happens before any threadId exists, so its names live in a
 * process-global set — one mastracode session per process, so process scope
 * is session scope. The probe treats these exactly like ledger entries.
 */
const bootInjected = new Set<string>();

export function recordBootInjectedNames(names: string[]): void {
  for (const n of names) if (n) bootInjected.add(n);
}

export function bootInjectedNames(): ReadonlySet<string> {
  return bootInjected;
}

/** Test seam: clear the process-global boot set. */
export function _resetSurfacedForTests(): void {
  bootInjected.clear();
  writeChain = Promise.resolve();
}
