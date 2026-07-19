/**
 * Boot-time memory injection — the plugin's `instructions()` entry point.
 *
 * At session start, mastracode calls `instructions(context)` and prepends the
 * returned string to the system prompt. This module builds a `## Memories`
 * section from whatever's in the store, ranked by the curator if armed or by
 * vector similarity otherwise, and never throws — every await is wrapped so a
 * storage failure surfaces as a short branded note instead of crashing the
 * session.
 *
 * Two fail-soft arms (R3):
 *  - The Phase 2 `failSoft` net inside `queryMemories` already converts a
 *    fast `ECONNREFUSED` into a branded `{ok:false}` result in milliseconds —
 *    so the boot query returns rather than rejects on a closed port.
 *  - A `Promise.race` timeout arm (default 3s, overridable via
 *    `MEMOREASE_BOOT_TIMEOUT_MS` for tests) covers true hangs. Any stray
 *    rejection from the race is converted to the same branded note.
 */

import { hostname } from "node:os";
import { basename } from "node:path";

import { resolveConfig, type PluginContext } from "./config.js";
import {
  createVectorStore,
  ensureSchema,
} from "./store.js";
import { queryMemories, type MemoryHit } from "./memory.js";
import {
  curateForBoot,
  curatorPreflightError,
  renderHit,
  resolveCuratorModel,
  resolveInjectBudget,
} from "./curator.js";

const BOOT_TIMEOUT_DEFAULT_MS = 3000;

function bootTimeoutMs(): number {
  const raw = Number.parseInt(process.env.MEMOREASE_BOOT_TIMEOUT_MS ?? "", 10);
  if (!Number.isFinite(raw) || raw <= 0) return BOOT_TIMEOUT_DEFAULT_MS;
  return raw;
}

/**
 * Build the short context string shown to the curator to help it judge
 * relevance. Stays cheap — `git rev-parse` is avoided (not worth a subprocess
 * at boot); cwd basename + hostname is enough signal.
 */
function buildSessionContext(): string {
  const host = hostname();
  const cwd = basename(process.cwd());
  return `host=${host}; cwd=${cwd}`;
}

/**
 * Build the probe query for the boot vector search. At boot we want a broad
 * net — the store is expected to be small (single user, hundreds at most) and
 * the curator + budget truncation do the actual ranking. The query text is
 * deliberately general: it should weakly match any stored memory so that
 * top-K returns the full set rather than filtering on a topic the user's
 * memories happen not to mention.
 *
 * Hostname and cwd-basename hints are folded in so that memories referencing
 * them (project context, machine profile facts) get a slight boost, without
 * excluding memories that don't.
 */
function buildBootQuery(): string {
  const host = hostname();
  const cwd = basename(process.cwd());
  return `memories about the user, their preferences, habits, current project ${cwd}, host ${host}, environment, past sessions, and any other durable context`;
}

/**
 * Wrap a promise with a timeout. Resolves to `{timedOut:true}` if the
 * timeout fires first; otherwise `{timedOut:false, value}`. Never rejects —
 * a rejection from the inner promise surfaces as `value:undefined` with
 * `timedOut:false` so the caller can distinguish a real hang from a throw.
 */
function withTimeout<T>(
  p: Promise<T>,
  ms: number,
): Promise<{ timedOut: true } | { timedOut: false; value: T | undefined }> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<{ timedOut: true }>((resolve) => {
    timer = setTimeout(() => resolve({ timedOut: true }), ms);
  });
  return Promise.race([
    p.then(
      (value) => ({ timedOut: false as const, value }),
      () => ({ timedOut: false as const, value: undefined as T | undefined }),
    ),
    timeout,
  ]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

/**
 * Format the disarmed-curator note (or omit it entirely if the curator is
 * armed). Appended to the injected section so the agent can tell the user
 * why curation isn't happening if asked.
 */
function curatorNote(config: {
  curatorModel?: string;
}): string | undefined {
  const err = curatorPreflightError(config);
  return err ? `\n\n${err}` : undefined;
}

/**
 * Render the `## Memories` section from a ranked list of hits.
 */
function renderMemoriesSection(
  hits: MemoryHit[],
  extras: { curatorNote?: string; fallbackNote?: string },
): string {
  const lines: string[] = ["## Memories", ""];
  for (const hit of hits) {
    lines.push(renderHit(hit));
  }
  lines.push("");
  lines.push(
    "Query memory before assuming prior context — names above are hints, not promises.",
  );
  if (extras.fallbackNote) lines.push("", extras.fallbackNote);
  if (extras.curatorNote) lines.push("", extras.curatorNote);
  return lines.join("\n");
}

function renderStorageUnreachableSection(detail: string): string {
  return [
    "## Memorease",
    "",
    "Storage unreachable — memory disabled this session.",
    detail,
    "",
    "Memory tools will return this error until storage is reachable. Fix the issue and restart the session to re-arm memory.",
  ].join("\n");
}

/**
 * Boot memory injection. The plugin's `instructions(context)` delegates here.
 *
 * Flow:
 *   1. Resolve storage config + open the store.
 *   2. Race the boot query (ensureSchema + queryMemories) against a timeout.
 *   3. On branded failure or timeout → return the storage-unreachable section.
 *   4. On success → curate (if armed) or rank-and-truncate, then render.
 *
 * Never throws. Never returns a rejected promise. The return is always a
 * string (possibly empty if everything succeeded but no memories were found
 * AND no curator note is warranted).
 */
export async function buildInstructions(context: PluginContext): Promise<string> {
  const resolved = resolveConfig(context);
  const store = createVectorStore(resolved);

  const modelId = resolveCuratorModel(context.config ?? {});
  const budget = resolveInjectBudget(
    (context.config as { injectBudget?: string } | undefined)?.injectBudget,
  );

  // The boot query runs both ensureSchema and queryMemories inside the
  // fail-soft net. If schema bootstrap or the query throws, queryMemories
  // still returns a branded result rather than rejecting.
  const bootQuery = (async () => {
    await ensureSchema(store);
    return queryMemories(store, buildBootQuery(), {
      topK: Math.ceil(budget / 200),
    });
  })();

  const raced = await withTimeout(bootQuery, bootTimeoutMs());
  if ("timedOut" in raced && raced.timedOut) {
    return renderStorageUnreachableSection(
      `Boot query did not complete within ${bootTimeoutMs()}ms.`,
    );
  }
  const queryResult = raced.value;
  if (!queryResult) {
    return renderStorageUnreachableSection(
      "Boot query rejected unexpectedly (this is a bug — please report).",
    );
  }
  if (!queryResult.ok) {
    return renderStorageUnreachableSection(queryResult.error);
  }

  const candidates = queryResult.value;
  if (candidates.length === 0) {
    // Nothing stored yet. Don't inject an empty section — but DO surface the
    // disarmed-curator note if applicable, so the user sees the hint early.
    const note = curatorNote(context.config ?? {});
    return note ? note.trim() : "";
  }

  const curate = await curateForBoot(
    modelId,
    candidates,
    budget,
    buildSessionContext(),
  );

  return renderMemoriesSection(curate.hits, {
    curatorNote: curate.usedCurator ? undefined : curatorNote(context.config ?? {}),
    fallbackNote: curate.fallbackNote,
  });
}
