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

import { resolveConfig, type PluginContext } from "./config.ts";
import {
  createVectorStore,
  ensureSchema,
} from "./store.ts";
import type { MastraVector } from "@mastra/core/vector";
import { queryMemories, type MemoryHit } from "./memory.ts";
import {
  curatorPreflightError,
  rankAndTruncate,
  renderHit,
  resolveInjectBudget,
} from "./curator.ts";
import { failSoft, type StorageResult } from "./errors.ts";

const BOOT_TIMEOUT_DEFAULT_MS = 3000;

/**
 * The behavioral directive. ALWAYS prepended to the injected section (on both
 * the empty-store and success paths) so every session — even a fresh one with
 * nothing in the store yet — knows memorease exists and when to reach for it.
 *
 * Without this, a session that boots into an empty store gets an empty
 * instruction string and never discovers the memory tools, which is the bug
 * this constant exists to close: the agent must be told to QUERY ON INSTINCT
 * or it will answer "I don't know" without ever checking memory.
 */
const MEMOREASE_DIRECTIVE = [
  "## Memorease — Global Knowledge Layer",
  "",
  "Memorease is your persistent store for knowledge that outlives the current session: facts about the user, lessons learned, recurring patterns, cross-project context, durable preferences, distilled skills. The admission test: would this help tomorrow, on a different project, or with a different question? If yes, it belongs here.",
  "",
  "Three behaviors, all required:",
  "",
  '1. QUERY ON INSTINCT. Any time you have a flicker of "I might know something about this" — a personal question, a tooling choice, an error pattern, a user preference, a past decision — run `memory_query` BEFORE you answer. A missed query when memory had the answer is a bug. "I don\'t know / I can\'t do that" is never final until you\'ve checked memory.',
  "",
  "2. WRITE ON LEARN. The moment a durable fact lands — a preference, an infra detail, a decision and its rationale, a lesson worth carrying forward — write it to memorease immediately via `memory_write`. Don't batch. If you find yourself doing the same thing across sessions, that's the signal to write it.",
  "",
  "3. DISTILL WHEN RECURRING. When a cluster of memories becomes a repeatable workflow, use `memory_distill_skill` to fold it into a skill. Skills make the knowledge active, not just retrievable.",
  "",
  "What does NOT belong — never write these:",
  "- SECRETS. Passwords, API keys, tokens, credentials — never, in any form, regardless of trust level. Store WHERE a credential lives (a path, a vault name), never the credential itself.",
  "- Task-in-progress state. A bug you're mid-fixing, a build you're waiting on, a session's working notes — that's thread context, not knowledge.",
  "- Project-local details. Specific file paths, one-off bugs, code that lives in the repo anyway.",
  "- Anything cheaply recomputed. Directory listings, versions, things one command re-derives.",
  "",
  "Hygiene — the store is read on every recall, so every entry costs context:",
  "- One fact per name. Before writing, `memory_query` for near-duplicates; UPDATE the existing name instead of writing a sibling.",
  "- When a fact changes or resolves, rewrite the memory to the new durable truth (or `memory_forget` it). A memory describing a resolved situation in past tense is rot.",
  "- Keep entries compact. Capture the durable core and the why; leave the play-by-play in the thread.",
].join("\n");

function bootTimeoutMs(): number {
  const raw = Number.parseInt(process.env.MEMOREASE_BOOT_TIMEOUT_MS ?? "", 10);
  if (!Number.isFinite(raw) || raw <= 0) return BOOT_TIMEOUT_DEFAULT_MS;
  return raw;
}

/**
 * Build the short context string shown to the curator to help it judge
 * relevance. Stays cheap — `git rev-parse` is avoided (not worth a subprocess
 * at boot); cwd basename + hostname is enough signal.
 *
 * Exported for the background curation signal (provider.ts), which appends
 * live conversation text to it before calling the curator.
 */
export function buildSessionContext(): string {
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
 *
 * Exported for the background curation signal (provider.ts), which re-runs
 * the same broad query to gather candidates for the curator LLM.
 */
export function buildBootQuery(): string {
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
  extras: { curatorNote?: string },
): string {
  const lines: string[] = ["## Memories", ""];
  for (const hit of hits) {
    lines.push(renderHit(hit));
  }
  lines.push("");
  lines.push(
    "Query memory before assuming prior context — names above are hints, not promises.",
  );
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
 * Fail-soft wrapper for `ensureSchema`. A throw here (CREATE INDEX permission
 * failure, pg connection refused during describeIndex, libsql I/O error) is
 * converted to a branded `{ok:false}` result so the boot IIFE never rejects.
 * The branded message then propagates through `renderStorageUnreachableSection`
 * with the real cause, rather than being discarded as an "unexpected bug".
 */
async function ensureSchemaOrFailSoft(
  store: MastraVector,
): Promise<StorageResult<void>> {
  return failSoft(async () => {
    await ensureSchema(store);
  });
}

/**
 * Boot memory injection. The plugin's `instructions(context)` delegates here.
 *
 * Flow:
 *   1. Resolve storage config + open the store.
 *   2. Race the boot query (ensureSchema + queryMemories) against a timeout.
 *   3. On branded failure or timeout → return the storage-unreachable section.
 *   4. On success → rank-and-truncate by similarity, then render. The curator
 *      LLM never runs here — it fires later as a background signal
 *      (provider.ts `bootCurateSubscriber`).
 *
 * Never throws. Never returns a rejected promise. The return is always a
 * string (possibly empty if everything succeeded but no memories were found
 * AND no curator note is warranted).
 */
export type BuildInstructionsOptions = {
  /**
   * Test-only seam. Override the store factory to deterministically exercise
   * the sync-throw wrapping in `buildInstructions` (M2 regression). Production
   * callers omit this and the real `createVectorStore` is used.
   */
  _createVectorStoreForTests?: typeof createVectorStore;
};

/**
 * Boot memory injection. The plugin's `instructions(context)` delegates here.
 *
 * Flow:
 *   1. Resolve storage config + open the store.
 *   2. Race the boot query (ensureSchema + queryMemories) against a timeout.
 *   3. On branded failure or timeout → return the storage-unreachable section.
 *   4. On success → rank-and-truncate by similarity, then render. The curator
 *      LLM never runs here — it fires later as a background signal
 *      (provider.ts `bootCurateSubscriber`).
 *
 * Never throws. Never returns a rejected promise. The return is always a
 * string (possibly empty if everything succeeded but no memories were found
 * AND no curator note is warranted).
 */
export async function buildInstructions(
  context: PluginContext,
  opts: BuildInstructionsOptions = {},
): Promise<string> {
  // Resolve config + open the store OUTSIDE the fail-soft net of queryMemories,
  // but inside our own: a sync throw from createVectorStore (e.g. libsql
  // parent-dir race on a fresh system) or a reject from ensureSchema (pg
  // CREATE INDEX permission failure, connection refused) must surface as a
  // branded storage-unreachable section, not a rejected instructions() promise.
  // The plugin contract is "never throws, never rejects" (JSDoc above).
  const storeFactory = opts._createVectorStoreForTests ?? createVectorStore;
  let store: MastraVector | undefined;
  try {
    const resolved = resolveConfig(context);
    store = await storeFactory(resolved);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    return renderStorageUnreachableSection(
      `memorease: storage init failed — ${detail}.`,
    );
  }

  const budget = resolveInjectBudget(
    (context.config as { injectBudget?: string } | undefined)?.injectBudget,
  );

  // The boot query runs both ensureSchema and queryMemories. Both are wrapped
  // in failSoft (ensureSchema via ensureSchemaOrFailSoft below) so that a
  // schema-bootstrap throw becomes a branded result rather than rejecting the
  // IIFE — which in turn would otherwise hit the "rejected unexpectedly" arm
  // and mislabel an actionable storage error as "a bug".
  const bootQuery = (async () => {
    const schema = await ensureSchemaOrFailSoft(store!);
    if (!schema.ok) return schema;
    return queryMemories(store!, buildBootQuery(), {
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
    // The boot IIFE rejected despite its internal try/catch — this is
    // genuinely unexpected (every step is wrapped) and indicates a bug in
    // our error handling rather than a user-actionable storage failure.
    return renderStorageUnreachableSection(
      "Boot query rejected unexpectedly — this indicates a bug in memorease's error handling, not a storage issue. Please report.",
    );
  }
  if (!queryResult.ok) {
    return renderStorageUnreachableSection(queryResult.error);
  }

  const candidates = queryResult.value;
  if (candidates.length === 0) {
    // Nothing stored yet — but the directive is load-bearing: a fresh session
    // must still know memorease exists and when to reach for the tools, or it
    // will answer "I don't know" without ever querying. Append the
    // disarmed-curator note if applicable.
    const note = curatorNote(context.config ?? {});
    return note
      ? `${MEMOREASE_DIRECTIVE}\n\n${note.trim()}`
      : MEMOREASE_DIRECTIVE;
  }

  // Similarity ranking only — the curator LLM never runs at boot (it costs a
  // multi-second model roundtrip that would block every session start). When
  // a curator model is armed, the curated selection arrives later as a
  // background signal (see provider.ts `bootCurateSubscriber`), where it can
  // also consider the live conversation.
  const hits = rankAndTruncate(candidates, budget);

  return `${MEMOREASE_DIRECTIVE}\n\n${renderMemoriesSection(hits, {
    curatorNote: curatorNote(context.config ?? {}),
  })}`;
}
