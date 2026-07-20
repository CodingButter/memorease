/**
 * "Gut-feeling" signal provider — experimental, fail-soft.
 *
 * Polls recent thread messages via `agent.memory.recall(...)`, embeds the most
 * recent user message, and pushes a notification nudge when a stored memory
 * scores above `TAP_THRESHOLD`. The nudge tells the agent a relevant memory
 * exists and suggests calling `memory_query` — it never writes to memory as a
 * side-channel (the agent stays in the loop).
 *
 * Fail-soft by design: every step (live-agent lookup, recall, embed, store
 * query, notify) is wrapped in try/catch. Any mastracode update that changes
 * `Memory.recall` shape, the `SignalProvider` base, or `notify` disarms the
 * provider without crashing the session. The deliberate-memory core
 * (tools + boot instructions) is entirely unaffected.
 *
 * Pivot from original Phase 5 (recorded as a deviation): the first design
 * pushed an `@mastra/memory` `Extractor` into the observer engine's private
 * extractor list — a reach-in through bundled dist code with no semver
 * guarantee. `SignalProvider` is the documented public surface for pushing
 * notification signals into agent threads; it keeps the agent in the loop
 * (query-then-act, not write-as-side-channel) and needs no private-field
 * traversal for the core path.
 */

import { SignalProvider } from "@mastra/core/signals";
import type { MastraVector } from "@mastra/core/vector";
import { embed } from "./embed.ts";
import { INDEX_NAME } from "./store.ts";
import { curateForBoot, INJECT_BUDGET_DEFAULT } from "./curator.ts";
import { buildBootQuery, buildSessionContext } from "./instructions.ts";
import type { MemoryHit } from "./memory.ts";

export type TapStatus =
  | "armed"
  | "disarmed-no-agent"
  | "disarmed-no-signal"
  | "disarmed-no-memory";

/** Cosine threshold above which a stored memory is considered "relevant". */
export const TAP_THRESHOLD = 0.5;
/** Per-thread dedup window: same top hit within this window does not re-notify. */
export const TAP_DEDUP_MS = 5 * 60 * 1000;

/** Signal priority — mirrors mastracode's NotificationPriority levels. */
export type SignalPriority = "low" | "medium" | "high" | "urgent";

/**
 * Similarity score above which a tap is considered strong enough to warrant
 * delivery into an active conversation rather than waiting in the inbox.
 */
export const TAP_MEDIUM_THRESHOLD = 0.65;
/** Score above which the resonance is strong enough to interrupt idle. */
export const TAP_HIGH_THRESHOLD = 0.8;

/**
 * Map the strongest hit's similarity score to a signal priority. The
 * delivery pipeline treats priority as "does this wake an idle thread or
 * wait for the next message" — so the tap's importance estimate IS the
 * score. "urgent" is deliberately never produced: a cosine similarity is
 * an informed hunch, not an emergency.
 */
export function tapPriority(score: number): SignalPriority {
  if (score >= TAP_HIGH_THRESHOLD) return "high";
  if (score >= TAP_MEDIUM_THRESHOLD) return "medium";
  return "low";
}
/** Default poll interval if none provided. */
export const DEFAULT_POLL_MS = 30_000;
/** How many recent messages to pull from each thread per poll. */
const RECALL_PER_PAGE = 10;

/** Minimal structural surface this module needs from a memory instance. */
type MemoryLike = {
  recall?: (args: {
    threadId: string | string[];
    resourceId?: string;
    perPage?: number | false;
  }) => Promise<{
    messages: Array<RecallMessageLike>;
  }>;
};

/** Minimal structural surface this module needs from a live mastra agent. */
type LiveAgentLike = {
  memory?: MemoryLike;
  getMemory?: () => Promise<MemoryLike | undefined>;
  sendNotificationSignal?: unknown;
};

/**
 * Resolve a usable memory instance from a live agent. Real `Agent` objects
 * expose memory behind the async `getMemory()` accessor (which resolves
 * function-based memory config); a plain `.memory` property is accepted first
 * for tests and older shapes. Returns undefined when neither yields an object
 * with a callable `recall`.
 */
async function resolveMemory(
  agent: LiveAgentLike | undefined,
): Promise<MemoryLike | undefined> {
  if (!agent) return undefined;
  if (agent.memory && typeof agent.memory.recall === "function") {
    return agent.memory;
  }
  if (typeof agent.getMemory === "function") {
    const mem = await agent.getMemory();
    if (mem && typeof mem.recall === "function") return mem;
  }
  return undefined;
}

/** Defensive shape of one recalled message (covers V1 and V2 formats). */
type RecallMessageLike = {
  id?: string;
  role?: string;
  content?:
    | string
    | {
        format?: number;
        content?: string;
        parts?: Array<{ type: string; text?: string }>;
      };
};

/** Minimal subscription shape this module consumes. */
type SubLike = {
  threadId: string;
  resourceId: string;
};

/**
 * The one "external resource" this provider monitors: the global memory
 * store itself. Every subscribed thread shares it — the per-thread signal
 * targeting comes from the subscription's threadId/resourceId, not from
 * distinct external resources.
 */
export const EXTERNAL_RESOURCE_ID = "memorease:global";

/** Mutable probe state — owned by the provider, passed to `probeSubscriber`. */
export type ProbeState = {
  /**
   * threadId → last notification, for dedup. `text` is the recent user text
   * at notify time: an idle conversation (no new user input) never re-taps,
   * no matter how much time passes.
   */
  lastNotifiedPerThread: Map<
    string,
    { name: string; at: number; text: string }
  >;
  lastPollAt?: number;
  lastError?: string;
  /** Reason the provider is currently disarmed, if any. */
  disarmedReason?: Exclude<TapStatus, "armed">;
  notifiedCount: number;
  /**
   * Threads whose one-shot boot curation has reached a terminal outcome
   * (notified, empty store, curator fallback). Threads with transient
   * failures (no recallable text yet, error) are NOT added, so the next poll
   * retries — cheap, since the LLM only runs after recall+query succeed.
   */
  bootCuratedThreads: Set<string>;
  bootCuratedCount: number;
};

export function newProbeState(): ProbeState {
  return {
    lastNotifiedPerThread: new Map(),
    notifiedCount: 0,
    bootCuratedThreads: new Set(),
    bootCuratedCount: 0,
  };
}

/**
 * Defensively extract a plain-text body from a recalled message. Handles both
 * the V1 string-content shape and the V2 parts-array shape. Returns the empty
 * string if no text is recoverable.
 */
export function extractText(msg: RecallMessageLike | undefined): string {
  if (!msg) return "";
  const c = msg.content;
  if (typeof c === "string") return c;
  if (c && typeof c === "object") {
    if (typeof c.content === "string" && c.content) return c.content;
    if (Array.isArray(c.parts)) {
      const texts: string[] = [];
      for (const p of c.parts) {
        if (p && typeof p === "object" && p.type === "text" && typeof p.text === "string") {
          texts.push(p.text);
        }
      }
      if (texts.length) return texts.join("\n");
    }
  }
  return "";
}

/**
 * Pick the most recent user message body from a recall result. Falls back to
 * the latest message of any role if no user message is present. Returns ""
 * if the list is empty or no text is recoverable.
 */
export function extractRecentUserText(
  messages: Array<RecallMessageLike> | undefined,
): string {
  if (!messages || messages.length === 0) return "";
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && m.role === "user") {
      const t = extractText(m);
      if (t) return t;
    }
  }
  // Fallback: latest of any role.
  return extractText(messages[messages.length - 1]);
}

/**
 * Format the nudge body shown to the agent. Never pastes memory content, and
 * deliberately does NOT present itself as "the memory" — a gut feeling is a
 * door, not a document. Naming a single memory as the payload would invite a
 * one-shot lookup; instead the hint says the current moment resonates with
 * stored knowledge and offers the hit names as *starting points* for a
 * broader memory_query dive.
 */
export function hintBody(hits: Array<{ name: string; score: number }>): string {
  const starters = hits.map((h) => `'${h.name}'`).join(", ");
  return (
    `memorease gut feeling: something about the current conversation ` +
    `resonates with stored memories — there may be relevant context you're ` +
    `missing. Starting points: ${starters}. Call memory_query with terms ` +
    `from the conversation to dig in; related memories beyond these may ` +
    `surface too.`
  );
}

/**
 * Format the boot-curation body. Same principle as `hintBody`: names only,
 * never memory content — the agent stays in the loop and recalls via
 * `memory_query`.
 */
export function bootCurationBody(names: string[]): string {
  const list = names.map((n) => `'${n}'`).join(", ");
  return (
    `memorease boot curation: given the current conversation, your most ` +
    `relevant stored memories are: ${list}. Call memory_query to recall any ` +
    `you haven't already.`
  );
}

/**
 * Resolve the live agent from a SignalProvider instance. The base class stores
 * the connected agent in a private field; subclasses reach it via the
 * protected `get agent()` accessor. We can't call that from a standalone
 * function, so this helper expects the provider to hand us its own `this`.
 */
type AgentGetter = () => LiveAgentLike | undefined;

/**
 * Probe a single subscriber: recall recent messages, embed the latest user
 * message, query the store, and notify on a threshold-passing hit (subject to
 * per-thread dedup). Never throws — every failure is recorded in `state`.
 *
 * Returns the resolution status for this probe:
 *  - "notified" — a notification was sent.
 *  - "skipped-below-threshold" — top hit was below TAP_THRESHOLD.
 *  - "skipped-deduped" — same hit was notified for this thread within window.
 *  - "skipped-no-text" — recall returned no usable text.
 *  - "disarmed-no-memory" — agent.memory.recall not available.
 *  - "error" — any step threw; `state.lastError` carries the message.
 */
export async function probeSubscriber(
  getAgent: AgentGetter,
  store: MastraVector,
  embedFn: (text: string) => Promise<number[]>,
  sub: SubLike,
  state: ProbeState,
  notify: (
    body: string,
    sub: SubLike,
    priority: SignalPriority,
  ) => Promise<void>,
): Promise<"notified" | "skipped-below-threshold" | "skipped-deduped" | "skipped-no-text" | "disarmed-no-memory" | "error"> {
  try {
    const memory = await resolveMemory(getAgent());
    if (!memory?.recall) {
      state.disarmedReason = "disarmed-no-memory";
      return "disarmed-no-memory";
    }
    const recalled = await memory.recall({
      threadId: sub.threadId,
      resourceId: sub.resourceId,
      perPage: RECALL_PER_PAGE,
    });
    const recent = extractRecentUserText(recalled?.messages);
    if (!recent) return "skipped-no-text";

    const vector = await embedFn(recent);
    // topK > 1 so a self-authored top hit can fall through to the next
    // candidate instead of silencing the probe entirely.
    const hits = await store.query({
      indexName: INDEX_NAME,
      queryVector: vector,
      topK: 3,
    });
    // Provenance gate: never tap a thread about a memory that thread itself
    // wrote — the knowledge is already in its context. Memories written
    // before this gate existed carry no sourceThreadId and pass through.
    const eligible = hits.filter(
      (h) =>
        h.metadata?.sourceThreadId !== sub.threadId &&
        h.score >= TAP_THRESHOLD,
    );
    if (eligible.length === 0) return "skipped-below-threshold";

    const name = String(eligible[0].metadata?.name ?? "");
    const now = Date.now();
    const last = state.lastNotifiedPerThread.get(sub.threadId);
    // Progress gate: a gut feeling reacts to what's happening. If no new
    // user text has arrived since the last tap, nothing has happened — stay
    // silent indefinitely, don't drip the same hint every dedup window.
    if (last && last.text === recent) {
      return "skipped-deduped";
    }
    // Resonance dedup: the conversation moved but the strongest hit is the
    // same — the feeling hasn't changed, wait out the window.
    if (last && last.name === name && now - last.at < TAP_DEDUP_MS) {
      return "skipped-deduped";
    }

    const starters = eligible.map((h) => ({
      name: String(h.metadata?.name ?? ""),
      score: h.score,
    }));
    await notify(hintBody(starters), sub, tapPriority(eligible[0].score));
    state.lastNotifiedPerThread.set(sub.threadId, { name, at: now, text: recent });
    state.notifiedCount += 1;
    return "notified";
  } catch (err) {
    state.lastError = err instanceof Error ? err.message : String(err);
    return "error";
  }
}

/** Test seam: matches `curateForBoot`'s signature. */
type CurateFn = typeof curateForBoot;

/**
 * One-shot boot curation for a single thread. This is where the curator LLM
 * actually runs — moved off the boot path (`instructions()` injects
 * similarity-ranked hits without an LLM call) so session start stays fast.
 * Running here has a second advantage: the curator sees the live
 * conversation, not boot-time guesswork.
 *
 * Flow: recall recent user text → broad boot query for candidates → curator
 * LLM pass → notify with the picked memory NAMES (never content). Terminal
 * outcomes (notified, empty store, curator fallback) mark the thread done in
 * `state.bootCuratedThreads`; transient ones (no text yet, error) leave it
 * unmarked so the next poll retries.
 *
 * Never throws — mirrors `probeSubscriber`'s fail-soft contract.
 */
export async function bootCurateSubscriber(
  getAgent: AgentGetter,
  store: MastraVector,
  embedFn: (text: string) => Promise<number[]>,
  sub: SubLike,
  state: ProbeState,
  notify: (body: string, sub: SubLike) => Promise<void>,
  opts: { modelId: string; budget: number; curateFn?: CurateFn },
): Promise<
  | "notified"
  | "skipped-already-curated"
  | "skipped-no-text"
  | "skipped-empty-store"
  | "skipped-curator-fallback"
  | "disarmed-no-memory"
  | "error"
> {
  if (state.bootCuratedThreads.has(sub.threadId)) {
    return "skipped-already-curated";
  }
  try {
    const memory = await resolveMemory(getAgent());
    if (!memory?.recall) {
      return "disarmed-no-memory";
    }
    const recalled = await memory.recall({
      threadId: sub.threadId,
      resourceId: sub.resourceId,
      perPage: RECALL_PER_PAGE,
    });
    const recent = extractRecentUserText(recalled?.messages);
    if (!recent) return "skipped-no-text";

    const vector = await embedFn(buildBootQuery());
    const rows = await store.query({
      indexName: INDEX_NAME,
      queryVector: vector,
      topK: Math.ceil(opts.budget / 200),
    });
    const candidates: MemoryHit[] = rows.map((r) => ({
      id: r.id,
      score: r.score,
      name: String(r.metadata?.name ?? ""),
      content: String(r.metadata?.content ?? ""),
      type: typeof r.metadata?.type === "string" ? r.metadata.type : undefined,
    }));
    if (candidates.length === 0) {
      state.bootCuratedThreads.add(sub.threadId);
      return "skipped-empty-store";
    }

    const sessionContext = `${buildSessionContext()}; recent conversation: ${recent.slice(0, 400)}`;
    const curate = await (opts.curateFn ?? curateForBoot)(
      opts.modelId,
      candidates,
      opts.budget,
      sessionContext,
    );
    if (!curate.usedCurator || curate.hits.length === 0) {
      // Curator fell back to similarity — that's exactly what boot already
      // injected, so a notification would be pure noise. Terminal: don't
      // re-spend the roundtrip next poll.
      state.bootCuratedThreads.add(sub.threadId);
      return "skipped-curator-fallback";
    }

    await notify(bootCurationBody(curate.hits.map((h) => h.name)), sub);
    state.bootCuratedThreads.add(sub.threadId);
    state.bootCuratedCount += 1;
    return "notified";
  } catch (err) {
    state.lastError = err instanceof Error ? err.message : String(err);
    return "error";
  }
}

export type ProviderStatus = {
  armed: boolean;
  disarmedReason?: Exclude<TapStatus, "armed">;
  lastPollAt?: number;
  lastError?: string;
  subscriptionCount: number;
  notifiedCount: number;
  bootCuratedCount: number;
};

type CreateProviderOpts = {
  store: MastraVector;
  /** Override the embedder (tests). Defaults to the module-level `embed`. */
  embedFn?: (text: string) => Promise<number[]>;
  pollIntervalMs?: number;
  /**
   * Curator model id for the background boot-curation signal. Undefined =
   * curator disarmed — boot curation never runs, gut-feeling taps still do.
   */
  curatorModelId?: string;
  /** Character budget for the curated selection. Defaults to the boot budget default. */
  injectBudget?: number;
};

/**
 * Build the SignalProvider subclass instance. Dynamic shape: we extend
 * `SignalProvider` and pass the resulting instance back. The base class is
 * `@experimental` and its `protected` members are reached via `this`, so the
 * subclass must be the one calling `this.notify(...)` and reading `this.agent`.
 *
 * `poll()` iterates active subscriptions and delegates to `probeSubscriber`
 * with closures bound to `this`. All failures are recorded in `state`; `poll`
 * itself never throws.
 */
export function createMemoreaseProvider({
  store,
  embedFn,
  pollIntervalMs,
  curatorModelId,
  injectBudget,
}: CreateProviderOpts): unknown {
  const embedImpl = embedFn ?? embed;
  const state = newProbeState();
  const pollInterval = pollIntervalMs ?? DEFAULT_POLL_MS;
  const budget = injectBudget ?? INJECT_BUDGET_DEFAULT;

  class MemoreaseSignalProvider extends SignalProvider {
    readonly id = "memorease-gut-feeling" as const;
    readonly name = "Memorease gut-feeling";
    readonly pollInterval = pollInterval;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private getAgent(): LiveAgentLike | undefined {
      // The base class exposes the connected agent via a protected getter.
      // From within the subclass we can reach it directly. Cast through
      // unknown because the bundled base type is narrower than our local
      // `LiveAgentLike` interface.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const a = (this as any).agent as LiveAgentLike | undefined;
      return a;
    }

    // `notify` is protected on the base class; the subclass calls it.
    // Cast through unknown to satisfy the loose base type.
    private async sendSignal(
      kind: string,
      body: string,
      target: SubLike,
      priority: SignalPriority = "low",
    ): Promise<void> {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const self = this as any;
      await self.notify(
        {
          source: "memorease",
          kind,
          summary: body,
          priority,
        },
        { threadId: target.threadId, resourceId: target.resourceId },
      );
    }

    /**
     * Public wrapper around the protected base `subscribe` — the arming
     * surface calls this with the current thread from the tool context.
     * Without at least one subscription, `poll()` iterates nothing and no
     * signal (gut-feeling or boot-curation) can ever fire.
     */
    subscribeThread(threadId: string, resourceId: string): void {
      const target = { threadId, resourceId };
      if (this.hasSubscription(target, EXTERNAL_RESOURCE_ID)) return;
      this.subscribe(target, EXTERNAL_RESOURCE_ID);
    }

    async poll(subscriptions: Array<SubLike>): Promise<void> {
      state.lastPollAt = Date.now();
      for (const sub of subscriptions) {
        // One-shot boot curation first (armed curator only) — this is the
        // curator LLM pass that used to block instructions() at session
        // start, now delivered as a background signal.
        if (curatorModelId) {
          await bootCurateSubscriber(
            () => this.getAgent(),
            store,
            embedImpl,
            sub,
            state,
            (body, target) => this.sendSignal("boot-curation", body, target),
            { modelId: curatorModelId, budget },
          );
        }
        await probeSubscriber(
          () => this.getAgent(),
          store,
          embedImpl,
          sub,
          state,
          (body, target, priority) =>
            this.sendSignal("gut-feeling", body, target, priority),
        );
      }
    }

    getStatus(): ProviderStatus {
      // `subscriptionCount` is protected on the base class.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const subCount = (this as any).subscriptionCount as number | undefined;
      return {
        armed: state.disarmedReason === undefined,
        disarmedReason: state.disarmedReason,
        lastPollAt: state.lastPollAt,
        lastError: state.lastError,
        subscriptionCount: subCount ?? 0,
        notifiedCount: state.notifiedCount,
        bootCuratedCount: state.bootCuratedCount,
      };
    }
  }

  return new MemoreaseSignalProvider();
}
