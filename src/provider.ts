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

export type TapStatus =
  | "armed"
  | "disarmed-no-agent"
  | "disarmed-no-signal"
  | "disarmed-no-memory";

/** Cosine threshold above which a stored memory is considered "relevant". */
export const TAP_THRESHOLD = 0.5;
/** Per-thread dedup window: same top hit within this window does not re-notify. */
export const TAP_DEDUP_MS = 5 * 60 * 1000;
/** Default poll interval if none provided. */
export const DEFAULT_POLL_MS = 30_000;
/** How many recent messages to pull from each thread per poll. */
const RECALL_PER_PAGE = 10;

/** Minimal structural surface this module needs from a live mastra agent. */
type LiveAgentLike = {
  memory?: {
    recall?: (args: {
      threadId: string | string[];
      resourceId?: string;
      perPage?: number | false;
    }) => Promise<{
      messages: Array<RecallMessageLike>;
    }>;
  };
  sendNotificationSignal?: unknown;
};

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

/** Mutable probe state — owned by the provider, passed to `probeSubscriber`. */
export type ProbeState = {
  /** threadId → { name, at } last notification, for dedup. */
  lastNotifiedPerThread: Map<string, { name: string; at: number }>;
  lastPollAt?: number;
  lastError?: string;
  /** Reason the provider is currently disarmed, if any. */
  disarmedReason?: Exclude<TapStatus, "armed">;
  notifiedCount: number;
};

export function newProbeState(): ProbeState {
  return {
    lastNotifiedPerThread: new Map(),
    notifiedCount: 0,
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

/** Format the nudge body shown to the agent. Never pastes memory content. */
export function hintBody(hit: {
  name: string;
  score: number;
}): string {
  return (
    `memorease: you may have a relevant memory named '${hit.name}' ` +
    `(score ${hit.score.toFixed(2)}). Consider calling memory_query to recall it.`
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
  notify: (body: string, sub: SubLike) => Promise<void>,
): Promise<"notified" | "skipped-below-threshold" | "skipped-deduped" | "skipped-no-text" | "disarmed-no-memory" | "error"> {
  try {
    const agent = getAgent();
    if (!agent?.memory || typeof agent.memory.recall !== "function") {
      state.disarmedReason = "disarmed-no-memory";
      return "disarmed-no-memory";
    }
    const recalled = await agent.memory.recall({
      threadId: sub.threadId,
      resourceId: sub.resourceId,
      perPage: RECALL_PER_PAGE,
    });
    const recent = extractRecentUserText(recalled?.messages);
    if (!recent) return "skipped-no-text";

    const vector = await embedFn(recent);
    const hits = await store.query({
      indexName: INDEX_NAME,
      queryVector: vector,
      topK: 1,
    });
    const top = hits[0];
    if (!top || top.score < TAP_THRESHOLD) return "skipped-below-threshold";

    const name = String(top.metadata?.name ?? "");
    const now = Date.now();
    const last = state.lastNotifiedPerThread.get(sub.threadId);
    if (last && last.name === name && now - last.at < TAP_DEDUP_MS) {
      return "skipped-deduped";
    }

    await notify(hintBody({ name, score: top.score }), sub);
    state.lastNotifiedPerThread.set(sub.threadId, { name, at: now });
    state.notifiedCount += 1;
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
};

type CreateProviderOpts = {
  store: MastraVector;
  /** Override the embedder (tests). Defaults to the module-level `embed`. */
  embedFn?: (text: string) => Promise<number[]>;
  pollIntervalMs?: number;
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
}: CreateProviderOpts): unknown {
  const embedImpl = embedFn ?? embed;
  const state = newProbeState();
  const pollInterval = pollIntervalMs ?? DEFAULT_POLL_MS;

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

    async poll(subscriptions: Array<SubLike>): Promise<void> {
      state.lastPollAt = Date.now();
      for (const sub of subscriptions) {
        await probeSubscriber(
          () => this.getAgent(),
          store,
          embedImpl,
          sub,
          state,
          async (body, target) => {
            // `notify` is protected on the base class; the subclass calls it.
            // Cast through unknown to satisfy the loose base type.
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const self = this as any;
            await self.notify(
              {
                source: "memorease",
                kind: "gut-feeling",
                summary: body,
                priority: "low",
              },
              { threadId: target.threadId, resourceId: target.resourceId },
            );
          },
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
      };
    }
  }

  return new MemoreaseSignalProvider();
}
