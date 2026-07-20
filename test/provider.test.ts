import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { readdirSync, readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import {
  probeSubscriber,
  bootCurateSubscriber,
  bootCurationBody,
  newProbeState,
  extractText,
  extractRecentUserText,
  hintBody,
  tapPriority,
  TAP_THRESHOLD,
  TAP_DEDUP_MS,
  TAP_MEDIUM_THRESHOLD,
  TAP_HIGH_THRESHOLD,
  createMemoreaseProvider,
  type ProbeState,
  type TapStatus,
} from "../src/provider.js";
import {
  armProvider,
  tapStatus,
  findLiveAgent,
  _resetArmingForTests,
} from "../src/observer.js";
import { _resetStoreForTests } from "../src/tools.js";
import { _resetForTests as resetConfig } from "../src/config.js";

/**
 * Provider tests cover the fail-soft surface of the gut-feeling signal layer.
 * The armed live-path (real mastracode observer firing `notify` inside an
 * actual thread) is verified in the Phase 6 demo, not here — these tests prove
 * the disarmed paths, the threshold/dedup behavior of `probeSubscriber`, and
 * the read-only `memory_tap_status` tool's contract.
 *
 * `probeSubscriber` is exported specifically so tests can drive it directly
 * with mock mastra/store/embed/notify — no real SignalProvider subclass or
 * polling loop needed.
 */

// --- helpers --------------------------------------------------------------

type MockMsg = {
  id?: string;
  role?: string;
  content?:
    | string
    | { format?: number; content?: string; parts?: Array<{ type: string; text?: string }> };
};

function makeAgent(opts: {
  recallImpl?: (args: {
    threadId: string | string[];
    resourceId?: string;
    perPage?: number | false;
  }) => Promise<{ messages: MockMsg[] }>;
  hasSendSignal?: boolean;
}) {
  const agent: {
    memory?: { recall: (args: unknown) => Promise<{ messages: MockMsg[] }> };
    sendNotificationSignal?: (...args: unknown[]) => void;
  } = {};
  if (opts.recallImpl) {
    agent.memory = {
      recall: ((args: unknown) =>
        opts.recallImpl!(args as Parameters<typeof opts.recallImpl>[0])) as typeof agent.memory extends {
        recall: infer R;
      }
        ? R
        : never,
    };
  }
  if (opts.hasSendSignal) {
    agent.sendNotificationSignal = () => {};
  }
  return agent;
}

type MockHit = {
  score: number;
  metadata?: { name?: string; content?: string; sourceThreadId?: string };
};

function makeMockStore(hits: MockHit[]) {
  return {
    query: async () => hits,
  };
}

function makeMockEmbedder(vector: number[]) {
  return async () => vector;
}

const SUB = { threadId: "t1", resourceId: "r1" };

// --- pure helpers ---------------------------------------------------------

describe("provider: text extraction", () => {
  it("extractText handles V1 string content", () => {
    expect(extractText({ role: "user", content: "hello world" })).toBe("hello world");
  });

  it("extractText handles V2 parts-array content", () => {
    const msg = {
      role: "user",
      content: {
        format: 2,
        parts: [
          { type: "text", text: "first" },
          { type: "text", text: "second" },
        ],
      },
    };
    expect(extractText(msg)).toBe("first\nsecond");
  });

  it("extractText returns empty string for unrecognised shapes", () => {
    expect(extractText(undefined)).toBe("");
    expect(extractText({})).toBe("");
    expect(extractText({ content: { format: 2, parts: [] } })).toBe("");
  });

  it("extractRecentUserText prefers the latest user message", () => {
    const msgs = [
      { role: "user", content: "old user" },
      { role: "assistant", content: "reply" },
      { role: "user", content: "new user" },
    ];
    expect(extractRecentUserText(msgs)).toBe("new user");
  });

  it("extractRecentUserText falls back to latest of any role", () => {
    const msgs = [
      { role: "assistant", content: "first" },
      { role: "assistant", content: "latest" },
    ];
    expect(extractRecentUserText(msgs)).toBe("latest");
  });

  it("extractRecentUserText returns empty for empty list", () => {
    expect(extractRecentUserText([])).toBe("");
    expect(extractRecentUserText(undefined)).toBe("");
  });

  it("tapPriority maps score to signal level, never urgent", () => {
    expect(tapPriority(TAP_THRESHOLD)).toBe("low");
    expect(tapPriority(TAP_MEDIUM_THRESHOLD - 0.01)).toBe("low");
    expect(tapPriority(TAP_MEDIUM_THRESHOLD)).toBe("medium");
    expect(tapPriority(TAP_HIGH_THRESHOLD - 0.01)).toBe("medium");
    expect(tapPriority(TAP_HIGH_THRESHOLD)).toBe("high");
    expect(tapPriority(0.99)).toBe("high");
    expect(tapPriority(1)).toBe("high");
  });

  it("hintBody is a door, not a document: hints at exploration, never content", () => {
    const body = hintBody([
      { name: "voice-first", score: 0.7234 },
      { name: "fleet-ssh-mesh", score: 0.61 },
    ]);
    expect(body).toContain("voice-first");
    expect(body).toContain("fleet-ssh-mesh");
    expect(body).toContain("memory_query");
    // Framed as starting points for a dive, not as the memory itself.
    expect(body).toContain("Starting points");
    expect(body).toMatch(/dig in|explore/);
  });
});

// --- probeSubscriber: threshold + dedup + fail-soft ----------------------

describe("provider: probeSubscriber", () => {
  let state: ProbeState;
  let notifyCalls: Array<{ body: string; sub: typeof SUB }>;

  beforeEach(() => {
    state = newProbeState();
    notifyCalls = [];
  });

  it("notifies on a hit above TAP_THRESHOLD", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "what does jamie prefer?" }],
      }),
    });
    const store = makeMockStore([
      { score: TAP_THRESHOLD + 0.1, metadata: { name: "voice-first" } },
    ]);
    const embedFn = makeMockEmbedder([1, 2, 3]);

    const result = await probeSubscriber(
      () => agent,
      store as never,
      embedFn,
      SUB,
      state,
      async (body, sub) => {
        notifyCalls.push({ body, sub });
      },
    );

    expect(result).toBe("notified");
    expect(notifyCalls).toHaveLength(1);
    expect(notifyCalls[0].body).toContain("voice-first");
    expect(state.notifiedCount).toBe(1);
  });

  it("resolves memory via async agent.getMemory() when .memory is absent (real Agent shape)", async () => {
    // Real mastra Agents expose memory behind getMemory(), not a .memory prop.
    const agent = {
      getMemory: async () => ({
        recall: async () => ({
          messages: [{ role: "user", content: "what does jamie prefer?" }],
        }),
      }),
    };
    const store = makeMockStore([
      { score: TAP_THRESHOLD + 0.1, metadata: { name: "voice-first" } },
    ]);

    const result = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1, 2, 3]),
      SUB,
      state,
      async (body, sub) => {
        notifyCalls.push({ body, sub });
      },
    );

    expect(result).toBe("notified");
    expect(notifyCalls).toHaveLength(1);
  });

  it("provenance gate: skips a hit the same thread wrote, taps the next candidate", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "tell me about the provider" }],
      }),
    });
    // Top hit was authored by THIS thread — must be skipped; runner-up wins.
    const store = makeMockStore([
      {
        score: TAP_THRESHOLD + 0.3,
        metadata: { name: "self-authored", sourceThreadId: SUB.threadId },
      },
      {
        score: TAP_THRESHOLD + 0.1,
        metadata: { name: "other-thread", sourceThreadId: "someone-else" },
      },
    ]);

    const result = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      async (body, sub) => {
        notifyCalls.push({ body, sub });
      },
    );

    expect(result).toBe("notified");
    expect(notifyCalls).toHaveLength(1);
    expect(notifyCalls[0].body).toContain("other-thread");
    expect(notifyCalls[0].body).not.toContain("self-authored");
  });

  it("provenance gate: silent when every above-threshold hit is self-authored", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "tell me about the provider" }],
      }),
    });
    const store = makeMockStore([
      {
        score: TAP_THRESHOLD + 0.3,
        metadata: { name: "self-authored", sourceThreadId: SUB.threadId },
      },
    ]);

    const result = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      async (body, sub) => {
        notifyCalls.push({ body, sub });
      },
    );

    expect(result).toBe("skipped-below-threshold");
    expect(notifyCalls).toHaveLength(0);
  });

  it("skips when top hit is below TAP_THRESHOLD", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    const store = makeMockStore([
      { score: TAP_THRESHOLD - 0.01, metadata: { name: "weak" } },
    ]);

    const result = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      async () => {
        notifyCalls.push({ body: "", sub: SUB });
      },
    );

    expect(result).toBe("skipped-below-threshold");
    expect(notifyCalls).toHaveLength(0);
  });

  it("dedupes same hit within TAP_DEDUP_MS", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    const store = makeMockStore([
      { score: 0.9, metadata: { name: "same-hit" } },
    ]);
    const notify = async (body: string) => {
      notifyCalls.push({ body, sub: SUB });
    };

    const r1 = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
    );
    const r2 = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
    );

    expect(r1).toBe("notified");
    expect(r2).toBe("skipped-deduped");
    expect(notifyCalls).toHaveLength(1);
  });

  it("passes score-derived priority to notify: strong hit taps high, borderline taps low", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    const priorities: string[] = [];
    const notify = async (
      _body: string,
      _sub: unknown,
      priority: string,
    ) => {
      priorities.push(priority);
    };

    // Strong resonance → high.
    await probeSubscriber(
      () => agent,
      makeMockStore([
        { score: TAP_HIGH_THRESHOLD + 0.05, metadata: { name: "strong" } },
      ]) as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
    );
    // Borderline resonance (different thread state) → low.
    const state2 = newProbeState();
    await probeSubscriber(
      () => agent,
      makeMockStore([
        { score: TAP_THRESHOLD + 0.01, metadata: { name: "borderline" } },
      ]) as never,
      makeMockEmbedder([1]),
      SUB,
      state2,
      notify,
    );

    expect(priorities).toEqual(["high", "low"]);
  });

  it("progress gate: idle conversation never re-taps, even after the dedup window", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    const store = makeMockStore([
      { score: 0.9, metadata: { name: "same-hit" } },
    ]);
    const notify = async (body: string) => {
      notifyCalls.push({ body, sub: SUB });
    };

    const r1 = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
    );
    // Simulate the dedup window having long expired — but no new user text.
    const rec = state.lastNotifiedPerThread.get(SUB.threadId)!;
    rec.at = Date.now() - TAP_DEDUP_MS * 10;
    const r2 = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
    );

    expect(r1).toBe("notified");
    expect(r2).toBe("skipped-deduped");
    expect(notifyCalls).toHaveLength(1);
  });

  it("progress gate: new user text after the window re-taps", async () => {
    let text = "hello";
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: text }],
      }),
    });
    const store = makeMockStore([
      { score: 0.9, metadata: { name: "same-hit" } },
    ]);
    const notify = async (body: string) => {
      notifyCalls.push({ body, sub: SUB });
    };

    const r1 = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
    );
    // Conversation moved AND the dedup window expired.
    text = "now talking about something new";
    const rec = state.lastNotifiedPerThread.get(SUB.threadId)!;
    rec.at = Date.now() - TAP_DEDUP_MS * 10;
    const r2 = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
    );

    expect(r1).toBe("notified");
    expect(r2).toBe("notified");
    expect(notifyCalls).toHaveLength(2);
  });

  it("skips when recall returns no usable text", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({ messages: [] }),
    });
    const store = makeMockStore([{ score: 0.99, metadata: { name: "x" } }]);

    const result = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      async () => {
        notifyCalls.push({ body: "", sub: SUB });
      },
    );

    expect(result).toBe("skipped-no-text");
    expect(notifyCalls).toHaveLength(0);
  });

  it("disarms (no-memory) when agent.memory.recall is missing", async () => {
    const agent = makeAgent({}); // no recall
    const store = makeMockStore([]);

    const result = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      async () => {
        notifyCalls.push({ body: "", sub: SUB });
      },
    );

    expect(result).toBe("disarmed-no-memory");
    expect(state.disarmedReason).toBe("disarmed-no-memory");
    expect(notifyCalls).toHaveLength(0);
  });

  it("swallows recall throws and records lastError", async () => {
    const agent = makeAgent({
      recallImpl: async () => {
        throw new Error("recall blew up");
      },
    });
    const store = makeMockStore([]);

    const result = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      async () => {
        notifyCalls.push({ body: "", sub: SUB });
      },
    );

    expect(result).toBe("error");
    expect(state.lastError).toContain("recall blew up");
    expect(notifyCalls).toHaveLength(0);
  });

  it("swallows store.query throws and records lastError", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    const store = {
      query: async () => {
        throw new Error("pg connection lost");
      },
    };

    const result = await probeSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      async () => {
        notifyCalls.push({ body: "", sub: SUB });
      },
    );

    expect(result).toBe("error");
    expect(state.lastError).toContain("pg connection lost");
  });

  it("swallows embed throws and records lastError", async () => {
    const agent = makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    const store = makeMockStore([{ score: 0.99 }]);
    const badEmbed = async () => {
      throw new Error("onnx load failed");
    };

    const result = await probeSubscriber(
      () => agent,
      store as never,
      badEmbed,
      SUB,
      state,
      async () => {
        notifyCalls.push({ body: "", sub: SUB });
      },
    );

    expect(result).toBe("error");
    expect(state.lastError).toContain("onnx load failed");
  });
});

// --- bootCurateSubscriber: background curation signal ---------------------

describe("provider: bootCurateSubscriber", () => {
  let state: ProbeState;
  let notifyCalls: Array<{ body: string; sub: typeof SUB }>;

  const OPTS_BASE = { modelId: "test/model", budget: 1200 };

  function chattyAgent() {
    return makeAgent({
      recallImpl: async () => ({
        messages: [{ role: "user", content: "working on the memorease plugin" }],
      }),
    });
  }

  function stubCurate(hits: Array<{ name: string; content?: string }>, usedCurator: boolean) {
    return async () => ({
      hits: hits.map((h, i) => ({
        id: String(i),
        score: 0.9 - i * 0.1,
        name: h.name,
        content: h.content ?? "",
      })),
      usedCurator,
    });
  }

  beforeEach(() => {
    state = newProbeState();
    notifyCalls = [];
  });

  const notify = async (body: string, sub: typeof SUB) => {
    notifyCalls.push({ body, sub });
  };

  it("notifies with curated memory names (never content) and marks the thread done", async () => {
    const store = makeMockStore([
      { score: 0.8, metadata: { name: "voice-first", content: "SECRET CONTENT" } },
    ]);

    const result = await bootCurateSubscriber(
      () => chattyAgent(),
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
      { ...OPTS_BASE, curateFn: stubCurate([{ name: "voice-first", content: "SECRET CONTENT" }], true) },
    );

    expect(result).toBe("notified");
    expect(notifyCalls).toHaveLength(1);
    expect(notifyCalls[0].body).toContain("voice-first");
    expect(notifyCalls[0].body).toContain("memory_query");
    expect(notifyCalls[0].body).not.toContain("SECRET CONTENT");
    expect(state.bootCuratedThreads.has(SUB.threadId)).toBe(true);
    expect(state.bootCuratedCount).toBe(1);
  });

  it("is one-shot per thread — second call skips without touching the curator", async () => {
    const store = makeMockStore([{ score: 0.8, metadata: { name: "m" } }]);
    let curateCalls = 0;
    const countingCurate = async () => {
      curateCalls++;
      return {
        hits: [{ id: "0", score: 0.9, name: "m", content: "" }],
        usedCurator: true,
      };
    };

    const r1 = await bootCurateSubscriber(
      () => chattyAgent(),
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
      { ...OPTS_BASE, curateFn: countingCurate },
    );
    const r2 = await bootCurateSubscriber(
      () => chattyAgent(),
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
      { ...OPTS_BASE, curateFn: countingCurate },
    );

    expect(r1).toBe("notified");
    expect(r2).toBe("skipped-already-curated");
    expect(curateCalls).toBe(1);
    expect(notifyCalls).toHaveLength(1);
  });

  it("suppresses the signal on curator fallback (similarity = what boot already injected)", async () => {
    const store = makeMockStore([{ score: 0.8, metadata: { name: "m" } }]);

    const result = await bootCurateSubscriber(
      () => chattyAgent(),
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
      { ...OPTS_BASE, curateFn: stubCurate([{ name: "m" }], false) },
    );

    expect(result).toBe("skipped-curator-fallback");
    expect(notifyCalls).toHaveLength(0);
    // Terminal — don't re-spend the LLM roundtrip next poll.
    expect(state.bootCuratedThreads.has(SUB.threadId)).toBe(true);
  });

  it("retries later when recall has no usable text yet (not terminal)", async () => {
    const agent = makeAgent({ recallImpl: async () => ({ messages: [] }) });
    const store = makeMockStore([{ score: 0.8, metadata: { name: "m" } }]);

    const result = await bootCurateSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
      { ...OPTS_BASE, curateFn: stubCurate([{ name: "m" }], true) },
    );

    expect(result).toBe("skipped-no-text");
    expect(state.bootCuratedThreads.has(SUB.threadId)).toBe(false);
    expect(notifyCalls).toHaveLength(0);
  });

  it("marks done on empty store without calling the curator", async () => {
    const store = makeMockStore([]);
    let curateCalls = 0;

    const result = await bootCurateSubscriber(
      () => chattyAgent(),
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
      {
        ...OPTS_BASE,
        curateFn: async () => {
          curateCalls++;
          return { hits: [], usedCurator: false };
        },
      },
    );

    expect(result).toBe("skipped-empty-store");
    expect(curateCalls).toBe(0);
    expect(state.bootCuratedThreads.has(SUB.threadId)).toBe(true);
  });

  it("swallows throws and records lastError (not terminal)", async () => {
    const agent = makeAgent({
      recallImpl: async () => {
        throw new Error("recall exploded");
      },
    });
    const store = makeMockStore([]);

    const result = await bootCurateSubscriber(
      () => agent,
      store as never,
      makeMockEmbedder([1]),
      SUB,
      state,
      notify,
      OPTS_BASE,
    );

    expect(result).toBe("error");
    expect(state.lastError).toContain("recall exploded");
    expect(state.bootCuratedThreads.has(SUB.threadId)).toBe(false);
  });

  it("bootCurationBody lists names and points at memory_query", () => {
    const body = bootCurationBody(["a-memory", "b-memory"]);
    expect(body).toContain("'a-memory'");
    expect(body).toContain("'b-memory'");
    expect(body).toContain("memory_query");
  });
});

// --- armProvider: live-agent ladder ---------------------------------------

describe("provider: armProvider", () => {
  beforeEach(() => {
    _resetArmingForTests();
    resetConfig();
    _resetStoreForTests();
  });

  afterEach(() => {
    _resetArmingForTests();
    resetConfig();
    _resetStoreForTests();
  });

  it("returns disarmed-no-agent when toolCtx is undefined", async () => {
    const store = makeMockStore([]);
    const result = await armProvider({
      toolCtx: undefined,
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });
    expect(result.status).toBe("disarmed-no-agent" satisfies TapStatus);
    expect(result.freshlyArmed).toBe(false);
  });

  it("returns disarmed-no-agent when mastra is missing", async () => {
    const store = makeMockStore([]);
    const result = await armProvider({
      toolCtx: { agent: { agentId: "a1" } },
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });
    expect(result.status).toBe("disarmed-no-agent");
  });

  it("returns disarmed-no-signal when agent lacks sendNotificationSignal", async () => {
    const store = makeMockStore([]);
    // mastra returns an agent, but without sendNotificationSignal
    const mastra = {
      getAgentById: () => ({ id: "a1", name: "main" }), // no signal method
    };
    const result = await armProvider({
      toolCtx: { agent: { agentId: "a1" }, mastra },
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });
    expect(result.status).toBe("disarmed-no-signal");
  });

  it("arms via mastra.getAgentById when agent has sendNotificationSignal", async () => {
    const store = makeMockStore([]);
    const connectedAgent = {
      id: "a1",
      name: "main",
      sendNotificationSignal: () => {},
      memory: { recall: async () => ({ messages: [] }) },
    };
    const mastra = {
      getAgentById: () => connectedAgent,
    };

    const result = await armProvider({
      toolCtx: { agent: { agentId: "a1" }, mastra },
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });

    expect(result.status).toBe("armed");
    expect(result.via).toBe("mastra.getAgentById");
    expect(result.freshlyArmed).toBe(true);
  });

  it("memoizes: second call does not re-arm", async () => {
    const store = makeMockStore([]);
    const connectedAgent = {
      id: "a1",
      sendNotificationSignal: () => {},
    };
    const mastra = {
      getAgentById: () => connectedAgent,
    };
    const toolCtx = { agent: { agentId: "a1" }, mastra };

    const r1 = await armProvider({
      toolCtx,
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });
    const r2 = await armProvider({
      toolCtx,
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });

    expect(r1.freshlyArmed).toBe(true);
    expect(r2.freshlyArmed).toBe(false);
    expect(r2.status).toBe("armed");
  });

  it("subscribes the calling thread on fresh arm (poll has something to iterate)", async () => {
    const store = makeMockStore([]);
    const mastra = {
      getAgentById: () => ({ id: "a1", sendNotificationSignal: () => {} }),
    };

    await armProvider({
      toolCtx: {
        agent: { agentId: "a1", threadId: "t1", resourceId: "r1" },
        mastra,
      },
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });

    const status = tapStatus();
    expect(status.status).toBe("armed");
    expect(
      (status.provider as { subscriptionCount: number }).subscriptionCount,
    ).toBe(1);
  });

  it("registers new threads on memoized calls and dedups repeats", async () => {
    const store = makeMockStore([]);
    const mastra = {
      getAgentById: () => ({ id: "a1", sendNotificationSignal: () => {} }),
    };
    const ctxFor = (threadId: string) => ({
      agent: { agentId: "a1", threadId, resourceId: "r1" },
      mastra,
    });

    await armProvider({ toolCtx: ctxFor("t1"), store: store as never, embedFn: makeMockEmbedder([1]) });
    // Same thread again — must not double-subscribe.
    await armProvider({ toolCtx: ctxFor("t1"), store: store as never, embedFn: makeMockEmbedder([1]) });
    // A second thread arriving later in the process — must get its own sub.
    await armProvider({ toolCtx: ctxFor("t2"), store: store as never, embedFn: makeMockEmbedder([1]) });

    const status = tapStatus();
    expect(
      (status.provider as { subscriptionCount: number }).subscriptionCount,
    ).toBe(2);
  });

  it("arms without a subscription when toolCtx lacks threadId (no throw)", async () => {
    const store = makeMockStore([]);
    const mastra = {
      getAgentById: () => ({ id: "a1", sendNotificationSignal: () => {} }),
    };

    const result = await armProvider({
      toolCtx: { agent: { agentId: "a1" }, mastra },
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });

    expect(result.status).toBe("armed");
    const status = tapStatus();
    expect(
      (status.provider as { subscriptionCount: number }).subscriptionCount,
    ).toBe(0);
  });
});

// --- findLiveAgent: lookup ladder -----------------------------------------

describe("provider: findLiveAgent ladder", () => {
  it("returns undefined when mastra is absent", () => {
    expect(findLiveAgent({ agent: { agentId: "x" } })).toBeUndefined();
  });

  it("returns undefined when no agent has sendNotificationSignal", () => {
    const mastra = {
      getAgentById: () => ({ id: "x" }), // no signal
      getAgent: () => undefined,
      getAgents: () => ({}),
    };
    expect(findLiveAgent({ agent: { agentId: "x" }, mastra })).toBeUndefined();
  });

  it("falls through getAgentById → getAgent → getAgents", () => {
    const withSignal = { id: "a1", sendNotificationSignal: () => {} };
    const mastra = {
      getAgentById: () => undefined,
      getAgent: () => undefined,
      getAgents: () => ({ main: withSignal }),
    };
    const found = findLiveAgent({ agent: { agentId: "a1" }, mastra });
    expect(found?.via).toBe("mastra.getAgents");
    expect(found?.agent).toBe(withSignal);
  });

  it("prefers a name match in getAgents over the single-agent fallback", () => {
    const named = { id: "a1", name: "main", sendNotificationSignal: () => {} };
    const other = { id: "a2", name: "other", sendNotificationSignal: () => {} };
    const mastra = {
      getAgents: () => ({ main: named, other }),
    };
    const found = findLiveAgent({
      agent: { agentId: "main" },
      mastra,
    });
    expect(found?.agent).toBe(named);
  });
});

// --- memory_tap_status -----------------------------------------------------

describe("provider: tapStatus probe", () => {
  beforeEach(() => {
    _resetArmingForTests();
  });

  it("returns disarmed-no-agent when never armed", () => {
    const s = tapStatus();
    expect(s.status).toBe("disarmed-no-agent");
    expect(s.provider).toBeUndefined();
  });

  it("exposes provider status snapshot after arming", async () => {
    const store = makeMockStore([]);
    const mastra = {
      getAgentById: () => ({
        id: "a1",
        sendNotificationSignal: () => {},
      }),
    };
    await armProvider({
      toolCtx: { agent: { agentId: "a1" }, mastra },
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });
    const s = tapStatus();
    expect(s.status).toBe("armed");
    expect(s.via).toBe("mastra.getAgentById");
    // Provider status snapshot exists (the concrete shape is the subclass's
    // getStatus() return — just assert it's an object with expected fields).
    expect(typeof s.provider).toBe("object");
    const ps = s.provider as { notifiedCount?: number; armed?: boolean };
    expect(ps).toHaveProperty("notifiedCount");
    expect(ps).toHaveProperty("armed");
  });
});

// --- createMemoreaseProvider: subclass smoke ------------------------------

describe("provider: createMemoreaseSignalProvider subclass", () => {
  it("builds an instance with the expected id and pollInterval", () => {
    const store = makeMockStore([]);
    const p = createMemoreaseProvider({
      store: store as never,
      embedFn: makeMockEmbedder([1]),
      pollIntervalMs: 12345,
    }) as {
      id: string;
      name: string;
      pollInterval: number;
      getStatus(): Record<string, unknown>;
    };

    expect(p.id).toBe("memorease-gut-feeling");
    expect(p.name).toBe("Memorease gut-feeling");
    expect(p.pollInterval).toBe(12345);
    expect(typeof p.getStatus).toBe("function");
  });

  it("getStatus reports initial state with zero notifications", () => {
    const store = makeMockStore([]);
    const p = createMemoreaseProvider({
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    }) as { getStatus(): { notifiedCount: number; armed: boolean } };

    const s = p.getStatus();
    expect(s.notifiedCount).toBe(0);
    expect(typeof s.armed).toBe("boolean");
  });

  it("sendSignal passes a per-thread coalesceKey so pending taps collapse instead of stacking", async () => {
    const store = makeMockStore([]);
    const p = createMemoreaseProvider({
      store: store as never,
      embedFn: makeMockEmbedder([1]),
    });

    // Capture what reaches the base class's protected `notify` — the
    // coalesceKey is what lets storage replace a still-pending notification
    // in place (threadId+source+kind+coalesceKey match) instead of piling
    // up a backlog.
    const notified: Array<{ input: Record<string, unknown>; target: unknown }> = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (p as any).notify = async (input: Record<string, unknown>, target: unknown) => {
      notified.push({ input, target });
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await (p as any).sendSignal(
      "gut-feeling",
      "hint body",
      { threadId: "thread-1", resourceId: "res-1" },
      "medium",
    );

    expect(notified).toHaveLength(1);
    expect(notified[0].input).toMatchObject({
      source: "memorease",
      kind: "gut-feeling",
      summary: "hint body",
      priority: "medium",
      coalesceKey: "memorease:gut-feeling:thread-1",
    });
    expect(notified[0].target).toEqual({ threadId: "thread-1", resourceId: "res-1" });
  });
});

// --- type-strip safety: no enum/namespace in src/ -------------------------

/**
 * Node's native type-stripping (the mastracode loader's fallback path when no
 * tsx-style transpiler is active) rejects `enum`, `namespace`, and parameter
 * properties. Guard against regressions by scanning src/*.ts statically.
 *
 * This is the meaningful test for the adversarial-review finding (the original
 * Phase 5 used a TypeScript `enum`). A dynamic `import()` test would not catch
 * the issue because tsx/esbuild tolerate enums — the failure only shows up in
 * strip-only mode.
 */
describe("provider: type-strip safety (no enum/namespace in src/)", () => {
  const srcDir = join(dirname(fileURLToPath(import.meta.url)), "..", "src");

  it("no src/*.ts file declares an enum or namespace", () => {
    const files = readdirSync(srcDir).filter((f) => f.endsWith(".ts"));
    expect(files.length).toBeGreaterThan(0);
    const offenders: string[] = [];
    for (const f of files) {
      const src = readFileSync(join(srcDir, f), "utf8");
      // Match `enum Foo`, `namespace Foo`, but NOT inside strings/comments.
      // Crude but effective: word-boundary `enum ` / `namespace ` at the start
      // of a statement (preceded by ; { } newline or start of file).
      const enumMatches = src.match(/(^|[;{}\n])\s*(?:export\s+)?enum\s+[A-Za-z_$]/);
      const nsMatches = src.match(/(^|[;{}\n])\s*(?:export\s+)?namespace\s+[A-Za-z_$]/);
      if (enumMatches) offenders.push(`${f}: enum (${enumMatches[0].trim()})`);
      if (nsMatches) offenders.push(`${f}: namespace (${nsMatches[0].trim()})`);
    }
    expect(offenders).toEqual([]);
  });
});
