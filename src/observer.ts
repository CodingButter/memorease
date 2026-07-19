/**
 * Arming surface for the gut-feeling signal provider.
 *
 * The provider is constructed lazily on the first memory tool call. The agent
 * handle isn't available at plugin load (instructions() runs without one), so
 * arming has to happen from inside a tool's execute path, where the tool
 * context carries `mastra` and the live agent id.
 *
 * Fail-soft by design: every lookup is feature-detected. If mastracode
 * renames `getAgentById`, removes `sendNotificationSignal`, or otherwise
 * changes the agent shape, `armProvider` returns a disarmed status and the
 * deliberate-memory core keeps working untouched.
 *
 * `findLiveAgent` mirrors the wren-brain reference plugin's lookup ladder:
 *   `mastra.getAgentById(id)` → `mastra.getAgent(id)` → `mastra.getAgents()`
 *                                                               (single-or-match)
 * The first agent in that chain with a `sendNotificationSignal` function wins.
 */

import type { MastraVector } from "@mastra/core/vector";

import { embed } from "./embed.ts";
import { createMemoreaseProvider, type TapStatus } from "./provider.ts";

/** Minimal tool-context surface this module needs. */
export type ToolCtxLike = {
  agent?: { agentId?: string };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  mastra?: any;
};

/** Minimal agent surface this module looks for. */
type LiveAgent = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sendNotificationSignal?: (...args: any[]) => unknown;
};

export type ArmResult = {
  status: TapStatus;
  via?: string;
  /** True if this call actually performed the connect+startPolling dance. */
  freshlyArmed: boolean;
};

/**
 * Walk the mastra registry's lookup ladder. First hit with a
 * `sendNotificationSignal` function wins. Returns `undefined` if no live
 * agent is reachable.
 */
export function findLiveAgent(
  toolCtx: ToolCtxLike | undefined,
): { agent: LiveAgent; via: string } | undefined {
  if (!toolCtx) return undefined;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mastra = toolCtx.mastra as any;
  if (!mastra) return undefined;
  const agentId = toolCtx.agent?.agentId;

  const attempts: Array<[string, () => unknown]> = [
    [
      "mastra.getAgentById",
      () => (agentId ? mastra.getAgentById?.(agentId) : undefined),
    ],
    [
      "mastra.getAgent",
      () => (agentId ? mastra.getAgent?.(agentId) : undefined),
    ],
    [
      "mastra.getAgents",
      () => {
        const all = mastra.getAgents?.();
        if (!all || typeof all !== "object") return undefined;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const list = Object.values(all) as any[];
        return (
          list.find((a) => a?.id === agentId || a?.name === agentId) ??
          (list.length === 1 ? list[0] : undefined)
        );
      },
    ],
  ];

  for (const [via, attempt] of attempts) {
    try {
      const agent = attempt() as LiveAgent | undefined;
      if (
        agent &&
        typeof agent.sendNotificationSignal === "function"
      ) {
        return { agent, via };
      }
    } catch {
      // try the next lookup path
    }
  }
  return undefined;
}

type ArmedState = {
  // `unknown` because the provider instance is an opaque subclass of
  // SignalProvider; we only need its `getStatus()` method via cast.
  provider: unknown;
  result: ArmResult;
};

let armed: ArmedState | undefined;
let armingPromise: Promise<ArmedState> | undefined;

/**
 * Build the provider and connect it to the live agent. Memoized by promise —
 * concurrent tool calls share the same arming attempt; subsequent calls
 * short-circuit with the cached result.
 *
 * `store` is required and must already be schema-bootstrapped. `embedFn`
 * defaults to the module-level fastembed embedder; tests inject a stub.
 */
export async function armProvider(args: {
  toolCtx: ToolCtxLike | undefined;
  store: MastraVector;
  embedFn?: (text: string) => Promise<number[]>;
  pollIntervalMs?: number;
  /** Curator model id for the background boot-curation signal (optional). */
  curatorModelId?: string;
  /** Character budget for the curated selection (optional). */
  injectBudget?: number;
}): Promise<ArmResult> {
  if (armingPromise) {
    await armingPromise;
  }
  if (armed) {
    // Memoized hit (either pre-existing or just resolved by the promise
    // above): the provider is already connected, but THIS call didn't perform
    // the connect dance — report `freshlyArmed: false`.
    return { ...armed.result, freshlyArmed: false };
  }
  armingPromise = (async () => {
    const { toolCtx, store, embedFn, pollIntervalMs, curatorModelId, injectBudget } = args;
    if (!toolCtx?.mastra) {
      const result: ArmResult = {
        status: "disarmed-no-agent",
        freshlyArmed: false,
      };
      armed = { provider: undefined, result };
      return armed;
    }
    const found = findLiveAgent(toolCtx);
    if (!found) {
      const result: ArmResult = {
        status: "disarmed-no-signal",
        freshlyArmed: false,
      };
      armed = { provider: undefined, result };
      return armed;
    }
    try {
      const provider = createMemoreaseProvider({
        store,
        embedFn: embedFn ?? embed,
        pollIntervalMs,
        curatorModelId,
        injectBudget,
      });
      // The provider's base class exposes `connect(agent)` and
      // `startPolling()` as public methods. Cast through unknown to satisfy
      // the loose base type — the concrete subclass is what we built.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const p = provider as any;
      if (typeof p.connect !== "function" || typeof p.startPolling !== "function") {
        const result: ArmResult = {
          status: "disarmed-no-signal",
          freshlyArmed: false,
        };
        armed = { provider: undefined, result };
        return armed;
      }
      p.connect(found.agent);
      p.startPolling();
      const result: ArmResult = {
        status: "armed",
        via: found.via,
        freshlyArmed: true,
      };
      armed = { provider, result };
      return armed;
    } catch {
      const result: ArmResult = {
        status: "disarmed-no-signal",
        freshlyArmed: false,
      };
      armed = { provider: undefined, result };
      return armed;
    }
  })();
  return (await armingPromise).result;
}

/**
 * Read-only status probe for the `memory_tap_status` tool. Returns a plain
 * object describing the current armed/disarmed state plus, when armed, the
 * provider's own status snapshot.
 */
export function tapStatus(): {
  status: TapStatus;
  via?: string;
  provider?: unknown;
} {
  if (!armed) {
    return { status: "disarmed-no-agent" };
  }
  const { provider, result } = armed;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const p = provider as any;
  const providerStatus =
    p && typeof p.getStatus === "function" ? p.getStatus() : undefined;
  return {
    status: result.status,
    via: result.via,
    provider: providerStatus,
  };
}

/**
 * Test-only escape hatch: drop the memoized arming state so the next
 * `armProvider` call re-attempts from scratch. Used by provider.test.ts to
 * exercise the disarmed → armed transition and the no-agent path.
 */
export function _resetArmingForTests(): void {
  armed = undefined;
  armingPromise = undefined;
}
