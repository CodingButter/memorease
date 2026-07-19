/**
 * Curator: optional LLM reranking of boot-time memory candidates.
 *
 * The curator is *armed* when a model id can be resolved; when armed,
 * `curateForBoot` asks the model to pick the few candidates that deserve
 * system-prompt space within `budget` characters, in the model's own order.
 * When *disarmed*, candidates are returned top-N by vector score, truncated
 * to the budget. Disarming never throws and never blocks the session.
 *
 * Default model chain (zero-config path): explicit `config.curatorModel`
 * wins; otherwise fall back to mastracode's `settings.json → models.observerModelOverride`
 * — the same cheap model mastracode already uses for the observer/reflector,
 * with credentials already provisioned. `resolveModel()` handles credentials
 * internally (OAuth → apikey slot → env per its JSDoc); this module never
 * reads `auth.json` or env vars directly.
 */

import type { MemoryHit } from "./memory.js";
import { readMastracodeSettings } from "./config.js";

/**
 * Minimal shape of the plugin config we read here. The full type lives in
 * `src/index.ts`; we only need the curator-relevant fields.
 */
export type CuratorConfig = {
  curatorModel?: string;
};

/**
 * MastraCode plugin context shape — minimal subset we read.
 */
type CuratorContext = {
  config?: CuratorConfig;
};

export const INJECT_BUDGET_DEFAULT = 1200;
export const INJECT_BUDGET_MIN = 200;
export const INJECT_BUDGET_MAX = 8000;

/**
 * Resolve the model id to use for curation, in priority order:
 *   1. `config.curatorModel` (non-empty string)
 *   2. `settings.json → models.observerModelOverride` (non-empty string)
 *   3. `undefined` (disarmed)
 *
 * Never throws. Any settings read or parse error → treated as "no observer
 * model configured" → disarmed.
 */
export function resolveCuratorModel(config: CuratorConfig): string | undefined {
  const explicit =
    typeof config.curatorModel === "string" ? config.curatorModel.trim() : "";
  if (explicit) return explicit;

  const settings = readMastracodeSettings();
  if (!settings) return undefined;
  const models = settings.models as
    | { observerModelOverride?: string }
    | undefined;
  const fromSettings =
    typeof models?.observerModelOverride === "string"
      ? models.observerModelOverride.trim()
      : "";
  return fromSettings || undefined;
}

/**
 * Returns `undefined` when the curator is armed, or a branded disarmed-string
 * when it isn't. Mirrors the wren-brain pattern; surfaces the zero-config
 * default path in the hint so the user knows about the observer-model
 * fallback before being asked to pick a model manually.
 */
export function curatorPreflightError(config: CuratorConfig): string | undefined {
  const modelId = resolveCuratorModel(config);
  if (modelId) return undefined;
  return "memorease: curator disarmed — no curator model selected and no mastracode observer model found in settings.json. Fix: /plugins → Memorease → curatorModel → pick a model, or set models.observerModelOverride in mastracode. Memory will still be injected using similarity ranking only.";
}

/**
 * Resolve `config.injectBudget` (a string from the plugin config schema,
 * since the loader's type union is `'model' | 'boolean' | 'string'`) to an
 * int, clamped to a sane range. Falls back to the default on any parse failure.
 */
export function resolveInjectBudget(raw: string | undefined): number {
  const n = Number.parseInt(String(raw ?? ""), 10);
  if (!Number.isFinite(n)) return INJECT_BUDGET_DEFAULT;
  return Math.max(INJECT_BUDGET_MIN, Math.min(INJECT_BUDGET_MAX, n));
}

/**
 * Build a system prompt for the curator. Phrased to elicit a ranked list of
 * memory names — nothing else — so parsing stays trivial and robust.
 */
function curatorSystemPrompt(budget: number): string {
  return [
    "You are the memory curator for an AI coding assistant's next session.",
    "Your job: select the few stored memories that will most help the assistant in this session, ranked most-relevant first.",
    `Total output budget: ~${budget} characters. Pick only what fits.`,
    "",
    "Context about the upcoming session is provided. Use it to judge relevance.",
    "Reply with ONLY a JSON object of shape:",
    '{"select": ["memory-name-1", "memory-name-2", ...]}',
    "Names must match the candidate list verbatim. No prose, no commentary.",
  ].join("\n");
}

/**
 * Parse the curator's reply. Tolerates a few common shape variations:
 *  - `{"select": ["a", "b"]}` — the documented shape
 *  - `{"select[]": [...]}` — stray suffix
 *  - bare JSON array `["a", "b"]`
 * Unknown names are dropped (the curator can't invent new memories). On any
 * parse failure, the caller falls back to similarity ranking.
 */
function parseCuratorReply(
  reply: string,
  candidates: MemoryHit[],
): MemoryHit[] | undefined {
  const byName = new Map(candidates.map((c) => [c.name, c]));

  const tryExtract = (obj: unknown): string[] | undefined => {
    if (!obj || typeof obj !== "object") return undefined;
    const o = obj as Record<string, unknown>;
    const arr = (o.select ?? o["select[]"]) as unknown;
    if (Array.isArray(arr)) {
      return arr.filter((x): x is string => typeof x === "string");
    }
    return undefined;
  };

  let parsed: unknown;
  try {
    parsed = JSON.parse(reply);
  } catch {
    // Try to find a JSON object embedded in surrounding prose.
    const match = reply.match(/\{[\s\S]*\}/);
    if (!match) return undefined;
    try {
      parsed = JSON.parse(match[0]);
    } catch {
      return undefined;
    }
  }

  const selected = tryExtract(parsed);
  if (!selected) return undefined;

  // Preserve curator order; drop unknown names silently.
  const out: MemoryHit[] = [];
  for (const name of selected) {
    const hit = byName.get(name);
    if (hit) out.push(hit);
  }
  if (out.length === 0) {
    // The curator returned nothing usable — surface nothing rather than
    // silently substituting our own ranking. The caller decides what to do.
    return undefined;
  }
  return out;
}

/**
 * Extract a plain-text reply from a GatewayLanguageModel `doGenerate` result.
 * The content is an array of parts; we concatenate text parts, skip reasoning.
 */
function replyText(content: unknown): string {
  if (!Array.isArray(content)) return "";
  let text = "";
  for (const part of content) {
    if (
      part &&
      typeof part === "object" &&
      (part as { type?: string }).type === "text" &&
      typeof (part as { text?: string }).text === "string"
    ) {
      text += (part as { text: string }).text;
    }
  }
  return text.trim();
}

/**
 * Lazily import `resolveModel` from the code-sdk subpath. Dynamic so this
 * module can be loaded in tests that stub the curator instead of provisioning
 * real provider credentials. Never throws — returns `undefined` on any failure.
 */
async function safeResolveModel(
  modelId: string,
): Promise<
  | { doGenerate: (opts: unknown) => Promise<{ content: unknown }> }
  | undefined
> {
  try {
    const mod = (await import("@mastra/code-sdk/agents/model")) as unknown as {
      resolveModel: (id: string) => {
        doGenerate: (opts: unknown) => Promise<{ content: unknown }>;
      };
    };
    return mod.resolveModel(modelId);
  } catch {
    return undefined;
  }
}

/**
 * Curate boot candidates within `budget` characters.
 *
 * - When `modelId` is undefined → disarmed path: top-N by vector score.
 * - When armed → ask the model to rank. Any model failure falls back to
 *   similarity ranking with a distinct branded note attached to the caller
 *   (this function returns `{ kind: "disarmed-fallback", note, hits }`).
 *
 * `sessionContext` is a short string (hostname, cwd basename, git branch)
 * the caller builds; it goes into the user message to focus the model.
 */
export type CurateResult = {
  hits: MemoryHit[];
  /** True iff the curator model was actually invoked and its reply used. */
  usedCurator: boolean;
  /**
   * Branded note to append to the injected section when the curator was
   * requested but couldn't run (model missing, network, parse failure).
   * Undefined in the success and intentionally-disarmed paths.
   */
  fallbackNote?: string;
};

export async function curateForBoot(
  modelId: string | undefined,
  candidates: MemoryHit[],
  budget: number,
  sessionContext: string,
): Promise<CurateResult> {
  // Disarmed (intentional) path — no model call.
  if (!modelId) {
    return { hits: rankAndTruncate(candidates, budget), usedCurator: false };
  }

  // Nothing to curate.
  if (candidates.length === 0) {
    return { hits: [], usedCurator: false };
  }

  const model = await safeResolveModel(modelId);
  if (!model) {
    return {
      hits: rankAndTruncate(candidates, budget),
      usedCurator: false,
      fallbackNote: `memorease: curator could not resolve model ${modelId} — falling back to similarity ranking.`,
    };
  }

  const userMsg = [
    "Upcoming session context:",
    sessionContext || "(no context available)",
    "",
    "Candidate memories (name — score — content):",
    ...candidates.map(
      (c) => `- ${c.name} — ${c.score.toFixed(3)} — ${c.content}`,
    ),
  ].join("\n");

  try {
    const result = await model.doGenerate({
      prompt: [
        { role: "system", content: [{ type: "text", text: curatorSystemPrompt(budget) }] },
        { role: "user", content: [{ type: "text", text: userMsg }] },
      ],
    });
    const picked = parseCuratorReply(replyText(result.content), candidates);
    if (!picked) {
      return {
        hits: rankAndTruncate(candidates, budget),
        usedCurator: false,
        fallbackNote: `memorease: curator reply was unparseable — falling back to similarity ranking.`,
      };
    }
    return { hits: truncate(picked, budget), usedCurator: true };
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    return {
      hits: rankAndTruncate(candidates, budget),
      usedCurator: false,
      fallbackNote: `memorease: curator call failed (${detail}) — falling back to similarity ranking.`,
    };
  }
}

/**
 * Sort candidates by descending vector score, then truncate to `budget`
 * characters of rendered content (see `renderHit`).
 */
function rankAndTruncate(candidates: MemoryHit[], budget: number): MemoryHit[] {
  const sorted = [...candidates].sort((a, b) => b.score - a.score);
  return truncate(sorted, budget);
}

/**
 * Greedily include hits until the next one would exceed `budget` characters
 * of rendered content. Always returns at least one hit if the input is
 * non-empty (the highest-scoring one) even if it alone exceeds the budget —
 * a single oversized memory is more useful than nothing.
 */
function truncate(hits: MemoryHit[], budget: number): MemoryHit[] {
  if (hits.length === 0) return [];
  const out: MemoryHit[] = [];
  let used = 0;
  for (const hit of hits) {
    const size = renderHit(hit).length + 1; // +1 for the leading newline
    if (out.length > 0 && used + size > budget) break;
    out.push(hit);
    used += size;
  }
  return out;
}

/**
 * Render a single memory as a markdown bullet. Used by the curator and the
 * instructions layer so the rendering rule lives in one place.
 */
export function renderHit(hit: MemoryHit): string {
  const parts = [`- **${hit.name}** — ${hit.content}`];
  if (hit.type && hit.type !== "fact") parts.push(`  (type: ${hit.type})`);
  return parts.join("\n");
}

/**
 * Convenience wrapper for callers that already have a context object.
 */
export function curatorModelFromContext(
  context: CuratorContext,
): string | undefined {
  return resolveCuratorModel(context.config ?? {});
}
