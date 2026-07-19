/**
 * Plugin tools. Four tools, one shared store + embedder resolved lazily from the
 * plugin context on first use:
 *
 *   memory_query         — semantic search; "query before assuming prior context"
 *   memory_write         — upsert by stable name; "write-on-learn"
 *   memory_forget        — delete by name
 *   memory_distill_skill — fold named memories into a SKILL.md on disk
 *
 * Every execute is wrapped in a fail-soft net (via the memory-ops layer). Tools
 * never throw on storage/embedder failure — they return a branded `{ok:false,
 * error}` result so the agent (and the user) see one consistent message and the
 * session keeps running memoryless until the issue is fixed.
 */

import { createTool, z } from "mastracode/plugin";
import type { MastraCodePluginContext } from "mastracode/plugin";
import { mkdir, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

import { resolveConfig, type PluginContext } from "./config.ts";
import {
  createVectorStore,
  ensureSchema,
} from "./store.ts";
import type { MastraVector } from "@mastra/core/vector";
import {
  forgetMemory,
  queryMemories,
  writeMemory,
} from "./memory.ts";
import { armProvider, tapStatus, type ToolCtxLike } from "./observer.ts";
import { resolveCuratorModel, resolveInjectBudget } from "./curator.ts";

/**
 * Lazily-resolved store handle. The first tool call pays the ONNX warm-up and
 * schema bootstrap; subsequent calls reuse the same store. A module-level
 * singleton is fine — mastracode runs each session in its own process, so there
 * is no cross-session leakage, and the store is just a connection pool.
 */
type StoreHandle = { store: MastraVector };

let handle: StoreHandle | undefined;
let handleConfigFingerprint: string | undefined;
// Negative cache: if schema bootstrap failed for this fingerprint, remember
// the branded error and short-circuit subsequent calls. Without this, every
// tool call against a persistently-down pg backend re-enters getStore,
// re-runs ensureSchema, and stalls the agent for the full connection timeout.
// The negative entry is cleared by _resetStoreForTests and on any successful
// resolution after a config change.
let handleError: { fingerprint: string; error: string } | undefined;
// Test-only: counts how many calls hit the negative cache (i.e. were
// short-circuited without re-attempting ensureSchema). Lets the behavioral
// test prove the cache works without timing or spying across ESM bindings.
export let _negativeCacheHitsForTests = 0;

function configFingerprint(ctx: PluginContext): string {
  const r = resolveConfig(ctx);
  return r.backend === "pg"
    ? `pg:${r.connectionString ?? ""}`
    : `libsql:${r.libsqlUrl ?? ""}`;
}

/**
 * Curator wiring for `armProvider` — the background boot-curation signal
 * needs the resolved curator model id and inject budget, since the curator
 * LLM no longer runs on the boot instructions path.
 */
function curatorArgs(ctx: PluginContext): {
  curatorModelId?: string;
  injectBudget: number;
} {
  const config = (ctx.config ?? {}) as { curatorModel?: string; injectBudget?: string };
  return {
    curatorModelId: resolveCuratorModel(config),
    injectBudget: resolveInjectBudget(config.injectBudget),
  };
}

/**
 * Resolve (and cache) the vector store. The first call per config fingerprint
 * also fires `armProvider(toolCtx, store)` — this is where the gut-feeling
 * signal provider gets connected to the live agent. Arming is memoized inside
 * `armProvider` itself, so repeated calls are a no-op.
 *
 * `toolCtx` is optional: boot instructions and tests call `getStore` without a
 * tool context, and arming is simply skipped (the provider arms on the first
 * real tool call instead).
 */
async function getStore(
  context: PluginContext,
  toolCtx?: ToolCtxLike,
): Promise<StoreHandle> {
  const fp = configFingerprint(context);
  const cached = handle && handleConfigFingerprint === fp ? handle : undefined;
  if (cached) {
    // Even on a cache hit, attempt arming once if a tool context is present
    // and we somehow haven't armed yet (e.g. first call after a cache hit
    // from instructions). Memoized inside armProvider — cheap.
    if (toolCtx) {
      void armProvider({ toolCtx, store: cached.store, ...curatorArgs(context) }).catch(
        () => {},
      );
    }
    return cached;
  }
  // Negative cache: a prior call against this fingerprint failed schema
  // bootstrap. Return the same branded error immediately — do NOT re-attempt
  // (which would stall the agent for the connection timeout on every call).
  if (handleError && handleError.fingerprint === fp) {
    _negativeCacheHitsForTests++;
    throw new Error(handleError.error);
  }
  const resolved = resolveConfig(context);
  let store: MastraVector;
  try {
    store = await createVectorStore(resolved);
    await ensureSchema(store);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    const branded = `storage init failed — ${detail}. The session continues without memory; this error will repeat on every memory tool call until the underlying issue is fixed and the session is restarted.`;
    handleError = { fingerprint: fp, error: branded };
    throw new Error(branded);
  }
  handle = { store };
  handleConfigFingerprint = fp;
  handleError = undefined;
  if (toolCtx) {
    // Fire-and-forget — arming failures must not break the tool call. The
    // deliberate-memory core works without the provider.
    void armProvider({ toolCtx, store, ...curatorArgs(context) }).catch(() => {});
  }
  return handle;
}

/**
 * Test-only escape hatch: drop the cached store so the next `getStore` call
 * re-resolves against the current config. Used by tools.test.ts to point the
 * tools at a temp libsql file and to inject unreachable backends.
 */
export function _resetStoreForTests(): void {
  handle = undefined;
  handleConfigFingerprint = undefined;
  handleError = undefined;
  _negativeCacheHitsForTests = 0;
}

/**
 * Resolve the skills directory. Honors `config.skillsDir` (if non-empty),
 * otherwise defaults to `~/.agents/skills`. Relative paths are resolved
 * against the user's home dir (not cwd) for stability across sessions.
 */
function resolveSkillsDir(skillsDir?: string): string {
  const raw = (skillsDir ?? "").trim();
  if (!raw) return join(homedir(), ".agents", "skills");
  return resolve(homedir(), raw);
}

/**
 * Slug validation for `memory_distill_skill`. Conservative: kebab-case,
 * alphanumerics, 1-64 chars. Anything fancier can be added later; the goal is
 * to reject path traversal and weird FS entries.
 */
function isValidSlug(slug: string): boolean {
  return /^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/.test(slug);
}

/**
 * Format the SKILL.md body. YAML-ish frontmatter carries provenance so the
 * distilled skill is auditable later ("which memories fed this? when?"). The
 * body is a plain summary + instructions section, no opinions on style beyond
 * that — the agent that calls the tool gets to phrase summary/instructions.
 */
function formatSkillFile(args: {
  slug: string;
  summary: string;
  instructions: string;
  fromNames: string[];
  distilledAt: string;
}): string {
  const { slug, summary, instructions, fromNames, distilledAt } = args;
  const provenance = fromNames.map((n) => `  - ${n}`).join("\n");
  return [
    "---",
    `name: ${slug}`,
    `description: ${summary.replace(/\n/g, " ").slice(0, 200)}`,
    "provenance:",
    "  distilled_from:",
    fromNames.length ? provenance : "  - []",
    `  distilled_at: ${distilledAt}`,
    "  distilled_by: memorease.memory_distill_skill",
    "---",
    "",
    "# " + slug,
    "",
    summary,
    "",
    "## Instructions",
    "",
    instructions,
    "",
  ].join("\n");
}

export function buildTools(context: MastraCodePluginContext) {
  const ctx = context as unknown as PluginContext & {
    config?: { skillsDir?: string };
  };

  const memory_query = createTool({
    id: "memory_query",
    description:
      "Search the memorease memory store semantically. Returns ranked memories with scores. Query before assuming prior context — names are hints, not promises. Use this whenever you are about to act on something the user may have previously told you to remember.",
    inputSchema: z.object({
      text: z
        .string()
        .min(1)
        .describe("The query text — what you want to look up."),
      k: z
        .number()
        .int()
        .positive()
        .max(50)
        .optional()
        .describe("Top-K results. Defaults to 5."),
    }),
    execute: async (input: { text: string; k?: number }, toolCtx?: ToolCtxLike) => {
      try {
        const { store } = await getStore(ctx, toolCtx);
        const res = await queryMemories(store, input.text, {
          topK: input.k ?? 5,
        });
        if (!res.ok) return res;
        return { ok: true as const, results: res.value };
      } catch (err) {
        return {
          ok: false as const,
          error: brandedFromError(err, "query"),
        };
      }
    },
  });

  const memory_write = createTool({
    id: "memory_write",
    description:
      "Write a durable memory under a stable name. Reuse the same name to update. Write-on-learn: the moment a durable fact lands (a stable preference, an infra detail, a decision and its rationale), store it — do not batch. Avoid transient, task-local, secret, or easily-recomputed details.",
    inputSchema: z.object({
      name: z
        .string()
        .min(1)
        .max(128)
        .describe(
          "Stable, descriptive fact id (kebab-case recommended). Reuse to update.",
        ),
      content: z
        .string()
        .min(1)
        .describe("The memory body. Plain text; one fact per call."),
      type: z
        .string()
        .optional()
        .describe(
          "Optional type tag (e.g. 'preference', 'infrastructure', 'lesson', 'reference'). Defaults to 'fact'.",
        ),
      metadata: z
        .record(z.string(), z.unknown())
        .optional()
        .describe("Optional structured metadata."),
    }),
    execute: async (input: {
      name: string;
      content: string;
      type?: string;
      metadata?: Record<string, unknown>;
    }, toolCtx?: ToolCtxLike) => {
      try {
        const { store } = await getStore(ctx, toolCtx);
        const res = await writeMemory(store, {
          name: input.name,
          content: input.content,
          type: input.type,
          metadata: input.metadata,
        });
        if (!res.ok) return res;
        return { ok: true as const, written: res.value };
      } catch (err) {
        return {
          ok: false as const,
          error: brandedFromError(err, "write"),
        };
      }
    },
  });

  const memory_forget = createTool({
    id: "memory_forget",
    description:
      "Delete a memory by its stable name. Idempotent — forgetting a name that doesn't exist is not an error. Use when the user corrects a stale fact or asks to remove something.",
    inputSchema: z.object({
      name: z
        .string()
        .min(1)
        .describe("Stable fact id to delete."),
    }),
    execute: async (input: { name: string }, toolCtx?: ToolCtxLike) => {
      try {
        const { store } = await getStore(ctx, toolCtx);
        const res = await forgetMemory(store, input.name);
        if (!res.ok) return res;
        return { ok: true as const, forgotten: res.value.forgotten };
      } catch (err) {
        return {
          ok: false as const,
          error: brandedFromError(err, "forget"),
        };
      }
    },
  });

  const memory_distill_skill = createTool({
    id: "memory_distill_skill",
    description:
      "Distill named memories into a SKILL.md file on disk under the configured skills directory (default ~/.agents/skills). Writes a provenance record back to the store so the lineage is queryable. The new skill becomes available at the harness's NEXT scan/session — it is not active immediately. Use this when you notice a cluster of memories that would be more useful as a reusable skill than as separate facts.",
    inputSchema: z.object({
      slug: z
        .string()
        .min(1)
        .max(64)
        .describe(
          "Short kebab-case identifier for the skill (lowercase, digits, hyphens).",
        ),
      fromNames: z
        .array(z.string().min(1))
        .min(1)
        .describe("Memory names to fold into the skill."),
      summary: z
        .string()
        .min(1)
        .describe("One-paragraph summary of what the skill is about."),
      instructions: z
        .string()
        .min(1)
        .describe(
          "Reusable guidance body. Written as the skill's instructions — what an agent reading this skill should know or do.",
        ),
    }),
    execute: async (input: {
      slug: string;
      fromNames: string[];
      summary: string;
      instructions: string;
    }, toolCtx?: ToolCtxLike) => {
      if (!isValidSlug(input.slug)) {
        return {
          ok: false as const,
          error: `memorease: invalid slug '${input.slug}'. Use lowercase letters, digits, and hyphens; 1-64 chars; no leading/trailing hyphen.`,
        };
      }
      try {
        const { store } = await getStore(ctx, toolCtx);
        const skillsDir = resolveSkillsDir(ctx.config?.skillsDir);
        const skillDir = join(skillsDir, input.slug);
        const skillPath = join(skillDir, "SKILL.md");
        const distilledAt = new Date().toISOString();

        const body = formatSkillFile({
          slug: input.slug,
          summary: input.summary,
          instructions: input.instructions,
          fromNames: input.fromNames,
          distilledAt,
        });

        await mkdir(skillDir, { recursive: true });
        await writeFile(skillPath, body, "utf8");

        // Provenance record back into the store. Content is the skill body so
        // `memory_query skill:foo` finds it; metadata tags it as a skill with
        // its lineage.
        const writeRes = await writeMemory(store, {
          name: `skill:${input.slug}`,
          content: body,
          type: "skill",
          metadata: {
            distilledFrom: input.fromNames,
            distilledAt,
            skillPath,
          },
        });
        if (!writeRes.ok) {
          // File was written; provenance failed. Be honest about the partial.
          return {
            ok: true as const,
            path: skillPath,
            provenance: "file-written; provenance record failed: " + writeRes.error,
          };
        }

        return {
          ok: true as const,
          path: skillPath,
          provenance: "recorded",
          note: "Available next scan/session — not active immediately.",
        };
      } catch (err) {
        return {
          ok: false as const,
          error: brandedFromError(err, "distill"),
        };
      }
    },
  });

  const memory_tap_status = createTool({
    id: "memory_tap_status",
    description:
      "Read-only probe of the experimental gut-feeling signal provider. Reports whether the provider is armed (actively polling for memory-relevant moments in the conversation) or disarmed, and why. No side effects — safe to call any time. The provider arms implicitly on the first memory_query or memory_write call; if it's disarmed, the deliberate-memory core still works (this is an experimental layer on top).",
    inputSchema: z.object({}).describe("No arguments."),
    execute: async () => {
      try {
        return { ok: true as const, status: tapStatus() };
      } catch (err) {
        return {
          ok: false as const,
          error: brandedFromError(err, "tap_status"),
        };
      }
    },
  });

  return {
    memory_query: { tool: memory_query },
    memory_write: { tool: memory_write },
    memory_forget: { tool: memory_forget },
    memory_distill_skill: { tool: memory_distill_skill },
    memory_tap_status: { tool: memory_tap_status },
  };
}

/**
 * Build a branded error string from a thrown error outside the fail-soft net.
 * Distinct prefixes per operation so the user can see at a glance which tool
 * failed and what kind of failure it was.
 */
function brandedFromError(err: unknown, op: string): string {
  const detail = err instanceof Error ? err.message : String(err);
  return `memorease: ${op} failed — ${detail}. The session continues without memory; the tool will keep returning this error until the underlying issue is fixed.`;
}
