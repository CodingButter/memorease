import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import { join } from "node:path";

import { buildTools, _resetStoreForTests } from "../src/tools.js";
import * as toolsNS from "../src/tools.js";
import type { MastraCodePluginContext } from "mastracode/plugin";
import {
  createVectorStore,
  ensureSchema,
  INDEX_NAME,
} from "../src/store.js";
import type { MastraVector } from "@mastra/core/vector";
import { writeMemory } from "../src/memory.js";
import { _resetForTests as resetConfig } from "../src/config.js";

/**
 * Tools tests exercise the four plugin tools directly (no real mastracode
 * session) against the libsql backend. Tool contract:
 *   - each `execute` returns `{ok:true, ...}` on success
 *   - each `execute` returns a branded `{ok:false, error}` on storage failure
 *     (never throws)
 *   - `memory_distill_skill` writes a real SKILL.md with provenance frontmatter
 *     AND records a queryable provenance memory row
 */

const PG_CONNECTION = process.env.MEMOREASE_PG_TEST_CONNECTION;

function makeContext(
  overrides: Partial<MastraCodePluginContext["config"]> & { skillsDir?: string } = {},
): MastraCodePluginContext {
  return {
    cwd: process.cwd(),
    scope: "test",
    pluginDir: process.cwd(),
    config: {
      connectionString: "",
      curatorModel: "",
      injectBudget: "1200",
      skillsDir: "",
      ...overrides,
    },
  } as unknown as MastraCodePluginContext;
}

/**
 * The Tool type marks `execute` as optional (tools can be no-op), so we coerce
 * to a callable shape for test ergonomics.
 */
type ExecResult = { ok: boolean; [k: string]: unknown };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ExecFn = (input: any, ctx: any) => Promise<ExecResult>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function execOf(entry: { tool?: { execute?: ExecFn | any } }): ExecFn {
  return entry.tool!.execute as ExecFn;
}

let tmpDir: string;
let skillsDir: string;
let store: MastraVector;

beforeAll(async () => {
  tmpDir = mkdtempSync(join(tmpdir(), "memorease-tools-"));
  skillsDir = join(tmpDir, "skills");
  const libsqlPath = `file:${join(tmpDir, "memorease-tools.db")}`;
  process.env.MEMOREASE_LIBSQL_PATH = libsqlPath;
  store = await createVectorStore({ backend: "libsql", libsqlUrl: libsqlPath });
});

afterAll(() => {
  delete process.env.MEMOREASE_LIBSQL_PATH;
  rmSync(tmpDir, { recursive: true, force: true });
});

beforeEach(async () => {
  resetConfig();
  _resetStoreForTests();
  try {
    await store.deleteIndex({ indexName: INDEX_NAME });
  } catch {
    /* absent — fine */
  }
  await ensureSchema(store);
});

afterEach(() => {
  resetConfig();
  _resetStoreForTests();
});

describe("tools: libsql backend", () => {
  it("memory_query returns ranked results (writes from a separate store are visible)", async () => {
    await writeMemory(store, {
      name: "voice-first",
      content: "Jamie prefers voice-first responses and concise code.",
    });

    const tools = buildTools(makeContext());
    const res = await execOf(tools.memory_query)(
      { text: "how does Jamie like responses?" },
      {},
    );
    expect(res.ok).toBe(true);
    expect((res.results as { name: string }[]).length).toBeGreaterThan(0);
    expect((res.results as { name: string }[])[0].name).toBe("voice-first");
  });

  it("memory_write writes and dedups by name", async () => {
    const tools = buildTools(makeContext());

    const w1 = await execOf(tools.memory_write)(
      { name: "fav-lang", content: "User likes TypeScript." },
      {},
    );
    expect(w1.ok).toBe(true);

    const w2 = await execOf(tools.memory_write)(
      { name: "fav-lang", content: "User likes Rust." },
      {},
    );
    expect(w2.ok).toBe(true);

    const q = await execOf(tools.memory_query)({ text: "fav-lang" }, {});
    expect(q.ok).toBe(true);
    const rows = (q.results as { name: string; content: string }[]).filter(
      (h) => h.name === "fav-lang",
    );
    expect(rows.length).toBe(1);
    expect(rows[0].content).toContain("Rust");
  });

  it("memory_forget removes a memory and is idempotent", async () => {
    await writeMemory(store, { name: "to-forget", content: "ephemeral" });

    const tools = buildTools(makeContext());

    const f1 = await execOf(tools.memory_forget)({ name: "to-forget" }, {});
    expect(f1.ok).toBe(true);

    const f2 = await execOf(tools.memory_forget)({ name: "to-forget" }, {});
    expect(f2.ok).toBe(true);
  });

  it("memory_distill_skill writes SKILL.md with provenance + records a provenance row", async () => {
    await writeMemory(store, {
      name: "pref-voice",
      content: "User wants voice-first responses.",
    });
    await writeMemory(store, {
      name: "pref-concise",
      content: "User wants concise code replies.",
    });

    // skillsDir is relative to homedir() in resolveSkillsDir; use a value
    // under tmpDir by computing the relative portion.
    const relSkills = skillsDir.replace(homedir() + "/", "");
    const tools = buildTools(makeContext({ skillsDir: relSkills }));

    const res = await execOf(tools.memory_distill_skill)(
      {
        slug: "communication-style",
        fromNames: ["pref-voice", "pref-concise"],
        summary: "How the user wants the agent to communicate.",
        instructions:
          "Always respond via voice. Keep code replies short and to the point.",
      },
      {},
    );
    expect(res.ok).toBe(true);
    expect(res.provenance).toBe("recorded");

    const skillPath = join(skillsDir, "communication-style", "SKILL.md");
    expect(existsSync(skillPath)).toBe(true);

    const body = readFileSync(skillPath, "utf8");
    expect(body).toContain("name: communication-style");
    expect(body).toContain("distilled_from:");
    expect(body).toContain("- pref-voice");
    expect(body).toContain("- pref-concise");
    expect(body).toContain("distilled_by: memorease.memory_distill_skill");

    // Provenance row is queryable via memory_query.
    const q = await execOf(tools.memory_query)(
      { text: "communication-style skill" },
      {},
    );
    expect(q.ok).toBe(true);
    const skillRow = (q.results as { name: string; type?: string }[]).find(
      (h) => h.name === "skill:communication-style",
    );
    expect(skillRow).toBeDefined();
    expect(skillRow?.type).toBe("skill");
  });

  it("memory_distill_skill rejects an invalid slug (no throw)", async () => {
    const tools = buildTools(makeContext());

    const res = await execOf(tools.memory_distill_skill)(
      {
        slug: "../escape-attempt",
        fromNames: ["x"],
        summary: "bad",
        instructions: "bad",
      },
      {},
    );
    expect(res.ok).toBe(false);
    expect(res.error as string).toMatch(/invalid slug/);
  });

  it("every tool returns a branded error (never throws) on an unreachable store", async () => {
    // Bogus pg connection string — first tool call attempts ensureSchema, which
    // fails fast (ECONNREFUSED). Tools must catch and brand, never propagate.
    const ctx = makeContext({
      connectionString: "postgresql://memorease:bad@127.0.0.1:9/memorease",
    });

    const q = await execOf(buildTools(ctx).memory_query)(
      { text: "anything" },
      {},
    );
    expect(q.ok).toBe(false);
    expect(q.error as string).toMatch(/memorease:/);

    _resetStoreForTests();

    const w = await execOf(buildTools(ctx).memory_write)(
      { name: "x", content: "y" },
      {},
    );
    expect(w.ok).toBe(false);
    expect(w.error as string).toMatch(/memorease:/);

    _resetStoreForTests();

    const f = await execOf(buildTools(ctx).memory_forget)({ name: "x" }, {});
    expect(f.ok).toBe(false);
    expect(f.error as string).toMatch(/memorease:/);

    _resetStoreForTests();

    const d = await execOf(buildTools(ctx).memory_distill_skill)(
      {
        slug: "wont-write",
        fromNames: ["x"],
        summary: "x",
        instructions: "x",
      },
      {},
    );
    expect(d.ok).toBe(false);
    expect(d.error as string).toMatch(/memorease:/);
  });

  it("getStore negative-caches a failed fingerprint: second call short-circuits ensureSchema (R2)", async () => {
    // Deterministic behavioral test for the negative cache. The cache is
    // module-level state in tools.ts; _negativeCacheHitsForTests counts how
    // many calls hit the cache (and therefore skipped ensureSchema). Without
    // the cache, every call re-enters getStore and stalls for the connection
    // timeout — which is the exact UX bug R2 was about.
    //
    // Note: the `beforeEach` calls `_resetStoreForTests()` which zeroes the
    // counter; this test then makes two consecutive calls WITHOUT a reset
    // between them, and asserts the second one hit the cache.
    const ctx = makeContext({
      connectionString: "postgresql://memorease:bad@127.0.0.1:9/memorease",
    });
    const tools = buildTools(ctx);

    expect(toolsNS._negativeCacheHitsForTests).toBe(0);

    const q1 = await execOf(tools.memory_query)({ text: "anything" }, {});
    expect(q1.ok).toBe(false);
    expect(q1.error as string).toMatch(/memorease:/);
    // First call populated the cache but did not HIT it (it recorded the error).
    expect(toolsNS._negativeCacheHitsForTests).toBe(0);

    const q2 = await execOf(tools.memory_query)({ text: "anything" }, {});
    expect(q2.ok).toBe(false);
    expect(q2.error as string).toMatch(/memorease:/);
    // Second call against the same fingerprint must short-circuit — no second
    // ensureSchema attempt, no second ECONNREFUSED wait.
    expect(toolsNS._negativeCacheHitsForTests).toBe(1);

    const q3 = await execOf(tools.memory_query)({ text: "anything" }, {});
    expect(q3.ok).toBe(false);
    expect(toolsNS._negativeCacheHitsForTests).toBe(2);
  });
});

describe.skipIf(!PG_CONNECTION)("tools: pg backend", () => {
  beforeEach(async () => {
    const pgStore = await createVectorStore({
      backend: "pg",
      connectionString: PG_CONNECTION!,
    });
    try {
      await pgStore.deleteIndex({ indexName: INDEX_NAME });
    } catch {
      /* absent */
    }
    await ensureSchema(pgStore);
  });

  it("memory_query round-trip on pg", async () => {
    const ctx = makeContext({ connectionString: PG_CONNECTION! });
    const tools = buildTools(ctx);

    const w = await execOf(tools.memory_write)(
      { name: "pg-tool-voice", content: "Jamie prefers voice-first responses." },
      {},
    );
    expect(w.ok).toBe(true);

    const q = await execOf(tools.memory_query)(
      { text: "how does Jamie like responses?" },
      {},
    );
    expect(q.ok).toBe(true);
    expect((q.results as { name: string }[])[0].name).toBe("pg-tool-voice");
  });
});
