import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildInstructions } from "../src/instructions.js";
import {
  createVectorStore,
  ensureSchema,
  INDEX_NAME,
} from "../src/store.js";
import type { MastraVector } from "@mastra/core/vector";
import { writeMemory } from "../src/memory.js";
import { _resetForTests as resetConfig } from "../src/config.js";

/**
 * Instructions tests cover the contract: `instructions(context)` returns a
 * string and never throws. The cases that matter are:
 *
 *  - storage reachable + disarmed curator → returns a `## Memories` section
 *    containing seeded memories + the disarmed-curator note
 *  - storage reachable + armed curator → same shape minus the note
 *    (covered by env-gated path in curator.test.ts; here we always disarm)
 *  - storage unreachable (R3 fast-reject) → returns a branded section in
 *    milliseconds (well under the 3s timeout)
 *  - boot-query hang → returns the branded section within ~timeout ms
 *  - empty store → returns empty string (or just the curator note if disarmed)
 */

const PG_CONNECTION = process.env.MEMOREASE_PG_TEST_CONNECTION;

function makeLibsqlConfig(tmpDir: string) {
  return {
    backend: "libsql" as const,
    libsqlUrl: `file:${join(tmpDir, "memorease-instr.db")}`,
  };
}

/** Path to the libsql file both the test fixtures and `buildInstructions`
 * resolve to. `buildInstructions` calls `resolveConfig` which honors
 * `MEMOREASE_LIBSQL_PATH` — setting it here makes the test-owned store and
 * the instructions-owned store point at the same file.
 */
function libsqlPathEnv(tmpDir: string): string {
  return `file:${join(tmpDir, "memorease-instr.db")}`;
}

function makeEmptySettings(): string {
  const dir = mkdirSync(
    join(tmpdir(), `memorease-instr-${Math.random().toString(36).slice(2)}/`),
    { recursive: true },
  ) as string;
  const path = join(dir, "settings.json");
  writeFileSync(path, JSON.stringify({ models: {} }));
  return path;
}

let tmpDir: string;
let store: MastraVector;

beforeAll(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "memorease-instr-"));
});

afterAll(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

beforeEach(async () => {
  resetConfig();
  delete process.env.MEMOREASE_SETTINGS_PATH;
  // Point both the test fixtures and `buildInstructions` at the same libsql
  // file in the per-test tmpDir. Without this, `buildInstructions` would
  // resolve to the real user data dir and never see the seeded memory.
  process.env.MEMOREASE_LIBSQL_PATH = libsqlPathEnv(tmpDir);
  // Use a fresh libsql file per test so we don't see cross-test state.
  store = await createVectorStore(makeLibsqlConfig(tmpDir));
  try {
    await store.deleteIndex({ indexName: INDEX_NAME });
  } catch {
    /* absent — fine */
  }
  await ensureSchema(store);
});

afterEach(() => {
  delete process.env.MEMOREASE_BOOT_TIMEOUT_MS;
  delete process.env.MEMOREASE_LIBSQL_PATH;
});

describe("instructions: storage reachable", () => {
  beforeEach(() => {
    // Force-disarm the curator so we test the deterministic path here.
    process.env.MEMOREASE_SETTINGS_PATH = makeEmptySettings();
  });

  it("returns a '## Memories' section containing seeded memories (libsql)", async () => {
    await writeMemory(store, {
      name: "voice-first",
      content: "Jamie prefers voice-first responses and concise code.",
    });

    const section = await buildInstructions({
      config: { connectionString: undefined, curatorModel: undefined, injectBudget: "2000" },
    });

    expect(section).toContain("## Memories");
    expect(section).toContain("voice-first");
    // Directive is always prepended — even when memories exist.
    expect(section).toContain("## Memorease");
    expect(section).toContain("QUERY ON INSTINCT");
    // Disarmed-curator note is appended when the curator isn't configured.
    expect(section).toMatch(/curator disarmed/);
  });

  it("returns the directive (but no Memories section) when the store is empty", async () => {
    const section = await buildInstructions({
      config: { connectionString: undefined, curatorModel: undefined },
    });
    // Fresh sessions must still learn memorease exists — the directive is
    // load-bearing. But no `## Memories` section because nothing is stored.
    expect(section).toContain("## Memorease");
    expect(section).toContain("QUERY ON INSTINCT");
    // Negative-space rules are part of the contract: what NOT to store.
    expect(section).toContain("What does NOT belong");
    expect(section).toContain("SECRETS");
    expect(section).not.toContain("## Memories");
  });

  it("appends no disarmed note when curatorModel is set", async () => {
    await writeMemory(store, {
      name: "any",
      content: "any content",
    });
    const section = await buildInstructions({
      config: { connectionString: undefined, curatorModel: "some-model-id" },
    });
    expect(section).toContain("## Memories");
    expect(section).not.toMatch(/curator disarmed/);
  });
});

describe("instructions: storage unreachable (R3 fast-reject)", () => {
  // Force-disarm so the only variable here is the storage layer.
  beforeEach(() => {
    process.env.MEMOREASE_SETTINGS_PATH = makeEmptySettings();
  });

  it("returns the branded section in milliseconds when pg port is closed", async () => {
    // Point at a closed local port. The libsql backend would construct at
    // boot (file-based), so to exercise a runtime network failure we use the
    // pg backend pointed at a port that nothing is listening on.
    const start = Date.now();
    const section = await buildInstructions({
      config: { connectionString: "postgresql://x:x@127.0.0.1:9/memorease" },
    });
    const elapsed = Date.now() - start;

    expect(section).toMatch(/Storage unreachable/);
    // Fast-reject: should complete well under the 3s timeout arm.
    expect(elapsed).toBeLessThan(2500);
  });

  it("does not throw — returns a string even on storage failure", async () => {
    const section = await buildInstructions({
      config: { connectionString: "postgresql://x:x@127.0.0.1:9/memorease" },
    });
    expect(typeof section).toBe("string");
  });
});

describe("instructions: boot-query timeout", () => {
  beforeEach(() => {
    process.env.MEMOREASE_SETTINGS_PATH = makeEmptySettings();
    // Tight timeout so the test stays fast.
    process.env.MEMOREASE_BOOT_TIMEOUT_MS = "150";
  });

  it("returns the branded section when the boot query hangs past the timeout", async () => {
    // A libsql URL pointing at an unwritable path that still appears valid
    // can hang; the more reliable hang simulator is a pg connection to a
    // black-hole IP. Use TEST-NET-1 (10.255.255.1) — typically unroutable
    // in CI so the connect hangs until our timeout fires.
    const start = Date.now();
    const section = await buildInstructions({
      config: { connectionString: "postgresql://x:x@10.255.255.1:5432/memorease" },
    });
    const elapsed = Date.now() - start;

    expect(section).toMatch(/Storage unreachable/);
    // Should resolve close to the timeout, not before.
    expect(elapsed).toBeGreaterThanOrEqual(140);
    // Generous upper bound — the timeout should fire well under 5s even
    // with scheduling jitter.
    expect(elapsed).toBeLessThan(5000);
  });
});

describe.skipIf(!PG_CONNECTION)("instructions: pg backend", () => {
  beforeEach(async () => {
    store = await createVectorStore({
      backend: "pg",
      connectionString: PG_CONNECTION!,
    });
    try {
      await store.deleteIndex({ indexName: INDEX_NAME });
    } catch {
      /* absent — fine */
    }
    await ensureSchema(store);
    process.env.MEMOREASE_SETTINGS_PATH = makeEmptySettings();
  });

  it("returns a '## Memories' section on pg", async () => {
    await writeMemory(store, {
      name: "pg-voice",
      content: "Jamie prefers voice-first responses.",
    });
    const section = await buildInstructions({
      config: { connectionString: PG_CONNECTION, curatorModel: undefined },
    });
    expect(section).toContain("## Memories");
    expect(section).toContain("pg-voice");
  });
});

describe("instructions: buildInstructions never throws", () => {
  // Regression: createVectorStore used to throw synchronously outside any
  // try/catch (libsql parent-dir race, pg config errors), which rejected the
  // instructions() promise. Now it's wrapped → branded section, never throws.
  beforeEach(() => {
    process.env.MEMOREASE_SETTINGS_PATH = makeEmptySettings();
  });

  it("returns a branded section when createVectorStore throws synchronously (M2)", async () => {
    // Deterministic M2 test: inject a factory that throws synchronously to
    // prove the buildInstructions wrapping catches it. The prior version of
    // this test pointed at a malformed pg URL, but resolveConfig returned it
    // as a usable connectionString — so the test was actually exercising the
    // M3 ensureSchema fail-soft path, not M2. This version deletes the M2
    // try/catch's effectiveness if removed: the injected factory throws
    // unconditionally, so only the M2 catch can produce a branded section.
    resetConfig();
    const section = await buildInstructions(
      { config: { connectionString: undefined, curatorModel: undefined } },
      {
        _createVectorStoreForTests: () => {
          throw new Error("forced sync throw from createVectorStore");
        },
      },
    );
    expect(section).toMatch(/Storage unreachable/);
    expect(section).toContain("forced sync throw from createVectorStore");
  });

  it("returns a branded section when pg connectionString points at an unreachable port (integration smoke)", async () => {
    // Integration check: end-to-end, a pg config that fails at connect time
    // surfaces as a branded section rather than a rejected promise. This
    // overlaps with M3 (ensureSchema fail-soft) but confirms the wrapping
    // composes correctly with the real PgVector error path.
    resetConfig();
    const section = await buildInstructions({
      config: {
        connectionString: "postgresql://x:x@127.0.0.1:9/memorease",
        curatorModel: undefined,
      },
    });
    expect(typeof section).toBe("string");
    expect(section).toMatch(/Storage unreachable/);
  });
});
