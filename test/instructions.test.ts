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
let store: ReturnType<typeof createVectorStore>;

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
  store = createVectorStore(makeLibsqlConfig(tmpDir));
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
    // Disarmed-curator note is appended when the curator isn't configured.
    expect(section).toMatch(/curator disarmed/);
  });

  it("returns empty string (or just the note) when the store is empty", async () => {
    const section = await buildInstructions({
      config: { connectionString: undefined, curatorModel: undefined },
    });
    // Either empty, or just the curator-disarmed note — never a section.
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
    store = createVectorStore({
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

  it("returns a branded section when createVectorStore throws (bad pg config)", async () => {
    // A pg backend with no resolvable connection string makes createVectorStore
    // throw inside resolveConfig→createVectorStore. buildInstructions must
    // catch and return a branded section, not reject.
    // We force this by asking for pg without a usable connection — the store
    // factory throws "pg backend selected but no connection string".
    // Achieve this by pointing explicit connectionString at empty after
    // resolveConfig memoization is reset.
    resetConfig();
    const section = await buildInstructions({
      // An explicitly-empty connectionString falls through to settings; with
      // MEMOREASE_SETTINGS_PATH pointing at an empty settings.json, resolveConfig
      // lands on libsql — which succeeds. To exercise the throw, we instead
      // pass a syntactically-broken connection that PgVector rejects at build.
      config: {
        connectionString: "not-a-valid-postgres-url",
        curatorModel: undefined,
      },
    });
    expect(typeof section).toBe("string");
    // Either it branded a storage failure or (if PgVector happened to defer
    // the error to connect time) it surfaced the unreachable section. Both
    // are acceptable — the contract is "no throw".
    expect(section.length).toBeGreaterThan(0);
  });
});
