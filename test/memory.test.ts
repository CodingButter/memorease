import { describe, it, expect, beforeAll, afterAll, beforeEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { MastraVector } from "@mastra/core/vector";
import {
  createVectorStore,
  ensureSchema,
  INDEX_NAME,
} from "../src/store.js";
import {
  queryMemories,
  writeMemory,
  forgetMemory,
} from "../src/memory.js";

const PG_CONNECTION = process.env.MEMOREASE_PG_TEST_CONNECTION;

/**
 * Each backend gets its own describe block so the failures are labeled clearly.
 * The libsql block always runs; the pg block is skipped unless the env is set.
 */

function makeLibsqlConfig(tmpDir: string) {
  return {
    backend: "libsql" as const,
    libsqlUrl: `file:${join(tmpDir, "memorease-test.db")}`,
  };
}

async function resetIndex(store: MastraVector) {
  // Drop+recreate for a clean slate per test. Both backends support deleteIndex.
  try {
    await store.deleteIndex({ indexName: INDEX_NAME });
  } catch {
    /* absent — fine */
  }
  await ensureSchema(store);
}

describe("memory ops: libsql backend", () => {
  let tmpDir: string;
  let store: MastraVector;

  beforeAll(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "memorease-mem-libsql-"));
    store = createVectorStore(makeLibsqlConfig(tmpDir));
  });

  afterAll(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  beforeEach(async () => {
    await resetIndex(store);
  });

  it("write→query round-trip returns the written memory ranked #1", async () => {
    const w = await writeMemory(store, {
      name: "voice-first",
      content: "Jamie prefers voice-first responses and concise code.",
    });
    expect(w.ok).toBe(true);

    const q = await queryMemories(store, "how does Jamie like responses?");
    expect(q.ok).toBe(true);
    if (!q.ok) return;
    expect(q.value.length).toBeGreaterThan(0);
    expect(q.value[0].name).toBe("voice-first");
    expect(q.value[0].score).toBeGreaterThan(0.3);
  });

  it("semantic ranking: relevant memory ranks above irrelevant ones", async () => {
    await writeMemory(store, {
      name: "rust-fan",
      content: "User loves Rust and systems programming with strict type systems.",
    });
    await writeMemory(store, {
      name: "python-fan",
      content: "User writes Python for data science and machine learning scripts.",
    });
    await writeMemory(store, {
      name: "guitarist",
      content: "User plays guitar in a jazz band on weekends.",
    });

    const q = await queryMemories(store, "what programming language does the user like?");
    expect(q.ok).toBe(true);
    if (!q.ok) return;
    expect(q.value.length).toBe(3);
    // Both programming memories should outrank the guitar one.
    const names = q.value.map((h) => h.name);
    const guitarIdx = names.indexOf("guitarist");
    const rustIdx = names.indexOf("rust-fan");
    const pythonIdx = names.indexOf("python-fan");
    expect(guitarIdx).toBeGreaterThan(rustIdx);
    expect(guitarIdx).toBeGreaterThan(pythonIdx);
  });

  it("dedup-by-name: second write of the same name replaces the first", async () => {
    await writeMemory(store, { name: "x", content: "alpha content" });
    await writeMemory(store, { name: "x", content: "beta content" });

    const q = await queryMemories(store, "alpha beta");
    expect(q.ok).toBe(true);
    if (!q.ok) return;
    const xRows = q.value.filter((h) => h.name === "x");
    expect(xRows.length).toBe(1);
    expect(xRows[0].content).toBe("beta content");
  });

  it("forget: deleted memory is absent from subsequent query", async () => {
    await writeMemory(store, { name: "ephemeral", content: "temporary fact" });
    const f = await forgetMemory(store, "ephemeral");
    expect(f.ok).toBe(true);

    const q = await queryMemories(store, "temporary");
    expect(q.ok).toBe(true);
    if (!q.ok) return;
    const found = q.value.find((h) => h.name === "ephemeral");
    expect(found).toBeUndefined();
  });

  it("forget is idempotent: deleting a missing name does not throw", async () => {
    const f = await forgetMemory(store, "never-existed");
    expect(f.ok).toBe(true);
  });

  it("caller metadata cannot overwrite system fields (name/content/type)", async () => {
    // Regression: an adversarial `metadata:{name:"evil"}` used to rebind the
    // row's lookup key because caller metadata was spread last. Verify the
    // system fields win and subsequent dedup/forget still find the row.
    const w = await writeMemory(store, {
      name: "legit",
      content: "real content",
      metadata: {
        name: "evil",
        content: "hijacked",
        type: "hijacked-type",
        custom: "kept",
      },
    });
    expect(w.ok).toBe(true);

    const q = await queryMemories(store, "real content");
    expect(q.ok).toBe(true);
    if (!q.ok) return;
    const hit = q.value.find((h) => h.name === "legit");
    expect(hit).toBeDefined();
    expect(hit?.content).toBe("real content");
    expect(hit?.type).toBe("fact");
    // Caller metadata that didn't collide is preserved.
    expect((hit?.metadata as { custom?: string })?.custom).toBe("kept");
    // And the hijacked row name does NOT exist as a separate memory.
    expect(q.value.find((h) => h.name === "evil")).toBeUndefined();

    // Dedup-by-name must still work — second write replaces the first.
    const w2 = await writeMemory(store, {
      name: "legit",
      content: "updated content",
      metadata: { name: "evil-again" },
    });
    expect(w2.ok).toBe(true);
    const q2 = await queryMemories(store, "updated content");
    expect(q2.ok).toBe(true);
    if (!q2.ok) return;
    const legitRows = q2.value.filter((h) => h.name === "legit");
    expect(legitRows.length).toBe(1);
    expect(legitRows[0].content).toBe("updated content");

    // Forget by the original name must succeed (would silently fail if the
    // row had been stored under the hijacked name).
    const f = await forgetMemory(store, "legit");
    expect(f.ok).toBe(true);
    const q3 = await queryMemories(store, "updated content");
    expect(q3.ok).toBe(true);
    if (!q3.ok) return;
    expect(q3.value.find((h) => h.name === "legit")).toBeUndefined();
  });
});

describe.skipIf(!PG_CONNECTION)(
  "memory ops: pg backend",
  () => {
    let store: MastraVector;

    beforeAll(() => {
      store = createVectorStore({
        backend: "pg",
        connectionString: PG_CONNECTION!,
      });
    });

    beforeEach(async () => {
      await resetIndex(store);
    });

    it("write→query round-trip on pg", async () => {
      const w = await writeMemory(store, {
        name: "pg-voice",
        content: "Jamie prefers voice-first responses.",
      });
      expect(w.ok).toBe(true);

      const q = await queryMemories(store, "how does Jamie like responses?");
      expect(q.ok).toBe(true);
      if (!q.ok) return;
      expect(q.value[0].name).toBe("pg-voice");
    });

    it("dedup-by-name on pg (delete-then-insert path)", async () => {
      await writeMemory(store, { name: "pg-x", content: "first" });
      await writeMemory(store, { name: "pg-x", content: "second" });

      const q = await queryMemories(store, "first second");
      expect(q.ok).toBe(true);
      if (!q.ok) return;
      const rows = q.value.filter((h) => h.name === "pg-x");
      expect(rows.length).toBe(1);
      expect(rows[0].content).toBe("second");
    });

    it("forget on pg", async () => {
      await writeMemory(store, { name: "pg-ephemeral", content: "temp" });
      const f = await forgetMemory(store, "pg-ephemeral");
      expect(f.ok).toBe(true);

      const q = await queryMemories(store, "temp");
      expect(q.ok).toBe(true);
      if (!q.ok) return;
      expect(q.value.find((h) => h.name === "pg-ephemeral")).toBeUndefined();
    });
  },
);
