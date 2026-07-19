import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  createVectorStore,
  ensureSchema,
  INDEX_NAME,
  DIMENSION,
} from "../src/store.js";
import type { MastraVector } from "@mastra/core/vector";
import type { ResolvedConfig } from "../src/config.js";

const PG_CONNECTION = process.env.MEMOREASE_PG_TEST_CONNECTION;

function makeLibsqlConfig(tmpDir: string): ResolvedConfig {
  return {
    backend: "libsql",
    libsqlUrl: `file:${join(tmpDir, "vectors.db")}`,
  };
}

describe("store: libsql backend (real LibSQLVector)", () => {
  let tmpDir: string;
  let store: MastraVector;

  beforeAll(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "memorease-libsql-"));
    store = createVectorStore(makeLibsqlConfig(tmpDir));
  });

  afterAll(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("ensureSchema creates the index with dimension 384 and cosine metric", async () => {
    await ensureSchema(store);
    const stats = await store.describeIndex({ indexName: INDEX_NAME });
    expect(stats.dimension).toBe(DIMENSION);
    expect(stats.metric).toBe("cosine");
  });

  it("ensureSchema is idempotent (second call is a no-op)", async () => {
    // Second call must not throw and must not re-create.
    const before = await store.describeIndex({ indexName: INDEX_NAME });
    await ensureSchema(store);
    const after = await store.describeIndex({ indexName: INDEX_NAME });
    expect(after.dimension).toBe(before.dimension);
    expect(after.metric).toBe(before.metric);
  });

  it("upsert + query round-trip returns the inserted vector", async () => {
    // Use orthogonal vectors so cosine similarity actually discriminates.
    // (Constant-fill vectors are collinear → cosine = 1.0 for both.)
    const low = new Array(DIMENSION).fill(0);
    const high = new Array(DIMENSION).fill(0);
    low[0] = 1;
    high[1] = 1;
    const query = new Array(DIMENSION).fill(0);
    query[1] = 1; // matches "high" (cosine=1), mismatches "low" (cosine=0)

    const ids = await store.upsert({
      indexName: INDEX_NAME,
      vectors: [low, high],
      metadata: [{ name: "low", content: "alpha" }, { name: "high", content: "beta" }],
    });
    expect(ids.length).toBe(2);

    const results = await store.query({
      indexName: INDEX_NAME,
      queryVector: query,
      topK: 1,
    });
    expect(results.length).toBe(1);
    expect(results[0].metadata?.name).toBe("high");
  });
});

describe.skipIf(!PG_CONNECTION)(
  "store: pg backend (real PgVector via Docker compose)",
  () => {
    let store: MastraVector;

    beforeAll(() => {
      const config: ResolvedConfig = {
        backend: "pg",
        connectionString: PG_CONNECTION!,
      };
      store = createVectorStore(config);
    });

    it("ensureSchema creates the index with dimension 384 and cosine metric", async () => {
      // Fresh DB on each compose-up; if the index already exists from a prior
      // run, ensureSchema is a no-op.
      await ensureSchema(store);
      const stats = await store.describeIndex({ indexName: INDEX_NAME });
      expect(stats.dimension).toBe(DIMENSION);
      expect(stats.metric).toBe("cosine");
    });

    it("ensureSchema is idempotent on pg", async () => {
      const before = await store.describeIndex({ indexName: INDEX_NAME });
      await ensureSchema(store);
      const after = await store.describeIndex({ indexName: INDEX_NAME });
      expect(after.dimension).toBe(before.dimension);
      expect(after.metric).toBe(before.metric);
    });

    it("upsert + query round-trip on pg", async () => {
      // Schema already created by the first two tests in this block.
      // Ensure memo has fired (no-op if already ensured).
      await ensureSchema(store);

      // Orthogonal vectors so cosine discriminates (see libsql test note).
      const a = new Array(DIMENSION).fill(0);
      const b = new Array(DIMENSION).fill(0);
      a[0] = 1;
      b[1] = 1;
      const query = new Array(DIMENSION).fill(0);
      query[1] = 1;

      const ids = await store.upsert({
        indexName: INDEX_NAME,
        vectors: [a, b],
        metadata: [{ name: "alpha", content: "first" }, { name: "beta", content: "second" }],
      });
      expect(ids.length).toBe(2);

      const results = await store.query({
        indexName: INDEX_NAME,
        queryVector: query,
        topK: 1,
      });
      expect(results.length).toBe(1);
      expect(results[0].metadata?.name).toBe("beta");
    });
  },
);

describe.skipIf(!!PG_CONNECTION)("store: pg backend (Docker not available)", () => {
  it.skip("pg backend tests require MEMOREASE_PG_TEST_CONNECTION env to be set", () => {
    // Skipped — surface the reason in test output.
  });
});
