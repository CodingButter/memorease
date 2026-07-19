import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { MastraVector } from "@mastra/core/vector";
import {
  createVectorStore,
  ensureSchema,
} from "../src/store.js";
import { queryMemories, writeMemory, forgetMemory } from "../src/memory.js";
import { setEmbedderForTests } from "../src/embed.js";

/**
 * Fail-soft behavior: every memory op must return a branded error result
 * (never throw) when the underlying storage or embedder fails.
 *
 * R4 (adversarial review): embedder/ONNX failures are outside the original
 * fail-soft net. This suite proves the net now catches them too.
 *
 * Note: `createVectorStore` itself is a setup-time call that can throw
 * (e.g. libsql opens the file synchronously). That construction-error path is
 * a boot-time concern owned by the instructions/plugin layer (Phase 3), not
 * the memory ops themselves. These tests cover the runtime contract: once a
 * store is constructed, every memory op on it is fail-soft.
 */

describe("fail-soft: unreachable pg (bogus connection string)", () => {
  // Ephemeral port nothing is listening on → fast ECONNREFUSED at first query.
  // PgVector's constructor does not eagerly connect, so construction succeeds
  // and the failure surfaces at op time — exactly what fail-soft catches.
  let store: MastraVector;

  beforeAll(() => {
    store = createVectorStore({
      backend: "pg",
      connectionString: "postgresql://u:p@127.0.0.1:1/none",
    });
  });

  it("queryMemories returns branded error and never throws", async () => {
    const r = await queryMemories(store, "anything");
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("memorease: storage unreachable");
    expect(r.error).toContain("Fix:");
  });

  it("writeMemory returns branded error and never throws", async () => {
    const r = await writeMemory(store, { name: "x", content: "y" });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("memorease: storage unreachable");
  });

  it("forgetMemory returns branded error and never throws", async () => {
    const r = await forgetMemory(store, "x");
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("memorease: storage unreachable");
  });
});

/**
 * Embedder-failure tests (R4). Uses the `setEmbedderForTests` seam — ESM
 * module exports are read-only bindings and cannot be monkey-patched directly.
 */
describe("fail-soft: embedder/ONNX failure (R4)", () => {
  let tmpDir: string;
  let store: MastraVector;

  beforeAll(async () => {
    tmpDir = mkdtempSync(join(tmpdir(), "memorease-embed-fail-"));
    store = createVectorStore({
      backend: "libsql",
      libsqlUrl: `file:${join(tmpDir, "test.db")}`,
    });
    await ensureSchema(store);
  });

  afterAll(() => {
    setEmbedderForTests(null); // restore default embedder
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("queryMemories catches embedder failure → branded 'embedding unavailable'", async () => {
    setEmbedderForTests(async () => {
      throw new Error("ONNX model load failed: corrupted model file");
    });
    const r = await queryMemories(store, "anything");
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("memorease: embedding unavailable");
    expect(r.error).toContain("ONNX");
    expect(r.error).toContain("Fix:");
  });

  it("writeMemory catches embedder failure → branded 'embedding unavailable'", async () => {
    setEmbedderForTests(async () => {
      throw new Error("failed to initialize onnxruntime WASM backend");
    });
    const r = await writeMemory(store, { name: "x", content: "y" });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("memorease: embedding unavailable");
  });
});
