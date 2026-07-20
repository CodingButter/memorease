/**
 * Memory operations: query, write, forget.
 *
 * All ops are async, take a `MastraVector` store, and **never throw to the
 * session**: every operation is wrapped in `failSoft` so embedder/ONNX failures
 * AND store/connection failures surface as branded `{ok:false, error}` results.
 * The caller (tool or instructions) decides how to render the error to the
 * agent — typically as a system-prompt note or a tool return value.
 *
 * Dedup-by-name uses **explicit delete-then-insert** via `deleteVectors({filter})`
 * on both backends. R1 (adversarial review): `upsert({deleteFilter})` is
 * silently ignored by LibSQLVector's `doUpsert` (verified against
 * `@mastra/libsql@1.16.0` — destructure list is `{indexName, vectors, metadata,
 * ids}` only). Using the explicit path gives identical behavior on both
 * backends and avoids a silent foot-gun on libsql.
 */

import type { MastraVector } from "@mastra/core/vector";
import type { VectorFilter } from "@mastra/core/vector";
import { embed } from "./embed.ts";
import { failSoft, type StorageResult } from "./errors.ts";
import { INDEX_NAME } from "./store.ts";

/**
 * Shape of a memory row as seen by tool callers and the boot injector.
 * `score` is the cosine similarity from the vector store (0..1 on pg; libsql
 * also returns 0..1 after normalization) — meaningful for ranking, not absolute.
 */
export type MemoryHit = {
  id: string;
  score: number;
  name: string;
  content: string;
  type?: string;
  metadata?: Record<string, unknown>;
  createdAt?: string;
  updatedAt?: string;
};

export type MemoryRecord = {
  name: string;
  content: string;
  type?: string;
  metadata?: Record<string, unknown>;
  /**
   * Thread that created (or last updated) this memory. Stamped by the tool
   * layer from the tool context. The gut-feeling tap uses it to skip
   * reminding a thread about a memory that thread itself wrote — that
   * knowledge is already in its context.
   */
  sourceThreadId?: string;
};

export type MemoryQueryOptions = {
  /** Semantic-search top-K. Defaults to 5. */
  topK?: number;
  /** Optional metadata filter applied at the store layer. */
  filter?: VectorFilter;
};

const FILTER_BY_NAME = (name: string): VectorFilter => ({
  name: { $eq: name },
});

/**
 * Delete all rows matching a memory name. Backend-agnostic. Used by both
 * `forgetMemory` and the `writeMemory` dedup step. Returns no count — the
 * vector store interface does not expose one, so callers treat the result as
 * best-effort and re-query if they need to verify deletion.
 */
export async function deleteByName(
  store: MastraVector,
  name: string,
): Promise<void> {
  await store.deleteVectors({
    indexName: INDEX_NAME,
    filter: FILTER_BY_NAME(name),
  });
}

function rowToHit(row: {
  id: string;
  score: number;
  metadata?: Record<string, unknown>;
}): MemoryHit {
  const m = row.metadata ?? {};
  return {
    id: row.id,
    score: row.score,
    name: String(m.name ?? ""),
    content: String(m.content ?? ""),
    type: typeof m.type === "string" ? m.type : undefined,
    metadata: m,
    createdAt: typeof m.createdAt === "string" ? m.createdAt : undefined,
    updatedAt: typeof m.updatedAt === "string" ? m.updatedAt : undefined,
  };
}

/**
 * Semantic memory search. Embeds `text`, queries the store, and returns ranked
 * hits. Never throws — returns a branded error result on embedder or store
 * failure.
 */
export async function queryMemories(
  store: MastraVector,
  text: string,
  opts: MemoryQueryOptions = {},
): Promise<StorageResult<MemoryHit[]>> {
  return failSoft(async () => {
    const vector = await embed(text);
    const results = await store.query({
      indexName: INDEX_NAME,
      queryVector: vector,
      topK: opts.topK ?? 5,
      filter: opts.filter,
    });
    return results.map(rowToHit);
  });
}

/**
 * Upsert a memory by name. Dedup is explicit: delete any existing rows with
 * the same name, then insert the new row. Atomic on the store side is not
 * guaranteed across the two calls — if the insert fails after the delete, the
 * memory is gone. That's acceptable for v1 (low write rate, single user) and
 * surfaced by the branded error result. Timestamps are ISO strings.
 */
export async function writeMemory(
  store: MastraVector,
  record: MemoryRecord,
): Promise<StorageResult<{ name: string; content: string }>> {
  return failSoft(async () => {
    const vector = await embed(record.content);
    await deleteByName(store, record.name);
    const now = new Date().toISOString();
    // System fields assigned LAST so caller-supplied metadata cannot collide
    // with them — an adversarial `metadata:{name:"evil"}` would otherwise
    // rebind the row's lookup key and break subsequent dedup/forget calls.
    const metadata = {
      ...(record.metadata ?? {}),
      name: record.name,
      content: record.content,
      type: record.type ?? "fact",
      createdAt: now,
      updatedAt: now,
      ...(record.sourceThreadId
        ? { sourceThreadId: record.sourceThreadId }
        : {}),
    };
    await store.upsert({
      indexName: INDEX_NAME,
      vectors: [vector],
      metadata: [metadata],
    });
    return { name: record.name, content: record.content };
  });
}

/**
 * Forget (delete) a memory by name. Returns `{ok:true, forgotten:name}` on
 * success. Never throws — returns a branded error result on store failure.
 *
 * Note: the vector store does not tell us whether anything was actually
 * deleted, so we cannot distinguish "deleted one" from "deleted zero". The
 * tool layer reports success either way.
 */
export async function forgetMemory(
  store: MastraVector,
  name: string,
): Promise<StorageResult<{ forgotten: string }>> {
  return failSoft(async () => {
    await deleteByName(store, name);
    return { forgotten: name };
  });
}
