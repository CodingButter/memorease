import { PgVector } from "@mastra/pg";
import { LibSQLVector } from "@mastra/libsql";
import type { MastraVector } from "@mastra/core/vector";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";

import type { ResolvedConfig } from "./config.js";

/**
 * Storage layer. Two backends behind one `MastraVector` interface:
 *  - libsql (default): local file, zero-config, in-process cosine.
 *  - pg: Postgres+pgvector, opt-in via connection string or auto-detected from
 *        mastracode's settings.json when the host is configured for pg.
 *
 * Embeddings are stored in-row with metadata (single query = semantic rank +
 * relational filter). Dedup-by-name is achieved via `deleteFilter` on upsert.
 */

export const INDEX_NAME = "memorease_memories";
export const DIMENSION = 384;
export const METRIC = "cosine" as const;

/**
 * Construct the backend vector store from a resolved config. Does NOT create
 * the index — that's the caller's responsibility (see `ensureSchema`).
 */
export function createVectorStore(
  config: ResolvedConfig,
): MastraVector {
  if (config.backend === "pg") {
    if (!config.connectionString) {
      throw new Error(
        "pg backend selected but no connection string was resolved",
      );
    }
    return new PgVector({
      id: "memorease",
      connectionString: config.connectionString,
    });
  }
  // libsql path: ensure the parent directory exists (data dir may not yet).
  const url = config.libsqlUrl;
  if (!url) throw new Error("libsql backend selected but no url was resolved");
  // Best-effort: create the parent dir for file: URLs (no-op for :memory:).
  if (url.startsWith("file:") && !url.includes(":memory:")) {
    const fsPath = url.slice("file:".length);
    void mkdir(dirname(fsPath), { recursive: true }).catch(() => {
      /* swallowed — libsql will fail loudly on init if the dir is unreachable */
    });
  }
  return new LibSQLVector({ id: "memorease", url });
}

/**
 * Idempotent index bootstrap. Checks `describeIndex`; on "absent" error, calls
 * `createIndex`. Any other error propagates. Memoized per-store so subsequent
 * ops skip it.
 *
 * R5: NOT called eagerly inside `createVectorStore`. Callers wrap it in the
 * same fail-soft net that catches store errors.
 */
const ensuredStores = new WeakMap<MastraVector, Promise<void>>();

export function ensureSchema(store: MastraVector): Promise<void> {
  const existing = ensuredStores.get(store);
  if (existing) return existing;

  const p = (async () => {
    try {
      await store.describeIndex({ indexName: INDEX_NAME });
      // Index exists — assume correct dimension. (Re-create is out of scope;
      // a dimension mismatch would surface as a query-time error.)
      return;
    } catch (err) {
      // Heuristic: "does not exist" / "not found" / "no such" / "unknown" →
      // create. Anything else → throw. Covers pg ("relation does not exist")
      // and libsql ("Table X not found") absent-index messages.
      const msg = err instanceof Error ? err.message : String(err);
      if (!/not exist|does not exist|not found|no such|unknown|absent|404/i.test(msg)) {
        throw err;
      }
      await store.createIndex({
        indexName: INDEX_NAME,
        dimension: DIMENSION,
        metric: METRIC,
      });
    }
  })();

  ensuredStores.set(store, p);
  return p;
}

/**
 * Reset the ensureSchema memo — for tests that need to re-bootstrap on a
 * fresh store.
 */
export function _resetSchemaMemoForTests(): void {
  // WeakMap has no clear(); tests should construct a fresh store instead.
  // This function exists for API symmetry / future bookkeeping.
}
