import { fastembed } from "@mastra/fastembed";

/**
 * Embedding model: `fastembed.small` — 384-dim, local ONNX inference, no API
 * keys. Same model mastracode uses internally for its observational memory
 * system (verified at `sdk/src/agents/memory.ts`).
 *
 * First call loads the ONNX model (~50MB download from cache, then in-process).
 * Subsequent calls reuse the cached model instance.
 */

let modelPromise: Promise<typeof fastembed.small> | undefined;

async function getModel() {
  if (!modelPromise) {
    // fastembed.small is a lazy EmbeddingModel getter on the fastembed object.
    // Accessing it may trigger ONNX init; memoize the resolved instance.
    modelPromise = Promise.resolve(fastembed.small);
  }
  return modelPromise;
}

/**
 * Internal embedder function. Indirected through `_embedder` so tests can
 * inject failures (e.g., simulated ONNX load errors) via `setEmbedderForTests`
 * — ESM module exports are read-only and cannot be monkey-patched directly.
 */
let _embedder: (text: string) => Promise<number[]> = defaultEmbedder;

async function defaultEmbedder(text: string): Promise<number[]> {
  const model = await getModel();
  const { embeddings } = await model.doEmbed({ values: [text] });
  return embeddings[0];
}

/**
 * Embed a single text string and return its 384-dim vector.
 */
export async function embed(text: string): Promise<number[]> {
  return _embedder(text);
}

/**
 * Embed multiple texts in one call (batched). Returns one 384-dim vector per
 * input, preserving order.
 */
export async function embedMany(texts: string[]): Promise<number[][]> {
  const model = await getModel();
  const { embeddings } = await model.doEmbed({ values: texts });
  return embeddings;
}

/** Reset internal state — used by tests to isolate model loading. */
export function _resetForTests(): void {
  modelPromise = undefined;
  _embedder = defaultEmbedder;
}

/**
 * Test-only seam: substitute the embedder with a stub. Pass `null` (or omit
 * the previous call's restoration) and call `_resetForTests()` to restore.
 */
export function setEmbedderForTests(
  fn: ((text: string) => Promise<number[]>) | null,
): void {
  _embedder = fn ?? defaultEmbedder;
}
