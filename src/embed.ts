// Both onnxruntime-node and @mastra/fastembed are heavy native-binding loads
// (~85ms combined at import time). They are dynamic-imported on the first
// embed call rather than at module eval, so plugin import stays cheap and
// sessions that never call a memory tool pay nothing.

import type { fastembed as fastembedType } from "@mastra/fastembed";

/**
 * Embedding model: `fastembed.small` — 384-dim, local ONNX inference, no API
 * keys.
 *
 * First call loads the ONNX model (~50MB download from cache, then in-process).
 * Subsequent calls reuse the cached model instance.
 */

type EmbedModel = typeof fastembedType.small;

let modelPromise: Promise<EmbedModel> | undefined;

async function getModel(): Promise<EmbedModel> {
  if (!modelPromise) {
    modelPromise = (async () => {
      // ONNX Runtime defaults its global log level to "warning", which emits a
      // noisy device-discovery probe warning on systems where
      // /sys/class/drm/card0/device/vendor is unreadable (common in
      // containers/sandboxes). The warning is cosmetic — inference falls back
      // to CPU — but it looks unprofessional on startup.
      //
      // Pin the level to "error" before any InferenceSession is created. The
      // `env` object is a process-wide singleton on onnxruntime-common, shared
      // with fastembed's internal ort import, so setting it here covers
      // fastembed too. Verified empirically: with this line, the card0 warning
      // disappears and the embedding output (384-dim) is unchanged. Genuine
      // errors still surface.
      const ort = await import("onnxruntime-node");
      ort.env.logLevel = "error";

      const { fastembed } = await import("@mastra/fastembed");
      return fastembed.small;
    })();
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
