/**
 * Branded error builders for memorease. Same shape used by tools and
 * instructions so the agent sees one consistent failure message regardless of
 * whether storage failed at boot (instructions) or mid-session (tool call).
 */

export type StorageFailure = {
  ok: false;
  error: string;
};

export type StorageOk<T> = {
  ok: true;
  value: T;
};

export type StorageResult<T> = StorageOk<T> | StorageFailure;

/**
 * Build a branded storage-unreachable error result. The detail is the
 * underlying cause; the hint is the actionable next step for the user.
 */
export function unreachableStorageError(
  detail: string,
  hint: string,
): StorageFailure {
  return {
    ok: false,
    error: `memorease: storage unreachable — ${detail}. Fix: ${hint}. The session continues without memory; memory tools will return this error until it's resolved.`,
  };
}

/**
 * Wrap any throwing operation in a fail-soft net. Used by every memory op so
 * that transient DB outages or ONNX load failures surface as branded results
 * rather than crashing the session.
 *
 * The `classify` callback inspects the caught error and returns both the
 * category prefix for the branded message and the actionable hint to display.
 * A ready-made `defaultClassifier` covers the common embedder vs. storage
 * split via heuristic message matching.
 */
export type ErrorClassifier = (
  err: unknown,
) => { prefix: string; hint: string };

/**
 * Heuristic classifier — embedder/ONNX errors get a distinct branded prefix
 * ("embedding unavailable") from storage errors ("storage unreachable"). Falls
 * back to storage for anything ambiguous.
 */
export function defaultClassifier(err: unknown): {
  prefix: string;
  hint: string;
} {
  const msg = err instanceof Error ? err.message : String(err);
  // ONNX load / fastembed init failures tend to mention onnx, model, or embed.
  if (
    /\b(onnx|fastembed|embed|model load|wasm)\b/i.test(msg)
  ) {
    return {
      prefix: "embedding unavailable",
      hint: "ensure the ONNX cache (~/.cache/mastra) is writable and the model files are intact; memorease runs memoryless until restart",
    };
  }
  return {
    prefix: "storage unreachable",
    hint: "check the connection string or ensure Postgres/libsql is running; memorease runs memoryless until resolved",
  };
}

export async function failSoft<T>(
  op: () => Promise<T>,
  classify: ErrorClassifier = defaultClassifier,
): Promise<StorageResult<T>> {
  try {
    const value = await op();
    return { ok: true, value };
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    let classified: { prefix: string; hint: string };
    try {
      classified = classify(err);
    } catch {
      classified = {
        prefix: "storage unreachable",
        hint: "check the plugin configuration and try again",
      };
    }
    return {
      ok: false,
      error: `memorease: ${classified.prefix} — ${detail}. Fix: ${classified.hint}. The session continues without memory; memory tools will return this error until it's resolved.`,
    };
  }
}
