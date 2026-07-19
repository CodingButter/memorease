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
 * The `hintFor` callback receives the caught error and returns the actionable
 * hint to display. If it throws, a generic hint is used.
 */
export async function failSoft<T>(
  op: () => Promise<T>,
  hintFor: (err: unknown) => string,
): Promise<StorageResult<T>> {
  try {
    const value = await op();
    return { ok: true, value };
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    let hint: string;
    try {
      hint = hintFor(err);
    } catch {
      hint = "check the plugin configuration and try again";
    }
    return unreachableStorageError(detail, hint);
  }
}
