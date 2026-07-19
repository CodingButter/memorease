import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type Backend = "libsql" | "pg";

export type ResolvedConfig = {
  backend: Backend;
  /**
   * Present iff backend is "pg". May be an explicit user-supplied connection
   * string or one auto-detected from mastracode's settings.json (shared DB).
   */
  connectionString?: string;
  /**
   * libsql file URL. Present iff backend is "libsql". Defaults to a path in
   * the user's data dir; overridable via MEMOREASE_LIBSQL_PATH for tests.
   */
  libsqlUrl?: string;
};

/**
 * MastraCode plugin context shape — minimal subset we read.
 */
export type PluginContext = {
  config?: {
    connectionString?: string;
  };
};

/**
 * Best-effort read of mastracode's settings.json. Returns undefined on any
 * failure (missing file, permission denied, malformed JSON). The caller
 * decides the fallback — libsql default for the storage decision.
 *
 * Path is computed lazily on each call (via `homedir()`) so tests that
 * override `process.env.HOME` are honored.
 */
function readMastracodeSettings(): Record<string, unknown> | undefined {
  try {
    const settingsPath = join(
      homedir(),
      ".local",
      "share",
      "mastracode",
      "settings.json",
    );
    const raw = readFileSync(settingsPath, "utf8");
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return undefined;
  }
}

/**
 * Pull a Postgres connection string out of mastracode's settings.json if the
 * user has configured `storage.backend: "pg"`. Returns undefined otherwise.
 *
 * mastracode's storage-factory accepts either a single `connectionString` or
 * discrete host/port/user/password fields. We honor both shapes.
 */
function pgConnectionStringFromSettings(
  settings: Record<string, unknown>,
): string | undefined {
  const storage = settings.storage as
    | {
        backend?: string;
        connectionString?: string;
        host?: string;
        port?: number | string;
        user?: string;
        password?: string;
        database?: string;
      }
    | undefined;
  if (!storage || storage.backend !== "pg") return undefined;
  if (typeof storage.connectionString === "string" && storage.connectionString) {
    return storage.connectionString;
  }
  if (storage.host) {
    const auth =
      storage.user || storage.password
        ? `${encodeURIComponent(storage.user ?? "")}:${
            encodeURIComponent(storage.password ?? "")
          }@`
        : "";
    const port = storage.port ? `:${storage.port}` : "";
    const db = storage.database ? `/${storage.database}` : "";
    return `postgresql://${auth}${storage.host}${port}${db}`;
  }
  return undefined;
}

/**
 * Default libsql file URL. Honors MEMOREASE_LIBSQL_PATH (tests) and falls back
 * to the user's data dir.
 */
function defaultLibsqlUrl(): string {
  const override = process.env.MEMOREASE_LIBSQL_PATH;
  if (override) return override.startsWith("file:") ? override : `file:${override}`;
  return `file:${join(homedir(), ".local", "share", "memorease", "memorease-vectors.db")}`;
}

let cached: ResolvedConfig | undefined;

/**
 * Resolve storage backend + connection details from plugin config and
 * mastracode's settings.json. Memoized per-process.
 *
 * Resolution order:
 *  1. Explicit `connectionString` in plugin config → `pg` with that string.
 *  2. mastracode settings.json `storage.backend === "pg"` with a usable
 *     connection → `pg` (shared DB).
 *  3. Otherwise → `libsql` default.
 */
export function resolveConfig(context: PluginContext): ResolvedConfig {
  if (cached) return cached;

  const explicit = context.config?.connectionString;
  if (typeof explicit === "string" && explicit.trim()) {
    cached = { backend: "pg", connectionString: explicit.trim() };
    return cached;
  }

  const settings = readMastracodeSettings();
  if (settings) {
    const shared = pgConnectionStringFromSettings(settings);
    if (shared) {
      cached = { backend: "pg", connectionString: shared };
      return cached;
    }
  }

  cached = { backend: "libsql", libsqlUrl: defaultLibsqlUrl() };
  return cached;
}

/**
 * Reset memoization — for tests that need to re-resolve with fresh inputs.
 */
export function _resetForTests(): void {
  cached = undefined;
}
