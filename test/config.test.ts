import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { resolveConfig, _resetForTests } from "../src/config.js";

/**
 * Config resolution is environment-dependent (reads ~/.local/share/mastracode/
 * settings.json). We isolate tests by overriding HOME so the settings path
 * resolves under a temp dir we control.
 */
function withSettings(dir: string, contents: unknown) {
  const settingsDir = join(dir, ".local", "share", "mastracode");
  mkdirSync(settingsDir, { recursive: true });
  writeFileSync(join(settingsDir, "settings.json"), JSON.stringify(contents));
}

describe("resolveConfig", () => {
  let realHome: string | undefined;
  let tmpHome: string;

  beforeEach(() => {
    realHome = process.env.HOME;
    tmpHome = mkdtempSync(join(tmpdir(), "memorease-cfg-"));
    process.env.HOME = tmpHome;
    // Clear any libsql path override from other test runs.
    delete process.env.MEMOREASE_LIBSQL_PATH;
    _resetForTests();
  });

  afterEach(() => {
    process.env.HOME = realHome;
    rmSync(tmpHome, { recursive: true, force: true });
    _resetForTests();
  });

  it("explicit connectionString → pg backend", () => {
    const cfg = resolveConfig({
      config: { connectionString: "postgresql://u:p@host:5432/db" },
    });
    expect(cfg.backend).toBe("pg");
    expect(cfg.connectionString).toBe("postgresql://u:p@host:5432/db");
  });

  it("explicit connectionString is trimmed", () => {
    const cfg = resolveConfig({
      config: { connectionString: "  postgresql://x  " },
    });
    expect(cfg.backend).toBe("pg");
    expect(cfg.connectionString).toBe("postgresql://x");
  });

  it("empty explicit connectionString + no settings → libsql default", () => {
    const cfg = resolveConfig({ config: { connectionString: "" } });
    expect(cfg.backend).toBe("libsql");
    expect(cfg.libsqlUrl).toMatch(/^file:.*memorease-vectors\.db$/);
  });

  it("settings.json backend:pg with connectionString → pg shared", () => {
    withSettings(tmpHome, {
      storage: {
        backend: "pg",
        connectionString: "postgresql://shared:pw@db.internal:5432/mastra",
      },
    });
    const cfg = resolveConfig({ config: {} });
    expect(cfg.backend).toBe("pg");
    expect(cfg.connectionString).toBe(
      "postgresql://shared:pw@db.internal:5432/mastra",
    );
  });

  it("settings.json backend:pg with discrete fields → pg shared (assembled)", () => {
    withSettings(tmpHome, {
      storage: {
        backend: "pg",
        host: "db.example.com",
        port: 5432,
        user: "alice",
        password: "s3cret",
        database: "mastracode",
      },
    });
    const cfg = resolveConfig({ config: {} });
    expect(cfg.backend).toBe("pg");
    expect(cfg.connectionString).toBe(
      "postgresql://alice:s3cret@db.example.com:5432/mastracode",
    );
  });

  it("settings.json backend:libsql → libsql default", () => {
    withSettings(tmpHome, { storage: { backend: "libsql" } });
    const cfg = resolveConfig({ config: {} });
    expect(cfg.backend).toBe("libsql");
    expect(cfg.connectionString).toBeUndefined();
  });

  it("settings.json missing entirely → libsql default (best-effort read)", () => {
    // No settings.json in tmpHome — should not throw.
    const cfg = resolveConfig({ config: {} });
    expect(cfg.backend).toBe("libsql");
  });

  it("settings.json malformed → libsql default (best-effort read)", () => {
    const settingsDir = join(tmpHome, ".local", "share", "mastracode");
    mkdirSync(settingsDir, { recursive: true });
    writeFileSync(join(settingsDir, "settings.json"), "{ not valid json");
    const cfg = resolveConfig({ config: {} });
    expect(cfg.backend).toBe("libsql");
  });

  it("explicit connectionString wins over settings.json", () => {
    withSettings(tmpHome, {
      storage: { backend: "pg", connectionString: "postgresql://from-settings" },
    });
    const cfg = resolveConfig({
      config: { connectionString: "postgresql://from-config" },
    });
    expect(cfg.backend).toBe("pg");
    expect(cfg.connectionString).toBe("postgresql://from-config");
  });

  it("MEMOREASE_LIBSQL_PATH override honored", () => {
    process.env.MEMOREASE_LIBSQL_PATH = "/tmp/custom/path.db";
    const cfg = resolveConfig({ config: {} });
    expect(cfg.libsqlUrl).toBe("file:/tmp/custom/path.db");
  });

  it("memoization: resolveConfig returns the same object on second call", () => {
    const a = resolveConfig({ config: { connectionString: "postgresql://x" } });
    const b = resolveConfig({ config: {} });
    expect(a).toBe(b);
  });
});
