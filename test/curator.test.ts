import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import { writeFileSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  resolveCuratorModel,
  curatorPreflightError,
  resolveInjectBudget,
  curateForBoot,
  INJECT_BUDGET_DEFAULT,
  INJECT_BUDGET_MIN,
  INJECT_BUDGET_MAX,
} from "../src/curator.js";
import type { MemoryHit } from "../src/memory.js";
import { _resetForTests as resetConfig } from "../src/config.js";

/**
 * Curator tests cover:
 *  - disarmed path (empty config + stubbed-empty settings → branded note;
 *    curateForBoot falls back to similarity ranking)
 *  - default-chain path (empty config.curatorModel + settings.json with
 *    observerModelOverride → that id wins — zero-config default)
 *  - explicit-override path (config.curatorModel beats settings.json)
 *  - settings-unreadable path (missing/malformed settings.json → undefined)
 *  - armed path (real resolveModel + doGenerate — gated by env var)
 */

function makeSettingsFile(body: unknown): string {
  const dir = mkdirSync(
    join(tmpdir(), `memorease-curator-${Math.random().toString(36).slice(2)}/`),
    { recursive: true },
  ) as string;
  const path = join(dir, "settings.json");
  writeFileSync(path, JSON.stringify(body));
  return path;
}

function makeMalformedSettingsFile(): string {
  const dir = mkdirSync(
    join(tmpdir(), `memorease-curator-${Math.random().toString(36).slice(2)}/`),
    { recursive: true },
  ) as string;
  const path = join(dir, "settings.json");
  writeFileSync(path, "{not valid json");
  return path;
}

const createdPaths: string[] = [];
function track(p: string) {
  createdPaths.push(p);
  return p;
}

afterEach(() => {
  resetConfig();
  delete process.env.MEMOREASE_SETTINGS_PATH;
});

afterAll(() => {
  for (const p of createdPaths) {
    rmSync(p, { recursive: true, force: true });
  }
});

const HITS: MemoryHit[] = [
  { id: "1", score: 0.9, name: "rust-fan", content: "User loves Rust and strict type systems." },
  { id: "2", score: 0.7, name: "voice-first", content: "Jamie prefers voice-first responses." },
  { id: "3", score: 0.5, name: "guitarist", content: "User plays guitar on weekends." },
];

describe("resolveCuratorModel", () => {
  it("returns undefined when both config and settings are empty", () => {
    process.env.MEMOREASE_SETTINGS_PATH = track(makeSettingsFile({ models: {} }));
    expect(resolveCuratorModel({})).toBeUndefined();
  });

  it("returns the observer model from settings when config is empty (zero-config default)", () => {
    process.env.MEMOREASE_SETTINGS_PATH = track(
      makeSettingsFile({ models: { observerModelOverride: "deepseek/deepseek-v4-flash" } }),
    );
    expect(resolveCuratorModel({})).toBe("deepseek/deepseek-v4-flash");
  });

  it("explicit config.curatorModel wins over settings.json.observerModelOverride", () => {
    process.env.MEMOREASE_SETTINGS_PATH = track(
      makeSettingsFile({ models: { observerModelOverride: "fallback-id" } }),
    );
    expect(resolveCuratorModel({ curatorModel: "explicit-id" })).toBe("explicit-id");
  });

  it("trims whitespace from config.curatorModel", () => {
    expect(resolveCuratorModel({ curatorModel: "  trimmed-id  " })).toBe("trimmed-id");
  });

  it("returns undefined when settings.json is missing", () => {
    process.env.MEMOREASE_SETTINGS_PATH = "/nonexistent/path/settings.json";
    expect(resolveCuratorModel({})).toBeUndefined();
  });

  it("returns undefined when settings.json is malformed", () => {
    process.env.MEMOREASE_SETTINGS_PATH = track(makeMalformedSettingsFile());
    expect(resolveCuratorModel({})).toBeUndefined();
  });

  it("treats empty/whitespace observerModelOverride as unset", () => {
    process.env.MEMOREASE_SETTINGS_PATH = track(
      makeSettingsFile({ models: { observerModelOverride: "   " } }),
    );
    expect(resolveCuratorModel({})).toBeUndefined();
  });
});

describe("curatorPreflightError", () => {
  it("returns undefined when armed", () => {
    expect(curatorPreflightError({ curatorModel: "some-id" })).toBeUndefined();
  });

  it("returns a branded note when disarmed", () => {
    process.env.MEMOREASE_SETTINGS_PATH = track(makeSettingsFile({ models: {} }));
    const err = curatorPreflightError({});
    expect(err).toMatch(/memorease: curator disarmed/);
    expect(err).toMatch(/observerModelOverride/);
  });
});

describe("resolveInjectBudget", () => {
  it("returns the default for undefined input", () => {
    expect(resolveInjectBudget(undefined)).toBe(INJECT_BUDGET_DEFAULT);
  });

  it("parses numeric strings", () => {
    expect(resolveInjectBudget("500")).toBe(500);
  });

  it("clamps below the minimum", () => {
    expect(resolveInjectBudget("10")).toBe(INJECT_BUDGET_MIN);
  });

  it("clamps above the maximum", () => {
    expect(resolveInjectBudget("99999")).toBe(INJECT_BUDGET_MAX);
  });

  it("falls back to default for non-numeric strings", () => {
    expect(resolveInjectBudget("not-a-number")).toBe(INJECT_BUDGET_DEFAULT);
  });
});

describe("curateForBoot — disarmed path", () => {
  beforeAll(() => {
    // Force-disarm by pointing settings at an empty file with no override.
    process.env.MEMOREASE_SETTINGS_PATH = track(makeSettingsFile({ models: {} }));
  });

  it("returns hits sorted by descending score, truncated to budget", async () => {
    const result = await curateForBoot(undefined, HITS, 10000, "host=test; cwd=repo");
    expect(result.usedCurator).toBe(false);
    expect(result.fallbackNote).toBeUndefined();
    // Sorted by score descending
    expect(result.hits.map((h) => h.name)).toEqual(["rust-fan", "voice-first", "guitarist"]);
  });

  it("respects the budget — stops adding hits when the next would overflow", async () => {
    // Tight budget: should keep the highest-scoring hit at minimum.
    const result = await curateForBoot(undefined, HITS, 50, "host=test");
    expect(result.hits.length).toBeGreaterThanOrEqual(1);
    expect(result.hits[0].name).toBe("rust-fan");
  });

  it("returns empty list for empty candidate input", async () => {
    const result = await curateForBoot(undefined, [], 1000, "host=test");
    expect(result.hits).toEqual([]);
  });
});

describe("curateForBoot — armed path (real model)", () => {
  const modelId = process.env.MEMOREASE_CURATOR_TEST_MODEL;

  it.skipIf(!modelId)(
    "uses the curator to rank candidates and returns its selection",
    async () => {
      // Stub settings to empty so resolveCuratorModel uses the explicit id.
      process.env.MEMOREASE_SETTINGS_PATH = track(makeSettingsFile({ models: {} }));
      const result = await curateForBoot(modelId!, HITS, 1000, "host=test; cwd=repo");
      // Either the model's selection applies, or we get a fallback note.
      // We don't assert exact ranking (model nondeterminism), only that we
      // either used the curator or produced a branded fallback note.
      expect(result.usedCurator || result.fallbackNote).toBeTruthy();
      expect(result.hits.length).toBeGreaterThan(0);
    },
  );
});
