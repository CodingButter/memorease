import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  recordSurfaced,
  getSurfaced,
  isSuppressed,
  recordBootInjectedNames,
  bootInjectedNames,
  _resetSurfacedForTests,
} from "../src/surfaced.js";

let seq = 0;
let ledgerPath: string;

beforeEach(() => {
  ledgerPath = join(tmpdir(), `memorease-surfaced-unit-${process.pid}-${++seq}.json`);
  process.env.MEMOREASE_SURFACED_PATH = ledgerPath;
  _resetSurfacedForTests();
});

afterEach(() => {
  delete process.env.MEMOREASE_SURFACED_PATH;
});

describe("surfaced: ledger round-trip", () => {
  it("records and reads back per thread", async () => {
    await recordSurfaced("t1", ["a", "b"]);
    const surfaced = await getSurfaced("t1");
    expect(Object.keys(surfaced).sort()).toEqual(["a", "b"]);
    expect(await getSurfaced("t2")).toEqual({});
  });

  it("first surfacing wins — re-recording does not refresh the timestamp", async () => {
    const early = new Date("2026-07-19T00:00:00Z");
    const late = new Date("2026-07-20T00:00:00Z");
    await recordSurfaced("t1", ["a"], early);
    await recordSurfaced("t1", ["a"], late);
    const surfaced = await getSurfaced("t1");
    expect(surfaced.a).toBe(early.toISOString());
  });

  it("skips empty threadIds and names without writing garbage", async () => {
    await recordSurfaced("", ["a"]);
    await recordSurfaced("t1", [""]);
    expect(await getSurfaced("t1")).toEqual({});
  });

  it("writes valid JSON with the versioned shape", async () => {
    await recordSurfaced("t1", ["a"]);
    const raw = JSON.parse(readFileSync(ledgerPath, "utf8"));
    expect(raw.version).toBe(1);
    expect(raw.threads.t1.a).toBeTruthy();
  });

  it("fails soft on an unwritable path", async () => {
    // A regular file used as a parent directory → mkdir fails ENOTDIR fast.
    const blocker = join(tmpdir(), `memorease-surfaced-blocker-${process.pid}-${seq}`);
    const { writeFileSync } = await import("node:fs");
    writeFileSync(blocker, "", "utf8");
    process.env.MEMOREASE_SURFACED_PATH = join(blocker, "ledger.json");
    await expect(recordSurfaced("t1", ["a"])).resolves.toBeUndefined();
    expect(await getSurfaced("t1")).toEqual({});
  });

  it("recovers from a corrupt ledger file", async () => {
    await recordSurfaced("t1", ["a"]);
    const { writeFileSync } = await import("node:fs");
    writeFileSync(ledgerPath, "not json {{{", "utf8");
    expect(await getSurfaced("t1")).toEqual({});
    await recordSurfaced("t1", ["b"]);
    expect(Object.keys(await getSurfaced("t1"))).toEqual(["b"]);
  });
});

describe("surfaced: isSuppressed", () => {
  it("unsurfaced names pass", () => {
    expect(isSuppressed({}, "a")).toBe(false);
  });

  it("surfaced names are suppressed", () => {
    expect(isSuppressed({ a: "2026-07-19T00:00:00Z" }, "a")).toBe(true);
  });

  it("updated-after-surfaced escapes suppression", () => {
    const surfaced = { a: "2026-07-19T00:00:00Z" };
    expect(isSuppressed(surfaced, "a", "2026-07-20T00:00:00Z")).toBe(false);
    expect(isSuppressed(surfaced, "a", "2026-07-18T00:00:00Z")).toBe(true);
  });

  it("garbage updatedAt stays suppressed", () => {
    expect(isSuppressed({ a: "2026-07-19T00:00:00Z" }, "a", "not-a-date")).toBe(true);
  });
});

describe("surfaced: boot-injected set", () => {
  it("records and exposes names, reset clears", () => {
    recordBootInjectedNames(["x", "", "y"]);
    expect(bootInjectedNames().has("x")).toBe(true);
    expect(bootInjectedNames().has("y")).toBe(true);
    expect(bootInjectedNames().has("")).toBe(false);
    _resetSurfacedForTests();
    expect(bootInjectedNames().size).toBe(0);
  });
});
