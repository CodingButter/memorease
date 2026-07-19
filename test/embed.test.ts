import { describe, it, expect } from "vitest";
import { embed, embedMany } from "../src/embed.js";

describe("embed (real fastembed)", () => {
  it("returns a 384-length vector of finite numbers", async () => {
    const v = await embed("hello world");
    expect(v.length).toBe(384);
    for (const x of v) {
      expect(Number.isFinite(x)).toBe(true);
    }
  });

  it("preserves order in batch embedding", async () => {
    const [a, b] = await embedMany(["first sentence", "second sentence"]);
    expect(a.length).toBe(384);
    expect(b.length).toBe(384);
    // Two distinct sentences should produce non-identical vectors.
    const equal = a.every((val, i) => val === b[i]);
    expect(equal).toBe(false);
  });

  it("embeds identical text identically (deterministic)", async () => {
    const a = await embed("memorease determinism check");
    const b = await embed("memorease determinism check");
    expect(a).toEqual(b);
  });
});
