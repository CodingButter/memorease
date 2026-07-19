import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts", "src/**/*.test.ts"],
    // PG tests share a single Postgres index (INDEX_NAME) across files; running
    // them in parallel means one file's beforeEach drops another's writes.
    // Libsql tests use per-file temp DBs so this only costs the PG path a few
    // seconds of serialization.
    fileParallelism: false,
  },
});
