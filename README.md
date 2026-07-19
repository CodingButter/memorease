# memorease

**Durable semantic memory for [MastraCode](https://github.com/mastra-cloud/mastracode) — as a plugin.**

Opt-in, inspectable, self-hosted, and removable. Memory that survives across
sessions and across machines, without the always-on opacity of an internal
memory engine.

---

## Why

MastraCode ships with an internal memory engine. It's useful, but it's opaque:
you can't easily see what's been remembered, you can't edit it, and the moment
you uninstall or change tools the memory goes with it. memorease takes a
different stance:

- **Memory is a tool.** The agent decides when to remember and what to look
  up. Nothing happens behind its back.
- **Memory is inspectable.** Every fact lives in a `memorease_memories`
  table (SQLite by default, Postgres opt-in) with a stable name, body, and
  embedding. You can read it, edit it, or wipe it.
- **Memory is yours.** The store lives in your data dir or your own Postgres.
  No telemetry, no API keys for embeddings, no third-party calls.
- **Memory survives.** Read by every future session on every machine that
  points at the same store.

## Quickstart

### Option A — Zero-config local SQLite (default)

```bash
mastracode plugin install github:CodingButter/memorease
```

That's it. memorease creates a SQLite file at
`~/.local/share/memorease/memorease-vectors.db` on first use, uses
[`@mastra/fastembed`](https://github.com/mastra-org/fastembed-js) for on-device
384-dim embeddings (no API keys, ~50MB ONNX model downloads on first call), and
falls back to vector-similarity ranking for the boot memories section.

### Option B — Postgres (shared / fleet / scaled)

memorease ships with a disposable Postgres + pgvector compose file:

```bash
git clone https://github.com/CodingButter/memorease
cd memorease
docker compose up -d                  # pgvector on localhost:5432
```

Then in your MastraCode plugin config, set the connection string:

```
postgresql://memorease:memorease@localhost:5432/memorease
```

The first time the plugin runs it creates the `vector` extension (if missing)
and the `memorease_memories` index.

## Config reference

Every key is optional. All four live in the standard MastraCode plugin config
schema and are editable from the TUI.

| Key               | Type     | Default        | Meaning                                                                                                                                                  |
| ----------------- | -------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `connectionString`| string   | `""`           | Postgres connection string. Empty = auto-detect (see [Storage backends](#storage-backends)).                                                             |
| `curatorModel`    | `model`  | `""`           | LLM used to rank which memories earn system-prompt space at boot. Empty = auto-default to mastracode's configured observer model; if that's empty, vector-similarity ranking. |
| `injectBudget`    | string   | `"1200"`       | Approximate character budget for the injected `## Memories` section. Coerced to int and clamped to 200–8000.                                             |
| `skillsDir`       | string   | `""` (`~/.agents/skills`) | Where `memory_distill_skill` writes `SKILL.md` files.                                                                                                  |

**You usually don't need to set any of them.** The defaults pick SQLite +
local embeddings + mastracode's observer model (or vector-similarity fallback)
without any configuration.

## Tools

The plugin exposes five tools. The agent calls them through the normal MastraCode
tool surface.

| Tool                   | Purpose                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `memory_query`         | Semantic search. Embeds the query text and returns top-k `{name, content, type, score, metadata}`. |
| `memory_write`         | Upsert a memory by stable name. Same-name writes replace the prior body (dedup-by-name).        |
| `memory_forget`        | Delete a memory by name. Idempotent — forgetting a missing name still returns success.          |
| `memory_distill_skill` | Fold named memories into a `SKILL.md` file on disk, with provenance metadata. Slug-validated.    |
| `memory_tap_status`    | Read-only probe of the experimental "gut-feeling" signal provider (armed / disarmed + reason).   |

Every tool returns branded errors (`{"ok": false, "error": "memorease: ..."}`)
on storage failure. **memorease never throws out of a tool call.**

## Architecture

### Storage backends

memorease uses Mastra's own `MastraVector` abstraction for storage, so you get
two tested backends for free with a single code path:

- **`PgVector`** — used when a Postgres `connectionString` is supplied or
  auto-detected. Embeddings live in-row as `vector(384)` columns; queries use
  pgvector's `<=>` cosine operator and a single round-trip.
- **`LibSQLVector`** — used by default. Embeddings live in-row in a local
  SQLite file in your data dir.

Resolution order:

1. Explicit `connectionString` in plugin config → **pg** with that string.
2. MastraCode `settings.json` has `storage.backend === "pg"` with a usable
   connection → **pg** with mastracode's own Postgres (shared DB).
3. Otherwise → **libsql** default.

### Embeddings

[`@mastra/fastembed`](https://github.com/mastra-org/fastembed-js) with the
`fastembed.small` model (384-dim, local ONNX). No API keys, no third-party
calls, no embedder config option. The first call downloads the ONNX weights
(~50MB); subsequent calls hit a local cache.

> **Why is the embedder not configurable?** Mixing embedding models inside one
> store produces meaningless similarity scores. One store, one embedding space,
> one obvious default — keeps the memory index coherent and the config surface
> small.

### Curator ranking

At session boot, memorease pulls candidate memories, ranks them, and injects
the top ones into the system prompt as a `## Memories` section. The ranking
model ("curator") follows this resolution chain:

1. Explicit `curatorModel` in plugin config.
2. MastraCode `settings.json → models.observerModelOverride`.
3. **Disarmed** — pure vector-similarity ranking (top-k by cosine).

The armed path issues a single `doGenerate` call with a fixed selection prompt
and a character budget (`injectBudget`). Credentials are handled by
`resolveModel` from `@mastra/code-sdk/agents/model` — memorease never reads
`auth.json` or API-key env vars directly.

### Fail-soft guarantees

memorease is designed so that storage or LLM failures never break the host
agent. Every storage and embedding operation is wrapped in `failSoft`, which
returns a branded `{"ok": false, "error": "memorease: ..."}` result on:

- unreachable database (`ECONNREFUSED`, timeouts, auth failures)
- embedder failure (ONNX load errors, model corruption)
- storage errors (malformed queries, missing index despite `ensureSchema`)

The `## Memories` boot section likewise degrades to a branded
`## Memories: unavailable` line if the store can't be reached within a 3s
timeout (covers both slow-DB timeouts and fast-reject `ECONNREFUSED`).

### Experimental: the "gut-feeling" signal provider

memorease ships a `MemoreaseSignalProvider` built on the official
[`SignalProvider`](https://github.com/mastra-org/mastra) surface. On every
poll (default 30s) the provider:

1. Recalls the latest user message from the active thread via
   `agent.memory.recall(...)`.
2. Embeds it and queries the store.
3. If the top hit scores above `TAP_THRESHOLD` (cosine ≥ 0.5) **and** this
   thread hasn't been notified about that same memory within `TAP_DEDUP_MS`
   (5 minutes), it fires a notification suggesting the agent call
   `memory_query`.

The provider never writes to memory as a side-channel — it just nudges the
agent to look. It arms on the first memory tool call and disarms cleanly on
every failure shape (no `memory.recall`, no `sendNotificationSignal`, throwing
agent, etc.).

**Why "experimental"?** The provider relies on agent-instance methods that
MastraCode exposes but doesn't formally document for plugin use. If a future
MastraCode release changes those shapes, the provider disarms with a branded
status (`memory_tap_status` will report `disarmed-no-memory`,
`disarmed-no-signal`, etc.) and the rest of memorease keeps working.

## Shared Postgres = shared memory (read this before fleet use)

When multiple MastraCode installs point memorease at the same Postgres
connection string — which is the **default** when mastracode is configured
for `storage.backend: "pg"` — they all read and write the same
`memorease_memories` index. That's *shared memory*, not per-user or
per-machine isolation.

- **Fleet use (intended):** this is the feature, not a bug. A memory written
  from your laptop is visible from your desktop a second later. That's how
  memorease becomes a cross-machine durable brain.
- **Single user on a shared DB:** if you want isolation from other users on
  the same Postgres, give memorease an explicit `connectionString` pointing
  at your own database.

## Scaling limits (v1)

- **LibSQL vector search is in-process.** Queries load all embeddings and
  compute cosine similarity in JS. Fine for typical single-user stores
  (≤ ~10k memories). For heavier use, switch to the Postgres backend —
  pgvector's ANN index keeps queries fast well past that.
- **The first embedding call downloads the ONNX model (~50MB).** Subsequent
  calls are cached.
- **The signal provider polls every 30s.** Cost scales with the number of
  active subscriptions, not the size of the store. Each poll issues one
  `recall` + one `embed` + one `query` per active thread.

## Verification

```bash
pnpm install
pnpm exec tsc --noEmit                  # typecheck
pnpm test                               # libsql-only suite
docker compose up -d                    # optional — enable pg tests
MEMOREASE_PG_TEST_CONNECTION=postgresql://memorease:memorease@localhost:5432/memorease \
  pnpm test                             # full suite (libsql + pg)
```

Live demo scripts (drive the plugin contract end-to-end without the TUI):

```bash
bash .mastracode/plans/memorease.proof/run-local.sh   # libsql demo → transcript-libsql.txt
```

## Roadmap

- **Fleet memory sharing** — first-class multi-machine story (today: works
  through shared Postgres, but there's no built-in sync for the libsql path).
- **Upstream `resolveModel`** — currently imported from
  `@mastra/code-sdk/agents/model`. If mastracode promotes this to a public
  plugin-context API, memorease will move to it.
- **Signal provider graduating from experimental** — pending a sanctioned
  notification surface in mastracode plugins. Today's reach-in is honest
  about its brittleness and disarms cleanly when shapes change.

## License

MIT — see [LICENSE](./LICENSE).
