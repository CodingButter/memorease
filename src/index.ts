/**
 * memorease — durable semantic memory for MastraCode, as a plugin.
 *
 * Plugin entry point. Exports a `defineMastraCodePlugin` with:
 *   - `config`: four keys — connectionString, curatorModel, injectBudget, skillsDir.
 *   - `instructions(context)`: at session boot, query the memory store, rank
 *     via the curator if armed (or vector similarity otherwise), and inject a
 *     `## Memories` section into the system prompt. Never throws.
 *   - `tools(context)`: five tools — memory_query, memory_write, memory_forget,
 *     memory_distill_skill, memory_tap_status.
 *
 * Auto-defaults:
 *   - Storage backend: explicit `connectionString` wins; otherwise reuse the
 *     Postgres mastracode is already configured for (settings.json
 *     `storage.backend: "pg"`); otherwise local SQLite in the user's data dir.
 *   - Curator model: explicit `curatorModel` wins; otherwise fall back to
 *     mastracode's `settings.json → models.observerModelOverride`; otherwise
 *     disarm (vector similarity ranking only).
 *
 * `resolveModel()` from `@mastra/code-sdk/agents/model` handles credential
 * injection internally (OAuth → apikey slot → env). This plugin never reads
 * `auth.json` or API-key env vars directly.
 */

import { defineMastraCodePlugin } from "mastracode/plugin";
import type { MastraCodePluginContext } from "mastracode/plugin";

import { buildInstructions } from "./instructions.js";
import { buildTools } from "./tools.js";

export default defineMastraCodePlugin({
  id: "memorease",
  name: "Memorease",
  version: "0.1.0",
  description:
    "Durable semantic memory for MastraCode — as a plugin. Inspectable, self-hosted, removable.",
  config: {
    connectionString: {
      type: "string",
      label: "Postgres connection string",
      description:
        "Leave empty to use local SQLite, or to reuse the Postgres mastracode already uses (auto-detected from settings.json).",
      default: "",
    },
    curatorModel: {
      type: "model",
      label: "Curator model",
      description:
        "Optional. Ranks which memories deserve system-prompt space at boot. If unset, memorease auto-defaults to mastracode's configured observer model (settings.json → models.observerModelOverride); if that's also unset, ranking falls back to vector similarity.",
      default: "",
    },
    injectBudget: {
      type: "string",
      label: "Boot injection budget (chars)",
      description:
        "Approximate character budget for the injected `## Memories` section. Coerced to int; clamped 200..8000.",
      default: "1200",
    },
    skillsDir: {
      type: "string",
      label: "Skills directory",
      description:
        "Where memory_distill_skill writes SKILL.md. Empty = ~/.agents/skills.",
      default: "",
    },
  },
  instructions: (context: MastraCodePluginContext) =>
    buildInstructions(context as unknown as Parameters<typeof buildInstructions>[0]),
  tools: (context: MastraCodePluginContext) => buildTools(context),
});
