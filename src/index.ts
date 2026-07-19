import { defineMastraCodePlugin } from "mastracode/plugin";

/**
 * memorease — durable semantic memory for MastraCode, as a plugin.
 *
 * Phase 0 stub: minimal valid plugin shape so the typecheck gate has a real
 * entry to resolve. Subsequent phases flesh out config, tools, and instructions.
 */
export default defineMastraCodePlugin({
  id: "memorease",
  name: "Memorease",
  version: "0.1.0",
  description: "Durable semantic memory for MastraCode — as a plugin.",
});
