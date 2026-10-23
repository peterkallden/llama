import { structuredToolResult } from "./structured-tool-result";
import { parseRelationshipVisualization, type RelationshipVisualization } from "../visualization/contract";

export function codeCallVisualization(toolName: string | undefined, payload: unknown): RelationshipVisualization | undefined {
  if (toolName !== "code.calls" || payload === undefined) return undefined;
  try {
    if (payload && typeof payload === "object" && !Array.isArray(payload) &&
        "schema_version" in payload && "kind" in payload) {
      return parseRelationshipVisualization(payload);
    }
    const result = structuredToolResult(payload);
    return result.visualization === undefined ? undefined : parseRelationshipVisualization(result.visualization);
  } catch {
    return undefined;
  }
}
