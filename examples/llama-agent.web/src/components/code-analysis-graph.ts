import type { AgentEvent } from "../types/agent";
import { structuredToolResult } from "./structured-tool-result";
import { parseRelationshipVisualization, type RelationshipVisualization } from "../visualization/contract";

export function codeCallVisualization(toolName: string | undefined, payload: unknown): RelationshipVisualization | undefined {
  if ((toolName !== "code.calls" && toolName !== "code.impact") || payload === undefined) return undefined;
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

// Session identity and delivery sequence belong to the event envelope, not the graph.
export function codeCallGraphsByTurn(events: AgentEvent[], sessionId: string) {
  const byTurn = new Map<string, Array<{ event: AgentEvent; visualization: RelationshipVisualization }>>();
  const seen = new Set<number>();
  for (const event of events) {
    if (!event.turn_id || event.session_id !== sessionId) continue;
    const type = String(event.event_type ?? event.type ?? "");
    if (type !== "tool.completed") continue;
    if (typeof event.sequence === "number" && seen.has(event.sequence)) continue;
    const toolName = event.tool_name ?? (typeof event.detail === "string" ? event.detail : undefined);
    const visualization = codeCallVisualization(toolName, event.visualization ?? event.tool_result);
    if (!visualization) continue;
    if (typeof event.sequence === "number") seen.add(event.sequence);
    const graphs = byTurn.get(event.turn_id) ?? [];
    graphs.push({ event, visualization });
    byTurn.set(event.turn_id, graphs);
  }
  return byTurn;
}
