import type { AgentEvent } from "../types/agent";

const executionStates = new Map<string, string>([
  ["tool.queued", "queued"], ["tool.started", "running"],
  ["tool.completed", "completed"], ["tool.failed", "failed"],
  ["turn.completed", "completed"], ["turn.failed", "failed"],
  ["turn.rejected", "rejected"], ["turn.cancelled", "cancelled"],
]);
const verdicts = new Set(["SATISFIED", "VIOLATED", "UNKNOWN"]);

export function classifyWorkflowEvent(event: AgentEvent) {
  const type = String(event.event_type ?? event.type ?? "event");
  const rawVerdict = String(event.verdict ?? event.verification_status ?? "").toUpperCase();
  return {
    type,
    execution: executionStates.get(type) ?? "observed",
    verdict: verdicts.has(rawVerdict) ? rawVerdict : "",
    phase: String(event.phase ?? event.cognitive_mode ?? event.route_id ?? event.event_category ?? "runtime"),
    text: String(event.detail ?? event.error ?? event.tool_name ?? ""),
  };
}
