import type { AgentEvent, DaemonToolStatus } from "../types/agent";

export function mcpAppLabel(tool: DaemonToolStatus): string {
  const uri = tool._meta?.ui?.resourceUri;
  const names: Record<string, string> = {
    "ui://llama-agent/apps/document-editor": "Document editor",
    "ui://llama-agent/apps/investigation-workspace": "Investigation Workspace",
    "ui://llama-agent/apps/data-explorer": "Data Explorer",
    "ui://llama-agent/apps/workflow-inspector": "Workflow Inspector",
  };
  return uri && names[uri] ? names[uri] : tool.name;
}

export function mcpAppKey(tool: DaemonToolStatus): string {
  return `${tool.provider_id ?? ""}\u0000${tool._meta?.ui?.resourceUri ?? ""}`;
}

function providerId(value: unknown): string {
  return typeof value === "string" ? value : "";
}

export function mcpAppInvocationKey(event: AgentEvent): string {
  return JSON.stringify([event.session_id, event.request_id, event.turn_id,
    event.sequence ?? event.client_received_at, event.operation_id,
    event.provider_id, event.server_tool_name]);
}

export function findAppForInvocationEvent(
  event: AgentEvent | undefined,
  tools: DaemonToolStatus[],
  sessionId?: string,
): DaemonToolStatus | undefined {
  if (sessionId !== undefined && event?.session_id !== sessionId) return undefined;
  if (!event || event.app_context_available !== true || !event.server_tool_name) return undefined;
  const eventProvider = providerId(event.provider_id);
  const origin = tools.find((tool) => providerId(tool.provider_id) === eventProvider &&
    (tool.server_tool_name || tool.name) === event.server_tool_name);
  const uri = origin?._meta?.ui?.resourceUri;
  if (!uri) return undefined;
  return tools.find((tool) => providerId(tool.provider_id) === providerId(origin.provider_id) &&
    tool._meta?.ui?.resourceUri === uri && (tool._meta?.ui?.visibility ?? []).includes("app"));
}

export function findLatestAppInvocation(
  eventsNewestFirst: AgentEvent[],
  tools: DaemonToolStatus[],
  app: DaemonToolStatus,
  sessionId?: string,
  localInvocation?: AgentEvent,
): AgentEvent | undefined {
  const appUri = app._meta?.ui?.resourceUri;
  if (!appUri) return undefined;
  const matches = (event: AgentEvent) => {
    if (sessionId !== undefined && event.session_id !== sessionId) return false;
    if (event.app_context_available !== true || !event.tool_input || typeof event.tool_input !== "object" ||
        Array.isArray(event.tool_input) || !event.server_tool_name) return false;
    if (providerId(event.provider_id) !== providerId(app.provider_id)) return false;
    const eventProvider = providerId(event.provider_id);
    const origin = tools.find((tool) => providerId(tool.provider_id) === eventProvider &&
      (tool.server_tool_name || tool.name) === event.server_tool_name);
    return origin?._meta?.ui?.resourceUri === appUri;
  };
  const latest = eventsNewestFirst.find(matches);
  const local = localInvocation && matches(localInvocation) ? localInvocation : undefined;
  return local && (!latest || Number(local.client_received_at ?? 0) >= Number(latest.client_received_at ?? 0))
    ? local : latest;
}
