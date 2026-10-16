import type { AgentEvent, DaemonToolStatus } from "../types/agent";

export function mcpAppKey(tool: DaemonToolStatus): string {
  return `${tool.provider_id ?? "native"}\u0000${tool._meta?.ui?.resourceUri ?? ""}`;
}

export function findAppForInvocationEvent(
  event: AgentEvent | undefined,
  tools: DaemonToolStatus[],
): DaemonToolStatus | undefined {
  if (!event || event.app_context_available !== true || typeof event.provider_id !== "string" || !event.server_tool_name) return undefined;
  const origin = tools.find((tool) => tool.provider_id === event.provider_id &&
    (tool.server_tool_name || tool.name) === event.server_tool_name);
  const uri = origin?._meta?.ui?.resourceUri;
  if (!uri) return undefined;
  return tools.find((tool) => tool.provider_id === origin.provider_id &&
    tool._meta?.ui?.resourceUri === uri && (tool._meta?.ui?.visibility ?? []).includes("app"));
}

export function findLatestAppInvocation(
  eventsNewestFirst: AgentEvent[],
  tools: DaemonToolStatus[],
  app: DaemonToolStatus,
): AgentEvent | undefined {
  const appUri = app._meta?.ui?.resourceUri;
  if (!appUri) return undefined;
  return eventsNewestFirst.find((event) => {
    if (event.app_context_available !== true || !event.tool_input || typeof event.provider_id !== "string" || !event.server_tool_name) return false;
    if (event.provider_id !== app.provider_id) return false;
    const origin = tools.find((tool) => tool.provider_id === event.provider_id &&
      (tool.server_tool_name || tool.name) === event.server_tool_name);
    return origin?._meta?.ui?.resourceUri === appUri;
  });
}
