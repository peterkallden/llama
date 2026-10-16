import { describe, expect, it } from "vitest";
import type { AgentEvent, DaemonToolStatus } from "../types/agent";
import { findAppForInvocationEvent, findLatestAppInvocation, mcpAppKey } from "./app-invocation";

const appTool = (provider_id: string, name: string, uri: string, server_tool_name = name): DaemonToolStatus => ({
  name,
  provider_id,
  server_tool_name,
  _meta: { ui: { resourceUri: uri, visibility: ["app"] } },
});

describe("MCP App invocation routing", () => {
  it("selects the newest event and isolates identical resource URIs by provider", () => {
    const tools = [
      appTool("provider-a", "app-a", "ui://shared/app", "run"),
      appTool("provider-b", "app-b", "ui://shared/app", "run"),
    ];
    const oldEvent: AgentEvent = { provider_id: "provider-a", server_tool_name: "run", app_context_available: true, tool_input: { value: "old" } };
    const newestEvent: AgentEvent = { provider_id: "provider-b", server_tool_name: "run", app_context_available: true, tool_input: { value: "new" } };
    expect(findLatestAppInvocation([newestEvent, oldEvent], tools, tools[1])).toBe(newestEvent);
    expect(findLatestAppInvocation([newestEvent, oldEvent], tools, tools[0])).toBe(oldEvent);
    expect(findAppForInvocationEvent(newestEvent, tools)).toBe(tools[1]);
    expect(mcpAppKey(tools[0])).not.toBe(mcpAppKey(tools[1]));
  });

  it("does not select an older app invocation for a newer unrelated event", () => {
    const tools = [appTool("provider-a", "app-a", "ui://a/app", "run")];
    const invocation: AgentEvent = { provider_id: "provider-a", server_tool_name: "run", app_context_available: true, tool_input: { value: 1 } };
    const unrelated: AgentEvent = { provider_id: "provider-a", event_type: "turn.completed" };
    expect(findAppForInvocationEvent(unrelated, tools)).toBeUndefined();
    expect(findLatestAppInvocation([unrelated], tools, tools[0])).toBeUndefined();
    expect(findLatestAppInvocation([invocation], tools, tools[0])).toBe(invocation);
  });
});
