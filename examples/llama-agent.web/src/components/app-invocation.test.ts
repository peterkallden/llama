import { describe, expect, it } from "vitest";
import type { AgentEvent, DaemonToolStatus } from "../types/agent";
import { findAppForInvocationEvent, findLatestAppInvocation, mcpAppInvocationKey, mcpAppKey } from "./app-invocation";

const appTool = (provider_id: string, name: string, uri: string, server_tool_name = name): DaemonToolStatus => ({
  name,
  provider_id,
  server_tool_name,
  _meta: { ui: { resourceUri: uri, visibility: ["app"] } },
});

describe("MCP App invocation routing", () => {
  it("gives repeated identical calls separate app instances using host event identity", () => {
    const event = { session_id: "s", request_id: "r", sequence: 3, tool_input: {}, tool_result: {} };
    expect(mcpAppInvocationKey(event)).not.toBe(mcpAppInvocationKey({ ...event, sequence: 4 }));
    expect(mcpAppInvocationKey(event)).toBe(mcpAppInvocationKey({ ...event }));
  });
  it("keeps session context isolated and selects the latest call within that session", () => {
    const tools = [appTool("provider", "app", "ui://shared/app", "run")];
    const old: AgentEvent = { session_id: "s1", provider_id: "provider", server_tool_name: "run", app_context_available: true, tool_input: {} };
    const latest = { ...old, tool_input: { latest: true } };
    const other = { ...latest, session_id: "s2" };
    expect(findLatestAppInvocation([other, latest, old], tools, tools[0], "s1")).toBe(latest);
    expect(findAppForInvocationEvent(other, tools, "s1")).toBeUndefined();
    expect(findLatestAppInvocation([other], tools, tools[0], "s1")).toBeUndefined();
  });
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

  it("routes native first-party tools whose provider ID is omitted", () => {
    const app = appTool("", "document.open", "ui://llama-agent/apps/document-editor");
    const tools = [app, { ...app, name: "document.import", server_tool_name: "document.import" }];
    const event: AgentEvent = {
      session_id: "s", provider_id: "", server_tool_name: "document.open", app_context_available: true,
      tool_input: { document_uri: "resource://doc" },
    };
    expect(findLatestAppInvocation([event], tools, app, "s")).toBe(event);
    expect(findAppForInvocationEvent(event, tools, "s")).toBe(app);
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
