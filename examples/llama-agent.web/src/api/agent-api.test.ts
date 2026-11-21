import { describe, expect, it } from "vitest";
import { AgentApi, AgentApiError, parseSseBlock, parseSseBuffer } from "./agent-api";

describe("web event framing", () => {
  it("parses multiline SSE data and preserves id", () => {
    expect(parseSseBlock("id: 42\nevent: agent\ndata: {\"type\":\ndata: \"turn.completed\"}"))
      .toEqual({ id: "42", event: "agent", data: "{\"type\":\n\"turn.completed\"}" });
  });

  it("keeps an incomplete SSE block for the next network chunk", () => {
    expect(parseSseBuffer("id: 1\ndata: {}\n\nid: 2\ndata: {"))
      .toEqual({ blocks: ["id: 1\ndata: {}"], rest: "id: 2\ndata: {" });
  });
});

describe("project, chat and resource navigation API", () => {
  it("uses host endpoints and preserves project scope for navigation and resource reads", async () => {
    const originalFetch = globalThis.fetch;
    const calls: Array<{ url: string; init?: RequestInit }> = [];
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(input), init });
      return new Response(JSON.stringify({ projects: [], chats: [], messages: [], resources: [] }), { status: 200 });
    }) as typeof fetch;
    try {
      const api = new AgentApi("/api/v1", "session-token");
      await api.projects();
      await api.createProject("Incident review");
      await api.chats("project-7");
      await api.createChat("web-8", "project-7");
      await api.migrateLegacyChat("web-legacy");
      await api.chatMessages("web-8", "project-7", "25", 25);
      await api.resources({ scope: "session", projectId: "project-7", sessionId: "web-8" });
      expect(calls.map((item) => item.url)).toEqual([
        "/api/v1/projects", "/api/v1/projects", "/api/v1/chats?project_id=project-7&limit=30",
        "/api/v1/chats", "/api/v1/chats/migrate-legacy", "/api/v1/chats/web-8/messages?project_id=project-7&cursor=25&limit=25",
        "/api/v1/resources?scope=session&project_id=project-7&session_id=web-8&limit=30",
      ]);
      expect(calls.every((item) => (item.init?.headers as Record<string, string>).Authorization === "Bearer session-token")).toBe(true);
      expect(JSON.parse(String(calls[3].init?.body))).toMatchObject({ session_id: "web-8", project_id: "project-7" });
    } finally { globalThis.fetch = originalFetch; }
  });
});

describe("host-brokered MCP App API", () => {
  it("keeps the app origin and arguments explicit for brokered calls", async () => {
    const originalFetch = globalThis.fetch;
    let sent: { url: string; init?: RequestInit } | undefined;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      sent = { url: String(input), init };
      return new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "Content-Type": "application/json" } });
    }) as typeof fetch;
    try {
      await new AgentApi("/api/v1", "secret").callMcpAppTool({
        app_tool_name: "provider.explorer",
        tool_name: "provider.dataset.schema",
        arguments: { dataset: "dataset://local/sales" },
        session_id: "web-test",
        project_id: "project-1",
      });
      expect(sent?.url).toBe("/api/v1/mcp-apps/call");
      expect(sent?.init?.headers).toMatchObject({ Authorization: "Bearer secret" });
      expect(JSON.parse(String(sent?.init?.body))).toEqual({
        app_tool_name: "provider.explorer",
        tool_name: "provider.dataset.schema",
        arguments: { dataset: "dataset://local/sales" },
        session_id: "web-test",
        project_id: "project-1",
      });
    } finally { globalThis.fetch = originalFetch; }
  });

  it("loads app UI resources through the host API, not a browser URL", async () => {
    const originalFetch = globalThis.fetch;
    let sentUrl = "";
    let sentBody = "";
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      sentUrl = String(input);
      sentBody = String(init?.body ?? "");
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;
    try {
      await new AgentApi().readMcpAppResource({ tool_name: "provider.explorer", resource_uri: "ui://provider/explorer", session_id: "s", project_id: "p" });
      expect(sentUrl).toBe("/api/v1/mcp-apps/resource");
      expect(JSON.parse(sentBody)).toEqual({ tool_name: "provider.explorer", resource_uri: "ui://provider/explorer", session_id: "s", project_id: "p" });
    } finally { globalThis.fetch = originalFetch; }
  });

  it("preserves structured daemon failures for MCP Apps callers", async () => {
    const originalFetch = globalThis.fetch;
    const failure = { ok: false, tool: { error: { code: "tool.denied", class: "policy", retryable: false, safe_summary: "Not allowed." } } };
    globalThis.fetch = (async () => new Response(JSON.stringify(failure), { status: 422 })) as typeof fetch;
    try {
      let thrown: unknown;
      try {
        await new AgentApi().callMcpAppTool({
          app_tool_name: "provider.explorer", tool_name: "read", arguments: {}, session_id: "s", project_id: "p",
        });
      } catch (error) { thrown = error; }
      expect(thrown).toBeInstanceOf(AgentApiError);
      expect((thrown as Error).message).toBe("Not allowed.");
      expect(thrown).toMatchObject({ status: 422, payload: failure });
    } finally { globalThis.fetch = originalFetch; }
  });
});
