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
      });
      expect(sent?.url).toBe("/api/v1/mcp-apps/call");
      expect(sent?.init?.headers).toMatchObject({ Authorization: "Bearer secret" });
      expect(JSON.parse(String(sent?.init?.body))).toEqual({
        app_tool_name: "provider.explorer",
        tool_name: "provider.dataset.schema",
        arguments: { dataset: "dataset://local/sales" },
        session_id: "web-test",
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
      await new AgentApi().readMcpAppResource({ tool_name: "provider.explorer", resource_uri: "ui://provider/explorer", session_id: "s" });
      expect(sentUrl).toBe("/api/v1/mcp-apps/resource");
      expect(JSON.parse(sentBody)).toEqual({ tool_name: "provider.explorer", resource_uri: "ui://provider/explorer", session_id: "s" });
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
          app_tool_name: "provider.explorer", tool_name: "read", arguments: {}, session_id: "s",
        });
      } catch (error) { thrown = error; }
      expect(thrown).toBeInstanceOf(AgentApiError);
      expect(thrown).toMatchObject({ status: 422, payload: failure });
    } finally { globalThis.fetch = originalFetch; }
  });
});
