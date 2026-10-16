import { describe, expect, it } from "vitest";
import { AgentApi, parseSseBlock, parseSseBuffer } from "./agent-api";

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
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      sentUrl = String(input);
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;
    try {
      await new AgentApi().readMcpAppResource({ app_tool_name: "provider.explorer", resource_uri: "ui://provider/explorer", session_id: "s" });
      expect(sentUrl).toBe("/api/v1/mcp-apps/resource");
    } finally { globalThis.fetch = originalFetch; }
  });
});
