import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { invocationNotifications, isCurrentAppRequest, normalizeMcpAppCsp, sandboxProxyDocument } from "./mcp-app-host";

describe("MCP App sandbox host", () => {
  it.each([
    ["data_app", "dataset"], ["workflow_app", "plan_id"],
  ])("delivers invocation input, results and failures to bundled %s", (appName, argument) => {
    const source = readFileSync(new URL("../../../../tools/agent/mcp/agent-mcp-server-tool-registry.cpp", import.meta.url), "utf8");
    const line = source.split("\n").find((entry) => entry.includes(`const std::string ${appName} =`))!;
    const script = line.slice(line.indexOf("<script>") + 8, line.lastIndexOf("</script>"));
    const parent = { postMessage: vi.fn() };
    const input = { value: "" };
    const out = { textContent: "" };
    const document = { querySelector: (selector: string) => selector === "#out" ? out : input };
    let receive = (_event: Record<string, unknown>) => {};
    new Function("parent", "addEventListener", "document", script)(parent,
      (_type: string, listener: typeof receive) => { receive = listener; }, document);
    const deliver = (data: Record<string, unknown>) => receive({ source: parent, data: { jsonrpc: "2.0", ...data } });
    deliver({ id: 1, result: { protocolVersion: "2026-01-26" } });
    expect(parent.postMessage).toHaveBeenLastCalledWith(expect.objectContaining({ method: "ui/notifications/initialized" }), "*");
    deliver({ method: "ui/notifications/tool-input", params: { arguments: { [argument]: "host-reference" } } });
    expect(input.value).toBe("host-reference");
    deliver({ method: "ui/notifications/tool-result", params: { structuredContent: { rows: 3 }, content: [] } });
    expect(JSON.parse(out.textContent)).toEqual({ rows: 3 });
    deliver({ id: 4, result: { isError: true, content: [{ type: "text", text: "Host rejected the call" }] } });
    expect(out.textContent).toBe("Host rejected the call");
  });
  it.each([
    '<html><head lang="en"><title>App</title></head><body>content</body></html>',
    '<p>content without head</p>',
  ])("runs the generated proxy, inserts CSP and relays only its own view: %s", (appHtml) => {
    const html = sandboxProxyDocument();
    const script = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
    const parent = { postMessage: vi.fn() };
    const view = { setAttribute: vi.fn(), style: {}, srcdoc: "", contentWindow: { postMessage: vi.fn() } };
    const document = { createElement: vi.fn(() => view), body: { replaceChildren: vi.fn() } };
    let receive = (_event: Record<string, unknown>) => {};
    new Function("parent", "addEventListener", "document", script)(parent,
      (_type: string, listener: typeof receive) => { receive = listener; }, document);
    expect(parent.postMessage).toHaveBeenCalledWith({ jsonrpc: "2.0", method: "ui/notifications/sandbox-proxy-ready" }, "*");
    receive({ source: parent, data: { method: "ui/notifications/sandbox-resource-ready", params: {
      html: appHtml, csp: normalizeMcpAppCsp({ csp: { connectDomains: ["https://*.example.org"] } }),
    } } });
    expect(view.srcdoc).toContain('http-equiv="Content-Security-Policy"');
    expect(view.srcdoc).toContain("connect-src https://*.example.org");
    expect(view.srcdoc.match(/<head\b/gi)).toHaveLength(1);
    expect(view.setAttribute).toHaveBeenCalledWith("sandbox", "allow-scripts");
    const message = { jsonrpc: "2.0", method: "ping", id: 2 };
    parent.postMessage.mockClear();
    receive({ source: {}, origin: "null", data: message });
    expect(parent.postMessage).not.toHaveBeenCalled();
    receive({ source: view.contentWindow, origin: "null", data: message });
    expect(parent.postMessage).toHaveBeenCalledWith(message, "*");
    receive({ source: parent, data: { jsonrpc: "2.0", id: 2, result: {} } });
    expect(view.contentWindow.postMessage).toHaveBeenCalledOnce();
  });

  it("publishes standard result params after input and preserves tool failures", () => {
    const result = { isError: true, content: [{ type: "text", text: "failed" }], structuredContent: { code: "bad" } };
    const messages = invocationNotifications({ input: { dataset: "dataset://a" }, result });
    expect(messages.map((message) => message.method)).toEqual(["ui/notifications/tool-input", "ui/notifications/tool-result"]);
    expect(messages[1].params).toEqual(result);
    expect(invocationNotifications({ result })).toEqual([]);
  });
  it("emits browser-parseable proxy JavaScript with working head matching", () => {
    const html = sandboxProxyDocument();
    expect(html.startsWith('<!doctype html><meta charset="utf-8"><script>')).toBe(true);
    const script = html.slice(html.indexOf("<script>") + "<script>".length, html.lastIndexOf("</script>"));
    expect(() => new Function(script)).not.toThrow();
    expect(script).toContain("/^https:\\/\\/(?:\\*\\.)?");
    expect(script).toContain(String.raw`/<head\b[^>]*>/i`);
    expect(new RegExp(String.raw`<head\b[^>]*>`, "i").test("<head lang=\"en\">" )).toBe(true);
  });

  it("normalizes wildcard HTTPS domains and rejects malformed or non-HTTPS sources", () => {
    expect(normalizeMcpAppCsp({ ui: { csp: {
      connectDomains: ["https://*.example.org", "api.example.org", "http://bad.example.org", "https://user@example.org", "https://bad.example.org/path"],
    } } }).connectDomains).toEqual(["https://*.example.org", "https://api.example.org"]);
  });

  it("drops pending responses after the app generation or frame changes", () => {
    const firstFrame = {} as Window;
    const secondFrame = {} as Window;
    const request = { generation: 4, target: firstFrame };
    expect(isCurrentAppRequest(request, 4, firstFrame)).toBe(true);
    expect(isCurrentAppRequest(request, 5, secondFrame)).toBe(false);
    expect(isCurrentAppRequest(request, 4, secondFrame)).toBe(false);
  });
});
