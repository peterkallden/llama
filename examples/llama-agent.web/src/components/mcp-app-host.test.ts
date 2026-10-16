import { describe, expect, it } from "vitest";
import { isCurrentAppRequest, normalizeMcpAppCsp, sandboxProxyDocument } from "./mcp-app-host";

describe("MCP App sandbox host", () => {
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
