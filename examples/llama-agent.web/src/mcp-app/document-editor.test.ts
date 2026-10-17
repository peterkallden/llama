import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import ts from "typescript";

type Fields = Record<string, any>;
const settle = async () => { for (let i = 0; i < 15; i++) await Promise.resolve(); };

function app() {
  const nodes = new Map<string, Fields>();
  const node = (selector: string) => {
    if (!nodes.has(selector)) nodes.set(selector, { textContent: "", dataset: {}, replaceChildren() {}, append() {},
      addEventListener(event: string, handler: () => void) { this[event] = handler; } });
    return nodes.get(selector)!;
  };
  let receive = (_event: Fields) => {};
  const requests: Fields[] = [];
  const drafts = new Map<string, Fields>();
  let handler = (request: Fields): unknown => {
    if (request.method === "ui/llama-agent/document-draft") {
      const key = request.params.document_uri;
      if (request.params.action === "write") drafts.set(key, request.params.draft);
      return { persistent: true, draft: drafts.get(key) };
    }
    return {};
  };
  const parent = { postMessage(request: Fields) {
    requests.push(request);
    if (request.id === undefined) return;
    queueMicrotask(() => {
      try { const value = handler(request); if (value !== undefined) receive({ source: parent, data: { jsonrpc: "2.0", id: request.id, result: value } }); }
      catch (error) { receive({ source: parent, data: { jsonrpc: "2.0", id: request.id, error: { message: String(error) } } }); }
    });
  } };
  class Editor {
    static current: Editor;
    markdown: string;
    listener = () => {};
    constructor(options: Fields) {
      expect(options.toolbarItems.every(Array.isArray)).toBe(true);
      this.markdown = options.initialValue; Editor.current = this;
    }
    getMarkdown() { return this.markdown; }
    destroy() {}
    on(_name: string, listener: () => void) { this.listener = listener; }
    edit(text: string) { this.markdown = text; this.listener(); }
  }
  const source = readFileSync(new URL("./document-editor.ts", import.meta.url), "utf8").replace(/^import[^\n]*\n/gm, "");
  const script = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;
  new Function("Editor", "document", "window", "parent", "crypto", "appCss", script)(Editor,
    { querySelector: node, createElement: () => node("style"), head: { append() {} } },
    { addEventListener(name: string, listener: typeof receive) { if (name === "message") receive = listener; } }, parent,
    { getRandomValues: (bytes: Uint8Array) => bytes.fill(requests.length) }, "");
  return {
    node, requests, drafts, Editor,
    handle(next: typeof handler) { const previous = handler; handler = (request) => next(request) ?? previous(request); },
    deliver(method: string, params: Fields) { receive({ source: parent, data: { jsonrpc: "2.0", method, params } }); },
    respond(id: number, value: unknown) { receive({ source: parent, data: { jsonrpc: "2.0", id, result: value } }); },
  };
}

const opened = { document_uri: "resource://r1", id: "report", revision: 1, title: "Report", markdown: "# Report\n\n## Results\n\nOriginal.\n", document: {} };

describe("document editor sandbox script", () => {
  it("restores a host draft from the UI extension's plain RPC result", async () => {
    const view = app();
    view.drafts.set(opened.document_uri, { markdown: "Restored local draft" });
    view.deliver("ui/notifications/tool-result", { structuredContent: opened }); await settle();
    expect(view.Editor.current.getMarkdown()).toBe("Restored local draft");
    expect(view.node("#status").textContent).toContain("local draft restored");
  });

  it("opens without sandbox localStorage and ignores a late open after user editing", async () => {
    const view = app();
    view.deliver("ui/notifications/tool-result", { structuredContent: opened }); await settle();
    expect(view.Editor.current.getMarkdown()).toBe(opened.markdown);
    view.handle((request) => request.method === "tools/call" ? undefined : undefined);
    view.node("#reload").click();
    const openRequest = view.requests.findLast((request) => request.method === "tools/call")!;
    view.Editor.current.edit("Newer draft"); await settle();
    view.respond(openRequest.id, { structuredContent: opened }); await settle();
    expect(view.Editor.current.getMarkdown()).toBe("Newer draft");
    expect(view.drafts.get(opened.document_uri)?.markdown).toBe("Newer draft");
  });

  it("retries the same save operation and preserves edits made while it is in flight", async () => {
    const view = app();
    view.deliver("ui/notifications/tool-result", { structuredContent: opened }); await settle();
    view.Editor.current.edit("First edit");
    let fail = true;
    view.handle((request) => {
      if (request.method === "tools/call" && request.params.name === "document.save") {
        if (fail) throw new Error("connection lost");
        return { structuredContent: { document_uri: "resource://r2", revision: 2, diff: {} } };
      }
    });
    view.node("#save").click(); await settle();
    const first = view.requests.findLast((request) => request.params?.name === "document.save")!;
    view.Editor.current.edit("Newer edit");
    fail = false; view.node("#save").click(); await settle();
    const retry = view.requests.findLast((request) => request.params?.name === "document.save")!;
    expect(retry.params.arguments).toEqual(first.params.arguments);
    expect(view.Editor.current.getMarkdown()).toBe("Newer edit");
    expect(view.drafts.get("resource://r2")?.markdown).toBe("Newer edit");
    expect(view.node("#status").textContent).toContain("newer local edits");
  });
});
