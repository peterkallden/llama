import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import ts from "typescript";

type Fields = Record<string, any>;
const settle = async () => { for (let i = 0; i < 15; i++) await Promise.resolve(); };

function app() {
  const nodes = new Map<string, Fields>();
  const node = (selector: string) => {
    if (!nodes.has(selector)) nodes.set(selector, { textContent: "", dataset: {}, replaceChildren() {}, append() {},
      click() { this.onclick?.(); }, showModal() { this.open = true; }, close() { this.open = false; },
      querySelector(child: string) { return node(child); },
      setAttribute() {},
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
    queueMicrotask(async () => {
      try { const value = await handler(request); if (value !== undefined) receive({ source: parent, data: { jsonrpc: "2.0", id: request.id, result: value } }); }
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
    setMarkdown(value: string) { this.markdown = value; }
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
    deliverRequest(id: number, method: string, params: Fields = {}) { receive({ source: parent, data: { jsonrpc: "2.0", id, method, params } }); },
    respond(id: number, value: unknown) { receive({ source: parent, data: { jsonrpc: "2.0", id, result: value } }); },
  };
}

const opened = { document_uri: "resource://r1", id: "report", kind: "report", revision: 1, title: "Report", markdown: "# Report\n\n## Results\n\nOriginal.\n", document: {} };

describe("document editor sandbox script", () => {
  it("reopens host-normalized saved Markdown without reviving a clean draft as unsaved", async () => {
    const view = app();
    view.handle((request) => request.params?.name === "document.save" ? { structuredContent: {
      document_uri: "resource://r2", revision: 2, diff: {},
    } } : undefined);
    view.deliver("ui/notifications/tool-result", { structuredContent: opened }); await settle();
    view.Editor.current.edit("Changed\n\n");
    view.node("#save").click(); await settle();
    expect(view.drafts.get("resource://r2")?.dirty).toBe(false);
    view.deliver("ui/notifications/tool-result", { structuredContent: {
      ...opened, document_uri: "resource://r2", revision: 2, markdown: "Changed\n",
    } }); await settle();
    expect(view.Editor.current.getMarkdown()).toBe("Changed\n");
    expect(view.node("#status").textContent).not.toContain("local draft restored");
  });
  it("restores a host draft from the UI extension's plain RPC result", async () => {
    const view = app();
    view.drafts.set(opened.document_uri, { markdown: "Restored local draft" });
    view.deliver("ui/notifications/tool-result", { structuredContent: opened }); await settle();
    expect(view.Editor.current.getMarkdown()).toBe("Restored local draft");
    expect(view.node("#status").textContent).toContain("local draft restored");
  });

  it("saves document title and Markdown as one revision and downloads the current unsaved Markdown through the host", async () => {
    const view = app();
    view.handle((request) => {
      if (request.method !== "tools/call") return undefined;
      if (request.params.name === "document.save") return { structuredContent: {
        document_uri: "resource://r2", markdown_uri: "resource://md2", revision: 2, reused: false, diff: {},
      } };
      return undefined;
    });
    view.deliver("ui/notifications/tool-result", { structuredContent: { ...opened, kind: "document", title: "First title" } }); await settle();
    const title = view.node("#document-title");
    title.value = "Changed title";
    title.oninput();
    view.Editor.current.edit("Saved body"); await settle();
    view.node("#save").click(); await settle();
    const save = view.requests.findLast((request) => request.params?.name === "document.save")!;
    expect(save.params.arguments.title).toBe("Changed title");
    expect(save.params.arguments.markdown).toBe("Saved body");
    expect(save.params.arguments.document_uri).toBe("resource://r1");

    title.value = "Unsaved title";
    title.oninput();
    view.Editor.current.edit("Exact unsaved body\n"); await settle();
    view.node("#download").click(); await settle();
    const download = view.requests.findLast((request) => request.method === "ui/llama-agent/document-download")!;
    expect(download.params).toEqual({ document_uri: "resource://r2", document_id: "report", title: "Unsaved title", markdown: "Exact unsaved body\n" });
  });

  it("closes a user document after an explicit save choice without starting a review turn", async () => {
    const view = app();
    view.handle((request) => {
      if (request.method !== "tools/call") return undefined;
      if (request.params.name === "document.save") return { structuredContent: {
        document_uri: "resource://r2", markdown_uri: "resource://md2", revision: 2, reused: false, diff: {},
      } };
      return undefined;
    });
    view.deliver("ui/notifications/tool-result", { structuredContent: { ...opened, kind: "document", title: "Notes" } }); await settle();
    view.Editor.current.edit("Saved note"); await settle();
    view.deliverRequest(81, "ui/llama-agent/close-document");
    expect(view.node("#switch-dialog").open).toBe(true);
    view.node('[data-switch-choice="save"]').click(); await new Promise((resolve) => setTimeout(resolve, 0)); await settle();
    expect(view.requests.some((request) => request.method === "tools/call" && request.params?.name === "document.save")).toBe(true);
    expect(view.requests.some((request) => request.method === "ui/message" && request.params?.action === "close-document")).toBe(true);
    expect(view.requests.some((request) => request.method === "ui/message" && request.params?.action === "review-document")).toBe(false);
    expect(view.requests.some((request) => request.id === 81 && request.result?.closed === true)).toBe(true);
  });

  it("waits for the matching tool result instead of fetching the document twice", async () => {
    const view = app();
    view.deliver("ui/notifications/tool-input", { arguments: { document_uri: opened.document_uri } });
    await settle();
    expect(view.requests.filter((request) => request.method === "tools/call")).toHaveLength(0);
    view.deliver("ui/notifications/tool-result", { structuredContent: opened });
    await settle();
    expect(view.Editor.current.getMarkdown()).toBe(opened.markdown);
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

  it("sends the saved revision and selected host block IDs as a typed review binding", async () => {
    const view = app();
    const documentValue = { sections: [{ id: "body", title: "Body", blocks: [{ id: "block-1", text: "Claim", author: "user", review_state: "unreviewed", support: "unverified" }] }] };
    const documentOpen = { ...opened, document: documentValue };
    view.handle((request) => {
      if (request.method !== "tools/call") return undefined;
      if (request.params.name === "document.save") return { structuredContent: {
        document_uri: "resource://r2", markdown_uri: "resource://md2", revision: 2, reused: false, diff: { blocks: [{ id: "block-1", text: "Changed" }] },
      } };
      if (request.params.name === "document.open") return { structuredContent: { ...documentOpen, document_uri: "resource://r2", revision: 2, markdown: "Changed" } };
      return undefined;
    });
    view.deliver("ui/notifications/tool-result", { structuredContent: documentOpen }); await settle();
    view.Editor.current.edit("Changed"); await settle();
    view.node("#review").click(); await settle();
    const message = view.requests.findLast((request) => request.method === "ui/message")!;
    expect(message.params.document_context).toEqual({ document_uri: "resource://r2", revision: 2, block_ids: [], base_document_uri: "resource://r1" });
    expect(message.params.action).toBe("review-document");
    expect(String(message.params.content.text)).toContain("Review changes between the bound saved document revisions");
    expect(String(message.params.content.text)).not.toContain("Document resource:");
  });

  it("saves before answering a host chat request and returns the new revision binding", async () => {
    const view = app();
    view.handle((request) => {
      if (request.method !== "tools/call") return undefined;
      if (request.params.name === "document.save") return { structuredContent: {
        document_uri: "resource://r2", revision: 2, reused: false, diff: { changed: true },
      } };
      if (request.params.name === "document.open") return { structuredContent: {
        ...opened, document_uri: "resource://r2", revision: 2, markdown: "Edited", document: {},
      } };
      return undefined;
    });
    view.deliver("ui/notifications/tool-result", { structuredContent: opened }); await settle();
    view.Editor.current.edit("Edited"); await settle();
    view.deliverRequest(77, "ui/llama-agent/prepare-chat"); await settle();
    const saveIndex = view.requests.findIndex((request) => request.params?.name === "document.save");
    const hostReplyIndex = view.requests.findIndex((request) => request.id === 77 && request.result?.document_uri === "resource://r2");
    expect(saveIndex).toBeGreaterThanOrEqual(0);
    expect(hostReplyIndex).toBeGreaterThan(saveIndex);
    expect(view.requests[hostReplyIndex].result).toEqual({ document_uri: "resource://r2", revision: 2, block_ids: [] });
  });

  it("queues review on close after saving and deduplicates a rapid second close", async () => {
    const view = app();
    view.handle((request) => {
      if (request.method === "tools/call" && request.params.name === "document.save") return { structuredContent: {
        document_uri: "resource://r2", revision: 2, reused: false, diff: { changed: true },
      } };
      if (request.method === "tools/call" && request.params.name === "document.open") return { structuredContent: {
        ...opened, document_uri: "resource://r2", revision: 2, markdown: "Edited",
        document: { sections: [{ blocks: [{ id: "block-2", text: "Edited" }] }] },
      } };
      return undefined;
    });
    view.deliver("ui/notifications/tool-result", { structuredContent: opened }); await settle();
    view.Editor.current.edit("Edited"); await settle();
    view.deliverRequest(76, "ui/llama-agent/close-document");
    view.deliverRequest(77, "ui/llama-agent/close-document");
    view.deliverRequest(78, "ui/llama-agent/close-document"); await settle();
    await new Promise((resolve) => setTimeout(resolve, 0)); await settle();
    const saves = view.requests.filter((request) => request.params?.name === "document.save");
    const reviews = view.requests.filter((request) => request.method === "ui/message" && request.params?.action === "review-document");
    const closes = view.requests.filter((request) => request.method === "ui/message" && request.params?.action === "close-document");
    expect(saves).toHaveLength(1);
    expect(reviews).toHaveLength(1);
    expect(closes).toHaveLength(1);
    expect(reviews[0].params.document_context).toEqual({ document_uri: "resource://r2", revision: 2, block_ids: [], base_document_uri: "resource://r1" });
    expect(reviews[0].params.operation_id).toBeTruthy();
    expect(view.requests.some((request) => request.id === 78 && request.result?.closed === true)).toBe(true);
    expect(view.node("#status").textContent).toContain("Review queued; closing editor");
  });

  it("does not acknowledge a host close if edits arrive after review registration", async () => {
    const view = app();
    view.handle((request) => {
      if (request.method === "tools/call" && request.params.name === "document.save") return { structuredContent: {
        document_uri: "resource://r2", revision: 2, reused: false, diff: { changed: true },
      } };
      if (request.method === "tools/call" && request.params.name === "document.open") return { structuredContent: {
        ...opened, document_uri: "resource://r2", revision: 2,
      } };
      if (request.method === "ui/message" && request.params.action === "review-document") {
        view.Editor.current.edit("Newer draft after review registration");
        return { queued: true };
      }
      return undefined;
    });
    view.deliver("ui/notifications/tool-result", { structuredContent: opened }); await settle();
    view.Editor.current.edit("Edited"); await settle();
    view.deliverRequest(79, "ui/llama-agent/close-document"); await settle();
    await new Promise((resolve) => setTimeout(resolve, 0)); await settle();
    expect(view.requests.some((request) => request.id === 79 && request.error)).toBe(true);
    expect(view.requests.some((request) => request.method === "ui/message" && request.params?.action === "close-document")).toBe(false);
    expect(view.Editor.current.getMarkdown()).toBe("Newer draft after review registration");
    expect(view.node("#editor").inert).toBe(false);
  });

  it("locks edits while close-review context is being registered and keeps the editor open on a changed draft", async () => {
    const view = app();
    let releaseOpen!: (value: unknown) => void;
    view.handle((request) => {
      if (request.method === "tools/call" && request.params.name === "document.save") return { structuredContent: {
        document_uri: "resource://r2", revision: 2, reused: false, diff: { changed: true },
      } };
      if (request.method === "tools/call" && request.params.name === "document.open") return new Promise((resolve) => { releaseOpen = resolve; });
      return undefined;
    });
    view.deliver("ui/notifications/tool-result", { structuredContent: opened }); await settle();
    view.Editor.current.edit("Edited"); await settle();
    view.deliverRequest(80, "ui/llama-agent/close-document"); await settle();
    expect(view.node("#editor").inert).toBe(true);
    view.Editor.current.edit("Changed while preparing review");
    releaseOpen({ structuredContent: { ...opened, document_uri: "resource://r2", revision: 2, document: {} } }); await settle();
    expect(view.requests.some((request) => request.method === "ui/message" && request.params?.action === "close-document")).toBe(false);
    expect(view.node("#status").textContent).toContain("changed while review context was being prepared");
    expect(view.node("#editor").inert).toBe(false);
  });
});
