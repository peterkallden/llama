import { readFileSync } from "node:fs";
import ts from "typescript";
import { expect, it } from "vitest";
import { createLatestRequestGate } from "../async/latest-request";

type Fields = Record<string, any>;
const settle = async () => { for (let i = 0; i < 50; i++) await Promise.resolve(); };

function workspace() {
  const nodes = new Map<string, Fields>();
  function element(): Fields {
    return { value: "", textContent: "", children: [], handlers: {}, classList: { toggle() {} },
      append(...children: Fields[]) { this.children.push(...children); },
      replaceChildren(...children: Fields[]) { this.children = children; },
      add(child: Fields) { this.children.push(child); }, setAttribute() {}, removeAttribute() {},
      querySelector() { return null; },
      querySelectorAll() { return this.children.flatMap((child: Fields) => [
        ...(child.type === "checkbox" ? [child] : []), ...child.querySelectorAll(),
      ]); },
      addEventListener(name: string, listener: (event: Fields) => unknown) { this.handlers[name] = listener; },
    };
  }
  const node = (selector: string) => {
    if (!nodes.has(selector)) nodes.set(selector, element());
    return nodes.get(selector)!;
  };
  let receive = (_event: Fields) => {};
  const requests: Fields[] = [];
  let handler = (request: Fields): unknown => {
    if (request.method === "ui/llama-agent/list-investigation-resources") return { resources: [], project_available: false };
    if (request.method !== "tools/call") return {};
    const args = request.params.arguments;
    if (request.params.name === "investigate.case.open") return { case: {
      case_id: args.case_id, revision: 1, title: args.case_id, source_id: "logs",
      window_start_ms: 100, window_end_ms: 1000, first_entity_id: "host:one",
    } };
    if (request.params.name === "investigate.case.items") return { items: [] };
    if (request.params.name === "investigate.source.list") return { sources: [] };
    return { nodes: [], edges: [] };
  };
  const respond = (id: number, result: unknown) => receive({ source: parent,
    data: { jsonrpc: "2.0", id, result } });
  const parent = { postMessage(request: Fields) {
    requests.push(request);
    if (request.id !== undefined) queueMicrotask(() => {
      const result = handler(request);
      if (result !== undefined) respond(request.id, result);
    });
  } };
  class Renderer {
    update() {}
    counts() { return { nodes: 0, edges: 0, complete: true, limitations: [] }; }
  }
  const source = readFileSync(new URL("./investigation-workspace.ts", import.meta.url), "utf8")
    .replace(/^import[^\n]*\n/gm, "");
  const script = ts.transpileModule(source, { compilerOptions: {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
  } }).outputText;
  new Function("RelationshipRenderer", "parseRelationshipVisualization", "createLatestRequestGate", "document",
    "window", "parent", "addEventListener", "Option", "crypto", script)(Renderer, (value: unknown) => value,
    createLatestRequestGate, { querySelector: node, createElement: element }, {}, parent,
    (_name: string, listener: typeof receive) => { receive = listener; },
    class { constructor(public text: string, public value: string) {} }, { randomUUID: () => "import-attempt" });
  return { node, requests, respond,
    deferTool(name: string, type?: string) {
      const previous = handler;
      handler = (request) => request.params?.name === name && (!type || request.params.arguments?.type === type)
        ? undefined : previous(request);
    },
    defer(method: string) { const previous = handler; handler = (request) => request.method === method ? undefined : previous(request); },
    open(caseId: string) { receive({ source: parent, data: { jsonrpc: "2.0", method: "ui/notifications/tool-input",
      params: { arguments: { case_id: caseId } } } }); },
  };
}

it("keeps source reads valid when the same case expands its graph", async () => {
  const view = workspace();
  view.deferTool("investigate.case.items", "sources");
  view.open("case-a"); await settle();
  const sourceRead = view.requests.find((request) => request.params?.name === "investigate.case.items" &&
    request.params.arguments.type === "sources");
  expect(view.requests.some((request) => request.params?.name === "investigate.graph.expand")).toBe(true);
  view.respond(sourceRead!.id, { items: [{ id: "provider-logs", kind: "search_connector", provider_id: "logs" }] });
  await settle();
  expect(view.node("#case-sources").children).toHaveLength(1);
});

it("does not attach an upload to another case opened while the upload was pending", async () => {
  const view = workspace();
  view.defer("ui/llama-agent/upload-investigation-resource");
  view.open("case-a"); await settle();
  const input = view.node("#resource-upload");
  input.files = [{ name: "evidence.txt", size: 10 }];
  const uploading = input.handlers.change({ currentTarget: input }); await settle();
  const upload = view.requests.find((request) => request.method === "ui/llama-agent/upload-investigation-resource");
  view.open("case-b"); await settle();
  view.respond(upload!.id, { resource_uri: "resource://session/uploaded" });
  await uploading; await settle();
  expect(view.requests.filter((request) => request.params?.name === "investigate.source.attach")).toHaveLength(0);
});

it("shows isolated actor candidates without requiring graph edges", async () => {
  const view = workspace();
  view.deferTool("investigate.case.items", "actors");
  view.open("case-a"); await settle();
  const read = view.requests.find((request) => request.params?.name === "investigate.case.items" &&
    request.params.arguments.type === "actors");
  view.respond(read!.id, { items: [{ id: "actor-a", display_name: "Alex", mention_ids: ["mention-a"] }], has_more: false });
  await settle();
  const actor = view.node("#actors").children[0];
  expect(actor.children[1].textContent).toBe("Alex");
  expect(actor.children[2].textContent).toContain("candidate");
  expect(actor.children[2].textContent).toContain("1 anchored mentions");
});

it("does not reopen the old case when an import completes after navigation", async () => {
  const view = workspace();
  view.deferTool("investigate.case.items", "sources");
  view.deferTool("investigate.log.import");
  view.open("case-a"); await settle();
  const read = view.requests.find((request) => request.params?.name === "investigate.case.items" &&
    request.params.arguments.type === "sources");
  view.respond(read!.id, { items: [{ id: "log-a", kind: "log_file", representation_uri: "resource://text/log-a" }] });
  await settle();
  const all = (item: Fields): Fields[] => [item, ...item.children.flatMap(all)];
  const button = all(view.node("#case-sources")).find((item) => item.textContent === "Import first page")!;
  const importing = button.handlers.click(); await settle();
  const request = view.requests.find((item) => item.params?.name === "investigate.log.import");
  expect(request?.params.arguments.case_id).toBe("case-a");
  view.open("case-b"); await settle();
  view.respond(request!.id, { case_id: "case-a", revision: 2, collected_events: 1 });
  await importing; await settle();
  expect(view.requests.filter((item) => item.params?.name === "investigate.case.open").map((item) => item.params.arguments.case_id))
    .toEqual(["case-a", "case-b"]);
});

it("requires actor selection and sends a user decision without starting a model turn", async () => {
  const view = workspace();
  view.deferTool("investigate.case.items", "actors");
  view.deferTool("investigate.association.review");
  view.open("case-a"); await settle();
  const read = view.requests.find((request) => request.params?.name === "investigate.case.items" && request.params.arguments.type === "actors");
  view.respond(read!.id, { items: [
    { id: "actor-a", display_name: "Alex", mention_ids: ["mention-a"] },
    { id: "actor-b", display_name: "Alex", mention_ids: ["mention-b"] },
  ] });
  await settle();
  expect(view.node("#same-actor").disabled).toBe(true);
  for (const row of view.node("#actors").children) {
    const checkbox = row.children[0]; checkbox.checked = true; checkbox.handlers.change();
  }
  expect(view.node("#same-actor").disabled).toBe(false);
  view.node("#association-comment").value = "I know both records refer to the same company.";
  view.node("#same-actor").handlers.click(); await settle();
  const request = view.requests.find((item) => item.params?.name === "investigate.association.review");
  expect(request?.params.arguments).toMatchObject({ case_id: "case-a", expected_revision: 1,
    actor_ids: ["actor-a", "actor-b"], kind: "identity", decision: "same_actor",
    comment: "I know both records refer to the same company." });
  expect(request?.params.arguments.reviewer_id).toBeUndefined();
  expect(view.requests.some((item) => item.method === "ui/message")).toBe(false);
  view.open("case-b"); await settle();
  view.respond(request!.id, { case_id: "case-a", revision: 2 }); await settle();
  expect(view.requests.filter((item) => item.params?.name === "investigate.case.open").map((item) => item.params.arguments.case_id))
    .toEqual(["case-a", "case-b"]);
});
