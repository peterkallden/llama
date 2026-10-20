import cytoscape, { type Core } from "cytoscape";
import "./investigation-workspace.css";

declare global { interface Window { __investigationWorkspace?: true } }
if (!window.__investigationWorkspace) {
  window.__investigationWorkspace = true;

  type Json = Record<string, any>;
  const q = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
  const status = q<HTMLDivElement>("#status");
  const pending = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void }>();
  let seq = 1;
  let generation = 0;
  let currentCase: Json | null = null;
  let selectedEntity = "";
  let selectedRelation = "";
  let selectedEvent = "";
  let graph: Core | null = null;
  let lastAppCallAt = 0;

  function rpc(method: string, params: Json = {}) {
    const id = seq++;
    parent.postMessage({ jsonrpc: "2.0", id, method, params }, "*");
    return new Promise<any>((resolve, reject) => pending.set(id, { resolve, reject }));
  }
  function setStatus(message: string, failed = false) {
    status.textContent = message;
    status.classList.toggle("error", failed);
  }
  function content(response: any): Json {
    if (response?.isError) throw new Error(response.content?.map((item: any) => item.text ?? "").join("\n") || "Host tool call failed.");
    let value = response?.structuredContent ?? response?.result?.structuredContent ?? response;
    if (typeof value === "string") { try { value = JSON.parse(value); } catch { return { text: value }; } }
    if (value?.content && Array.isArray(value.content) && !value.structuredContent) {
      const text = value.content.map((item: any) => item.text ?? "").join("\n");
      try { return JSON.parse(text); } catch { return { text }; }
    }
    return value && typeof value === "object" ? value : {};
  }
  async function call(tool: string, args: Json, token = generation): Promise<Json | null> {
    lastAppCallAt = Date.now();
    try {
      const result = content(await rpc("tools/call", { name: tool, arguments: args }));
      lastAppCallAt = Date.now();
      return token === generation ? result : null;
    } catch (error) { lastAppCallAt = Date.now(); throw error; }
  }
  function msValue(selector: string) {
    const value = q<HTMLInputElement>(selector).value;
    const time = value ? new Date(value).getTime() : 0;
    return Number.isFinite(time) ? time : 0;
  }
  function setDate(selector: string, ms: number) {
    if (ms > 0) q<HTMLInputElement>(selector).value = new Date(ms).toISOString().slice(0, 16);
  }
  function showCase(value: Json) {
    currentCase = value;
    selectedEntity = ""; selectedRelation = ""; selectedEvent = "";
    const revision = Number(value.revision ?? 1);
    q("#case-summary").textContent = `${value.title} · r${revision} · ${value.case_id}`;
    q<HTMLInputElement>("#case-id").value = value.case_id;
    q<HTMLInputElement>("#case-title").value = value.title;
    q<HTMLInputElement>("#case-question").value = value.question ?? "";
    q<HTMLInputElement>("#source-id").value = value.source_id;
    setDate("#filter-start", value.window_start_ms); setDate("#filter-end", value.window_end_ms);
    q<HTMLButtonElement>("#save-hypothesis").disabled = false;
    q<HTMLButtonElement>("#compose-report").disabled = false;
    q("#report-result").textContent = "No report created for this revision.";
    renderClaims(value.claims ?? []);
    renderGraph({ nodes: [], edges: [], case_revision: revision, truncated: false, limitations: [] });
    setStatus(`Case ${value.case_id} opened at revision ${revision}.`);
  }
  async function openCase(id: string) {
    const token = ++generation;
    try {
      setStatus("Loading case…");
      const result = await call("investigation.case.open", { case_id: id }, token);
      if (!result) return;
      if (!result.case) throw new Error("The host returned no case.");
      showCase(result.case);
      await expand();
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  function renderGraph(value: Json) {
    const elements = [
      ...(value.nodes ?? []).map((node: Json) => ({ data: { id: node.id, label: node.label, kind: node.type } })),
      ...(value.edges ?? []).map((edge: Json) => ({ data: { id: edge.id, source: edge.source, target: edge.target,
        label: `${edge.event_count} event${edge.event_count === 1 ? "" : "s"}`, classification: edge.classification } })),
    ];
    if (!graph) {
      graph = cytoscape({ container: q<HTMLDivElement>("#graph"), elements,
        style: [
          { selector: "node", style: { "background-color": "#3979b8", label: "data(label)", color: "#e9f2ff",
            "font-size": "11px", "text-wrap": "wrap", "text-max-width": "100px", width: 36, height: 36, "border-width": 2, "border-color": "#79b7f1" } },
          { selector: "node[kind = 0]", style: { "background-color": "#a06fbd", "border-color": "#d4a6ee" } },
          { selector: "node[kind = 1]", style: { "background-color": "#3979b8" } },
          { selector: "edge", style: { width: 2, "line-color": "#6f8eaf", "target-arrow-color": "#6f8eaf",
            "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(label)", color: "#b9cce0", "font-size": "9px" } },
          { selector: "edge:selected", style: { "line-color": "#f0b95e", "target-arrow-color": "#f0b95e", width: 4 } },
          { selector: "node:selected", style: { "border-color": "#ffd27d", "border-width": 4 } },
        ], layout: { name: "cose", animate: false, fit: true, padding: 32 }, userZoomingEnabled: true, userPanningEnabled: true });
      graph.on("tap", "node", (event) => { selectedEntity = event.target.id(); selectedRelation = ""; void expand(selectedEntity); });
      graph.on("tap", "edge", (event) => { selectedRelation = event.target.id(); selectedEntity = ""; void loadEvents("", selectedRelation); });
    } else {
      graph.elements().remove();
      graph.add(elements);
      graph.layout({ name: "cose", animate: false, fit: true, padding: 32 }).run();
    }
    q("#graph-status").textContent = `${value.nodes?.length ?? 0} objects · ${value.edges?.length ?? 0} relations${value.truncated ? " · truncated by bounds" : ""}`;
  }
  async function expand(center = selectedEntity) {
    if (!currentCase) return;
    const token = generation;
    try {
      const value = await call("investigation.graph.expand", {
        case_id: currentCase.case_id, case_revision: currentCase.revision,
        center_entity_id: center || currentCase.entities?.[0]?.id || "",
        max_depth: 2, max_nodes: 100, max_edges: 200,
        window_start_ms: msValue("#filter-start") || currentCase.window_start_ms,
        window_end_ms: msValue("#filter-end") || currentCase.window_end_ms,
      }, token);
      if (!value) return;
      renderGraph(value);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function loadEvents(entityId: string, relationId = "") {
    if (!currentCase) return;
    const token = generation;
    try {
      setStatus("Loading source events…");
      const args: Json = { case_id: currentCase.case_id, limit: 200,
        window_start_ms: msValue("#filter-start") || currentCase.window_start_ms,
        window_end_ms: msValue("#filter-end") || currentCase.window_end_ms };
      if (relationId) args.relation_id = relationId; else args.entity_id = entityId;
      const value = await call("investigation.events", args, token);
      if (!value) return;
      renderTimeline(value.events ?? [], Boolean(value.truncated));
      setStatus(`${value.events?.length ?? 0} evidence events${value.truncated ? " · more events are available" : ""}.`);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  function renderTimeline(events: Json[], truncated: boolean) {
    const list = q<HTMLDivElement>("#timeline"); list.replaceChildren();
    events.sort((a, b) => Number(a.time?.event_time_ms ?? 0) - Number(b.time?.event_time_ms ?? 0));
    for (const event of events) {
      const button = document.createElement("button"); button.className = `event${selectedEvent === event.id ? " selected" : ""}`;
      const time = document.createElement("time"); time.textContent = event.time?.has_event_time
        ? new Date(event.time.event_time_ms).toLocaleString() : "Unknown event time";
      const title = document.createElement("strong"); title.textContent = event.event_type || "Event";
      const id = document.createElement("small"); id.textContent = `Source document: ${event.source_document_id}`;
      button.append(time, title, id); button.addEventListener("click", () => { selectedEvent = event.id; showEvidence(event); });
      list.append(button);
    }
    if (truncated) { const note = document.createElement("p"); note.textContent = "Event list reached its display bound."; list.append(note); }
  }
  function showEvidence(event: Json) {
    selectedEvent = event.id;
    q("#evidence").textContent = JSON.stringify({
      event_id: event.id, source_document_id: event.source_document_id, event_type: event.event_type,
      event_time: event.time, entities: event.entity_ids, raw_resource_uri: event.raw_resource_uri,
      normalized_resource_uri: event.normalized_resource_uri,
      note: "This is a source-linked observation. It does not by itself establish causality.",
    }, null, 2);
  }
  function renderClaims(claims: Json[]) {
    const target = q<HTMLDivElement>("#claims"); target.replaceChildren();
    for (const claim of claims) {
      const box = document.createElement("div"); box.className = "claim";
      const state = document.createElement("small"); state.textContent = String(claim.state) === "2" ? "Reviewed conclusion" : "Hypothesis";
      const text = document.createElement("div"); text.textContent = claim.text;
      box.append(state, text); target.append(box);
    }
  }
  async function saveHypothesis() {
    if (!currentCase) return;
    const text = q<HTMLTextAreaElement>("#hypothesis").value.trim();
    if (!text) return;
    const token = generation;
    try {
      const value = await call("investigation.claim.hypothesize", {
        case_id: currentCase.case_id, expected_revision: currentCase.revision,
        claim_id: `hypothesis-${Date.now()}`, text,
        ...(selectedEvent ? { supporting_event_ids: [selectedEvent] } : {}),
      }, token);
      if (!value) return;
      q<HTMLTextAreaElement>("#hypothesis").value = "";
      await openCase(currentCase.case_id);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function composeReport() {
    if (!currentCase) return;
    const token = generation;
    try {
      const result = await call("investigation.report.compose", {
        case_id: currentCase.case_id, expected_revision: currentCase.revision,
      }, token);
      if (!result) return;
      q("#report-result").textContent = `${result.complete ? "No recorded gaps" : "Partial / qualified"} · frozen revision ${result.case_revision}\n${result.report_uri}\n${result.content_fingerprint}`;
      setStatus(`Report frozen from case revision ${result.case_revision}. The resource can be opened through the existing document workflow.`);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function createOrOpen() {
    const caseId = q<HTMLInputElement>("#case-id").value.trim();
    if (caseId) { q("#case-form").setAttribute("hidden", ""); await openCase(caseId); return; }
    const start = msValue("#window-start"), end = msValue("#window-end");
    try {
      const value = await call("investigation.case.create", {
        title: q<HTMLInputElement>("#case-title").value.trim(),
        question: q<HTMLInputElement>("#case-question").value.trim(),
        source_id: q<HTMLInputElement>("#source-id").value.trim(),
        window_start_ms: start, window_end_ms: end,
      });
      if (!value) return;
      q("#case-form").setAttribute("hidden", ""); await openCase(value.case_id);
    } catch (error) { setStatus(error instanceof Error ? error.message : String(error), true); }
  }

  q("#create-case").addEventListener("click", () => { q("#case-form").removeAttribute("hidden"); q<HTMLInputElement>("#case-id").value = ""; });
  q("#open-case").addEventListener("click", () => { q("#case-form").removeAttribute("hidden"); q<HTMLInputElement>("#case-id").focus(); });
  q("#case-submit").addEventListener("click", () => void createOrOpen());
  q("#case-cancel").addEventListener("click", () => q("#case-form").setAttribute("hidden", ""));
  q("#apply-filter").addEventListener("click", () => void expand());
  q("#save-hypothesis").addEventListener("click", () => void saveHypothesis());
  q("#compose-report").addEventListener("click", () => void composeReport());
  q<HTMLTextAreaElement>("#hypothesis").addEventListener("input", (event) => {
    q<HTMLButtonElement>("#save-hypothesis").disabled = !currentCase || !(event.target as HTMLTextAreaElement).value.trim();
  });
  addEventListener("message", (event) => {
    if (event.source !== parent) return;
    const message = event.data;
    if (message?.jsonrpc !== "2.0") return;
    if (typeof message.id === "number" && pending.has(message.id)) {
      const request = pending.get(message.id)!; pending.delete(message.id);
      if (message.error) request.reject(new Error(String(message.error.message ?? "Host request failed.")));
      else request.resolve(message.result);
      return;
    }
    if (message.method === "ui/notifications/tool-input") {
      const args = message.params?.arguments ?? {};
      if (typeof args.case_id === "string") { q<HTMLInputElement>("#case-id").value = args.case_id; void openCase(args.case_id); }
    }
    if (message.method === "ui/notifications/tool-result" && message.params?.structuredContent?.case_id) {
      if (Date.now() - lastAppCallAt > 1000) void openCase(message.params.structuredContent.case_id);
    }
  });
  void rpc("ui/initialize", { protocolVersion: "2026-01-26", appInfo: { name: "llama-agent-investigation-workspace", version: "0.1" }, appCapabilities: { tools: {} } })
    .then(() => { parent.postMessage({ jsonrpc: "2.0", method: "ui/notifications/initialized", params: {} }, "*"); setStatus("Ready. Open a case or create one."); })
    .catch((error) => setStatus(error instanceof Error ? error.message : String(error), true));
}
