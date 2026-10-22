import "./investigation-workspace.css";
import { RelationshipRenderer } from "../visualization/relationship-renderer";
import { parseRelationshipVisualization } from "../visualization/contract";

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
  let graph: RelationshipRenderer | null = null;
  let lastAppCallAt = 0;
  let caseListCursor = "";
  let reportCursor = "";
  let eventsCursor = "";
  let eventQuery: Json = {};
  let snapshotPending = false;
  let renderedVisualizationUri = "";
  for (const selector of ["#filter-start", "#filter-end", "#apply-filter"])
    q<HTMLInputElement | HTMLButtonElement>(selector).disabled = true;

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
    const date = new Date(ms);
    q<HTMLInputElement>(selector).value = Number.isFinite(ms) && Number.isFinite(date.getTime())
      ? new Date(ms - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16) : "";
  }
  function resetEvents() {
    eventsCursor = ""; eventQuery = {}; selectedEvent = "";
    q("#timeline").replaceChildren(); q("#evidence").textContent = "";
    q<HTMLButtonElement>("#events-more").hidden = true;
    q<HTMLButtonElement>("#events-more").disabled = false;
  }
  function showCase(value: Json) {
    currentCase = value;
    renderedVisualizationUri = "";
    selectedEntity = ""; selectedRelation = ""; selectedEvent = "";
    q<HTMLInputElement>("#visualization-uri").value = "";
    resetEvents();
    const revision = Number(value.revision ?? 1);
    q("#case-summary").textContent = `${value.title} · r${revision} · ${value.case_id}`;
    q<HTMLInputElement>("#case-id").value = value.case_id;
    q<HTMLInputElement>("#case-title").value = value.title;
    q<HTMLInputElement>("#case-question").value = value.question ?? "";
    q<HTMLInputElement>("#source-id").value = value.source_id;
    setDate("#filter-start", value.window_start_ms); setDate("#filter-end", value.window_end_ms);
    for (const selector of ["#filter-start", "#filter-end", "#apply-filter"])
      q<HTMLInputElement | HTMLButtonElement>(selector).disabled = false;
    q<HTMLButtonElement>("#save-hypothesis").disabled = false;
    q<HTMLButtonElement>("#compose-report").disabled = false;
    q("#report-result").textContent = "No report created for this revision.";
    reportCursor = "";
    q<HTMLButtonElement>("#frozen-reports-more").hidden = true;
    const reportStatus = document.createElement("p"); reportStatus.textContent = "Loading indexed reports…";
    q("#frozen-reports").replaceChildren(reportStatus);
    renderClaims([]);
    renderGraph({ nodes: [], edges: [], case_revision: revision, truncated: false, limitations: [] }, true);
    setStatus(`Case ${value.case_id} opened at revision ${revision}.`);
  }
  async function loadCaseClaims() {
    if (!currentCase) return;
    const token = generation;
    try {
      const value = await call("investigation.case.items", { case_id: currentCase.case_id,
        case_revision: currentCase.revision, type: "claims", limit: 100 }, token);
      if (value) renderClaims(value.items ?? []);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function loadReports(continuation = false) {
    if (!currentCase) return;
    if (continuation && !reportCursor) return;
    const token = generation;
    const target = q<HTMLDivElement>("#frozen-reports");
    try {
      const value = await call("investigation.case.items", { case_id: currentCase.case_id,
        case_revision: currentCase.revision, type: "reports", limit: 50,
        ...(continuation ? { cursor: reportCursor } : {}) }, token);
      if (!value || token !== generation) return;
      if (!continuation) target.replaceChildren();
      reportCursor = String(value.next_cursor ?? "");
      q<HTMLButtonElement>("#frozen-reports-more").hidden = !value.has_more;
      const reports = Array.isArray(value.items) ? value.items : [];
      if (!reports.length && !continuation) {
        const empty = document.createElement("p"); empty.textContent = "No frozen reports are indexed for this revision.";
        target.append(empty); return;
      }
      for (const report of reports) {
        const button = document.createElement("button"); button.className = "frozen-report";
        const title = document.createElement("strong"); title.textContent = `Frozen report · revision ${report.case_revision}`;
        const state = document.createElement("small"); state.className = report.complete ? "" : "qualified";
        state.textContent = report.complete ? "No recorded gaps" : `Partial / qualified${Array.isArray(report.limitations) && report.limitations.length ? ` · ${report.limitations.length} limitation(s)` : ""}`;
        const fingerprint = document.createElement("small"); fingerprint.textContent = String(report.content_fingerprint ?? "");
        const uri = document.createElement("small"); uri.textContent = String(report.report_uri ?? "");
        button.append(title, state, fingerprint, uri);
        button.addEventListener("click", async () => {
          const documentUri = String(report.report_uri ?? "");
          if (!documentUri) return;
          try {
            await rpc("ui/message", { action: "open-investigation-report", content: { type: "text", text: documentUri } });
            setStatus(`Opening frozen report from revision ${report.case_revision}…`);
          } catch (error) { setStatus(error instanceof Error ? error.message : String(error), true); }
        });
        target.append(button);
      }
    } catch (error) {
      if (token !== generation) return;
      target.replaceChildren();
      const failure = document.createElement("p"); failure.textContent = `Could not load frozen reports: ${error instanceof Error ? error.message : String(error)}`;
      target.append(failure);
    }
  }
  async function openCase(id: string) {
    const token = ++generation;
    try {
      setStatus("Loading case…");
      const result = await call("investigation.case.open", { case_id: id }, token);
      if (!result) return;
      if (!result.case) throw new Error("The host returned no case.");
      showCase(result.case);
      q("#case-form").setAttribute("hidden", "");
      await loadCaseClaims();
      await loadReports();
      if (token !== generation) return;
      if (result.case.first_entity_id) await expand(result.case.first_entity_id);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function listCases(after = "") {
    const token = ++generation;
    try {
      setStatus("Loading cases in this scope…");
      const value = await call("investigation.case.list", { limit: 50, after_case_id: after }, token);
      if (!value) return;
      const target = q<HTMLDivElement>("#case-list-items");
      if (!after) target.replaceChildren();
      caseListCursor = String(value.next_cursor ?? "");
      for (const item of value.cases ?? []) {
        const button = document.createElement("button"); button.className = "case-item";
        const title = document.createElement("strong"); title.textContent = String(item.title || item.case_id);
        const meta = document.createElement("small"); meta.textContent = `${item.case_id} · r${item.revision} · ${item.source_id}`;
        button.append(title, meta); button.addEventListener("click", () => {
          q("#case-list").setAttribute("hidden", ""); void openCase(String(item.case_id));
        }); target.append(button);
      }
      q("#case-list").removeAttribute("hidden");
      q<HTMLButtonElement>("#case-list-more").hidden = !caseListCursor;
      setStatus(`${target.children.length} cases visible in this scope.`);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  function renderGraph(value: Json, replace = false) {
    if (!graph) {
      graph = new RelationshipRenderer(q<HTMLDivElement>("#graph"), { onSelect: (selection) => {
        if (!currentCase) return;
        resetEvents();
        if (selection.kind === "node") {
          selectedEntity = selection.id; selectedRelation = ""; void expand(selectedEntity);
        } else {
          selectedRelation = selection.id; selectedEntity = ""; void loadEvents("", selectedRelation);
        }
      } });
    }
    const limitations = Array.isArray(value.limitations) ? value.limitations.map(String) : [];
    const truncated = value.truncated === true;
    if (truncated && limitations.length === 0) limitations.push("The host truncated this graph to its configured bounds.");
    const isVisualization = value.schema_version === 1 && value.kind === "relationship";
    graph.update(isVisualization ? value : { schema_version: 1, kind: "relationship", title: "Investigation relations",
      data_revision: String(value.case_revision ?? currentCase?.revision ?? ""),
      nodes: (value.nodes ?? []).map((node: Json) => ({ id: String(node.id),
        label: String(node.label ?? node.display_name ?? node.id), kind: String(node.type ?? "") })),
      edges: (value.edges ?? []).map((edge: Json) => ({ id: String(edge.id), source: String(edge.source),
        target: String(edge.target), label: `${edge.event_count} event${edge.event_count === 1 ? "" : "s"}`, directed: true })),
      complete: !truncated, truncated, limitations,
    }, { replace });
    const counts = graph.counts();
    q("#graph-status").textContent = `${counts.nodes} objects · ${counts.edges} relations${counts.truncated ? " · truncated by bounds" : !counts.complete ? " · incomplete projection" : ""}${counts.limitations.length ? ` · ${counts.limitations.join(" ")}` : ""}`;
    q<HTMLButtonElement>("#save-snapshot").disabled = snapshotPending || counts.nodes === 0;
  }
  function showVisualization(result: Json) {
    const visualization = parseRelationshipVisualization(result.visualization);
    // Keep the current case intact unless both validation and rendering succeed.
    renderGraph(visualization, true);
    ++generation;
    currentCase = null;
    selectedEntity = ""; selectedRelation = ""; selectedEvent = "";
    resetEvents(); renderClaims([]);
    q<HTMLButtonElement>("#save-hypothesis").disabled = true;
    q<HTMLButtonElement>("#compose-report").disabled = true;
    q("#report-result").textContent = "No report is attached to this visualization.";
    q<HTMLInputElement>("#visualization-uri").value = "";
    for (const selector of ["#filter-start", "#filter-end", "#apply-filter"])
      q<HTMLInputElement | HTMLButtonElement>(selector).disabled = true;
    q<HTMLInputElement>("#filter-start").value = "";
    q<HTMLInputElement>("#filter-end").value = "";
    q("#case-summary").textContent = visualization.title;
    const uri = String(result.visualization_uri ?? "");
    renderedVisualizationUri = uri;
    if (uri) q<HTMLInputElement>("#visualization-uri").value = uri;
    q<HTMLButtonElement>("#save-snapshot").disabled = !uri || !graph || graph.counts().nodes === 0;
    setStatus(`${uri ? `Opened ${uri}` : "Opened saved visualization"}${visualization.complete === false ? " · projection is incomplete" : ""}.`);
  }
  async function expand(center = selectedEntity) {
    if (!currentCase) return;
    const token = ++generation;
    try {
      const value = await call("investigation.graph.expand", {
        case_id: currentCase.case_id, case_revision: currentCase.revision,
        center_entity_id: center || currentCase.first_entity_id || "",
        max_depth: 2, max_nodes: 100, max_edges: 200,
        window_start_ms: msValue("#filter-start") || currentCase.window_start_ms,
        window_end_ms: msValue("#filter-end") || currentCase.window_end_ms,
      }, token);
      if (!value) return;
      renderGraph(value);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function loadEvents(entityId: string, relationId = "", continuation = false) {
    if (!currentCase) return;
    if (continuation && (!eventsCursor || eventQuery.case_id !== currentCase.case_id ||
        eventQuery.case_revision !== currentCase.revision || q<HTMLButtonElement>("#events-more").disabled)) return;
    const token = ++generation;
    try {
      setStatus("Loading source events…");
      if (!continuation) {
        resetEvents();
        eventQuery = { case_id: currentCase.case_id, case_revision: currentCase.revision, limit: 100,
        window_start_ms: msValue("#filter-start") || currentCase.window_start_ms,
        window_end_ms: msValue("#filter-end") || currentCase.window_end_ms };
        if (relationId) eventQuery.relation_id = relationId; else eventQuery.entity_id = entityId;
      }
      q<HTMLButtonElement>("#events-more").disabled = true;
      const args: Json = { ...eventQuery, ...(continuation && eventsCursor ? { cursor: eventsCursor } : {}) };
      const value = await call("investigation.events", args, token);
      if (!value) return;
      eventsCursor = String(value.next_cursor ?? "");
      renderTimeline(value.events ?? [], Boolean(value.has_more), continuation);
      setStatus(`${value.events?.length ?? 0} evidence events on this page${value.has_more ? " · more events are available" : ""}.`);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
    finally { if (token === generation) q<HTMLButtonElement>("#events-more").disabled = false; }
  }
  function renderTimeline(events: Json[], hasMore: boolean, append = false) {
    const list = q<HTMLDivElement>("#timeline"); if (!append) list.replaceChildren();
    for (const event of events) {
      const button = document.createElement("button"); button.className = `event${selectedEvent === event.id ? " selected" : ""}`;
      const time = document.createElement("time"); time.textContent = event.time?.has_event_time
        ? new Date(event.time.event_time_ms).toLocaleString() : "Unknown event time";
      const title = document.createElement("strong"); title.textContent = event.event_type || "Event";
      const id = document.createElement("small"); id.textContent = `Source: ${event.source_index ? `${event.source_index}/` : ""}${event.source_document_id}`;
      button.append(time, title, id); button.addEventListener("click", () => { selectedEvent = event.id; showEvidence(event); });
      list.append(button);
    }
    q<HTMLButtonElement>("#events-more").hidden = !hasMore;
  }
  function showEvidence(event: Json) {
    selectedEvent = event.id;
    q("#evidence").textContent = JSON.stringify({
      event_id: event.id, source_index: event.source_index, source_document_id: event.source_document_id, event_type: event.event_type,
      event_time: event.time, entities: event.entity_ids, raw_resource_uri: event.raw_resource_uri,
      normalized_resource_uri: event.normalized_resource_uri,
      note: "This is a source-linked observation. It does not by itself establish causality.",
    }, null, 2);
  }
  function selectedContext() {
    if (!currentCase) throw new Error("Open an investigation case before adding it to chat.");
    return { case_id: String(currentCase.case_id), revision: Number(currentCase.revision),
      entity_ids: selectedEntity ? [selectedEntity] : [],
      relation_ids: selectedRelation ? [selectedRelation] : [],
      event_ids: selectedEvent ? [selectedEvent] : [] };
  }
  function renderClaims(claims: Json[]) {
    const target = q<HTMLDivElement>("#claims"); target.replaceChildren();
    for (const claim of claims) {
      const box = document.createElement("div"); box.className = "claim";
      const state = document.createElement("small");
      state.textContent = ["Observation", "Correlation", "Hypothesis", "Reviewed conclusion"][Number(claim.state)] ?? "Unknown claim state";
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
        visualization_uris: q<HTMLTextAreaElement>("#snapshot-refs").value.split(/[\r\n,]+/).map((uri) => uri.trim()).filter(Boolean),
      }, token);
      if (!result) return;
      q("#report-result").textContent = `${result.complete ? "No recorded gaps" : "Partial / qualified"} · frozen revision ${result.case_revision}\nDocument: ${result.report_uri}\nMarkdown: ${result.markdown_uri}\nSources: ${result.source_manifest_uri}\n${result.content_fingerprint}${result.visualization_uris?.length ? `\nSnapshots:\n${result.visualization_uris.join("\n")}` : ""}`;
      setStatus(`Structured report frozen from case revision ${result.case_revision}; open its document URI in the existing report editor or export flow.`);
      await loadReports();
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
  q("#browse-cases").addEventListener("click", () => void listCases());
  q("#open-visualization").addEventListener("click", async () => {
    const uri = q<HTMLInputElement>("#visualization-uri").value.trim();
    if (!uri) { setStatus("Enter a saved visualization resource URI.", true); return; }
    const token = ++generation;
    try {
      setStatus("Opening saved visualization…");
      const result = await call("visualization.open", { visualization_uri: uri }, token);
      if (result) showVisualization(result);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  });
  q("#save-snapshot").addEventListener("click", async () => {
    if (!graph || snapshotPending) return;
    const visualizationUri = renderedVisualizationUri;
    const projection = graph.snapshotProjection();
    if (!visualizationUri && !projection) { setStatus("There is no validated graph projection to snapshot.", true); return; }
    const token = generation;
    const viewMetadata = { ...graph.snapshotViewMetadata(), graph_nodes: graph.counts().nodes,
      graph_edges: graph.counts().edges, time_from: q<HTMLInputElement>("#filter-start").value,
      time_to: q<HTMLInputElement>("#filter-end").value };
    snapshotPending = true;
    q<HTMLButtonElement>("#save-snapshot").disabled = true;
    try {
      const image = await graph.exportPng();
      if (token !== generation) return;
      if (image.size > 512 * 1024) throw new Error("The snapshot exceeds the 512 KiB persisted PNG limit.");
      const bytes = new Uint8Array(await image.arrayBuffer());
      if (token !== generation) return;
      let binary = "";
      for (let offset = 0; offset < bytes.length; offset += 0x8000)
        binary += String.fromCharCode(...bytes.subarray(offset, Math.min(bytes.length, offset + 0x8000)));
      const result = await call("visualization.snapshot", {
        ...(visualizationUri ? { visualization_uri: visualizationUri } : { visualization: projection }),
        image_base64: btoa(binary),
        view_json: JSON.stringify(viewMetadata),
      }, token);
      if (!result || token !== generation) return;
      const refs = q<HTMLTextAreaElement>("#snapshot-refs");
      const existing = refs.value.split(/[\r\n,]+/).map((uri) => uri.trim()).filter(Boolean);
      if (!existing.includes(result.snapshot_uri)) refs.value = [...existing, result.snapshot_uri].join("\n");
      setStatus(`PNG snapshot persisted: ${result.snapshot_uri}`);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
    finally {
      snapshotPending = false;
      q<HTMLButtonElement>("#save-snapshot").disabled = !graph || graph.counts().nodes === 0;
    }
  });
  q("#case-list-more").addEventListener("click", () => { if (caseListCursor) void listCases(caseListCursor); });
  q("#open-case").addEventListener("click", () => { q("#case-form").removeAttribute("hidden"); q<HTMLInputElement>("#case-id").focus(); });
  q("#case-submit").addEventListener("click", () => void createOrOpen());
  q("#case-cancel").addEventListener("click", () => q("#case-form").setAttribute("hidden", ""));
  q("#apply-filter").addEventListener("click", () => {
    if (!currentCase) return;
    resetEvents(); selectedRelation = "";
    renderGraph({ nodes: [], edges: [] }, true);
    void expand();
  });
  q("#events-more").addEventListener("click", () => void loadEvents("", "", true));
  q("#frozen-reports-more").addEventListener("click", () => void loadReports(true));
  q("#save-hypothesis").addEventListener("click", () => void saveHypothesis());
  q("#compose-report").addEventListener("click", () => void composeReport());
  q<HTMLTextAreaElement>("#hypothesis").addEventListener("input", (event) => {
    q<HTMLButtonElement>("#save-hypothesis").disabled = !currentCase || !(event.target as HTMLTextAreaElement).value.trim();
  });
  addEventListener("message", (event) => {
    if (event.source !== parent) return;
    const message = event.data;
    if (message?.jsonrpc !== "2.0") return;
    if (message.method === "ui/llama-agent/prepare-investigation-context" && typeof message.id === "number") {
      try { parent.postMessage({ jsonrpc: "2.0", id: message.id, result: selectedContext() }, "*"); }
      catch (error) { parent.postMessage({ jsonrpc: "2.0", id: message.id,
        error: { code: -32000, message: error instanceof Error ? error.message : String(error) } }, "*"); }
      return;
    }
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
    if (message.method === "ui/notifications/tool-result") {
      const result = message.params?.structuredContent ?? message.params?.result?.structuredContent;
      if (result?.visualization?.kind === "relationship") {
        try { showVisualization(result); }
        catch (error) { setStatus(error instanceof Error ? error.message : String(error), true); }
      } else if (result?.case_id && Date.now() - lastAppCallAt > 1000) {
        void openCase(result.case_id);
      }
    }
  });
  void rpc("ui/initialize", { protocolVersion: "2026-01-26", appInfo: { name: "llama-agent-investigation-workspace", version: "0.1" }, appCapabilities: { tools: {} } })
    .then(() => { parent.postMessage({ jsonrpc: "2.0", method: "ui/notifications/initialized", params: {} }, "*"); setStatus("Ready. Open a case or create one."); })
    .catch((error) => setStatus(error instanceof Error ? error.message : String(error), true));
}
