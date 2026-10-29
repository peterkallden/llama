import "./investigation-workspace.css";
import { RelationshipRenderer } from "../visualization/relationship-renderer";
import { parseRelationshipVisualization } from "../visualization/contract";
import { createLatestRequestGate } from "../async/latest-request";

declare global { interface Window { __investigationWorkspace?: true } }
if (!window.__investigationWorkspace) {
  window.__investigationWorkspace = true;

  type Json = Record<string, any>;
  const q = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
  const status = q<HTMLDivElement>("#status");
  const pending = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void }>();
  let seq = 1;
  let generation = 0;
  const beginGraphRead = createLatestRequestGate(() => generation);
  const beginEventRead = createLatestRequestGate(() => generation);
  const beginResourceRead = createLatestRequestGate(() => generation);
  const beginCaseListRead = createLatestRequestGate(() => generation);
  const beginResourceChoiceRead = createLatestRequestGate(() => generation);
  let currentCase: Json | null = null;
  let selectedEntity = "";
  let selectedRelation = "";
  let selectedEvent = "";
  let graph: RelationshipRenderer | null = null;
  let actorGraph: RelationshipRenderer | null = null;
  const selectedActors = new Set<string>();
  const actorCandidates = new Map<string, Json>();
  const sourceStatements = new Map<string, Json>();
  const actorMentions = new Map<string, Json>();
  const actorGroups = new Map<string, Json>();
  const actorRelationships = new Map<string, Json>();
  let actorGroupsLoaded = false, actorRelationshipsLoaded = false;
  const actorProjectionLimitations = new Set<string>();
  let reviewBusy = false;
  let reviewAttempt: Json | null = null;
  let lastAppCallAt = 0;
  let caseListCursor = "";
  let reportCursor = "";
  let eventsCursor = "";
  type AnalysisItemType = "statements" | "comparisons" | "actors" | "actor_mentions" | "evidence_links" | "actor_groups" | "actor_relationships" | "association_reviews";
  const analysisCursors: Partial<Record<AnalysisItemType, string>> = {};
  const analysisReads: Partial<Record<AnalysisItemType, number>> = {};
  let resourceCursor = "";
  let caseSources: Json[] = [];
  let resourceRows: Json[] = [];
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
    beginEventRead();
    eventsCursor = ""; eventQuery = {}; selectedEvent = "";
    q("#timeline").replaceChildren(); q("#evidence").textContent = "";
    q<HTMLButtonElement>("#events-more").hidden = true;
    q<HTMLButtonElement>("#events-more").disabled = false;
  }
  function showCase(value: Json) {
    currentCase = value;
    caseSources = [];
    selectedActors.clear(); actorCandidates.clear(); sourceStatements.clear(); actorMentions.clear(); actorGroups.clear(); actorRelationships.clear();
    reviewAttempt = null; reviewBusy = false;
    actorGroupsLoaded = false; actorRelationshipsLoaded = false; actorProjectionLimitations.clear();
    q<HTMLTextAreaElement>("#association-comment").value = "";
    q("#actor-association-detail").textContent = "";
    updateReviewSelection();
    renderActorAssociations();
    for (const type of Object.keys(analysisCursors) as AnalysisItemType[]) delete analysisCursors[type];
    for (const selector of ["#case-sources", "#selected-sources", "#statements", "#comparisons", "#actors", "#actor_mentions", "#evidence_links", "#followup-steps", "#actor_groups", "#actor_relationships", "#association_reviews"])
      q(selector).replaceChildren();
    q<HTMLButtonElement>("#start-analysis").disabled = true;
    q<HTMLButtonElement>("#statements-more").hidden = true;
    q<HTMLButtonElement>("#comparisons-more").hidden = true;
    for (const type of ["actors", "actor_mentions", "evidence_links", "actor_groups", "actor_relationships", "association_reviews"])
      q<HTMLButtonElement>(`#${type}-more`).hidden = true;
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
    q("#source-panel").removeAttribute("hidden");
    q("#analysis-panel").removeAttribute("hidden");
    q("#report-result").textContent = "No report created for this revision.";
    reportCursor = "";
    q<HTMLButtonElement>("#frozen-reports-more").hidden = true;
    const reportStatus = document.createElement("p"); reportStatus.textContent = "Loading indexed reports…";
    q("#frozen-reports").replaceChildren(reportStatus);
    renderClaims([]);
    renderGraph({ nodes: [], edges: [], case_revision: revision, truncated: false, limitations: [] }, true);
    setStatus(`Case ${value.case_id} opened at revision ${revision}.`);
    void loadCaseSources();
    void loadAnalysisItems("statements");
    void loadAnalysisItems("comparisons");
    void loadAnalysisItems("actors");
    void loadAnalysisItems("actor_mentions");
    void loadAnalysisItems("evidence_links");
    void loadAnalysisItems("actor_groups");
    void loadAnalysisItems("actor_relationships");
    void loadAnalysisItems("association_reviews");
  }
  async function loadCaseClaims() {
    if (!currentCase) return;
    const token = generation;
    try {
      const value = await call("investigate.case.items", { case_id: currentCase.case_id,
        case_revision: currentCase.revision, type: "claims", limit: 100 }, token);
      if (value) renderClaims(value.items ?? []);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function loadCaseSources() {
    if (!currentCase) return;
    const token = generation;
    try {
      const value = await call("investigate.case.items", { case_id: currentCase.case_id,
        case_revision: currentCase.revision, type: "sources", limit: 100 }, token);
      if (!value || token !== generation) return;
      caseSources = Array.isArray(value.items) ? value.items : [];
      const target = q<HTMLDivElement>("#case-sources"); target.replaceChildren();
      const selected = q<HTMLDivElement>("#selected-sources"); selected.replaceChildren();
      for (const source of caseSources) {
        const row = document.createElement("div"); row.className = "source-row";
        const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.value = String(source.id ?? ""); checkbox.checked = true;
        checkbox.addEventListener("change", updateAnalysisButton);
        const label = document.createElement("label"); label.className = "source-choice";
        const name = document.createElement("strong"); name.textContent = String(source.display_name || source.provider_id || source.id);
        const detail = document.createElement("small");
        detail.textContent = `${source.kind} · ${source.processing_status ?? "unknown"} · ${source.complete === false ? "partial coverage" : "coverage not assessed"} · ${source.content_hash || source.provider_id || "no hash"}`;
        label.append(name, detail); row.append(checkbox, label);
        checkbox.addEventListener("change", () => {
          const selectedCheckbox = [...selected.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].find((item) => item.value === checkbox.value);
          if (selectedCheckbox) selectedCheckbox.checked = checkbox.checked;
        });
        if (source.kind !== "search_connector" && !source.representation_uri) {
          const prepare = document.createElement("button"); prepare.type = "button"; prepare.textContent = "Prepare text";
          prepare.addEventListener("click", () => void prepareSource(String(source.id ?? ""))); row.append(prepare);
        }
        if (source.kind === "log_file" && source.representation_uri) {
          const controls = document.createElement("details");
          const summary = document.createElement("summary"); summary.textContent = "Import structured log events";
          const format = document.createElement("select"); format.setAttribute("aria-label", "Log format");
          for (const kind of ["jsonl", "csv"]) { const option = document.createElement("option"); option.value = kind; option.textContent = kind.toUpperCase(); format.append(option); }
          const fields = document.createElement("textarea"); fields.setAttribute("aria-label", "Log field mapping");
          fields.value = JSON.stringify({ "event.type": "/event/type", "@timestamp": "/@timestamp" }, null, 2);
          let profile: Json | null = null;
          try { profile = source.log_parser_profile_json ? JSON.parse(source.log_parser_profile_json) : null; } catch { /* The host reports invalid saved profiles. */ }
          if (profile) { format.value = String(profile.format); fields.value = JSON.stringify(profile.fields, null, 2); format.disabled = true; fields.readOnly = true; }
          const help = document.createElement("small"); help.textContent = "Explicit mapping: normalized field → exact JSON key/JSON pointer or CSV header. event.type is required. Map event.outcome to preserve authentication success/failure; missing outcomes do not prove success. Up to 200 records per page; unknown formats are not guessed.";
          const progress = document.createElement("small"); progress.textContent = `Processed ${source.log_import_next_record ?? 0} records · rejected ${source.log_import_rejected_records ?? 0}`;
          const button = document.createElement("button"); button.type = "button";
          button.textContent = profile ? "Import next page" : "Import first page";
          button.disabled = source.log_import_exhausted === true;
          let attempt: Json | null = null;
          button.addEventListener("click", async () => {
            if (!currentCase || button.disabled) return;
            const token = generation, caseId = currentCase.case_id;
            try {
              if (!attempt) attempt = { case_id: caseId, expected_revision: currentCase.revision,
                source_id: source.id, format: format.value, fields: JSON.parse(fields.value), limit: 200,
                idempotency_key: crypto.randomUUID() };
              button.disabled = true;
              const result = await call("investigate.log.import", attempt, token);
              if (!result || token !== generation) return;
              attempt = null;
              await openCase(String(caseId));
            } catch (error) {
              if (token === generation) { button.disabled = false; button.textContent = "Retry this import page"; setStatus(error instanceof Error ? error.message : String(error), true); }
            }
          });
          controls.append(summary, help, format, fields, progress, button); row.append(controls);
        }
        target.append(row);
        const selectedRow = document.createElement("label"); selectedRow.className = "selected-source";
        const selectedCheckbox = document.createElement("input"); selectedCheckbox.type = "checkbox"; selectedCheckbox.value = String(source.id ?? ""); selectedCheckbox.checked = true;
        selectedCheckbox.addEventListener("change", () => {
          const sourceCheckbox = [...target.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].find((item) => item.value === selectedCheckbox.value);
          if (sourceCheckbox) sourceCheckbox.checked = selectedCheckbox.checked;
          updateAnalysisButton();
        });
        const selectedName = document.createElement("span"); selectedName.textContent = String(source.display_name || source.provider_id || source.id);
        selectedRow.append(selectedCheckbox, selectedName); selected.append(selectedRow);
      }
      updateAnalysisButton();
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  function updateAnalysisButton() {
    const selected = [...q<HTMLDivElement>("#case-sources").querySelectorAll<HTMLInputElement>('input[type="checkbox"]')]
      .filter((item) => item.checked).length;
    q<HTMLButtonElement>("#start-analysis").disabled = !currentCase || selected === 0;
  }
  async function loadAnalysisItems(type: AnalysisItemType, continuation = false) {
    if (!currentCase || analysisReads[type] === generation) return;
    const cursor = analysisCursors[type] ?? "";
    if (continuation && !cursor) return;
    const token = generation;
    const target = q<HTMLDivElement>(`#${type}`);
    analysisReads[type] = token;
    q<HTMLButtonElement>(`#${type}-more`).disabled = true;
    try {
      const value = await call("investigate.case.items", { case_id: currentCase.case_id,
        case_revision: currentCase.revision, type, limit: 50, ...(continuation ? { cursor } : {}) }, token);
      if (!value || token !== generation) return;
      if (!continuation) {
        target.replaceChildren();
        if (type === "actors") actorCandidates.clear();
        if (type === "statements") sourceStatements.clear();
        if (type === "actor_mentions") actorMentions.clear();
        if (type === "actor_groups") actorGroups.clear();
        if (type === "actor_relationships") actorRelationships.clear();
      }
      analysisCursors[type] = String(value.next_cursor ?? "");
      if (type === "actor_groups") actorGroupsLoaded = true;
      if (type === "actor_relationships") actorRelationshipsLoaded = true;
      if (type === "actor_groups" || type === "actor_relationships")
        for (const limitation of value.limitations ?? []) actorProjectionLimitations.add(String(limitation));
      const rows = Array.isArray(value.items) ? value.items : [];
      if (!rows.length && !continuation) {
        const empty = document.createElement("p"); empty.textContent = `No ${type.replaceAll("_", " ")} recorded.`; target.append(empty);
      }
      for (const item of rows) {
        const box = document.createElement("div"); box.className = "analysis-item";
        if (type === "statements") {
          sourceStatements.set(String(item.id),item);
          const title = document.createElement("strong"); title.textContent = `${item.kind}: ${item.text}`;
          const quote = document.createElement("blockquote"); quote.textContent = `“${item.anchor?.quote ?? ""}”`;
          const meta = document.createElement("small"); meta.textContent = `${item.anchor?.source_id ?? "source"} · ${item.anchor?.resource_uri ?? ""} · byte ${item.anchor?.byte_offset ?? 0}${item.anchor?.page_number ? ` · page ${item.anchor.page_number}` : ""}`;
          box.append(title, quote, meta);
        } else if (type === "actors") {
          actorCandidates.set(String(item.id),item);
          const choose = document.createElement("input"); choose.type = "checkbox"; choose.value = String(item.id);
          choose.checked = selectedActors.has(String(item.id)); choose.setAttribute("aria-label", `Select actor ${item.display_name ?? item.id}`);
          choose.addEventListener("change", () => {
            if (choose.checked) selectedActors.add(String(item.id)); else selectedActors.delete(String(item.id));
            updateReviewSelection();
          });
          box.append(choose);
          const title = document.createElement("strong"); title.textContent = String(item.display_name ?? item.id);
          const detail = document.createElement("small"); detail.textContent = `${item.id} · candidate · ${item.mention_ids?.length ?? 0} anchored mentions`;
          box.append(title, detail);
        } else if (type === "actor_mentions") {
          actorMentions.set(String(item.id),item);
          const title = document.createElement("strong"); title.textContent = `${item.name} (${item.kind})`;
          const detail = document.createElement("small"); detail.textContent = `${item.id} · statement ${item.statement_id} · ${item.anchor?.source_id} · byte ${item.anchor?.byte_offset}`;
          box.append(title, detail);
        } else if (type === "evidence_links") {
          const title = document.createElement("strong"); title.textContent = `${item.kind}: ${item.left_id} ↔ ${item.right_id}`;
          const detail = document.createElement("small"); detail.textContent = `Proposed, no merge · ${item.rationale} · evidence ${(item.supporting_statement_ids ?? []).join(", ")}`;
          box.append(title, detail);
          if (item.kind === "possible_same_actor") {
            const review = document.createElement("button"); review.type = "button"; review.textContent = "Review this identity proposal";
            review.addEventListener("click", () => {
              if (reviewBusy || reviewAttempt) return;
              selectedActors.clear(); selectedActors.add(String(item.left_id)); selectedActors.add(String(item.right_id));
              updateReviewSelection();
            });
            box.append(review);
          }
        } else if (type === "actor_groups") {
          actorGroups.set(String(item.id),item);
          const title = document.createElement("strong"); title.textContent = `${item.display_name}: ${item.actor_count} candidate(s)`;
          const detail = document.createElement("small"); detail.textContent = `${item.assessment_origin} · ${item.mention_count} mentions · ${(item.actor_ids ?? []).join(", ")}${item.members_truncated ? " · member list abbreviated" : ""}`;
          box.append(title,detail);
        } else if (type === "actor_relationships") {
          actorRelationships.set(String(item.id),item);
          const title = document.createElement("strong"); title.textContent = `${item.source} → ${item.target}: ${item.type}`;
          const detail = document.createElement("small"); detail.textContent = `Analyst confirmed · ${item.reviewer_id}`;
          box.append(title,detail);
        } else if (type === "association_reviews") {
          const title = document.createElement("strong"); title.textContent = `${item.decision}: ${item.left_actor_id} ↔ ${item.right_actor_id}`;
          const detail = document.createElement("small"); detail.textContent = `${item.id} · ${item.reviewer_id} · ${new Date(item.created_at_ms).toLocaleString()} · r${item.published_revision}${item.supersedes_id ? ` · supersedes ${item.supersedes_id}` : ""} · ${item.comment || "No comment"}`;
          box.append(title,detail);
          const reconsider = document.createElement("button"); reconsider.type = "button"; reconsider.textContent = "Select this pair for reconsideration";
          reconsider.addEventListener("click", () => {
            selectedActors.clear(); selectedActors.add(String(item.left_actor_id)); selectedActors.add(String(item.right_actor_id));
            if (item.kind === "relationship") q<HTMLSelectElement>("#association-relation-type").value = String(item.relation_type);
            updateReviewSelection();
            q("#actor-association-detail").textContent = `Selected ${item.left_actor_id} and ${item.right_actor_id}. Choose a new decision; prior records remain in history.`;
          });
          box.append(reconsider);
        } else {
          const title = document.createElement("strong"); title.textContent = `${item.kind}: ${item.left_statement_id} ↔ ${item.right_statement_id}`;
          const detail = document.createElement("small"); detail.textContent = `${item.assessment_origin ?? "model_proposal"} · ${item.rationale ?? "No rationale provided."}`;
          box.append(title, detail);
        }
        target.append(box);
      }
      q<HTMLButtonElement>(`#${type}-more`).hidden = !value.has_more;
      updateReviewSelection();
      if (type === "actor_groups" || type === "actor_relationships") renderActorAssociations();
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
    finally {
      if (analysisReads[type] === token) delete analysisReads[type];
      if (token === generation) q<HTMLButtonElement>(`#${type}-more`).disabled = false;
    }
  }
  function updateReviewSelection() {
    const target = q<HTMLDivElement>("#association-selection"); target.replaceChildren();
    for (const id of selectedActors) {
      const actor = actorCandidates.get(id);
      const line = document.createElement("p"); line.textContent = `${actor?.display_name ?? id} · ${id}`; target.append(line);
      for (const mentionId of actor?.mention_ids ?? []) {
        const mention = actorMentions.get(String(mentionId));
        if (!mention) continue;
        const statement = sourceStatements.get(String(mention.statement_id));
        const quote = document.createElement("blockquote"); quote.textContent = `“${statement?.anchor?.quote ?? mention.anchor?.quote ?? mention.name}” · ${mention.anchor?.resource_uri} · byte ${mention.anchor?.byte_offset} · statement ${mention.statement_id}`;
        target.append(quote);
      }
    }
    if (!selectedActors.size) target.textContent = "Select actor candidates to review. More mentions can be loaded below.";
    const locked = reviewBusy || reviewAttempt !== null;
    for (const checkbox of q("#actors").querySelectorAll<HTMLInputElement>('input[type="checkbox"]')) checkbox.disabled = locked;
    for (const id of ["same-actor", "different-actors", "confirm-actor-relation", "unresolve-identity", "withdraw-actor-relation"])
      q<HTMLButtonElement>(`#${id}`).disabled = !currentCase || locked || (id === "same-actor" ? selectedActors.size < 2 || selectedActors.size > 32 : selectedActors.size !== 2);
    q<HTMLButtonElement>("#retry-association-review").hidden = !reviewAttempt;
    q<HTMLButtonElement>("#retry-association-review").disabled = reviewBusy;
    q<HTMLButtonElement>("#edit-association-review").hidden = !reviewAttempt;
    q<HTMLButtonElement>("#edit-association-review").disabled = reviewBusy;
    q<HTMLTextAreaElement>("#association-comment").disabled = locked;
    q<HTMLSelectElement>("#association-relation-type").disabled = locked;
  }
  async function reviewAssociations(decision: string, kind: "identity" | "relationship") {
    if (!currentCase || reviewBusy || reviewAttempt) return;
    const actors = [...selectedActors];
    if (actors.length < 2 || actors.length > 32 || (decision !== "same_actor" && actors.length !== 2)) {
      setStatus("Select two candidates, or up to 32 for a shared identity.",true); return;
    }
    reviewAttempt = { case_id: currentCase.case_id, expected_revision: currentCase.revision,
      idempotency_key: crypto.randomUUID(), actor_ids: actors, kind, decision,
      ...(kind === "relationship" ? { relation_type: q<HTMLSelectElement>("#association-relation-type").value } : {}),
      comment: q<HTMLTextAreaElement>("#association-comment").value };
    await publishAssociationReview();
  }
  async function publishAssociationReview() {
    if (!reviewAttempt || reviewBusy) return;
    const token = generation, attempt = reviewAttempt;
    reviewBusy = true; updateReviewSelection();
    try {
      const result = await call("investigate.association.review",attempt,token);
      if (!result || token !== generation) return;
      reviewAttempt = null;
      await openCase(String(attempt.case_id));
    } catch (error) {
      if (token === generation) setStatus(`${error instanceof Error ? error.message : String(error)} Retry the same decision, or reload the case to inspect its current state.`,true);
    } finally {
      if (token === generation) { reviewBusy = false; updateReviewSelection(); }
    }
  }
  function renderActorAssociations() {
    if (!actorGraph) actorGraph = new RelationshipRenderer(q<HTMLDivElement>("#actor-association-graph"), { onSelect: (selection) => {
      const value = selection.kind === "node" ? actorGroups.get(selection.id) : actorRelationships.get(selection.id);
      q("#actor-association-detail").textContent = JSON.stringify(value ?? {},null,2);
    } });
    const groups = [...actorGroups.values()].slice(0,500);
    const ids = new Set(groups.map((group) => String(group.id)));
    const relationships = [...actorRelationships.values()];
    const edges = relationships.filter((edge) => ids.has(String(edge.source)) && ids.has(String(edge.target))).slice(0,1000);
    const complete = actorGroupsLoaded && actorRelationshipsLoaded && actorProjectionLimitations.size === 0 && !analysisCursors.actor_groups && !analysisCursors.actor_relationships && actorGroups.size <= 500 &&
      actorRelationships.size <= 1000 && edges.length === relationships.length;
    actorGraph.update({ schema_version: 1, kind: "relationship", title: "Reviewed actor associations",
      data_revision: String(currentCase?.revision ?? ""),
      nodes: groups.map((group) => ({ id: String(group.id), label: String(group.display_name), kind: String(group.assessment_origin) })),
      edges: edges.map((edge) => ({ id: String(edge.id), source: String(edge.source), target: String(edge.target),
        label: `Analyst confirmed: ${edge.type}`, directed: true, provenance_refs: [String(edge.id)] })),
      complete, truncated: !complete, limitations: complete ? [] : [...actorProjectionLimitations,"Only loaded actor groups and relations are shown; load more pages to inspect further associations."] }, { replace: true });
  }
  async function browseResources(continuation = false) {
    if (continuation && !resourceCursor) return;
    const target = q<HTMLDivElement>("#resource-results");
    const query = q<HTMLInputElement>("#resource-query").value.trim();
    const scope = q<HTMLSelectElement>("#resource-scope").value;
    const isCurrent = beginResourceRead();
    try {
      const value = await rpc("ui/llama-agent/list-investigation-resources", { scope, query, limit: 30,
        ...(continuation ? { cursor: resourceCursor } : {}) });
      if (!isCurrent()) return;
      if (!continuation) { target.replaceChildren(); resourceRows = []; }
      resourceCursor = String(value?.next_cursor ?? "");
      const resources = Array.isArray(value?.resources) ? value.resources : [];
      resourceRows.push(...resources);
      for (const item of resources) {
        const row = document.createElement("div"); row.className = "resource-result";
        const label = document.createElement("span"); label.textContent = `${item.name ?? "Resource"} · ${item.mime_type ?? "unknown type"}`;
        const add = document.createElement("button"); add.type = "button"; add.textContent = "Add to case";
        add.addEventListener("click", () => void attachResource(String(item.uri ?? "")));
        row.append(label, add); target.append(row);
      }
      if (!resources.length && !continuation) { const empty = document.createElement("p"); empty.textContent = "No authorized resources match this search."; target.append(empty); }
      q<HTMLButtonElement>("#resource-more").hidden = !resourceCursor;
    } catch (error) { if (isCurrent()) { target.textContent = error instanceof Error ? error.message : String(error); setStatus("Could not browse authorized resources.", true); } }
  }
  async function populateCaseResources() {
    const isCurrent = beginResourceChoiceRead();
    const selector = q<HTMLSelectElement>("#case-resource");
    selector.replaceChildren(new Option("Choose a host-authorized resource", ""));
    const firstPage = await rpc("ui/llama-agent/list-investigation-resources", { scope: "session", limit: 50 });
    if (!isCurrent()) return;
    const projectOption = q<HTMLSelectElement>("#resource-scope").querySelector<HTMLOptionElement>('option[value="project"]');
    if (projectOption) projectOption.disabled = !firstPage?.project_available;
    const scopes = ["session", ...(firstPage?.project_available ? ["project"] : [])];
    for (const scope of scopes) {
      let cursor = "";
      for (let page = 0; page < 10; page++) {
        const result = scope === "session" && page === 0 ? firstPage :
          await rpc("ui/llama-agent/list-investigation-resources", { scope, limit: 50, ...(cursor ? { cursor } : {}) });
        if (!isCurrent()) return;
        for (const resource of Array.isArray(result?.resources) ? result.resources : []) {
          const uri = String(resource.uri ?? "");
          if (!uri) continue;
          selector.add(new Option(`${resource.name ?? "Resource"} · ${resource.mime_type ?? "unknown"} · ${scope}`, uri));
        }
        cursor = String(result?.next_cursor ?? "");
        if (!cursor) break;
      }
    }
  }
  async function populateConfiguredSources() {
    const selectors = [q<HTMLSelectElement>("#source-id"), q<HTMLSelectElement>("#add-search-source")];
    const current = selectors.map((selector) => selector.value);
    for (const selector of selectors) selector.replaceChildren(new Option("No search source", ""));
    const result = await call("investigate.source.list", { limit: 100 });
    for (const source of Array.isArray(result?.sources) ? result.sources : []) {
      const id = String(source.source_id ?? "");
      if (id) for (const selector of selectors) selector.add(new Option(String(source.display_name ?? id), id));
    }
    selectors.forEach((selector, index) => { selector.value = current[index]; });
  }
  async function attachResource(uri: string, token = generation, kind = q<HTMLSelectElement>("#attach-kind").value) {
    if (!currentCase || !uri || token !== generation) return;
    try {
      const result = await call("investigate.source.attach", { case_id: currentCase.case_id,
        expected_revision: currentCase.revision, resource_uri: uri, kind }, token);
      if (result) await openCase(currentCase.case_id);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function attachSearchSource() {
    if (!currentCase) return;
    const providerId = q<HTMLSelectElement>("#add-search-source").value;
    if (!providerId) { setStatus("Choose an authorized search source first.", true); return; }
    const input = { case_id: currentCase.case_id, expected_revision: currentCase.revision, provider_id: providerId } as Json;
    if (!currentCase.source_id) {
      const start = msValue("#attach-window-start"), end = msValue("#attach-window-end");
      if (!start || !end || start >= end) { setStatus("The first search source requires a valid, non-empty time window.", true); return; }
      input.window_start_ms = start; input.window_end_ms = end;
    }
    const token = generation;
    try {
      const result = await call("investigate.source.attach", input, token);
      if (result) await openCase(currentCase.case_id);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function prepareSource(sourceId: string) {
    if (!currentCase || !sourceId) return;
    const token = generation;
    try {
      const result = await call("investigate.document.prepare", { case_id: currentCase.case_id,
        expected_revision: currentCase.revision, source_id: sourceId }, token);
      if (result) { setStatus(`Readable representation prepared (${result.processor_version || "existing text"}).`); await openCase(currentCase.case_id); }
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function loadReports(continuation = false) {
    if (!currentCase) return;
    if (continuation && !reportCursor) return;
    const token = generation;
    const target = q<HTMLDivElement>("#frozen-reports");
    try {
      const value = await call("investigate.case.items", { case_id: currentCase.case_id,
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
      const result = await call("investigate.case.open", { case_id: id }, token);
      if (!result) return;
      if (!result.case) throw new Error("The host returned no case.");
      showCase(result.case);
      q("#case-form").setAttribute("hidden", "");
      await loadCaseClaims();
      await loadReports();
      if (token !== generation) return;
      if (result.case.first_entity_id) await expand(result.case.first_entity_id);
      if (token === generation) await loadEvents("");
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function listCases(after = "") {
    const token = generation;
    const isCurrent = beginCaseListRead();
    try {
      setStatus("Loading cases in this scope…");
      const value = await call("investigate.case.list", { limit: 50, after_case_id: after }, token);
      if (!value || !isCurrent()) return;
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
    } catch (error) { if (isCurrent()) setStatus(error instanceof Error ? error.message : String(error), true); }
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
    q("#source-panel").setAttribute("hidden", "");
    q("#analysis-panel").setAttribute("hidden", "");
    q<HTMLButtonElement>("#start-analysis").disabled = true;
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
    const token = generation;
    const isCurrent = beginGraphRead();
    try {
      const value = await call("investigate.graph.expand", {
        case_id: currentCase.case_id, case_revision: currentCase.revision,
        center_entity_id: center || currentCase.first_entity_id || "",
        max_depth: 2, max_nodes: 100, max_edges: 200,
        window_start_ms: msValue("#filter-start") || currentCase.window_start_ms,
        window_end_ms: msValue("#filter-end") || currentCase.window_end_ms,
      }, token);
      if (!value || !isCurrent()) return;
      renderGraph(value);
    } catch (error) { if (isCurrent()) setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function loadEvents(entityId: string, relationId = "", continuation = false) {
    if (!currentCase) return;
    if (continuation && (!eventsCursor || eventQuery.case_id !== currentCase.case_id ||
        eventQuery.case_revision !== currentCase.revision || q<HTMLButtonElement>("#events-more").disabled)) return;
    const token = generation;
    if (!continuation) resetEvents();
    const isCurrent = beginEventRead();
    try {
      setStatus("Loading source events…");
      if (!continuation) {
        eventQuery = { case_id: currentCase.case_id, case_revision: currentCase.revision, limit: 100,
        window_start_ms: msValue("#filter-start") || currentCase.window_start_ms,
        window_end_ms: msValue("#filter-end") || currentCase.window_end_ms };
        if (relationId) eventQuery.relation_id = relationId; else eventQuery.entity_id = entityId;
      }
      q<HTMLButtonElement>("#events-more").disabled = true;
      const args: Json = { ...eventQuery, ...(continuation && eventsCursor ? { cursor: eventsCursor } : {}) };
      const value = await call("investigate.events", args, token);
      if (!value || !isCurrent()) return;
      eventsCursor = String(value.next_cursor ?? "");
      renderTimeline(value.events ?? [], Boolean(value.has_more), continuation);
      setStatus(`${value.events?.length ?? 0} evidence events on this page${value.has_more ? " · more events are available" : ""}.`);
    } catch (error) { if (isCurrent()) setStatus(error instanceof Error ? error.message : String(error), true); }
    finally { if (isCurrent()) q<HTMLButtonElement>("#events-more").disabled = false; }
  }
  function renderTimeline(events: Json[], hasMore: boolean, append = false) {
    const list = q<HTMLDivElement>("#timeline"); if (!append) list.replaceChildren();
    for (const event of events) {
      const button = document.createElement("button"); button.className = `event${selectedEvent === event.id ? " selected" : ""}`;
      const time = document.createElement("time"); time.textContent = event.time?.has_event_time
        ? new Date(event.time.event_time_ms).toLocaleString() : "Unknown event time";
      const title = document.createElement("strong"); title.textContent = `${event.event_type || "Event"} · outcome: ${event.outcome || "unknown"}`;
      const id = document.createElement("small"); id.textContent = `Source: ${event.source_index ? `${event.source_index}/` : ""}${event.source_document_id}`;
      button.append(time, title, id); button.addEventListener("click", () => { selectedEvent = event.id; showEvidence(event); });
      list.append(button);
    }
    q<HTMLButtonElement>("#events-more").hidden = !hasMore;
  }
  function showEvidence(event: Json) {
    selectedEvent = event.id;
    q("#evidence").textContent = JSON.stringify({
      event_id: event.id, source_index: event.source_index, source_document_id: event.source_document_id, event_type: event.event_type, outcome: event.outcome || "unknown",
      event_time: event.time, entities: event.entity_ids, raw_resource_uri: event.raw_resource_uri,
      normalized_resource_uri: event.normalized_resource_uri,
      source_resource_uri: event.source_resource_uri, source_byte_offset: event.source_byte_offset,
      source_byte_length: event.source_byte_length, source_line: event.source_line, parser_version: event.parser_version,
      note: "This is a source-linked observation. It does not by itself establish causality.",
    }, null, 2);
  }
  function selectedContext() {
    if (!currentCase) throw new Error("Open an investigation case before adding it to chat.");
    return { case_id: String(currentCase.case_id), revision: Number(currentCase.revision),
      entity_ids: selectedEntity ? [selectedEntity] : [],
      relation_ids: selectedRelation ? [selectedRelation] : [],
      event_ids: selectedEvent ? [selectedEvent] : [],
      source_ids: [...q<HTMLDivElement>("#case-sources").querySelectorAll<HTMLInputElement>('input[type="checkbox"]')]
        .filter((item) => item.checked).map((item) => item.value),
      analysis_profile: q<HTMLSelectElement>("#analysis-profile").value };
  }
  async function startAnalysis() {
    try {
      const context = selectedContext();
      if (!context.source_ids.length) throw new Error("Select at least one source for this analysis.");
      const profiles: Record<string, string> = {
        identity_analysis: "identity analysis",
        communication_analysis: "communication analysis",
        actor_relationship: "actor and relationship analysis",
        narrative_comparison: "narrative and contradiction analysis",
        document_provenance: "document provenance analysis",
        event_chain: "event chain analysis",
        hypothesis_testing: "hypothesis testing",
        report_composition: "report composition",
        integration_trace: "integration tracing",
      };
      const profile = q<HTMLSelectElement>("#analysis-profile").value;
      await rpc("ui/message", { action: "start-investigation", content: { type: "text",
        text: `Run the selected ${profiles[profile] ?? "investigation"} workflow for the selected sources in this case. Follow its declared steps, preserve source anchors and uncertainty, and do not present model proposals as verified findings.` } });
      setStatus("Analysis request sent through the normal chat workflow.");
    } catch (error) { setStatus(error instanceof Error ? error.message : String(error), true); }
  }
  async function suggestFollowup() {
    if (!currentCase) return;
    const token = generation;
    try {
      const context = selectedContext();
      const result = await call("investigate.analyze", { case_id: context.case_id,
        analysis: "source_followup", source_ids: context.source_ids, max_expansions: 10000 }, token);
      if (!result || token !== generation) return;
      const target = q<HTMLDivElement>("#followup-steps"); target.replaceChildren();
      for (const step of result.steps ?? []) {
        const row = document.createElement("p"); row.textContent = `${step.operation}: ${(step.source_ids ?? [step.source_id]).join(", ")} · ${step.reason}`; target.append(row);
      }
      if (!result.steps?.length) target.textContent = "No unprocessed sources or source pairs found within this structural check.";
      if (!result.complete) { const note = document.createElement("p"); note.textContent = "Suggestions stopped at their budget or result limit."; target.append(note); }
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
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
      const value = await call("investigate.claim.hypothesize", {
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
      const result = await call("investigate.report.compose", {
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
    const sourceId = q<HTMLSelectElement>("#source-id").value.trim();
    const resourceUri = q<HTMLSelectElement>("#case-resource").value;
    if (!sourceId && !resourceUri) { setStatus("Choose a configured search source or a host-authorized resource.", true); return; }
    if (sourceId && (!start || !end || start >= end)) { setStatus("A configured log source requires a valid, non-empty time window.", true); return; }
    try {
      const value = await call("investigate.case.create", {
        title: q<HTMLInputElement>("#case-title").value.trim(),
        question: q<HTMLInputElement>("#case-question").value.trim(),
        ...(sourceId ? { source_id: sourceId, window_start_ms: start, window_end_ms: end } : {}),
        ...(resourceUri ? { resource_uri: resourceUri, resource_kind: q<HTMLSelectElement>("#case-resource-kind").value } : {}),
      });
      if (!value) return;
      q("#case-form").setAttribute("hidden", ""); await openCase(value.case_id);
    } catch (error) { setStatus(error instanceof Error ? error.message : String(error), true); }
  }

  q("#create-case").addEventListener("click", () => {
    q("#case-form").removeAttribute("hidden"); q<HTMLInputElement>("#case-id").value = "";
    void populateCaseResources().catch((error) => setStatus(error instanceof Error ? error.message : String(error), true));
    void populateConfiguredSources().catch((error) => setStatus(error instanceof Error ? error.message : String(error), true));
  });
  q("#browse-cases").addEventListener("click", () => void listCases());
  q("#open-visualization").addEventListener("click", async () => {
    const uri = q<HTMLInputElement>("#visualization-uri").value.trim();
    if (!uri) { setStatus("Enter a saved visualization resource URI.", true); return; }
    const token = ++generation;
    try {
      setStatus("Opening saved visualization…");
      const result = await call("visualize.open", { visualization_uri: uri }, token);
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
      const result = await call("visualize.snapshot", {
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
    void loadEvents("");
  });
  q("#events-more").addEventListener("click", () => void loadEvents("", "", true));
  q("#frozen-reports-more").addEventListener("click", () => void loadReports(true));
  q("#resource-search").addEventListener("click", () => void browseResources());
  q("#resource-more").addEventListener("click", () => void browseResources(true));
  q("#attach-search-source").addEventListener("click", () => void attachSearchSource());
  q("#start-analysis").addEventListener("click", () => void startAnalysis());
  q("#suggest-followup").addEventListener("click", () => void suggestFollowup());
  q("#same-actor").addEventListener("click", () => void reviewAssociations("same_actor","identity"));
  q("#different-actors").addEventListener("click", () => void reviewAssociations("different_actors","identity"));
  q("#confirm-actor-relation").addEventListener("click", () => void reviewAssociations("related","relationship"));
  q("#unresolve-identity").addEventListener("click", () => void reviewAssociations("unresolved","identity"));
  q("#withdraw-actor-relation").addEventListener("click", () => void reviewAssociations("unresolved","relationship"));
  q("#retry-association-review").addEventListener("click", () => void publishAssociationReview());
  q("#edit-association-review").addEventListener("click", () => {
    if (reviewBusy) return;
    reviewAttempt = null; updateReviewSelection();
    setStatus("Decision draft unlocked. If the previous request reached the host, a new submission will require reloading the current case revision.");
  });
  for (const type of ["actors", "actor_mentions", "evidence_links", "actor_groups", "actor_relationships", "association_reviews"] as AnalysisItemType[])
    q(`#${type}-more`).addEventListener("click", () => void loadAnalysisItems(type, true));
  q("#statements-more").addEventListener("click", () => void loadAnalysisItems("statements", true));
  q("#comparisons-more").addEventListener("click", () => void loadAnalysisItems("comparisons", true));
  q<HTMLInputElement>("#resource-upload").addEventListener("change", async (event) => {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    if (!currentCase) {
      setStatus("Open or create a case before uploading a source.", true);
      input.value = "";
      return;
    }
    const token = generation;
    const caseId = currentCase.case_id;
    const revision = currentCase.revision;
    const kind = q<HTMLSelectElement>("#attach-kind").value;
    try {
      if (file.size > 4 * 1024 * 1024) throw new Error("Investigation uploads are limited to 4 MiB per source.");
      const uploaded = await rpc("ui/llama-agent/upload-investigation-resource", { file });
      if (uploaded?.resource_uri && currentCase?.case_id === caseId && currentCase?.revision === revision)
        await attachResource(String(uploaded.resource_uri), token, kind);
    } catch (error) { if (token === generation) setStatus(error instanceof Error ? error.message : String(error), true); }
    finally { input.value = ""; }
  });
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
    .then(async () => {
      parent.postMessage({ jsonrpc: "2.0", method: "ui/notifications/initialized", params: {} }, "*");
      void populateConfiguredSources().catch((error) => setStatus(error instanceof Error ? error.message : String(error), true));
      const page = await rpc("ui/llama-agent/list-investigation-resources", { scope: "session", limit: 1 });
      const projectOption = q<HTMLSelectElement>("#resource-scope").querySelector<HTMLOptionElement>('option[value="project"]');
      if (projectOption) projectOption.disabled = !page?.project_available;
      setStatus("Ready. Open or create a case, then select sources and an analysis.");
    })
    .catch((error) => setStatus(error instanceof Error ? error.message : String(error), true));
}
