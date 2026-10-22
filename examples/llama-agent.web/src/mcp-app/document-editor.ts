import Editor from "@toast-ui/editor";
import "@toast-ui/editor/dist/toastui-editor.css";
import "@toast-ui/editor/dist/theme/toastui-editor-dark.css";
import appCss from "./document-editor.css?inline";

type RpcResult = { structuredContent?: Record<string, unknown>; content?: Array<{ text?: string }>; isError?: boolean };
const appStyle = document.createElement("style");appStyle.textContent=appCss;document.head.append(appStyle);
const pending = new Map<number, { resolve: (value: RpcResult) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();
let requestId = 1;
let editor: Editor | undefined;
let documentUri = "";
let markdownUri = "";
let revision = 0;
let documentId = "";
let documentKind = "document";
let savedMarkdown = "";
let savedTitle = "Untitled document";
let selectedBlockIds = new Set<string>();
let lastKnownBlockIds = new Set<string>();
let dirty = false;
let editSequence = 0;
let activeProposal = "";
let openGeneration = 0;
let mutationBusy = false;
let reviewBusy = false;
let suppressDraftChanges = false;
type SaveAttempt = { document_uri: string; expected_revision: number; title?: string; markdown: string; operation_id: string };
let saveAttempt: SaveAttempt | undefined;
let applyAttempt: { document_uri: string; proposal_uri: string; expected_revision: number; operation_id: string } | undefined;
let reviewAttempt: { document_uri: string; base_uri: string; revision: number; operation_id: string } | undefined;
let closing = false;
let closePromise: Promise<boolean> | undefined;
let switchPromise: Promise<boolean> | undefined;

function updateButtons() {
  for (const id of ["save", "review", "reload", "accept-proposal"]) {
    const button = document.querySelector<HTMLButtonElement>(`#${id}`);
    if (button) button.disabled = mutationBusy || reviewBusy || !editor;
  }
  const download = document.querySelector<HTMLButtonElement>("#download");
  if (download) download.disabled = !editor;
}

function lockEditor(locked: boolean) {
  const mount = document.querySelector<HTMLElement>("#editor");
  if (mount) mount.inert = locked;
}

function replaceEditorMarkdown(markdown: string) {
  if (!editor) throw new Error("Open a document before replacing its Markdown.");
  // Toast UI Editor 3 exposes setMarkdown at runtime, but its bundled default
  // export type omits the method from some TypeScript resolution paths.
  const editorApi = editor as Editor & { setMarkdown?: (value: string, cursorToEnd?: boolean) => void };
  if (typeof editorApi.setMarkdown !== "function") throw new Error("The editor cannot restore the saved Markdown draft.");
  editorApi.setMarkdown(markdown, false);
}

async function draft(action: "read" | "write", value?: Record<string, unknown>, uri = documentUri, base = revision): Promise<Record<string, unknown>> {
  const result = await rpc("ui/llama-agent/document-draft", { action, document_uri: uri, revision: base, ...(value ? { draft: value } : {}) });
  if (result.isError) structured(result);
  return result as Record<string, unknown>;
}

function retainDraft() {
  if (!editor || !documentUri) return;
  void draft("write", { title: currentTitle(), markdown: editor.getMarkdown(), dirty, pending_save: saveAttempt, pending_review: reviewAttempt, pending_apply: applyAttempt })
    .then((result) => { if (!result.persistent) status("Draft retained in host memory; browser persistence is unavailable.", true); })
    .catch((error) => status(`Draft storage failed; keep this editor open. ${String(error)}`, true));
}

function rpc(method: string, params: Record<string, unknown> = {}): Promise<RpcResult> {
  const id = requestId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error("The host did not respond. Your draft and operation ID are retained for retry."));
    }, method === "ui/message" ? 30 * 60 * 1000 : 210000);
    pending.set(id, { resolve, reject, timer });
    parent.postMessage({ jsonrpc: "2.0", id, method, params }, "*");
  });
}

function operationId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function status(message: string, isError = false) {
  const node = document.querySelector<HTMLElement>("#status");
  if (node) { node.textContent = message; node.dataset.error = isError ? "true" : "false"; }
  publishEditorStatus(message, isError);
}

function publishEditorStatus(message = "", isError = false) {
  if (!documentUri) return;
  const title = currentTitle();
  parent.postMessage({ jsonrpc: "2.0", method: "ui/llama-agent/editor-status", params: {
    document_uri: documentUri, document_id: documentId, kind: documentKind, revision, title,
    dirty, busy: mutationBusy, message, error: isError,
  } }, "*");
}

function currentTitle(): string {
  const node = document.querySelector<HTMLInputElement>("#document-title");
  return (node?.value ?? node?.textContent ?? savedTitle).trim();
}

function structured(result: RpcResult): Record<string, unknown> {
  if (result.isError) throw new Error(result.content?.map((item) => item.text ?? "").join("\n") || "Tool call failed.");
  return result.structuredContent ?? {};
}

function showBibliography(items: unknown) {
  const list = document.querySelector<HTMLElement>("#sources");
  if (!list) return;
  list.replaceChildren();
  if (!Array.isArray(items) || !items.length) { list.textContent = "No source details."; return; }
  for (const value of items) {
    if (!value || typeof value !== "object") continue;
    const source = value as Record<string, unknown>;
    const item = document.createElement("li");
    const title = String(source.title || source.source_id || "Source");
    const uri = String(source.original_uri || source.resolved_uri || source.file_name || source.resource_uri || "");
    const label = document.createElement("span"); label.textContent = title;
    item.append(label);
    if (/^https?:\/\//i.test(uri)) {
      const link = document.createElement("a"); link.href = uri; link.textContent = uri; link.rel = "noreferrer"; link.target = "_blank"; item.append(link);
    } else if (uri) { const detail = document.createElement("small"); detail.textContent = uri; item.append(detail); }
    list.append(item);
  }
}

function showEvidenceStatus(documentValue: unknown) {
  const list = document.querySelector<HTMLElement>("#evidence-status");
  if (!list) return;
  list.replaceChildren();
  const doc = documentValue && typeof documentValue === "object" ? documentValue as Record<string, unknown> : {};
  const sections = Array.isArray(doc.sections) ? doc.sections : [];
  for (const sectionValue of sections) {
    if (!sectionValue || typeof sectionValue !== "object") continue;
    const section = sectionValue as Record<string, unknown>;
    const blocks = Array.isArray(section.blocks) ? section.blocks : [];
    for (const blockValue of blocks) {
      if (!blockValue || typeof blockValue !== "object") continue;
      const block = blockValue as Record<string, unknown>;
      const item = document.createElement("li");
      const select = document.createElement("input");
      select.type = "checkbox";
      select.checked = selectedBlockIds.has(String(block.id ?? ""));
      select.setAttribute("aria-label", `Include block ${String(block.id ?? "")} in review`);
      select.addEventListener("change", () => {
        const id = String(block.id ?? "");
        if (select.checked) selectedBlockIds.add(id); else selectedBlockIds.delete(id);
      });
      const text = document.createElement("span");
      text.textContent = String(block.text ?? "").slice(0, 180);
      const state = document.createElement("small");
      state.textContent = `${String(block.author ?? "unknown")} · ${String(block.review_state ?? "unreviewed")} · ${String(block.support ?? "unverified")}`;
      item.append(select, text, state);
      list.append(item);
    }
  }
  if (!list.childElementCount) list.textContent = "No saved claim blocks.";
}

async function openDocument(doc: Record<string, unknown>, uri: string) {
  if (mutationBusy) return;
  if (dirty && uri !== documentUri && !(await prepareDocumentSwitch())) return;
  const generation = ++openGeneration;
  const sequence = editSequence;
  const loadedRevision = Number(doc.revision ?? 0);
  const response = await draft("read", undefined, uri, loadedRevision);
  if (generation !== openGeneration || sequence !== editSequence || mutationBusy) return;
  documentUri = uri;
  markdownUri = String(doc.markdown_uri ?? "");
  revision = Number(doc.revision ?? 0);
  documentId = String(doc.id ?? "");
  documentKind = String(doc.kind ?? "document");
  const initialDocument = doc.document && typeof doc.document === "object" ? doc.document as Record<string, unknown> : {};
  selectedBlockIds = new Set<string>();
  for (const section of Array.isArray(initialDocument.sections) ? initialDocument.sections : []) {
    if (!section || typeof section !== "object") continue;
    for (const block of Array.isArray((section as Record<string, unknown>).blocks) ? (section as Record<string, unknown>).blocks as unknown[] : []) {
      if (block && typeof block === "object" && typeof (block as Record<string, unknown>).id === "string") selectedBlockIds.add((block as Record<string, unknown>).id as string);
    }
  }
  lastKnownBlockIds = new Set(selectedBlockIds);
  const markdown = String(doc.markdown ?? "");
  savedMarkdown = markdown;
  savedTitle = String(doc.title ?? "Untitled document");
  const stored = response.draft && typeof response.draft === "object" ? response.draft as Record<string, unknown> : {};
  // A saved projection may differ textually after host Markdown normalization.
  // Restore only actual unsaved drafts; pending review metadata remains valid.
  const content = stored.dirty !== false && typeof stored.markdown === "string" ? stored.markdown : markdown;
  const titleNode = document.querySelector<HTMLInputElement>("#document-title");
  if (titleNode) { titleNode.value = stored.dirty !== false && typeof stored.title === "string" ? stored.title : savedTitle; titleNode.disabled = documentKind !== "document"; }
  const pendingSave = stored.pending_save as SaveAttempt | undefined;
  saveAttempt = pendingSave?.document_uri === uri && pendingSave.expected_revision === revision &&
    typeof pendingSave.markdown === "string" && typeof pendingSave.operation_id === "string" ? pendingSave : undefined;
  const pendingReview = stored.pending_review as Record<string, unknown> | undefined;
  reviewAttempt = pendingReview?.document_uri === uri && pendingReview.revision === revision
    ? {
      document_uri: uri, base_uri: String(pendingReview.base_uri ?? uri), revision,
      operation_id: typeof pendingReview.operation_id === "string" ? pendingReview.operation_id : operationId(),
    }
    : undefined;
  const pendingApply = stored.pending_apply as typeof applyAttempt;
  applyAttempt = pendingApply?.document_uri === uri && pendingApply.expected_revision === revision ? pendingApply : undefined;
  if (applyAttempt) activeProposal = applyAttempt.proposal_uri;
  if (editor) editor.destroy();
  const mount = document.querySelector<HTMLElement>("#editor");
  if (!mount) return;
  editor = new Editor({
    el: mount,
    height: "auto",
    initialValue: content,
    initialEditType: "markdown",
    previewStyle: window.matchMedia?.("(max-width: 760px)").matches ? "tab" : "vertical",
    theme: "dark",
    usageStatistics: false,
    hideModeSwitch: false,
    toolbarItems: [["heading", "bold", "italic", "strike"], ["hr", "quote"], ["ul", "ol", "task"], ["table", "link"], ["code", "codeblock"]],
  });
  dirty = content !== markdown || currentTitle() !== savedTitle;
  editor.on("change", () => {
    if (suppressDraftChanges) return;
    editSequence++;
    dirty = true;
    retainDraft();
    status(`Unsaved draft · base revision ${revision}`);
    publishEditorStatus();
  });
  if (titleNode) titleNode.oninput = () => {
    if (suppressDraftChanges) return;
    editSequence++;
    dirty = true;
    retainDraft();
    status(`Unsaved draft · base revision ${revision}`);
    publishEditorStatus();
  };
  showBibliography(doc.bibliography);
  showEvidenceStatus(doc.document);
  status(`${String(doc.title ?? "Document")} · revision ${revision}${dirty ? " · local draft restored" : ""}`);
  updateButtons();
}

async function callOpen(uri: string) {
  if (mutationBusy || (dirty && !(await prepareDocumentSwitch()))) return;
  const generation = ++openGeneration;
  const sequence = editSequence;
  status("Loading saved document…");
  const result = structured(await rpc("tools/call", { name: "document.open", arguments: { document_uri: uri } }));
  if (generation !== openGeneration || sequence !== editSequence || mutationBusy) return;
  await openDocument(result, String(result.document_uri ?? uri));
}

async function saveRevision() {
  if (!editor || !documentUri || !documentId) throw new Error("Open a document before saving.");
  if (mutationBusy) throw new Error("A document operation is already running. Try again when it finishes.");
  mutationBusy = true; ++openGeneration; updateButtons();
  publishEditorStatus();
  try {
  const content = editor.getMarkdown();
  const title = documentKind === "document" ? currentTitle() : undefined;
  if (title !== undefined && (!title || new TextEncoder().encode(title).byteLength > 256)) throw new Error("Document title must contain 1-256 UTF-8 bytes.");
  const savedEditSequence = editSequence;
  saveAttempt ??= { document_uri: documentUri, expected_revision: revision, ...(title !== undefined ? { title } : {}), markdown: content, operation_id: operationId() };
  const attempt = saveAttempt;
  await draft("write", { title: currentTitle(), markdown: content, dirty, pending_save: attempt, pending_review: reviewAttempt });
  const result = structured(await rpc("tools/call", { name: "document.save", arguments: attempt }));
  documentUri = String(result.document_uri ?? "");
  markdownUri = String(result.markdown_uri ?? "");
  revision = Number(result.revision ?? revision + 1);
  if (attempt.title !== undefined) savedTitle = attempt.title;
  savedMarkdown = attempt.markdown;
  dirty = editSequence !== savedEditSequence || editor.getMarkdown() !== attempt.markdown || currentTitle() !== savedTitle;
  saveAttempt = undefined;
  if (!result.reused) {
    if (!reviewAttempt) reviewAttempt = { document_uri: documentUri, base_uri: attempt.document_uri, revision, operation_id: operationId() };
    else { reviewAttempt.document_uri = documentUri; reviewAttempt.revision = revision; }
  }
  await draft("write", { title: currentTitle(), markdown: editor.getMarkdown(), dirty, pending_review: reviewAttempt });
  status(dirty ? `Saved revision ${revision}; newer local edits remain unsaved.` : `Saved immutable revision ${revision}`);
  return result;
  } finally { mutationBusy = false; updateButtons(); publishEditorStatus(); }
}

async function refreshedSelection() {
  if (!documentUri) return;
  const latest = structured(await rpc("tools/call", { name: "document.open", arguments: { document_uri: documentUri } }));
  const latestDocument = latest.document && typeof latest.document === "object" ? latest.document as Record<string, unknown> : {};
  const available = new Set<string>();
  for (const section of Array.isArray(latestDocument.sections) ? latestDocument.sections : []) {
    if (!section || typeof section !== "object") continue;
    for (const block of Array.isArray((section as Record<string, unknown>).blocks) ? (section as Record<string, unknown>).blocks as unknown[] : []) {
      if (block && typeof block === "object" && typeof (block as Record<string, unknown>).id === "string") available.add((block as Record<string, unknown>).id as string);
    }
  }
  const hadAllBlocksSelected = selectedBlockIds.size === lastKnownBlockIds.size &&
    [...lastKnownBlockIds].every((id) => selectedBlockIds.has(id));
  selectedBlockIds = new Set([...selectedBlockIds].filter((id) => available.has(id)));
  if (hadAllBlocksSelected) for (const id of available) selectedBlockIds.add(id);
  lastKnownBlockIds = available;
  showEvidenceStatus(latestDocument);
  // An empty selection means the complete document to the host, which reads
  // it through its normal chunk policy. Do not serialize thousands of IDs.
  if (selectedBlockIds.size === available.size) return [];
  if (selectedBlockIds.size > 128) throw new Error("Select at most 128 blocks, or select the complete document.");
  if (!selectedBlockIds.size && available.size) throw new Error("Select document blocks before requesting chat or review.");
  return [...selectedBlockIds];
}

async function registerReview(closeAfter = false): Promise<boolean> {
  if (reviewBusy) throw new Error("A document review operation is already running.");
  reviewBusy = true; lockEditor(true); updateButtons();
  let closeAccepted = false;
  try {
    if (dirty) await saveRevision();
    if (dirty) throw new Error("Newer edits remain unsaved. Save again before requesting review.");
    if (!reviewAttempt) {
      if (closeAfter) {
        await rpc("ui/message", { action: "close-document" });
        closeAccepted = true;
      }
      status("No unreviewed document changes.");
      return closeAccepted;
    }
    const expectedEditSequence = editSequence;
    const blockIds = await refreshedSelection();
    if (dirty || editSequence !== expectedEditSequence) {
      throw new Error("The document changed while review context was being prepared. Save again before requesting review.");
    }
    const attempt = reviewAttempt;
    const content = "Review changes between the bound saved document revisions. Read the host-provided comparison resource through the normal chunked reading path. Verify changed claims using available support and counterevidence, and propose bounded patches without applying them.";
    await rpc("ui/message", {
      action: "review-document", operation_id: attempt.operation_id,
      content: { type: "text", text: content },
      document_context: { document_uri: attempt.document_uri, revision: attempt.revision,
        block_ids: blockIds ?? [], base_document_uri: attempt.base_uri },
    });
    reviewAttempt = undefined;
    await draft("write", { title: currentTitle(), markdown: editor?.getMarkdown() ?? "", dirty, pending_review: undefined });
    if (closeAfter) {
      if (dirty || editSequence !== expectedEditSequence) {
        status("Review queued; newer edits remain unsaved. Save them before closing.", true);
        return false;
      }
      await rpc("ui/message", { action: "close-document" });
      closeAccepted = true;
    }
    status(closeAfter ? "Review queued; closing editor." : "Review queued in chat.");
    return closeAccepted;
  } finally {
    reviewBusy = false;
    if (!closeAccepted) lockEditor(false);
    updateButtons();
  }
}

async function save(review: boolean) {
  if (dirty) await saveRevision();
  if (review) await registerReview(false);
  else status(dirty ? `Saved revision ${revision}; newer local edits remain unsaved.` : `Revision ${revision} saved. No agent turn was started.`);
}

function chooseSwitchAction(): Promise<"save" | "discard" | "cancel"> {
  const dialog = document.querySelector<HTMLDialogElement>("#switch-dialog");
  if (!dialog || typeof dialog.showModal !== "function") return Promise.resolve("cancel");
  return new Promise((resolve) => {
    let settled = false;
    const finish = (choice: "save" | "discard" | "cancel") => {
      if (settled) return;
      settled = true;
      dialog.close();
      resolve(choice);
    };
    for (const choice of ["save", "discard", "cancel"] as const) {
      const button = dialog.querySelector<HTMLButtonElement>(`[data-switch-choice="${choice}"]`);
      if (button) button.onclick = () => finish(choice);
    }
    dialog.oncancel = (event) => { event.preventDefault(); finish("cancel"); };
    dialog.showModal();
  });
}

function prepareDocumentSwitch(): Promise<boolean> {
  if (switchPromise) return switchPromise;
  if (mutationBusy || reviewBusy) return Promise.resolve(false);
  switchPromise = resolveDocumentSwitch().finally(() => { switchPromise = undefined; });
  return switchPromise;
}

async function resolveDocumentSwitch(): Promise<boolean> {
  if (!dirty) return true;
  const choice = await chooseSwitchAction();
  if (choice === "cancel") return false;
  if (choice === "save") {
    try {
      await saveRevision();
      return !dirty;
    } catch (cause) {
      status(String(cause), true);
      return false;
    }
  }
  if (!editor || mutationBusy) return false;
  mutationBusy = true; lockEditor(true); updateButtons();
  try {
    suppressDraftChanges = true;
    replaceEditorMarkdown(savedMarkdown);
    const title = document.querySelector<HTMLInputElement>("#document-title");
    if (title) title.value = savedTitle;
    suppressDraftChanges = false;
    dirty = false; saveAttempt = undefined; ++editSequence;
    await draft("write", { title: savedTitle, markdown: savedMarkdown, dirty: false });
    status(`Discarded local changes · revision ${revision}`);
    publishEditorStatus();
    return true;
  } catch (cause) {
    suppressDraftChanges = false;
    status(`Could not discard the local draft: ${String(cause)}`, true);
    return false;
  } finally {
    mutationBusy = false; lockEditor(false); updateButtons();
  }
}

async function downloadCurrentDocument() {
  if (!editor || !documentUri || !documentId) return;
  try {
    const result = await rpc("ui/llama-agent/document-download", {
      document_uri: documentUri, document_id: documentId, title: currentTitle(), markdown: editor.getMarkdown(),
    });
    if (result.isError) structured(result);
    const value = result as Record<string, unknown>;
    status(typeof value.filename === "string" ? `Downloaded ${value.filename}` : "Downloaded current Markdown.");
  } catch (cause) { status(String(cause), true); }
}

function closeEditor(): Promise<boolean> {
  if (closePromise) return closePromise;
  if (mutationBusy) return Promise.resolve(false);
  closing = true; updateButtons();
  closePromise = (async () => {
    try {
      if (documentKind === "document") {
        if (!(await prepareDocumentSwitch())) return false;
        await rpc("ui/message", { action: "close-document" });
        status("Document closed.");
        return true;
      }
      return await registerReview(true);
    }
    catch (cause) { status(String(cause), true); return false; }
    finally { closing = false; closePromise = undefined; updateButtons(); }
  })();
  return closePromise;
}

window.addEventListener("message", (event) => {
  if (event.source !== parent) return;
  const message = event.data as Record<string, unknown>;
  if (message?.jsonrpc !== "2.0") return;
  if (typeof message.id === "number" && pending.has(message.id)) {
    const call = pending.get(message.id)!; pending.delete(message.id); clearTimeout(call.timer);
    if (message.error && typeof message.error === "object") call.reject(new Error(String((message.error as Record<string, unknown>).message ?? "MCP request failed.")));
    else call.resolve(message.result as RpcResult);
    return;
  }
  if (typeof message.id === "number" && message.method === "ui/llama-agent/prepare-chat") {
    void (async () => {
      try {
        if (dirty) await saveRevision();
        if (dirty) throw new Error("Newer edits remain unsaved. Retry sending after saving.");
        const expectedEditSequence = editSequence;
        const blockIds = await refreshedSelection();
        if (dirty || editSequence !== expectedEditSequence) throw new Error("The document changed while chat context was being prepared. Retry after saving.");
        replyToHost(message.id as number, { document_uri: documentUri, revision, block_ids: blockIds ?? [] });
      } catch (cause) { replyToHost(message.id as number, undefined, cause instanceof Error ? cause.message : String(cause)); }
    })();
    return;
  }
  if (typeof message.id === "number" && message.method === "ui/llama-agent/prepare-document-switch") {
    void prepareDocumentSwitch().then((ready) => replyToHost(message.id as number,
      ready ? { ready: true } : undefined, ready ? undefined : "Document switching was canceled or the draft could not be saved."));
    return;
  }
  if (typeof message.id === "number" && message.method === "ui/llama-agent/close-document") {
    void closeEditor().then((closed) => {
      if (closed) replyToHost(message.id as number, { closed: true });
      else replyToHost(message.id as number, undefined, "The document could not be closed.");
    });
    return;
  }
  if (message.method === "ui/notifications/tool-input") {
    // The host sends the matching tool-result notification immediately after
    // the input. It already carries the authorized document projection; making
    // a second document.open call here is redundant and can race the result.
    status("Loading document result…");
  }
  if (message.method === "ui/notifications/tool-result") {
    const params = message.params as RpcResult | undefined;
    if (params?.structuredContent && typeof params.structuredContent === "object") {
      const doc = params.structuredContent as Record<string, unknown>;
      if (typeof doc.document_uri === "string" && typeof doc.markdown === "string") void openDocument(doc, doc.document_uri).catch((error) => status(String(error), true));
      if (typeof doc.proposal_uri === "string") {
        activeProposal = doc.proposal_uri;
        const panel = document.querySelector<HTMLElement>("#proposal");
        const summary = document.querySelector<HTMLElement>("#proposal-summary");
        const base = document.querySelector<HTMLElement>("#proposal-base");
        if (panel) panel.hidden = false;
        if (summary) summary.textContent = String(doc.summary ?? "The agent proposed a document edit.");
        if (base) base.textContent = `Based on document revision ${String(doc.base_revision ?? "unknown")}. Acceptance creates a new revision and does not verify factual claims.`;
        const changes = document.querySelector<HTMLElement>("#proposal-changes");
        if (changes) changes.textContent = JSON.stringify(doc.diff ?? {}, null, 2);
      }
    }
  }
});

const downloadButton = document.querySelector<HTMLButtonElement>("#download");
downloadButton?.addEventListener("click", () => { void downloadCurrentDocument(); });

document.querySelector<HTMLButtonElement>("#save")?.addEventListener("click", () => void save(false).catch((error) => status(String(error), true)));
document.querySelector<HTMLButtonElement>("#review")?.addEventListener("click", () => void save(true).catch((error) => status(String(error), true)));
document.querySelector<HTMLButtonElement>("#toggle-side")?.addEventListener("click", () => {
  const side = document.querySelector<HTMLElement>("#side-content");
  const button = document.querySelector<HTMLButtonElement>("#toggle-side");
  const expanded = button?.getAttribute("aria-expanded") !== "true";
  if (side) side.hidden = !expanded;
  button?.setAttribute("aria-expanded", String(expanded));
  const state = document.querySelector<HTMLElement>("#side-state"); if (state) state.textContent = expanded ? "−" : "+";
});
document.querySelector<HTMLButtonElement>("#reload")?.addEventListener("click", () => void callOpen(documentUri).catch((error) => status(String(error), true)));
document.querySelector<HTMLButtonElement>("#licenses")?.addEventListener("click", () => { const dialog=document.querySelector<HTMLDialogElement>("#license-dialog");dialog?.showModal(); });
document.querySelector<HTMLButtonElement>("#close-licenses")?.addEventListener("click", () => document.querySelector<HTMLDialogElement>("#license-dialog")?.close());
document.querySelector<HTMLButtonElement>("#reject-proposal")?.addEventListener("click", () => {
  activeProposal = "";
  const panel = document.querySelector<HTMLElement>("#proposal");
  if (panel) panel.hidden = true;
});
document.querySelector<HTMLButtonElement>("#accept-proposal")?.addEventListener("click", () => void (async () => {
  if (!activeProposal || !documentUri) return;
  if (dirty || mutationBusy) throw new Error("Save your current draft before accepting a proposal.");
  mutationBusy = true; ++openGeneration; updateButtons();
  try {
  const appliedEditSequence = editSequence;
  applyAttempt ??= { document_uri: documentUri, proposal_uri: activeProposal, expected_revision: revision, operation_id: operationId() };
  await draft("write", { title: currentTitle(), markdown: editor?.getMarkdown() ?? "", pending_apply: applyAttempt, pending_review: reviewAttempt });
  const result = structured(await rpc("tools/call", { name: "document.apply_patch", arguments: applyAttempt }));
  applyAttempt = undefined;
  activeProposal = "";
  const panel = document.querySelector<HTMLElement>("#proposal");
  if (panel) panel.hidden = true;
  documentUri = String(result.document_uri ?? documentUri);
  revision = Number(result.revision ?? revision + 1);
  dirty = editSequence !== appliedEditSequence;
  if (dirty) await draft("write", { title: currentTitle(), markdown: editor?.getMarkdown() ?? "" });
  } finally { mutationBusy = false; updateButtons(); }
  await callOpen(documentUri);
})().catch((error) => status(String(error), true)));
window.addEventListener("beforeunload", (event) => { if (dirty) event.preventDefault(); });

function replyToHost(id: number, result?: unknown, error?: string) {
  parent.postMessage({ jsonrpc: "2.0", id, ...(error ? { error: { code: -32000, message: error } } : { result }) }, "*");
}

void rpc("ui/initialize", { protocolVersion: "2026-01-26", appInfo: { name: "llama-agent-document-editor", version: "0.1" }, appCapabilities: { tools: {} } })
  .then(() => parent.postMessage({jsonrpc:"2.0",method:"ui/notifications/initialized",params:{}},"*"))
  .catch((error) => status(String(error), true));
