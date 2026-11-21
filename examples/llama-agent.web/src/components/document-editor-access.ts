// The sandbox has an opaque origin. Draft storage belongs to its web host,
// scoped to the session, provider and host-confirmed open document.
type Fields = Record<string, unknown>;
const transientDrafts = new Map<string, string>();

export class DocumentEditorAccess {
  private uri = "";
  private id = "";
  private revision = 0;
  private completedWrite = "";
  private savedMarkdown = "";
  private draftMarkdown = "";
  private documentTitle = "";
  private draftTitle = "";
  private dirty = false;
  constructor(private readonly session: string, private readonly provider: string) {}

  observe(name: string, args: Fields, result: Fields) {
    if (name === "document.open" && typeof result.document_uri === "string" &&
        result.document_uri === args.document_uri && typeof result.id === "string" &&
        Number.isSafeInteger(result.revision) && Number(result.revision) > 0) {
      if (this.uri && this.dirty) return;
      this.uri = result.document_uri; this.id = result.id; this.revision = Number(result.revision);
      this.savedMarkdown = String(result.markdown ?? ""); this.draftMarkdown = this.savedMarkdown;
      this.documentTitle = String(result.title ?? "Document"); this.draftTitle = this.documentTitle;
      this.dirty = false;
      this.completedWrite = "";
    } else if ((name === "document.save" || name === "document.apply_patch") &&
        args.document_uri === this.uri && typeof result.document_uri === "string" &&
        Number.isSafeInteger(result.revision) && Number(result.revision) >= this.revision) {
      this.uri = result.document_uri; this.revision = Number(result.revision);
      if (typeof args.markdown === "string") this.savedMarkdown = args.markdown;
      if (typeof args.title === "string") this.documentTitle = args.title;
      this.draftMarkdown = this.savedMarkdown; this.draftTitle = this.documentTitle;
      this.dirty = false;
      this.completedWrite = JSON.stringify([name, args]);
    }
  }

  authorize(name: string, args: Fields) {
    if (this.completedWrite && this.completedWrite === JSON.stringify([name, args])) return;
    if (["document.save", "document.apply_patch", "document.propose_patch"].includes(name) &&
        (!this.uri || args.document_uri !== this.uri || args.expected_revision !== this.revision)) {
      throw new Error("The editor may modify only its currently opened document revision.");
    }
  }

  draft(params: Fields, storage?: Storage): Fields {
    if (!this.uri || params.document_uri !== this.uri || params.revision !== this.revision) {
      throw new Error("Draft access does not match the opened document revision.");
    }
    const key = `llama-agent-document:${JSON.stringify([this.session, this.provider, this.id, this.revision])}`;
    let persistent = Boolean(storage);
    let stored = transientDrafts.get(key);
    if (params.action === "write") {
      if (!params.draft || typeof params.draft !== "object" || Array.isArray(params.draft)) throw new Error("Invalid draft.");
      const draft = params.draft as Fields;
      if (typeof draft.markdown !== "string" || new TextEncoder().encode(draft.markdown).byteLength > 4 * 1024 * 1024) throw new Error("Draft exceeds the document limit.");
      if (draft.title !== undefined && (typeof draft.title !== "string" || draft.title.length > 256)) throw new Error("Draft title exceeds the document limit.");
      stored = JSON.stringify(draft);
      this.draftMarkdown = draft.markdown;
      this.draftTitle = typeof draft.title === "string" ? draft.title : this.documentTitle;
      transientDrafts.set(key, stored);
      try { storage?.setItem(key, stored); } catch { persistent = false; }
    } else if (params.action === "read") {
      try { stored ??= storage?.getItem(key) ?? undefined; } catch { persistent = false; }
    } else throw new Error("Unsupported draft operation.");
    let draft: unknown;
    try { draft = stored ? JSON.parse(stored) : undefined; } catch { draft = undefined; }
    if (draft && typeof draft === "object" && typeof (draft as Fields).markdown === "string") {
      this.draftMarkdown = (draft as Fields).markdown as string;
      this.draftTitle = typeof (draft as Fields).title === "string" ? (draft as Fields).title as string : this.documentTitle;
      this.dirty = (draft as Fields).dirty === true || ((draft as Fields).dirty !== false &&
        (this.draftMarkdown !== this.savedMarkdown || this.draftTitle !== this.documentTitle));
    }
    return { draft, persistent };
  }

  authorizeDownload(params: Fields): { filename: string; markdown: string } {
    if (!this.uri || params.document_uri !== this.uri || params.document_id !== this.id) {
      throw new Error("Download request does not match the currently opened document.");
    }
    if (typeof params.markdown !== "string" || new TextEncoder().encode(params.markdown).byteLength > 4 * 1024 * 1024 ||
        typeof params.title !== "string" || params.title.length > 256) {
      throw new Error("Download content is invalid or exceeds the document limit.");
    }
    let filename = params.title.trim().replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").replace(/[. ]+$/g, "").slice(0, 120) || "Untitled document";
    if (/^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/i.test(filename)) filename = `_${filename}`;
    return { filename: `${filename}.md`, markdown: params.markdown };
  }
}
