import { describe, expect, it } from "vitest";
import { DocumentEditorAccess } from "./document-editor-access";

const open = (access: DocumentEditorAccess, uri = "resource://r1") => access.observe("document.open",
  { document_uri: uri }, { document_uri: uri, id: "report", revision: 1 });

describe("document editor host grant and draft storage", () => {
  it("requires a host-confirmed open and binds writes to that document", () => {
    const access = new DocumentEditorAccess("session", "native");
    const args = { document_uri: "resource://r1", expected_revision: 1, operation_id: "op" };
    expect(() => access.authorize("document.save", args)).toThrow();
    open(access);
    expect(() => access.authorize("document.save", args)).not.toThrow();
    expect(() => access.authorize("document.apply_patch", { ...args, document_uri: "resource://other" })).toThrow();
    access.observe("document.save", args, { document_uri: "resource://r2", revision: 2 });
    expect(() => access.authorize("document.save", args)).not.toThrow(); // exact retry after lost response
    expect(() => access.authorize("document.save", { ...args, operation_id: "other" })).toThrow();
  });

  it("retains drafts when host storage throws and isolates sessions and revisions", () => {
    const storage = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("quota"); } } as unknown as Storage;
    const access = new DocumentEditorAccess("draft-session", "native"); open(access);
    const params = { document_uri: "resource://r1", revision: 1 };
    expect(access.draft({ ...params, action: "write", draft: { markdown: "Local changes" } }, storage).persistent).toBe(false);
    const reopened = new DocumentEditorAccess("draft-session", "native"); open(reopened);
    expect(reopened.draft({ ...params, action: "read" }, storage).draft).toEqual({ markdown: "Local changes" });
    const other = new DocumentEditorAccess("other-session", "native"); open(other);
    expect(other.draft({ ...params, action: "read" }, storage).draft).toBeUndefined();
    expect(() => access.draft({ ...params, revision: 2, action: "read" }, storage)).toThrow();
  });

  it("keeps the open-document grant when a later invocation arrives during editing", () => {
    const access = new DocumentEditorAccess("late-invocation", "native"); open(access);
    access.draft({ document_uri: "resource://r1", revision: 1, action: "write", draft: { markdown: "Unsaved" } });
    open(access); // A late reload must not clear the dirty grant's protection.
    access.observe("document.open", { document_uri: "resource://other" }, { document_uri: "resource://other", id: "other", revision: 1 });
    expect(() => access.authorize("document.save", { document_uri: "resource://r1", expected_revision: 1 })).not.toThrow();
    expect(() => access.authorize("document.save", { document_uri: "resource://other", expected_revision: 1 })).toThrow();
  });

  it("prefers the newest in-memory draft after a persistent write failed", () => {
    const access = new DocumentEditorAccess("quota-session", "native"); open(access);
    const storage = { getItem: () => JSON.stringify({ markdown: "Old" }), setItem: () => { throw new Error("quota"); } } as unknown as Storage;
    const params = { document_uri: "resource://r1", revision: 1 };
    access.draft({ ...params, action: "write", draft: { markdown: "Newest" } }, storage);
    expect(access.draft({ ...params, action: "read" }, storage).draft).toEqual({ markdown: "Newest" });
  });

  it("retains title edits and exports only the currently opened document draft", () => {
    const access = new DocumentEditorAccess("download-session", "native");
    access.observe("document.open", { document_uri: "resource://doc-r1" }, {
      document_uri: "resource://doc-r1", id: "doc-1", revision: 1, title: "Untitled document", markdown: "Saved",
    });
    const params = { document_uri: "resource://doc-r1", revision: 1 };
    access.draft({ ...params, action: "write", draft: { title: "Notes?", markdown: "Unsaved exact\n", dirty: true } });
    expect(access.draft({ ...params, action: "read" }).draft).toEqual({ title: "Notes?", markdown: "Unsaved exact\n", dirty: true });
    expect(access.authorizeDownload({ document_uri: params.document_uri, document_id: "doc-1", title: "Notes?", markdown: "Unsaved exact\n" }))
      .toEqual({ filename: "Notes_.md", markdown: "Unsaved exact\n" });
    expect(access.authorizeDownload({ document_uri: params.document_uri, document_id: "doc-1", title: "NUL", markdown: "draft" }).filename).toBe("_NUL.md");
    expect(() => access.authorizeDownload({ document_uri: params.document_uri, document_id: "other", title: "x", markdown: "secret" })).toThrow();
  });
});
