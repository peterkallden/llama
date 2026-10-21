// @vitest-environment node
import { createHash, webcrypto } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadReportSnapshots } from "./report-snapshots";

const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/4XcAAAAASUVORK5CYII=", "base64");
const sha = (bytes: string | Uint8Array) => createHash("sha256").update(bytes).digest("hex");
function fixture() {
  vi.stubGlobal("crypto", webcrypto);
  const manifest = JSON.stringify({ schema: "investigation-report-source.v1", case_revision: 4,
    visualization_snapshots: [{ uri: "agent-resource://resource/png", visualization_uri: "agent-resource://resource/graph",
      sha256: sha(png), size_bytes: png.length }] });
  const document = { snapshot_uri: "agent-resource://resource/manifest", source_resource_uri: "agent-resource://resource/manifest",
    snapshot_fingerprint: `sha256:${sha(manifest)}`, source_content_fingerprint: `sha256:${sha(manifest)}` };
  const read = vi.fn(async (uri: string) => uri === document.snapshot_uri
    ? new Blob([manifest], { type: "application/json" }) : new Blob([Uint8Array.from(png).buffer], { type: "image/png" }));
  return { document, read };
}
afterEach(() => vi.unstubAllGlobals());
describe("saved report snapshot images", () => {
  it("loads the frozen PNG and retains its source projection after document edits", async () => {
    const { document, read } = fixture();
    const images = await loadReportSnapshots({ ...document, revision: 5, content_fingerprint: "edited-document" }, read);
    expect(images).toHaveLength(1);
    expect(images[0].blob?.type).toBe("image/png");
    expect(images[0].sourceUri).toBe("agent-resource://resource/graph");
    expect(images[0].caption).toContain("revision 4");
  });
  it("rejects a changed manifest before reading image references", async () => {
    const { document, read } = fixture();
    await expect(loadReportSnapshots({ ...document, snapshot_fingerprint: "sha256:wrong" }, read)).rejects.toThrow("integrity");
    expect(read).toHaveBeenCalledTimes(1);
  });
  it("shows a per-image error when snapshot access has been revoked", async () => {
    const { document, read } = fixture();
    const guarded = async (uri: string) => {
      if (uri !== document.snapshot_uri) throw new Error("Resource access denied");
      return read(uri);
    };
    const images = await loadReportSnapshots(document, guarded);
    expect(images[0].blob).toBeUndefined();
    expect(images[0].error).toContain("access denied");
  });
  it("rejects image bytes that differ from the frozen resource hash", async () => {
    const { document, read } = fixture();
    const corrupted = Buffer.from(png); corrupted[40] ^= 1;
    const images = await loadReportSnapshots(document, (uri) => uri === document.snapshot_uri ? read(uri)
      : Promise.resolve(new Blob([Uint8Array.from(corrupted).buffer], { type: "image/png" })));
    expect(images[0].blob).toBeUndefined();
    expect(images[0].error).toContain("integrity");
  });
});
