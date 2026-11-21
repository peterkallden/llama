import { object, type Fields } from "./report-view";

export type ReportSnapshot = { uri: string; sourceUri: string; caption: string; blob?: Blob; error?: string };
const resourceUri = (value: unknown): value is string => typeof value === "string" &&
  /^(?:agent-resource|resource):\/\/[^\s<>]+$/.test(value) && value.length <= 512;
async function fingerprint(blob: Blob): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

// Read only immutable references from a manifest whose bytes match the opened
// document. Image URLs are created by the view only after its generation check.
export async function loadReportSnapshots(document: Fields, readBlob: (uri: string) => Promise<Blob>): Promise<ReportSnapshot[]> {
  if (!resourceUri(document.snapshot_uri)) throw new Error("The report has no valid saved source manifest.");
  const manifestBlob = await readBlob(document.snapshot_uri);
  if (manifestBlob.size > 1024 * 1024) throw new Error("The report source manifest exceeds its reading limit.");
  const expected = `sha256:${await fingerprint(manifestBlob)}`;
  if (document.snapshot_fingerprint !== expected || document.source_content_fingerprint !== expected ||
      document.source_resource_uri !== document.snapshot_uri)
    throw new Error("The report source manifest failed integrity validation.");
  const manifest = object(JSON.parse(await manifestBlob.text()));
  if (manifest.schema !== "investigation-report-source.v1" ||
      !Array.isArray(manifest.visualization_snapshots) || manifest.visualization_snapshots.length > 8)
    throw new Error("The snapshot manifest does not match this report revision.");
  const results: ReportSnapshot[] = [];
  for (const [index, raw] of manifest.visualization_snapshots.entries()) {
    const entry = object(raw);
    if (!resourceUri(entry.uri) || !resourceUri(entry.visualization_uri) ||
        typeof entry.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(entry.sha256) ||
        typeof entry.size_bytes !== "number" || !Number.isSafeInteger(entry.size_bytes) || Number(entry.size_bytes) < 45 || Number(entry.size_bytes) > 512 * 1024)
      throw new Error("The report contains an invalid snapshot reference.");
    const image: ReportSnapshot = { uri: entry.uri, sourceUri: entry.visualization_uri,
      caption: `Snapshot ${index + 1} · frozen case revision ${manifest.case_revision}` };
    try {
      const blob = await readBlob(entry.uri);
      if (blob.type !== "image/png" || blob.size !== entry.size_bytes || blob.size > 512 * 1024 ||
          await fingerprint(blob) !== entry.sha256)
        throw new Error("The snapshot failed media type, size or integrity validation.");
      const signature = new Uint8Array(await blob.slice(0, 8).arrayBuffer());
      if (![137, 80, 78, 71, 13, 10, 26, 10].every((byte, position) => signature[position] === byte))
        throw new Error("The saved snapshot is not a PNG image.");
      image.blob = blob;
    } catch (cause) { image.error = cause instanceof Error ? cause.message : String(cause); }
    results.push(image);
  }
  return results;
}
