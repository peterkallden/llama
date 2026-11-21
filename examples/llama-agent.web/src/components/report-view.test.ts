import { describe, expect, it } from "vitest";
import { blockTargets, externalUri, reportReferences, savedToolPayload, sourceTargets } from "./report-view";

describe("saved report discovery", () => {
  it("uses structured media type and session identity, deduplicating created/attached events", () => {
    const report = { session_id: "a", turn_id: "turn", resource_uri: "agent-resource://opaque",
      resource_name: "report.json", resource_mime_type: "application/vnd.llama-agent.report+json" };
    const result = reportReferences([
      { ...report, type: "resource.attached" }, { ...report, type: "resource.created" },
      { ...report, session_id: "b", resource_uri: "agent-resource://other-session" },
      { session_id: "a", detail: "agent-resource://invented-report.json" },
      { ...report, resource_uri: "agent-resource://ordinary", resource_mime_type: "application/json" },
    ], "a");
    expect(result).toEqual([{ uri: report.resource_uri, turnId: "turn", name: "report.json" }]);
  });
  it("keeps separately published immutable revisions and command-owned reports", () => {
    expect(reportReferences([1, 2].map((revision) => ({ session_id: "a", request_id: "request",
      resource_uri: `agent-resource://r${revision}`, resource_mime_type: "application/vnd.llama-agent.report+json" })), "a")).toHaveLength(2);
  });
});

describe("report provenance", () => {
  const document = { source_bibliography: [
    { source_id: "source", title: "Study", resource_uri: "agent-resource://paper", original_uri: "https://example.org/paper", evidence_ids: ["e1"] },
    { source_id: "unrelated", title: "Unrelated", resource_uri: "agent-resource://other" },
  ] };
  const snapshot = {
    research_evidence: [{ id: "e1", source_id: "source" }],
    observations: [
      { id: "read", source: "resource.read", summary: '{"text":"Saved source excerpt"}', evidence_ids: ["e1"], resource_refs: [{ uri: "agent-resource://paper" }] },
      { id: "query", source: "data.query", summary: '{"rows":[{"value":42}]}', dataset_refs: [{ uri: "dataset://result", name: "Result" }] },
      { id: "unrelated", source: "web.search", summary: "Other result" },
    ],
  };
  it("joins claims/evidence to their actual source and saved observation", () => {
    const result = blockTargets({ evidence_ids: ["e1"] }, document, snapshot);
    expect(result.map((item) => item.id)).toEqual(["source", "observation:read", "agent-resource://paper"]);
    expect(result.some((item) => item.title === "Unrelated")).toBe(false);
    expect(result.some((item) => item.kind === "tool" && item.title === "web.search")).toBe(false);
  });
  it("opens an explicitly linked dataset without inventing links to all tool events", () => {
    const result = blockTargets({ source_refs: ["query"] }, document, snapshot);
    expect(result.find((item) => item.kind === "dataset")?.uri).toBe("dataset://result");
    expect(result.filter((item) => item.kind === "tool").map((item) => item.title)).toEqual(["data.query"]);
  });
  it("never derives supporting material from prose or a missing relationship", () => {
    expect(blockTargets({ text: "Source e1 supports this claim", source_refs: [] }, document, snapshot)).toEqual([]);
    expect(sourceTargets(document)).toHaveLength(2);
  });
  it("retains bounded/non-JSON observations rather than treating them as a full replay", () => {
    expect(savedToolPayload({ summary: "Bounded text result…" })).toEqual({ summary: "Bounded text result…" });
    expect(savedToolPayload(snapshot.observations[1])).toEqual({ rows: [{ value: 42 }] });
  });
});

it("only allows explicit external web links without embedded credentials or active schemes", () => {
  expect(externalUri("https://example.org/paper?q=one")).toBe("https://example.org/paper?q=one");
  for (const uri of ["javascript:alert(1)", "data:text/html,test", "file:///etc/passwd", "https://secret@example.org", "https://example.org/\nfoo"]) expect(externalUri(uri)).toBeUndefined();
});
