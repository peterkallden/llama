import { describe, expect, it } from "vitest";
import { codeCallVisualization } from "./code-analysis-graph";

const graph = {
  schema_version: 1,
  kind: "relationship",
  title: "Static call relationships",
  nodes: [
    { id: "function:src/a.cpp:1:1", label: "src/a.cpp:1:1", kind: "function" },
    { id: "function:src/b.cpp:2:1", label: "src/b.cpp:2:1", kind: "function" },
  ],
  edges: [{
    id: "call:0", source: "function:src/a.cpp:1:1", target: "function:src/b.cpp:2:1",
    directed: true, relation: "static_call", label: "calls", provenance_refs: ["src/a.cpp#L1", "src/b.cpp#L2"],
  }],
  complete: true,
  truncated: false,
  limitations: [],
};

describe("code analysis graph projection", () => {
  it("parses a code.calls visualization using the shared relationship contract", () => {
    const result = codeCallVisualization("code.calls", graph);
    expect(result?.edges[0]?.relation).toBe("static_call");
    expect(result?.edges[0]?.provenance_refs).toEqual(["src/a.cpp#L1", "src/b.cpp#L2"]);
  });

  it("does not render another tool family or malformed visualization data", () => {
    expect(codeCallVisualization("repository.search", { visualization: graph })).toBeUndefined();
    expect(codeCallVisualization("code.calls", { visualization: { ...graph, edges: [{ id: "bad" }] } })).toBeUndefined();
  });
});
