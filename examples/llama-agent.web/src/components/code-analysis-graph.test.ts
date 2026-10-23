import { describe, expect, it } from "vitest";
import { codeCallVisualization, codeCallGraphsByTurn } from "./code-analysis-graph";

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

  it("renders code.impact with the same generic relationship renderer contract", () => {
    expect(codeCallVisualization("code.impact", { visualization: graph })?.edges).toHaveLength(1);
  });

  it("does not render another tool family or malformed visualization data", () => {
    expect(codeCallVisualization("repository.search", { visualization: graph })).toBeUndefined();
    expect(codeCallVisualization("code.calls", { visualization: { ...graph, edges: [{ id: "bad" }] } })).toBeUndefined();
  });
});

describe("code graph event routing", () => {
  it("isolates sessions and ignores replayed deliveries and unfinished operations", () => {
    const event = { event_type: "tool.completed", session_id: "current", turn_id: "turn",
      detail: "code.calls", sequence: 7, visualization: graph };
    const groups = codeCallGraphsByTurn([event, { ...event }, { ...event, session_id: "other", sequence: 8 },
      { ...event, event_type: "tool.started", sequence: 9 }, { ...event, session_id: undefined, sequence: 10 }], "current");
    expect(groups.get("turn")).toHaveLength(1);
    expect(groups.get("turn")?.[0]?.visualization.title).toBe(graph.title);
  });
});
