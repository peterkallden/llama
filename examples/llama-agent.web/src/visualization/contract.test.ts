import { describe, expect, it } from "vitest";
import { parseRelationshipVisualization } from "./contract";
import fixture from "../../../../tests/fixtures/visualization-relationship-v1.json";

const valid = () => JSON.parse(JSON.stringify(fixture)) as { nodes: Array<Record<string, unknown>>; edges: Array<Record<string, unknown>> };

describe("relationship visualization contract", () => {
  it("accepts bounded graphs and preserves their projection", () => {
    const parsed = parseRelationshipVisualization(valid());
    expect(parsed.edges[0].target).toBe("svc-store");
    expect(parsed.complete).toBe(true);
  });
  it("rejects missing edge endpoints", () => {
    const input = valid(); input.edges[0].target = "missing";
    expect(() => parseRelationshipVisualization(input)).toThrow(/endpoint/);
  });
  it("rejects node and edge id collisions", () => {
    const input = valid(); input.edges[0].id = "svc-api";
    expect(() => parseRelationshipVisualization(input)).toThrow(/edge/);
  });
  it("requires a limitation when the projection is incomplete", () => {
    expect(() => parseRelationshipVisualization({ ...valid(), complete: false })).toThrow(/limitation/);
  });
  it("rejects unrecognized fields", () => {
    const input = valid(); (input.nodes[0] as Record<string, unknown>).html = "<script>";
    expect(() => parseRelationshipVisualization(input)).toThrow(/node/);
  });
  it("rejects contradictory completeness and measures UTF-8 bytes", () => {
    expect(() => parseRelationshipVisualization({ ...valid(), complete: true, truncated: true,
      limitations: ["bounded"] })).toThrow(/complete/);
    const input = valid(); input.nodes[0].label = "€".repeat(171);
    expect(() => parseRelationshipVisualization(input)).toThrow(/node/);
  });
});
