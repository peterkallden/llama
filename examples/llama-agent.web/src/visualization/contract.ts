export type RelationshipNode = { id: string; label?: string; kind?: string };
export type RelationshipEdge = {
  id: string; source: string; target: string; label?: string; directed?: boolean;
};
export type RelationshipLayout = "cose" | "circle" | "grid" | "breadthfirst";
export type RelationshipVisualization = {
  schema_version: 1;
  kind: "relationship";
  title: string;
  data_revision?: string;
  nodes: RelationshipNode[];
  edges: RelationshipEdge[];
  complete?: boolean;
  truncated?: boolean;
  limitations?: string[];
  view?: { layout?: RelationshipLayout; selected_nodes?: string[]; selected_edges?: string[] };
};

export const VISUALIZATION_LIMITS = { bytes: 1024 * 1024, nodes: 500, edges: 1000 } as const;
const layouts = new Set(["cose", "circle", "grid", "breadthfirst"]);
const keys = (value: Record<string, unknown>, allowed: string[]) => Object.keys(value).every((key) => allowed.includes(key));
const text = (value: unknown, maximum: number, optional = false): value is string =>
  typeof value === "string" && (optional || value.length > 0) &&
  new TextEncoder().encode(value).byteLength <= maximum && !value.includes("\0");

export function parseRelationshipVisualization(input: unknown): RelationshipVisualization {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("Visualization must be an object.");
  const spec = input as Record<string, unknown>;
  if (!keys(spec, ["schema_version", "kind", "title", "data_revision", "nodes", "edges", "complete", "truncated", "limitations", "view"]) ||
      spec.schema_version !== 1 || spec.kind !== "relationship" || !text(spec.title, 256) ||
      (spec.data_revision !== undefined && !text(spec.data_revision, 256, true)) ||
      !Array.isArray(spec.nodes) || !Array.isArray(spec.edges) ||
      (spec.complete !== undefined && typeof spec.complete !== "boolean") ||
      (spec.truncated !== undefined && typeof spec.truncated !== "boolean") ||
      (spec.limitations !== undefined && (!Array.isArray(spec.limitations) || spec.limitations.length > 32 ||
        !spec.limitations.every((item) => text(item, 512))))) {
    throw new Error("Visualization is missing required fields or contains unsupported fields.");
  }
  if (spec.nodes.length > VISUALIZATION_LIMITS.nodes || spec.edges.length > VISUALIZATION_LIMITS.edges) {
    throw new Error("Visualization exceeds its node or edge limit.");
  }
  const nodeIds = new Set<string>();
  const allIds = new Set<string>();
  const nodes = spec.nodes.map((raw): RelationshipNode => {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("Visualization node must be an object.");
    const node = raw as Record<string, unknown>;
    if (!keys(node, ["id", "label", "kind"]) || !text(node.id, 256) ||
        (node.label !== undefined && !text(node.label, 512, true)) ||
        (node.kind !== undefined && !text(node.kind, 64, true)) || allIds.has(node.id)) {
      throw new Error("Visualization contains an invalid or duplicate node.");
    }
    nodeIds.add(node.id); allIds.add(node.id);
    return { id: node.id, label: node.label as string | undefined, kind: node.kind as string | undefined };
  });
  const edgeIds = new Set<string>();
  const edges = spec.edges.map((raw): RelationshipEdge => {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("Visualization edge must be an object.");
    const edge = raw as Record<string, unknown>;
    if (!keys(edge, ["id", "source", "target", "label", "directed"]) || !text(edge.id, 256) ||
        !text(edge.source, 256) || !text(edge.target, 256) ||
        (edge.label !== undefined && !text(edge.label, 512, true)) ||
        (edge.directed !== undefined && typeof edge.directed !== "boolean") || allIds.has(edge.id) ||
        !nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
      throw new Error("Visualization contains an invalid edge or a missing endpoint.");
    }
    allIds.add(edge.id); edgeIds.add(edge.id);
    return { id: edge.id, source: edge.source, target: edge.target,
      label: edge.label as string | undefined, directed: edge.directed as boolean | undefined };
  });
  let view: RelationshipVisualization["view"];
  if (spec.view !== undefined) {
    if (!spec.view || typeof spec.view !== "object" || Array.isArray(spec.view)) throw new Error("Visualization view must be an object.");
    const rawView = spec.view as Record<string, unknown>;
    if (!keys(rawView, ["layout", "selected_nodes", "selected_edges"]) ||
        (rawView.layout !== undefined && (typeof rawView.layout !== "string" || !layouts.has(rawView.layout))) ||
        (rawView.selected_nodes !== undefined && !Array.isArray(rawView.selected_nodes)) ||
        (rawView.selected_edges !== undefined && !Array.isArray(rawView.selected_edges))) {
      throw new Error("Visualization view contains unsupported fields.");
    }
    const selectedNodes = (rawView.selected_nodes ?? []) as unknown[];
    const selectedEdges = (rawView.selected_edges ?? []) as unknown[];
    if (!selectedNodes.every((id) => typeof id === "string" && nodeIds.has(id)) ||
        new Set(selectedNodes).size !== selectedNodes.length ||
        !selectedEdges.every((id) => typeof id === "string" && edgeIds.has(id)) ||
        new Set(selectedEdges).size !== selectedEdges.length) throw new Error("Visualization selection contains unknown or duplicate ids.");
    view = { layout: rawView.layout as RelationshipLayout | undefined,
      selected_nodes: selectedNodes as string[], selected_edges: selectedEdges as string[] };
  }
  const complete = spec.complete !== false;
  const truncated = spec.truncated === true;
  const limitations = (spec.limitations ?? []) as string[];
  if (complete && truncated) throw new Error("Truncated visualization cannot be complete.");
  if ((!complete || truncated) && limitations.length === 0) throw new Error("Incomplete visualization must state a limitation.");
  const result: RelationshipVisualization = { schema_version: 1, kind: "relationship", title: spec.title,
    data_revision: spec.data_revision as string | undefined, nodes, edges, complete, truncated, limitations, view };
  if (new TextEncoder().encode(JSON.stringify(result)).byteLength > VISUALIZATION_LIMITS.bytes) {
    throw new Error("Visualization exceeds its serialized byte limit.");
  }
  return result;
}
