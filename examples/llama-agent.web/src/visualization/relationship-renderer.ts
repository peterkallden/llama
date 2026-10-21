import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import { parseRelationshipVisualization, type RelationshipLayout, type RelationshipVisualization } from "./contract";

export type RelationshipSelection = { kind: "node" | "edge"; id: string };
export type RelationshipRendererOptions = { onSelect?: (selection: RelationshipSelection) => void };

const nodeElementId = (id: string) => `visualization-node:${id}`;
const edgeElementId = (id: string) => `visualization-edge:${id}`;

export class RelationshipRenderer {
  private readonly graph: Core;
  private readonly onSelect?: RelationshipRendererOptions["onSelect"];
  private readonly tooltip: HTMLDivElement;
  constructor(container: HTMLElement, options: RelationshipRendererOptions = {}) {
    this.onSelect = options.onSelect;
    this.tooltip = document.createElement("div");
    this.tooltip.setAttribute("role", "tooltip");
    this.tooltip.setAttribute("aria-live", "polite");
    this.tooltip.hidden = true;
    this.tooltip.style.cssText = "position:absolute;z-index:2;left:8px;top:8px;max-width:80%;padding:6px 9px;border-radius:5px;background:#172536;color:#f4f7fb;font:12px sans-serif;pointer-events:none";
    container.append(this.tooltip);
    this.graph = cytoscape({ container, elements: [], style: [
      { selector: "node", style: { "background-color": "#3979b8", label: "data(label)", color: "#e9f2ff",
        "font-size": "11px", "text-wrap": "wrap", "text-max-width": "100px", width: 36, height: 36,
        "border-width": 2, "border-color": "#79b7f1" } },
      { selector: "edge[directed = true]", style: { width: 2, "line-color": "#6f8eaf", "target-arrow-color": "#6f8eaf",
        "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(label)", color: "#b9cce0", "font-size": "9px" } },
      { selector: "edge[directed = false]", style: { width: 2, "line-color": "#6f8eaf", "curve-style": "bezier",
        label: "data(label)", color: "#b9cce0", "font-size": "9px" } },
      { selector: "edge:selected", style: { "line-color": "#f0b95e", "target-arrow-color": "#f0b95e", width: 4 } },
      { selector: "node:selected", style: { "border-color": "#ffd27d", "border-width": 4 } },
    ], layout: { name: "grid", animate: false }, userZoomingEnabled: true, userPanningEnabled: true });
    this.graph.on("tap", "node", (event) => {
      this.onSelect?.({ kind: "node", id: String(event.target.data("sourceId")) });
    });
    this.graph.on("tap", "edge", (event) => {
      this.onSelect?.({ kind: "edge", id: String(event.target.data("sourceId")) });
    });
    this.graph.on("mouseover", "node, edge", (event) => {
      const element = event.target;
      const label = String(element.data("label") ?? element.data("sourceId"));
      const kind = element.isNode() ? String(element.data("kind") ?? "object") : "relation";
      this.tooltip.textContent = `${label} · ${kind}`;
      this.tooltip.hidden = false;
    });
    this.graph.on("mouseout", "node, edge", () => { this.tooltip.hidden = true; });
  }

  update(input: RelationshipVisualization | unknown, options: { replace?: boolean; fit?: boolean } = {}): void {
    const spec = parseRelationshipVisualization(input);
    const layout = spec.view?.layout ?? "cose";
    const elements: ElementDefinition[] = [
      ...spec.nodes.map((node) => ({ data: { id: nodeElementId(node.id), sourceId: node.id,
        label: node.label ?? node.id, kind: node.kind ?? "" } })),
      ...spec.edges.map((edge) => ({ data: { id: edgeElementId(edge.id), sourceId: edge.id,
        source: nodeElementId(edge.source), target: nodeElementId(edge.target),
        label: edge.label ?? "", directed: edge.directed !== false } })),
    ];
    const firstRender = this.graph.elements().empty();
    this.graph.batch(() => {
      if (options.replace) this.graph.elements().remove();
      for (const element of elements) {
        const existing = this.graph.getElementById(String(element.data.id));
        if (existing.empty()) this.graph.add(element);
        else existing.data(element.data);
      }
    });
    this.graph.layout({ name: layout as RelationshipLayout, animate: false,
      fit: options.fit ?? firstRender, padding: 32 }).run();
    for (const id of spec.view?.selected_nodes ?? []) this.graph.getElementById(nodeElementId(id)).select();
    for (const id of spec.view?.selected_edges ?? []) this.graph.getElementById(edgeElementId(id)).select();
  }

  counts(): { nodes: number; edges: number } {
    return { nodes: this.graph.nodes().length, edges: this.graph.edges().length };
  }

  dispose(): void { this.graph.destroy(); this.tooltip.remove(); }
}
