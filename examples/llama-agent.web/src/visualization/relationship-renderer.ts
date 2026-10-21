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
  private projection: RelationshipVisualization | null = null;
  constructor(container: HTMLElement, options: RelationshipRendererOptions = {}) {
    this.onSelect = options.onSelect;
    this.tooltip = document.createElement("div");
    this.tooltip.setAttribute("role", "tooltip");
    this.tooltip.setAttribute("aria-live", "polite");
    this.tooltip.hidden = true;
    this.tooltip.style.cssText = "position:absolute;z-index:2;left:8px;top:8px;max-width:80%;padding:6px 9px;border-radius:5px;background:#172536;color:#f4f7fb;font:12px sans-serif;pointer-events:none";
    this.graph = cytoscape({ container, elements: [], style: [
      { selector: "node", style: { "background-color": "#3979b8", label: "data(label)", color: "#e9f2ff",
        "font-size": "11px", "text-wrap": "wrap", "text-max-width": "100px", width: 36, height: 36,
        "border-width": 2, "border-color": "#79b7f1" } },
      { selector: "edge[?directed]", style: { width: 2, "line-color": "#6f8eaf", "target-arrow-color": "#6f8eaf",
        "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(label)", color: "#b9cce0", "font-size": "9px" } },
      { selector: "edge[!directed]", style: { width: 2, "line-color": "#6f8eaf", "curve-style": "bezier",
        label: "data(label)", color: "#b9cce0", "font-size": "9px" } },
      { selector: "edge:selected", style: { "line-color": "#f0b95e", "target-arrow-color": "#f0b95e", width: 4 } },
      { selector: "node:selected", style: { "border-color": "#ffd27d", "border-width": 4 } },
    ], layout: { name: "grid", animate: false }, userZoomingEnabled: true, userPanningEnabled: true });
    container.append(this.tooltip);
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
    let spec = parseRelationshipVisualization(input);
    if (!options.replace && this.projection) {
      if (spec.data_revision !== this.projection.data_revision)
        throw new Error("Cannot combine graph projections from different data revisions.");
      const nodes = new Map(this.projection.nodes.map((node) => [node.id, node]));
      const edges = new Map(this.projection.edges.map((edge) => [edge.id, edge]));
      for (const node of spec.nodes) nodes.set(node.id, node);
      for (const edge of spec.edges) edges.set(edge.id, edge);
      // Validate the whole accumulated graph before mutating Cytoscape, including
      // cross-page ID collisions, endpoint references and serialized byte limits.
      const limitations = [...new Set([...(this.projection.limitations ?? []), ...(spec.limitations ?? [])])];
      spec = parseRelationshipVisualization({ ...spec, nodes: [...nodes.values()], edges: [...edges.values()],
        complete: this.projection.complete !== false && spec.complete !== false,
        truncated: this.projection.truncated === true || spec.truncated === true, limitations });
    }
    const layout = spec.view?.layout ?? "cose";
    const elements: ElementDefinition[] = [
      ...spec.nodes.map((node) => ({ data: { id: nodeElementId(node.id), sourceId: node.id,
        label: node.label ?? node.id, kind: node.kind ?? "" } })),
      ...spec.edges.map((edge) => ({ data: { id: edgeElementId(edge.id), sourceId: edge.id,
        source: nodeElementId(edge.source), target: nodeElementId(edge.target),
        label: edge.label ?? "", directed: edge.directed !== false } })),
    ];
    const firstRender = options.replace === true || this.graph.elements().empty();
    this.tooltip.hidden = true;
    this.graph.batch(() => {
      if (options.replace) this.graph.elements().remove();
      for (const element of elements) {
        const existing = this.graph.getElementById(String(element.data.id));
        if (existing.empty()) this.graph.add(element);
        else {
          if (existing.isEdge()) existing.move({ source: String(element.data.source), target: String(element.data.target) });
          existing.data(element.data);
        }
      }
    });
    this.graph.layout({ name: layout as RelationshipLayout, animate: false,
      fit: options.fit ?? firstRender, padding: 32 }).run();
    if (spec.view) this.graph.elements().unselect();
    for (const id of spec.view?.selected_nodes ?? []) this.graph.getElementById(nodeElementId(id)).select();
    for (const id of spec.view?.selected_edges ?? []) this.graph.getElementById(edgeElementId(id)).select();
    this.projection = spec;
  }

  counts(): { nodes: number; edges: number; complete: boolean; truncated: boolean; limitations: string[] } {
    return { nodes: this.graph.nodes().length, edges: this.graph.edges().length,
      complete: this.projection?.complete !== false, truncated: this.projection?.truncated === true,
      limitations: [...(this.projection?.limitations ?? [])] };
  }

  snapshotProjection(): RelationshipVisualization | null {
    return this.projection ? parseRelationshipVisualization(JSON.parse(JSON.stringify(this.projection))) : null;
  }

  snapshotViewMetadata(): Record<string, unknown> {
    const pan = this.graph.pan();
    return { data_revision: this.projection?.data_revision ?? "", layout: this.projection?.view?.layout ?? "cose",
      zoom: this.graph.zoom(), pan: { x: pan.x, y: pan.y },
      selected_nodes: this.graph.nodes(":selected").map((node) => String(node.data("sourceId"))),
      selected_edges: this.graph.edges(":selected").map((edge) => String(edge.data("sourceId"))) };
  }

  async exportPng(): Promise<Blob> {
    if (this.graph.nodes().empty()) throw new Error("There are no graph nodes to export.");
    if (this.graph.width() <= 0 || this.graph.height() <= 0)
      throw new Error("The graph must be visible before exporting an image.");
    // Cytoscape treats maxWidth/maxHeight as the requested output dimensions when
    // they are supplied (rather than merely as upper bounds). Start at a useful
    // report resolution, then step down if the actual graph is too dense for the
    // persisted snapshot contract. This keeps both canvas memory and resource
    // size bounded without rejecting ordinary small graphs due to an accidental
    // 4096px upscale.
    const maxBytes = 512 * 1024;
    for (let maxDimension = 1600; maxDimension >= 400; maxDimension = Math.floor(maxDimension * 0.75)) {
      const image = await this.graph.png({ output: "blob-promise", full: false,
        maxWidth: maxDimension, maxHeight: maxDimension, bg: "#0d1726" });
      if (image.size > 0 && image.size <= maxBytes) return image;
    }
    throw new Error("The graph snapshot remains above the 512 KiB persisted PNG limit at reduced resolution.");
  }

  dispose(): void { this.graph.destroy(); this.tooltip.remove(); }
}
