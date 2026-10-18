import type { AgentEvent } from "../types/agent";

export type Fields = Record<string, unknown>;
export type InspectionTarget = {
  id: string; title: string; kind: "dataset" | "resource" | "tool" | "source";
  uri?: string; originalUri?: string; mimeType?: string; detail?: Fields;
};
export type ReportReference = { uri: string; turnId: string; name: string };
export const object = (value: unknown): Fields => value && typeof value === "object" && !Array.isArray(value) ? value as Fields : {};
export const objects = (value: unknown): Fields[] => Array.isArray(value) ? value.map(object).filter((item) => Object.keys(item).length > 0) : [];
export const strings = (value: unknown): string[] => Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];

export function externalUri(value: unknown): string | undefined {
  if (typeof value !== "string" || /[\u0000-\u0020\u007f]/.test(value)) return;
  try {
    const url = new URL(value);
    if ((url.protocol === "https:" || url.protocol === "http:") && !url.username && !url.password) return url.href;
  } catch { /* no external link for opaque/local references */ }
}

export function reportReferences(events: AgentEvent[], sessionId: string): ReportReference[] {
  const found = new Map<string, ReportReference>();
  for (const event of [...events].reverse()) {
    if (event.session_id !== sessionId) continue;
    if (event.resource_mime_type !== "application/vnd.llama-agent.report+json" &&
        event.resource_mime_type !== "application/vnd.llama-agent.document+json") continue;
    if (typeof event.resource_uri !== "string" || !event.resource_uri) continue;
    if (!found.has(event.resource_uri)) found.set(event.resource_uri, {
      uri: event.resource_uri, turnId: event.turn_id || event.request_id || event.resource_uri,
      name: typeof event.resource_name === "string" ? event.resource_name : "Document",
    });
  }
  return [...found.values()];
}

export function sourceTargets(document: Fields): InspectionTarget[] {
  return objects(document.source_bibliography).map((item): InspectionTarget => {
    const uri = typeof item.resource_uri === "string" && item.resource_uri ? item.resource_uri : undefined;
    return { id: String(item.source_id), title: String(item.title || item.file_name || item.source_id),
      kind: uri?.startsWith("dataset://") ? "dataset" : "source", uri,
      originalUri: externalUri(item.original_uri) ?? externalUri(item.resolved_uri), detail: item };
  });
}

// Only host-retained structured references establish a relationship. Text,
// neighboring event timestamps and the currently active tool are not joins.
export function blockTargets(block: Fields, document: Fields, snapshot: Fields): InspectionTarget[] {
  const refs = new Set(strings(block.source_refs));
  const evidence = new Set(strings(block.evidence_ids));
  for (const item of objects(snapshot.research_evidence)) {
    if (evidence.has(String(item.id))) refs.add(String(item.source_id));
  }
  const targets = new Map<string, InspectionTarget>();
  const add = (target: InspectionTarget) => { if (!targets.has(target.id)) targets.set(target.id, target); };
  for (const item of objects(document.source_bibliography)) {
    if (!refs.has(String(item.source_id)) && !refs.has(String(item.resource_uri)) &&
        !strings(item.evidence_ids).some((id) => evidence.has(id))) continue;
    const uri = typeof item.resource_uri === "string" && item.resource_uri ? item.resource_uri : undefined;
    add({ id: String(item.source_id), title: String(item.title || item.file_name || item.source_id),
      kind: uri?.startsWith("dataset://") ? "dataset" : "source", uri,
      originalUri: externalUri(item.original_uri) ?? externalUri(item.resolved_uri), detail: item });
    if (uri) refs.add(uri);
  }
  for (const observation of objects(snapshot.observations)) {
    const resources = objects(observation.resource_refs);
    const datasets = objects(observation.dataset_refs);
    if (!refs.has(String(observation.id)) && !strings(observation.evidence_ids).some((id) => evidence.has(id)) &&
        ![...resources, ...datasets].some((ref) => refs.has(String(ref.uri)) ||
          (typeof ref.parent_uri === "string" && refs.has(ref.parent_uri)))) continue;
    add({ id: `observation:${String(observation.id)}`, title: String(observation.source || "Tool result"),
      kind: "tool", detail: observation });
    for (const ref of datasets) {
      if (typeof ref.uri === "string") add({ id: ref.uri, title: String(ref.name || ref.uri), kind: "dataset", uri: ref.uri, detail: ref });
    }
    for (const ref of resources) {
      if (typeof ref.uri === "string") add({ id: ref.uri, title: String(ref.name || ref.uri), kind: "resource",
        uri: ref.uri, mimeType: typeof ref.mime_type === "string" ? ref.mime_type : undefined, detail: ref });
    }
  }
  for (const ref of refs) {
    if ([...targets.values()].some((target) => target.id === ref || target.uri === ref)) continue;
    if (/^(?:agent-resource|resource|dataset):\/\//.test(ref)) add({ id: ref, title: ref,
      kind: ref.startsWith("dataset://") ? "dataset" : "resource", uri: ref });
  }
  return [...targets.values()];
}

export function savedToolPayload(observation: Fields): Fields {
  if (typeof observation.summary === "string") {
    try {
      const value = object(JSON.parse(observation.summary));
      return value.ok === true && value.result && typeof value.result === "object" ? object(value.result) : value;
    } catch { return { summary: observation.summary }; }
  }
  return object(observation.summary);
}
