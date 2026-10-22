import type { AgentEvent, DaemonStatus, ToolCallResult, TurnRequest, UploadedResource } from "../types/agent";

export type EventSink = (event: AgentEvent) => void;
export type WebProject = { id: string; name: string; created_at: number; last_active_at: number };
export type WebChat = { id: string; project_id: string; title: string; created_at: number; last_active_at: number; active_turn_id?: string; attachments?: Array<{ uri: string; name: string; mime_type?: string }> };
export type WebChatMessage = { id: string; turn_id: string; role: "user" | "assistant"; content: string; recorded_at: number; sequence?: number; evidence_refs?: string[] };

export class AgentApiError extends Error {
  constructor(message: string, readonly status: number, readonly payload: unknown) {
    super(message);
    this.name = "AgentApiError";
  }
}

function safeErrorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const value = payload as Record<string, unknown>;
  if (typeof value.error === "string" && value.error.trim()) return value.error;
  const tool = value.tool && typeof value.tool === "object" ? value.tool as Record<string, unknown> : undefined;
  const toolError = tool?.error && typeof tool.error === "object" ? tool.error as Record<string, unknown> : undefined;
  if (typeof toolError?.safe_summary === "string" && toolError.safe_summary.trim()) return toolError.safe_summary;
  return fallback;
}

export function parseSseBlock(block: string): { id?: string; event?: string; data: string } | null {
  const lines = block.split(/\r?\n/);
  const data: string[] = [];
  let id: string | undefined;
  let event: string | undefined;
  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    const value = separator < 0 ? "" : line.slice(separator + 1).replace(/^ /, "");
    if (field === "id") id = value;
    if (field === "event") event = value;
    if (field === "data") data.push(value);
  }
  return data.length || id || event ? { id, event, data: data.join("\n") } : null;
}

export function parseSseBuffer(buffer: string): { blocks: string[]; rest: string } {
  const parts = buffer.split(/\r?\n\r?\n/);
  return { blocks: parts.slice(0, -1), rest: parts.at(-1) ?? "" };
}

export class AgentApi {
  constructor(private readonly baseUrl = "/api/v1", private readonly token = "") {}

  private headers(json = false): HeadersInit {
    return {
      Accept: "application/json",
      ...(json ? { "Content-Type": "application/json" } : {}),
      ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}),
    };
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: { ...this.headers(Boolean(init.body)), ...(init.headers ?? {}) },
    });
    const text = await response.text();
    let payload: unknown = text;
    try { payload = text ? JSON.parse(text) : {}; } catch { /* retain diagnostic text */ }
    if (!response.ok) {
      const message = safeErrorMessage(payload, `${response.status} ${response.statusText}`);
      throw new AgentApiError(message, response.status, payload);
    }
    return payload as T;
  }

  status(): Promise<DaemonStatus> { return this.request<DaemonStatus>("/status"); }

  projects(): Promise<{ projects: WebProject[] }> { return this.request("/projects"); }

  createProject(name: string): Promise<{ project: WebProject }> {
    return this.request("/projects", { method: "POST", body: JSON.stringify({ name }) });
  }

  chats(projectId = "", cursor = "", limit = 30): Promise<{ chats: WebChat[]; next_cursor: string }> {
    const query = new URLSearchParams({ project_id: projectId, limit: String(limit), ...(cursor ? { cursor } : {}) });
    return this.request(`/chats?${query}`);
  }

  createChat(sessionId: string, projectId = "", title = "New chat"): Promise<{ chat: WebChat }> {
    return this.request("/chats", { method: "POST", body: JSON.stringify({ session_id: sessionId, project_id: projectId, title }) });
  }

  migrateLegacyChat(sessionId: string): Promise<{ chat: WebChat; migrated?: boolean }> {
    return this.request("/chats/migrate-legacy", { method: "POST", body: JSON.stringify({ session_id: sessionId }) });
  }

  setChatResource(input: { sessionId: string; projectId: string; uri: string; scope: "project" | "session"; action: "attach" | "detach" }): Promise<{ chat: WebChat }> {
    return this.request(`/chats/${encodeURIComponent(input.sessionId)}/resources`, {
      method: "POST", body: JSON.stringify({ project_id: input.projectId, resource_uri: input.uri, scope: input.scope, action: input.action }),
    });
  }

  chatMessages(sessionId: string, projectId = "", cursor = "", limit = 50): Promise<{ chat: WebChat; messages: WebChatMessage[]; history_available: boolean; next_cursor?: string }> {
    const query = new URLSearchParams({ project_id: projectId, ...(cursor ? { cursor } : {}), limit: String(limit) });
    return this.request(`/chats/${encodeURIComponent(sessionId)}/messages?${query}`);
  }

  resources(input: { scope: "project" | "session"; projectId: string; sessionId?: string; query?: string; mimeType?: string; cursor?: string; limit?: number }): Promise<{ resources: Array<Record<string, unknown>>; next_cursor: string }> {
    const query = new URLSearchParams({ scope: input.scope, project_id: input.projectId,
      ...(input.sessionId ? { session_id: input.sessionId } : {}), ...(input.query ? { q: input.query } : {}),
      ...(input.mimeType ? { mime_type: input.mimeType } : {}), ...(input.cursor ? { cursor: input.cursor } : {}),
      limit: String(input.limit ?? 30) });
    return this.request(`/resources?${query}`);
  }

  submitTurn(turn: TurnRequest, signal?: AbortSignal): Promise<unknown> {
    return this.request("/turns", { method: "POST", body: JSON.stringify(turn), ...(signal ? { signal } : {}) });
  }

  callReadOnlyTool(input: {
    tool_name: string;
    arguments: Record<string, unknown>;
    namespace_id: string;
    session_id: string;
    project_id: string;
  }): Promise<ToolCallResult> {
    return this.request("/tools/call", { method: "POST", body: JSON.stringify(input) });
  }

  readMcpAppResource(input: { tool_name: string; resource_uri: string; session_id: string; project_id: string }): Promise<ToolCallResult> {
    return this.request("/mcp-apps/resource", { method: "POST", body: JSON.stringify(input) });
  }

  callMcpAppTool(input: { app_tool_name: string; tool_name: string; arguments: Record<string, unknown>; session_id: string; project_id: string }): Promise<ToolCallResult> {
    return this.request("/mcp-apps/call", { method: "POST", body: JSON.stringify(input) });
  }

  cancelTurn(turnId: string, sessionId: string, projectId = ""): Promise<unknown> {
    return this.request(`/chats/${encodeURIComponent(sessionId)}/turns/${encodeURIComponent(turnId)}/cancel`, {
      method: "POST", body: JSON.stringify({ project_id: projectId }),
    });
  }

  putResource(resource: UploadedResource): Promise<unknown> {
    return this.request("/resources", { method: "POST", body: JSON.stringify(resource) });
  }

  async downloadResource(uri: string, context: Record<string, string> = {}): Promise<Blob> {
    const query = new URLSearchParams({ uri, ...context });
    const response = await fetch(`${this.baseUrl}/resources/download?${query}`, {
      headers: this.headers(),
    });
    if (!response.ok) throw new Error(`Artifact download failed: ${response.status}`);
    return response.blob();
  }

  async connectEvents(sink: EventSink, signal: AbortSignal, onConnected?: () => void): Promise<void> {
    const response = await fetch(`${this.baseUrl}/events`, {
      headers: { ...this.headers(), Accept: "text/event-stream" },
      signal,
    });
    if (!response.ok || !response.body) throw new Error(`SSE connection failed: ${response.status}`);
    onConnected?.();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (!signal.aborted) {
        const part = await reader.read();
        if (part.done) break;
        buffer += decoder.decode(part.value, { stream: true });
        const parsed = parseSseBuffer(buffer);
        buffer = parsed.rest;
        for (const block of parsed.blocks) {
          const message = parseSseBlock(block);
          if (!message?.data) continue;
          try {
            const payload = JSON.parse(message.data) as AgentEvent & { event?: AgentEvent };
            // The daemon SSE envelope carries the presentation event in its
            // nested `event` value. Pass that value to the UI so category,
            // type and detail are rendered from the event itself rather than
            // falling back to the envelope's generic message type.
            const event = payload.event && typeof payload.event === "object"
              ? { ...payload, ...payload.event }
              : payload;
            sink(event);
          } catch { /* ignore malformed event */ }
        }
      }
    } finally { reader.releaseLock(); }
  }
}
