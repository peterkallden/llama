import type { AgentEvent, DaemonStatus, TurnRequest, UploadedResource } from "../types/agent";

export type EventSink = (event: AgentEvent) => void;

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
      const message = typeof payload === "object" && payload && "error" in payload
        ? String((payload as { error: unknown }).error) : `${response.status} ${response.statusText}`;
      throw new Error(message);
    }
    return payload as T;
  }

  status(): Promise<DaemonStatus> { return this.request<DaemonStatus>("/status"); }

  submitTurn(turn: TurnRequest): Promise<unknown> {
    return this.request("/turns", { method: "POST", body: JSON.stringify(turn) });
  }

  cancelTurn(turnId: string): Promise<unknown> {
    return this.request(`/turns/${encodeURIComponent(turnId)}/cancel`, { method: "POST", body: "{}" });
  }

  putResource(resource: UploadedResource): Promise<unknown> {
    return this.request("/resources", { method: "POST", body: JSON.stringify(resource) });
  }

  async connectEvents(sink: EventSink, signal: AbortSignal): Promise<void> {
    const response = await fetch(`${this.baseUrl}/events`, {
      headers: { ...this.headers(), Accept: "text/event-stream" },
      signal,
    });
    if (!response.ok || !response.body) throw new Error(`SSE connection failed: ${response.status}`);
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
          try { sink(JSON.parse(message.data) as AgentEvent); } catch { /* ignore malformed event */ }
        }
      }
    } finally { reader.releaseLock(); }
  }
}
