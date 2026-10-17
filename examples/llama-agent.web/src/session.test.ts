import { afterEach, expect, it, vi } from "vitest";

const transport = vi.hoisted(() => ({
  calls: [] as Array<Record<string, unknown>>,
  replies: [] as Array<(value: unknown) => void>,
  sink: (_event: Record<string, unknown>) => {},
}));

vi.mock("./api/agent-api", () => ({ AgentApi: class {
  status() { return Promise.resolve({}); }
  connectEvents(sink: typeof transport.sink, _signal: AbortSignal, connected: () => void) {
    transport.sink = sink; connected();
    return new Promise<void>(() => {});
  }
  submitTurn(turn: Record<string, unknown>, signal?: AbortSignal) {
    transport.calls.push(turn);
    return new Promise((resolve, reject) => {
      transport.replies.push(resolve);
      signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
    });
  }
} }));

afterEach(() => { vi.unstubAllGlobals(); });

it("waits for the active HTTP turn before dispatching a queued review, despite an earlier SSE completion", async () => {
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  void session.connect();
  const ordinary = session.submit("Question");
  await session.submitReview("Review", { document_uri: "resource://doc", revision: 2, block_ids: [] }, "review-operation");
  transport.sink({ event_type: "turn.completed", session_id: "another-session" });
  expect(session.busy).toBe(true);
  transport.sink({ event_type: "turn.completed", session_id: session.sessionId, turn_id: transport.calls[0].turn_id });
  expect(session.busy).toBe(true);
  expect(transport.calls).toHaveLength(1);
  expect(session.documentReviews[0].state).toBe("queued");
  transport.replies[0]({ ok: true, response: "Answer" });
  await ordinary;
  expect(transport.calls).toHaveLength(2);
  expect(transport.calls[1].turn_id).toBe("review-operation");
  transport.replies[1]({ ok: true, response: "Reviewed" });
  for (let i = 0; i < 10; i++) await Promise.resolve();
  expect(session.documentReviews[0].state).toBe("completed");
  await session.submitReview("Review", { document_uri: "resource://doc", revision: 2, block_ids: [] }, "review-operation");
  expect(transport.calls).toHaveLength(2);
  const cancelled = session.submit("A long request");
  transport.sink({ event_type: "turn.cancelled", session_id: session.sessionId,
    turn_id: transport.calls[2].turn_id, detail: "turn deadline exceeded" });
  await cancelled;
  expect(session.busy).toBe(false);
  expect(session.error).toBe("turn deadline exceeded");
});
