import { afterEach, expect, it, vi } from "vitest";

const transport = vi.hoisted(() => ({
  calls: [] as Array<Record<string, unknown>>,
  replies: [] as Array<(value: unknown) => void>,
  upload: undefined as (() => Promise<unknown>) | undefined,
  sink: (_event: Record<string, unknown>) => {},
}));

vi.mock("./api/agent-api", () => ({ AgentApi: class {
  status() { return Promise.resolve({}); }
  putResource() { return transport.upload?.() ?? Promise.resolve({ resource: { uri: "resource://upload/test" } }); }
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

afterEach(() => { transport.upload = undefined; vi.unstubAllGlobals(); vi.useRealTimers(); });

it("retains report metadata from the completed HTTP turn when the event stream was unavailable", async () => {
  transport.calls.length = 0;
  transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const { reportReferences } = await import("./components/report-view");
  const session = useAgentSession();
  const pending = session.submit("Create a report");
  transport.replies[0]({ ok: true, response: "The report is ready.", events: [{
    type: "resource.created", sequence: 23, request_id: "report-request",
    resource_uri: "agent-resource://report", resource_name: "Report.json",
    resource_mime_type: "application/vnd.llama-agent.report+json",
  }] });
  expect(await pending).toBe(true);
  expect(reportReferences(session.events, session.sessionId)).toEqual([{
    uri: "agent-resource://report", name: "Report.json", turnId: String(transport.calls[0].turn_id),
  }]);
});

it("automatically orients each uploaded attachment as its own read-only host task", async () => {
  transport.calls.length = 0;
  transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  const file = { name: "notes.md", type: "text/markdown", size: 20,
    text: async () => "# Notes\nA short document." } as File;
  await session.upload(file);
  await new Promise((resolve) => setTimeout(resolve, 1600));
  expect(transport.calls).toHaveLength(1);
  expect(transport.calls[0].attachment_orientation).toBe(true);
  expect(transport.calls[0].resource_refs).toEqual(["resource://upload/test"]);
  expect(session.messages[0].content).toBe("Initial orientation: notes.md");
  transport.replies[0]({ ok: true, response: "This is a short note." });
  for (let i = 0; i < 10; i++) await Promise.resolve();
  expect(session.messages[1].content).toBe("This is a short note.");
  expect(session.attachmentOrientations[0].state).toBe("completed");
});

it("skips the pending default orientation when the user sends an explicit question with the attachment", async () => {
  transport.calls.length = 0;
  transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  const file = { name: "notes.md", type: "text/markdown", size: 20,
    text: async () => "# Notes\nA short document." } as File;
  await session.upload(file);
  session.selectedResourceRefs = ["resource://upload/test"];
  const send = session.submit("Who is this document about?");
  expect(transport.calls).toHaveLength(1);
  expect(transport.calls[0].prompt).toBe("Who is this document about?");
  expect(transport.calls[0].attachment_orientation).toBeUndefined();
  expect(transport.calls[0].attachment_context_required).toBe(true);
  expect(transport.calls[0].resource_refs).toEqual(["resource://upload/test"]);
  expect(session.attachmentOrientations).toHaveLength(0);
  transport.replies[0]({ ok: true, response: "The document is about Dag Hammarskjöld." });
  await send;
  for (let i = 0; i < 10; i++) await Promise.resolve();
});

it("queues a user question submitted during attachment orientation and sends it afterward", async () => {
  transport.calls.length = 0;
  transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  await session.upload({ name: "notes.md", type: "text/markdown", size: 20,
    text: async () => "# Notes\nSubject text." } as File);
  await new Promise((resolve) => setTimeout(resolve, 1600));
  expect(transport.calls).toHaveLength(1);
  session.selectedResourceRefs = ["resource://upload/test"];
  await session.submit("Who is this document about?", [], false, undefined, undefined,
    session.selectedResourceRefs);
  expect(transport.calls).toHaveLength(1);
  expect(session.pendingUserTurns).toHaveLength(1);
  expect(session.messages.at(-2)?.content).toBe("Who is this document about?");
  transport.replies[0]({ ok: true, response: "Initial orientation." });
  for (let i = 0; i < 20; i++) await Promise.resolve();
  expect(transport.calls).toHaveLength(2);
  expect(transport.calls[1].prompt).toBe("Who is this document about?");
  expect(transport.calls[1].attachment_context_required).toBe(true);
  expect(transport.calls[1].resource_refs).toEqual(["resource://upload/test"]);
  transport.replies[1]({ ok: true, response: "It is about the subject in the note." });
  for (let i = 0; i < 20; i++) await Promise.resolve();
  expect(session.pendingUserTurns).toHaveLength(0);
});

it("does not attach saved session files to unrelated turns, but uses the explicit current-turn selection", async () => {
  transport.calls.length = 0;
  transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  await session.upload({ name: "notes.md", type: "text/markdown", size: 20,
    text: async () => "# Notes" } as File);
  session.attachmentOrientations[0].state = "completed";
  const unrelated = session.submit("A question unrelated to the file");
  expect(transport.calls[0].resource_refs).toBeUndefined();
  transport.replies[0]({ ok: true, response: "Answer" });
  await unrelated;
  const selected = session.submit("Now use this file", [], false, undefined, undefined,
    ["resource://upload/test"]);
  expect(transport.calls[1].resource_refs).toEqual(["resource://upload/test"]);
  expect(transport.calls[1].attachment_context_required).toBe(true);
  transport.replies[1]({ ok: true, response: "Grounded answer" });
  await selected;
});

it("persists document context on a queued user turn and keeps failed turns retryable", async () => {
  transport.calls.length = 0;
  transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  const first = session.submit("first");
  const documentContext = { document_uri: "resource://doc", revision: 3, block_ids: ["b1"] };
  const queued = await session.submit("review this", [], false, documentContext, "queued-review",
    ["resource://attachment"], "Review draft");
  expect(queued).toBe(true);
  expect(session.pendingUserTurns[0].document_context).toEqual(documentContext);
  expect(session.pendingUserTurns[0].attachment_context_required).toBe(true);
  transport.replies[0]({ ok: true, response: "first done" });
  await first;
  for (let i = 0; i < 20; i++) await Promise.resolve();
  expect(transport.calls[1].document_context).toEqual(documentContext);
  expect(transport.calls[1].resource_refs).toEqual(["resource://attachment"]);
  transport.replies[1]({ ok: false, error: "turn deadline exceeded" });
  for (let i = 0; i < 20; i++) await Promise.resolve();
  expect(session.pendingUserTurns[0].state).toBe("failed");
  expect(session.pendingUserTurns[0].error).toContain("deadline");
  session.retryPendingUserTurn("queued-review");
  for (let i = 0; i < 20; i++) await Promise.resolve();
  expect(transport.calls).toHaveLength(3);
  expect(transport.calls[2].document_context).toEqual(documentContext);
});

it("ignores late upload and turn completions after the user starts a new session", async () => {
  transport.calls.length = 0;
  transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  let finishUpload!: (value: unknown) => void;
  transport.upload = () => new Promise((resolve) => { finishUpload = resolve; });
  const upload = session.upload({ name: "late.txt", type: "text/plain", size: 4,
    text: async () => "late" } as File);
  await Promise.resolve();
  await Promise.resolve();
  session.newSession();
  finishUpload({ resource: { uri: "resource://old-session/upload" } });
  await upload;
  expect(session.attachments).toHaveLength(0);

  transport.upload = undefined;
  const oldSessionTurn = session.submit("old session question");
  const oldSessionId = session.sessionId;
  session.newSession();
  expect(session.sessionId).not.toBe(oldSessionId);
  transport.replies[0]({ ok: true, response: "late response" });
  await oldSessionTurn;
  expect(session.messages).toHaveLength(0);
  expect(session.busy).toBe(false);
});

it("cancels a queued orientation when its attachment is removed", async () => {
  transport.calls.length = 0;
  transport.replies.length = 0;
  vi.useFakeTimers();
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  await session.upload({ name: "notes.txt", type: "text/plain", size: 8,
    text: async () => "some text" } as File);
  expect(session.attachmentOrientations).toHaveLength(1);
  session.removeAttachment(0);
  expect(session.attachmentOrientations).toHaveLength(0);
  await vi.advanceTimersByTimeAsync(2000);
  expect(transport.calls).toHaveLength(0);
});

it("dispatches a review queued during a queued chat turn", async () => {
  transport.calls.length = 0; transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  const first = session.submit("first");
  await session.submit("second");
  transport.replies[0]({ ok: true, response: "first done" });
  await first;
  expect(transport.calls).toHaveLength(2);
  await session.submitReview("review", { document_uri: "resource://doc", revision: 1, block_ids: [] }, "later-review");
  transport.replies[1]({ ok: true, response: "second done" });
  for (let i = 0; i < 20; i++) await Promise.resolve();
  expect(transport.calls).toHaveLength(3);
  expect(transport.calls[2].turn_id).toBe("later-review");
  transport.replies[2]({ ok: true, response: "review done" });
  for (let i = 0; i < 20; i++) await Promise.resolve();
});

it("isolates a draining review from a new session and ignores unrelated turn events", async () => {
  transport.calls.length = 0; transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
  const { useAgentSession } = await import("./session");
  const session = useAgentSession();
  void session.connect();
  await session.submitReview("old review", { document_uri: "resource://old", revision: 1, block_ids: [] }, "old-review");
  session.newSession();
  await session.submitReview("new review", { document_uri: "resource://new", revision: 1, block_ids: [] }, "new-review");
  expect(transport.calls).toHaveLength(2);
  for (let i = 0; i < 20; i++) await Promise.resolve();
  expect(session.documentReviews[0].state).toBe("running");
  expect(session.busy).toBe(true);
  transport.sink({ event_type: "turn.failed", session_id: session.sessionId, turn_id: "old-review", error: "stale" });
  expect(session.error).toBe("");
  transport.replies[1]({ ok: true, response: "review done" });
  for (let i = 0; i < 20; i++) await Promise.resolve();
  expect(session.documentReviews[0].state).toBe("completed");
  transport.sink({ event_type: "turn.started", session_id: session.sessionId, turn_id: "old-review" });
  expect(session.busy).toBe(false);
});

it("waits for the active HTTP turn before dispatching a queued review, despite an earlier SSE completion", async () => {
  transport.calls.length = 0;
  transport.replies.length = 0;
  const values = new Map<string, string>();
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  vi.resetModules();
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
