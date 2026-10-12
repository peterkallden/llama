import { describe, expect, it } from "vitest";
import { parseSseBlock, parseSseBuffer } from "./agent-api";

describe("web event framing", () => {
  it("parses multiline SSE data and preserves id", () => {
    expect(parseSseBlock("id: 42\nevent: agent\ndata: {\"type\":\ndata: \"turn.completed\"}"))
      .toEqual({ id: "42", event: "agent", data: "{\"type\":\n\"turn.completed\"}" });
  });

  it("keeps an incomplete SSE block for the next network chunk", () => {
    expect(parseSseBuffer("id: 1\ndata: {}\n\nid: 2\ndata: {"))
      .toEqual({ blocks: ["id: 1\ndata: {}"], rest: "id: 2\ndata: {" });
  });
});
