import { describe, expect, it } from "vitest";
import { classifyWorkflowEvent } from "./workflow-event";

describe("workflow event classification", () => {
  it("matches exact execution states and keeps verdict separate", () => {
    expect(classifyWorkflowEvent({ event_type: "tool.completed", verdict: "SATISFIED" })).toMatchObject({
      execution: "completed", verdict: "SATISFIED",
    });
  });

  it("does not infer completion or verification from substrings", () => {
    expect(classifyWorkflowEvent({ event_type: "tool.observed", status: "unverified", verdict: "UNSATISFIED" })).toMatchObject({
      execution: "observed", verdict: "",
    });
  });
});
