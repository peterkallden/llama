import { describe, expect, it } from "vitest";
import { projectGraphSearchArguments } from "./project-graph-search";

describe("project graph relation search", () => {
  it("searches outgoing and incoming relations without reversing their meaning", () => {
    expect(projectGraphSearchArguments("alice", "from", "authored")).toEqual({ from: "alice", relation: "authored" });
    expect(projectGraphSearchArguments("paper", "to", "authored")).toEqual({ to: "paper", relation: "authored" });
  });
  it("allows any relation without adding an empty filter", () => {
    expect(projectGraphSearchArguments("alice", "from", "")).toEqual({ from: "alice" });
    expect(projectGraphSearchArguments("alice", "from", " authored ")).toEqual({ from: "alice", relation: " authored " });
  });
});
