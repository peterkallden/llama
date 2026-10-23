import { expect, it } from "vitest";
import { createLatestRequestGate } from "./latest-request";

it("discards an earlier response without invalidating an independent view", () => {
  let scope = "case-a:r1";
  const graph = createLatestRequestGate(() => scope);
  const sources = createLatestRequestGate(() => scope);
  const previousGraph = graph();
  const sourceRead = sources();
  const nextGraph = graph();
  expect(previousGraph()).toBe(false);
  expect(sourceRead()).toBe(true);
  expect(nextGraph()).toBe(true);
  scope = "case-a:r2";
  expect(sourceRead()).toBe(false);
  expect(nextGraph()).toBe(false);
});
