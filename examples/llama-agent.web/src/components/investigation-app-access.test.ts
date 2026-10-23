import { expect, it } from "vitest";
import { InvestigationAppAccess } from "./investigation-app-access";

it("requires an acknowledged case and exact revision for user decisions", () => {
  const access = new InvestigationAppAccess();
  const review = { case_id: "case-a", expected_revision: 2 };
  expect(() => access.authorize("investigation.association.review", review)).toThrow();
  const epoch = access.begin("investigation.case.open");
  access.observe("investigation.case.open", { case_id: "case-a" }, { case: { case_id: "case-a", revision: 2 } }, epoch);
  expect(() => access.authorize("investigation.association.review", review)).not.toThrow();
  expect(() => access.authorize("investigation.association.review", { ...review, case_id: "case-b" })).toThrow();
  expect(() => access.authorize("investigation.association.review", { ...review, expected_revision: 1 })).toThrow();
});

it("does not let a late open or write restore a previous case binding", () => {
  const access = new InvestigationAppAccess();
  const first = access.begin("investigation.case.open");
  const second = access.begin("investigation.case.open");
  access.observe("investigation.case.open", { case_id: "case-b" }, { case: { case_id: "case-b", revision: 3 } }, second);
  access.observe("investigation.case.open", { case_id: "case-a" }, { case: { case_id: "case-a", revision: 4 } }, first);
  access.observe("investigation.association.review", { case_id: "case-a" }, { case_id: "case-a", revision: 5 }, first);
  expect(() => access.authorize("investigation.association.review", { case_id: "case-b", expected_revision: 3 })).not.toThrow();
  expect(() => access.authorize("investigation.association.review", { case_id: "case-a", expected_revision: 5 })).toThrow();
});

it("tracks published revisions monotonically without treating tool arguments as authority", () => {
  const access = new InvestigationAppAccess();
  const epoch = access.begin("investigation.case.open");
  access.observe("investigation.case.open", { case_id: "case-a" }, { case: { case_id: "case-a", revision: 2 } }, epoch);
  access.observe("investigation.source.attach", { case_id: "case-a", expected_revision: 100 }, { case_id: "case-a", revision: 3 }, epoch);
  access.observe("investigation.statement.record", { case_id: "case-a" }, { case_id: "case-a", revision: 2 }, epoch);
  expect(() => access.authorize("investigation.association.review", { case_id: "case-a", expected_revision: 3 })).not.toThrow();
  expect(() => access.authorize("investigation.association.review", { case_id: "case-a", expected_revision: 100 })).toThrow();
});
