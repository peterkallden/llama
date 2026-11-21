import { describe, expect, it } from "vitest";
import { datasetTable, sortedDatasetRows } from "./dataset-table";

describe("dataset result tables", () => {
  it("preserves declared column order, missing cells, and additional row fields", () => {
    expect(datasetTable({ columns: ["amount", "name"], rows: [{ name: "a", extra: true }, { amount: 10 }] })).toEqual({
      columns: ["amount", "name", "extra"], rows: [{ name: "a", extra: true }, { amount: 10 }],
    });
    expect(datasetTable({ columns: [{ name: "a" }, { name: "b" }], rows: [[1, null], [2, "x"]] }).rows)
      .toEqual([{ a: 1, b: null }, { a: 2, b: "x" }]);
  });
  it("shows typed schema and inspection metadata as tables", () => {
    expect(datasetTable({ columns: [{ name: "value", type: "integer", nullable: true }] }).columns)
      .toEqual(["name", "type", "nullable"]);
    expect(datasetTable({ name: "fixture", rows: 3 }).rows).toEqual([{ Field: "name", Value: "fixture" }, { Field: "rows", Value: 3 }]);
    expect(datasetTable({ columns: ["a"], rows: [] })).toEqual({ columns: ["a"], rows: [] });
  });
  it("sorts numbers numerically, keeps null last and leaves source rows unchanged", () => {
    const rows = [{ value: 100 }, { value: null }, { value: 2 }, { value: 10 }];
    expect(sortedDatasetRows(rows, "value").map(r => r.value)).toEqual([2, 10, 100, null]);
    expect(sortedDatasetRows(rows, "value", true).map(r => r.value)).toEqual([100, 10, 2, null]);
    expect(rows.map(r => r.value)).toEqual([100, null, 2, 10]);
    expect(sortedDatasetRows(["2.1", "-10", "2.05", "-2", "1e3"].map(value => ({ value })), "value").map(r => r.value))
      .toEqual(["-10", "-2", "2.05", "2.1", "1e3"]);
    expect(sortedDatasetRows([{ name: "item10" }, { name: "item2" }], "name").map(r => r.name)).toEqual(["item2", "item10"]);
  });
});
