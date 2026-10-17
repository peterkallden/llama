export type TableRow = Record<string, unknown>;
export type DatasetTable = { columns: string[]; rows: TableRow[] };

export function datasetTable(result: Record<string, unknown>): DatasetTable {
  if (Array.isArray(result.rows)) {
    const columns = Array.isArray(result.columns)
      ? result.columns.map((column) => typeof column === "string" ? column
        : column && typeof column === "object" ? String(column.name ?? "") : "").filter(Boolean)
      : [];
    const rows = result.rows.map((row): TableRow => Array.isArray(row)
      ? Object.fromEntries(row.map((value, index) => [columns[index] ?? String(index + 1), value]))
      : row && typeof row === "object" ? row as TableRow : { value: row });
    for (const row of rows) for (const column of Object.keys(row)) if (!columns.includes(column)) columns.push(column);
    return { columns, rows };
  }
  if (Array.isArray(result.columns)) {
    const rows = result.columns.map((column): TableRow => typeof column === "string" ? { name: column } : column as TableRow);
    return { columns: [...new Set(rows.flatMap((row) => Object.keys(row ?? {})))], rows };
  }
  return { columns: ["Field", "Value"], rows: Object.entries(result).map(([Field, Value]) => ({ Field, Value })) };
}

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
function numericValue(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim())) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}
export function sortedDatasetRows(rows: TableRow[], column: string, descending = false): TableRow[] {
  if (!column) return rows;
  return rows.map((row, index) => ({ row, index })).sort((a, b) => {
    const left = a.row[column], right = b.row[column];
    if (left == null || right == null) return left == null && right == null ? a.index - b.index : left == null ? 1 : -1;
    const leftNumber = numericValue(left), rightNumber = numericValue(right);
    const comparison = leftNumber !== undefined && rightNumber !== undefined
      ? leftNumber - rightNumber : collator.compare(tableCell(left), tableCell(right));
    return (descending ? -comparison : comparison) || a.index - b.index;
  }).map(({ row }) => row);
}

export function tableCell(value: unknown): string {
  return value == null ? "—" : typeof value === "object" ? JSON.stringify(value) : String(value);
}
