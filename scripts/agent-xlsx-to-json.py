#!/usr/bin/env python3
"""Bounded XLSX worksheet normalizer for the host resource processor.

This intentionally uses only Python's standard library. It emits the existing
worksheet envelope consumed by agent-dataset-importer; it does not become a
second dataset store or a model-facing tool.
"""
import datetime
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
MAX_SHEETS = 64
MAX_ROWS = 100000
MAX_COLUMNS = 512
MAX_CELLS = 1000000

def col_index(ref):
    letters = re.match(r"([A-Z]+)", ref.upper())
    if not letters:
        return 0
    value = 0
    for c in letters.group(1):
        value = value * 26 + ord(c) - 64
    return value - 1

def value_type(value):
    if value is None or value == "": return "null"
    if value in ("true", "false"): return "boolean"
    try:
        int(value); return "integer"
    except ValueError: pass
    try:
        float(value); return "decimal"
    except ValueError: return "string"

def typed(value, kind):
    if kind == "null": return None
    if kind == "boolean": return value == "true"
    if kind == "integer": return int(value)
    if kind == "decimal": return float(value)
    return value

def main():
    if len(sys.argv) != 3:
        raise ValueError("usage: agent-xlsx-to-json.py INPUT OUTPUT")
    source, output = sys.argv[1:]
    with zipfile.ZipFile(source, "r") as package:
        names = set(package.namelist())
        if "xl/workbook.xml" not in names:
            raise ValueError("XLSX workbook.xml is missing")
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(package.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", NS):
                shared.append("".join(item.itertext()))
        workbook = ET.fromstring(package.read("xl/workbook.xml"))
        rels = {}
        if "xl/_rels/workbook.xml.rels" in names:
            relroot = ET.fromstring(package.read("xl/_rels/workbook.xml.rels"))
            for rel in relroot.findall("rel:Relationship", NS):
                rels[rel.attrib["Id"]] = rel.attrib["Target"]
        worksheets = []
        for index, sheet in enumerate(workbook.findall("m:sheets/m:sheet", NS)):
            if index >= MAX_SHEETS: raise ValueError("worksheet count exceeds host limit")
            rel_id = sheet.attrib.get("{" + NS["r"] + "}id")
            target = rels.get(rel_id, "worksheets/sheet%d.xml" % (index + 1)).lstrip("/")
            if not target.startswith("xl/"): target = "xl/" + target
            if target not in names: raise ValueError("worksheet part is missing")
            root = ET.fromstring(package.read(target))
            cells = {}
            max_col = max_row = -1
            for cell in root.findall(".//m:c", NS):
                ref = cell.attrib.get("r", "A1")
                col = col_index(ref); row_match = re.search(r"(\d+)$", ref)
                row = int(row_match.group(1)) - 1 if row_match else 0
                raw = cell.find("m:v", NS)
                value = "" if raw is None else raw.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                if cell.attrib.get("t") == "b": value = "true" if value == "1" else "false"
                cells[(row, col)] = value
                max_col = max(max_col, col); max_row = max(max_row, row)
            if max_col + 1 > MAX_COLUMNS or max_row + 1 > MAX_ROWS or (max_col + 1) * (max_row + 1) > MAX_CELLS:
                raise ValueError("worksheet shape exceeds host limit")
            if max_col < 0:
                continue
            headers = [cells.get((0, col), "") or "column_%d" % (col + 1) for col in range(max_col + 1)]
            rows = []
            kinds = ["null"] * len(headers)
            for row in range(1, max_row + 1):
                values = [cells.get((row, col), "") for col in range(max_col + 1)]
                for col, value in enumerate(values):
                    kind = value_type(value)
                    if kind == "string": kinds[col] = kind
                    elif kinds[col] == "null": kinds[col] = kind
                    elif kinds[col] != kind: kinds[col] = "string"
                rows.append(values)
            worksheets.append({"name": sheet.attrib.get("name", "Sheet_%d" % (index + 1)),
                "index": index, "table_index": index, "columns": [], "rows": rows})
            worksheets[-1]["columns"] = [{"name": name, "type": kinds[i], "nullable": True}
                for i, name in enumerate(headers)]
            worksheets[-1]["rows"] = [{headers[i]: typed(value, kinds[i]) for i, value in enumerate(row)}
                for row in rows]
    with open(output, "w", encoding="utf-8", newline="\n") as stream:
        json.dump({"worksheets": worksheets}, stream, ensure_ascii=False, separators=(",", ":"))

if __name__ == "__main__":
    try: main()
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile, ET.ParseError) as exc:
        print("xlsx processor: " + str(exc), file=sys.stderr)
        sys.exit(2)
