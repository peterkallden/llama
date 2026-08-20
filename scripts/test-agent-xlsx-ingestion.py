#!/usr/bin/env python3
"""Small model-free smoke for the bounded XLSX worksheet normalizer."""
import base64
import json
import pathlib
import subprocess
import sys
import tempfile
import zipfile

def main():
    normalizer = pathlib.Path(__file__).with_name("agent-xlsx-to-json.py")
    with tempfile.TemporaryDirectory(prefix="llama-agent-xlsx-") as directory:
        root = pathlib.Path(directory)
        source = root / "sales.xlsx"
        output = root / "workbook.json"
        parts = {
            "[Content_Types].xml": "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>",
            "xl/workbook.xml": "<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'><sheets><sheet name='Sales' sheetId='1' r:id='rId1'/></sheets></workbook>",
            "xl/_rels/workbook.xml.rels": "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Target='worksheets/sheet1.xml' Type='worksheet'/></Relationships>",
            "xl/worksheets/sheet1.xml": "<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheetData><row r='1'><c r='A1' t='inlineStr'><is><t>country</t></is></c><c r='B1' t='inlineStr'><is><t>population</t></is></c></row><row r='2'><c r='A2' t='inlineStr'><is><t>Sweden</t></is></c><c r='B2'><v>10500000</v></c></row></sheetData></worksheet>",
        }
        with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as package:
            for name, value in parts.items(): package.writestr(name, value)
        subprocess.run([sys.executable, str(normalizer), str(source), str(output)], check=True)
        envelope = json.loads(output.read_text(encoding="utf-8"))
        assert len(envelope["worksheets"]) == 1
        sheet = envelope["worksheets"][0]
        assert sheet["name"] == "Sales"
        assert sheet["columns"][1]["type"] == "integer"
        assert sheet["rows"][0]["country"] == "Sweden"
        assert sheet["rows"][0]["population"] == 10500000
        print("xlsx ingestion smoke passed: Sales, 1 row, typed integer population")

if __name__ == "__main__":
    main()
