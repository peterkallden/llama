# Document table round-trip fixture

`TAB6695_sv.csv` is a UTF-8-normalized copy of the external statistics source
`TAB6695_sv.csv`. The source file is Windows-1252 encoded; the fixture is kept
UTF-8 so the model-free smoke is reproducible on Windows and Linux.

`TAB6695_sv.docx` contains the same table after a CSV-to-Markdown-to-DOCX
conversion. The round-trip check uses the `2024` value `5622` for `fisk och
skaldjur` and verifies that it remains available in the Pandoc document table.
