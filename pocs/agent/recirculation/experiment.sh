#!/usr/bin/env bash
set -euo pipefail
# Usage: bash experiment.sh BUILD_DIR MODEL.gguf OUTPUT_DIR
build_dir=$(realpath "${1:?build directory required}")
model=$(realpath "${2:?model required}")
out=${3:?output directory required}
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
mkdir -p "$out"
out=$(realpath "$out")
ctest --test-dir "$build_dir" -R '^llama-agent-recirculation-(contract|runner|memory)-ctest$' --no-tests=error --output-on-failure > "$out/contracts.log" 2>&1
sha256sum "$model" "$root/pocs/agent/recirculation/data/development.jsonl" "$root/pocs/agent/recirculation/data/evaluation.jsonl" > "$out/inputs.sha256"
git -C "$root" rev-parse HEAD > "$out/revision.txt"
tar -czf "$out/experiment-sources.tar.gz" -C "$root" pocs/agent/recirculation pocs/agent/recirculation-contract.h
binary="$build_dir/bin/llama-recirculation-poc"
"$binary" --model "$model" --trace "$out/development.jsonl" --experiment-mode tune --dev-cases-file "$root/pocs/agent/recirculation/data/development.jsonl" --selection-file "$out/selection.json" > "$out/development.stdout.log" 2> "$out/development.stderr.log"
"$binary" --model "$model" --trace "$out/evaluation.jsonl" --experiment-mode evaluate --eval-cases-file "$root/pocs/agent/recirculation/data/evaluation.jsonl" --selection-file "$out/selection.json" > "$out/evaluation.stdout.log" 2> "$out/evaluation.stderr.log"
python3 - "$out" <<'PY'
import json
import pathlib
import sys
out = pathlib.Path(sys.argv[1])
reports = []
for line in (out / 'evaluation.jsonl').open():
    row = json.loads(line)
    if row.get('event') in ('pair_comparison', 'held_out_evaluation'):
        reports.append(row)
(out / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
if len(reports) != 3:
    raise SystemExit('missing comparison reports')
print('Experiment complete:', out / 'report.json')
PY
