#!/usr/bin/env bash
set -euo pipefail
# Usage: bash experiment.sh BUILD_DIR MODEL.gguf OUTPUT_DIR
build_dir=$(realpath "${1:?build directory required}")
model=$(realpath "${2:?model required}")
out=${3:?output directory required}
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
if [[ -e "$out/locked-selection.json" || -e "$out/evaluation.jsonl" ]]; then
    echo "Refusing to overwrite a locked experiment; use a new versioned output directory" >&2
    exit 1
fi
mkdir -p "$out"
out=$(realpath "$out")
ctest --test-dir "$build_dir" -R '^llama-agent-recirculation-(contract|runner|memory|window|analysis)-ctest$' --no-tests=error --output-on-failure > "$out/contracts.log" 2>&1
sha256sum "$model" "$root/pocs/agent/recirculation/data/development-v2.jsonl" "$root/pocs/agent/recirculation/data/evaluation-v2.jsonl" > "$out/inputs.sha256"
git -C "$root" rev-parse HEAD > "$out/revision.txt"
git -C "$root" diff --binary > "$out/worktree.diff"
tar -czf "$out/experiment-sources.tar.gz" -C "$root" pocs/agent/recirculation pocs/agent/recirculation-contract.h
binary="$build_dir/bin/llama-recirculation-poc"
"$binary" --model "$model" --trace "$out/development.jsonl" --experiment-mode tune --dev-cases-file "$root/pocs/agent/recirculation/data/development-v2.jsonl" --selection-file "$out/selection.json" > "$out/development.stdout.log" 2> "$out/development.stderr.log"
python3 "$root/pocs/agent/recirculation/analysis.py" lock "$out/selection.json" "$out/locked-selection.json"
sha256sum "$out/locked-selection.json" > "$out/locked-selection.sha256"
"$binary" --model "$model" --trace "$out/evaluation.jsonl" --experiment-mode evaluate --eval-cases-file "$root/pocs/agent/recirculation/data/evaluation-v2.jsonl" --selection-file "$out/locked-selection.json" > "$out/evaluation.stdout.log" 2> "$out/evaluation.stderr.log"
python3 "$root/pocs/agent/recirculation/analysis.py" report "$out/evaluation.jsonl" "$out/report.json"
