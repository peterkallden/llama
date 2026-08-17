#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build-agent-packaging}"
model="${2:-${LLAMA_AGENT_MODEL:-}}"
mmproj="${3:-${LLAMA_AGENT_MMPROJ:-}}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
agent="${repo_root}/${build_dir}/bin/llama-agent"
fixture_root="${repo_root}/pocs/agent/smoke/data/fixtures/multimodal"

if [[ ! -x "$agent" ]]; then
    echo "multimodal smoke not-run: agent executable not found: $agent"
    exit 77
fi
if [[ -z "$model" || ! -f "$model" ]]; then
    echo "multimodal smoke not-run: set LLAMA_AGENT_MODEL or pass MODEL"
    exit 77
fi
if [[ -z "$mmproj" || ! -f "$mmproj" ]]; then
    echo "multimodal smoke not-run: set LLAMA_AGENT_MMPROJ or pass MMPROJ"
    exit 77
fi

for fixture in "$fixture_root/cats.jpg" "$fixture_root/scb-cpi.png"; do
    if [[ ! -f "$fixture" ]]; then
        echo "multimodal smoke failed: missing fixture: $fixture" >&2
        exit 1
    fi
done

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/llama-agent-multimodal-smoke.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

run_image_case() {
    local name="$1"
    local fixture="$2"
    local mime="$3"
    local prompt="$4"
    local pattern="$5"
    local output="$work_dir/${name}.out"

    echo "multimodal smoke: running ${name}"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${repo_root}/${build_dir}/bin" \
        "$agent" run \
        --model "$model" \
        --mmproj "$mmproj" \
        --agent-inference-backend server-context \
        --agent-profile static \
        --agent-plan off \
        --thinking-mode reflective \
        --n-predict 64 \
        --threads 4 \
        --resource "$fixture" \
        --resource-mime-type "$mime" \
        --prompt "$prompt" >"$output" 2>&1

    if ! grep -Eiq "$pattern" "$output"; then
        echo "multimodal smoke failed: ${name} response did not match ${pattern}" >&2
        tail -80 "$output" >&2
        exit 1
    fi
    echo "multimodal_${name}=passed"
}

run_image_case \
    cats \
    "$fixture_root/cats.jpg" \
    image/jpeg \
    "Describe the main subject in this image. Include the word cat or cats in your answer." \
    'cat|cats|kitten|katt'

run_image_case \
    scb_cpi \
    "$fixture_root/scb-cpi.png" \
    image/png \
    "Read this statistics image. Mention CPI or consumer prices and include one visible numeric value if possible." \
    'cpi|consumer|price|konsument|pris|[0-9]'

echo "multimodal_smoke=passed"
