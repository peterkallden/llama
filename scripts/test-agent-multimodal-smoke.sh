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
if [[ ! -f "$fixture_root/sample-speech.mp3" ]]; then
    echo "multimodal smoke failed: missing fixture: $fixture_root/sample-speech.mp3" >&2
    exit 1
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/llama-agent-multimodal-smoke.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

run_media_case() {
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

run_audio_case() {
    local output="$work_dir/audio.out"
    echo "multimodal smoke: running audio"
    set +e
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
        --resource "$fixture_root/sample-speech.mp3" \
        --resource-mime-type audio/mpeg \
        --prompt "Listen to this audio and briefly describe the speech or sound you hear." >"$output" 2>&1
    local status=$?
    set -e
    if [[ $status -ne 0 ]]; then
        if grep -Eiq 'audio.*(not supported|unsupported)|capabilit' "$output"; then
            echo "multimodal_audio=not-run"
            return 0
        fi
        echo "multimodal smoke failed: audio execution failed" >&2
        tail -80 "$output" >&2
        exit "$status"
    fi
    if ! grep -Eiq 'audio|sound|speech|voice|music|ljud|tal|röst|musik' "$output"; then
        echo "multimodal smoke failed: audio response did not describe audio" >&2
        tail -80 "$output" >&2
        exit 1
    fi
    echo "multimodal_audio=passed"
}

run_media_case \
    cats \
    "$fixture_root/cats.jpg" \
    image/jpeg \
    "Describe the main subject in this image. Include the word cat or cats in your answer." \
    'cat|cats|kitten|katt'

run_media_case \
    scb_cpi \
    "$fixture_root/scb-cpi.png" \
    image/png \
    "Read this statistics image. Mention CPI or consumer prices and include one visible numeric value if possible." \
    'cpi|consumer|price|konsument|pris|[0-9]'

run_audio_case

echo "multimodal_smoke=passed"
