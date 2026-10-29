#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
cat > "$work_dir/llama-agent" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@"
EOF
chmod +x "$work_dir/llama-agent"
agent_smoke_run_logged "$work_dir/default.log" "$work_dir/llama-agent" run --model fixture
grep -Fxq -- '--agent-import' "$work_dir/default.log"
grep -Fxq -- "$(agent_smoke_repo_root)/docs/examples/agent-bootstrap-workflows-v1.json" "$work_dir/default.log"
touch "$work_dir/custom.json"
agent_smoke_run_logged "$work_dir/explicit.log" "$work_dir/llama-agent" run --agent-import "$work_dir/custom.json"
[[ $(grep -Fxc -- '--agent-import' "$work_dir/explicit.log") == 1 ]]
grep -Fxq -- "$work_dir/custom.json" "$work_dir/explicit.log"
