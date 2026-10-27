#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-flydelta-dose-smoke-common.sh"
agent_dose_run_profile paired
