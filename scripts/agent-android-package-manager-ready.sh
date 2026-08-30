#!/usr/bin/env bash

set -euo pipefail

attempts="${1:-30}"
if ! [[ "$attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "package-manager readiness attempts must be a positive integer" >&2
    exit 2
fi

for ((attempt = 1; attempt <= attempts; ++attempt)); do
    if adb shell cmd package list packages >/dev/null 2>&1; then
        echo "Android Package Manager is ready after attempt ${attempt}/${attempts}."
        exit 0
    fi
    sleep 2
done

echo "Android Package Manager did not become ready after ${attempts} attempts." >&2
adb shell service list 2>&1 || true
adb logcat -d -v brief -t 400 2>&1 || true
exit 1
