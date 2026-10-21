#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${1:-${repo_root}/build-agent-vulkan-cozo}"
podman_bin="${PODMAN:-podman}"
ctest_bin="${CTEST:-ctest}"
suffix="$$"
elasticsearch_name="llama-investigation-elasticsearch-${suffix}"
opensearch_name="llama-investigation-opensearch-${suffix}"

cleanup() {
    "${podman_bin}" rm -f "${elasticsearch_name}" "${opensearch_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_engine() {
    local dialect="$1" image="$2" port="$3" name="$4"
    local endpoint="http://127.0.0.1:${port}"
    local -a engine_env
    if [[ "${dialect}" == elasticsearch ]]; then
        engine_env=(-e discovery.type=single-node -e xpack.security.enabled=false
            -e 'ES_JAVA_OPTS=-Xms512m -Xmx512m')
    else
        engine_env=(-e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true
            -e DISABLE_INSTALL_DEMO_CONFIG=true -e 'OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m')
    fi

    "${podman_bin}" run -d --name "${name}" -p "127.0.0.1:${port}:9200" \
        "${engine_env[@]}" \
        "${image}" >/dev/null

    local ready=0
    for _ in $(seq 1 36); do
        if curl --silent --show-error --fail --max-time 2 "${endpoint}/" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 5
    done
    if [[ "${ready}" != 1 ]]; then
        "${podman_bin}" logs --tail 100 "${name}" >&2 || true
        echo "${dialect} did not become ready at ${endpoint}" >&2
        return 1
    fi

    echo "${dialect} server:"
    curl --silent --show-error --fail "${endpoint}/"
    echo
    curl --silent --show-error --fail -X PUT "${endpoint}/llama-agent-investigation-smoke" \
        -H 'Content-Type: application/json' \
        -d '{"mappings":{"properties":{"event":{"properties":{"id":{"type":"keyword"},"kind":{"type":"keyword"}}},"host":{"properties":{"name":{"type":"keyword"}}},"@timestamp":{"type":"date"}}}}' >/dev/null
    curl --silent --show-error --fail -X POST \
        "${endpoint}/llama-agent-investigation-smoke/_bulk?refresh=true" \
        -H 'Content-Type: application/x-ndjson' --data-binary @- <<'NDJSON' >/dev/null
{"index":{"_id":"evt-001"}}
{"event":{"id":"evt-001","kind":"auth"},"host":{"name":"SRV-APP-07"},"@timestamp":"2026-10-20T08:00:00Z"}
{"index":{"_id":"evt-002"}}
{"event":{"id":"evt-002","kind":"auth"},"host":{"name":"SRV-APP-08"},"@timestamp":"2026-10-20T08:01:00Z"}
{"index":{"_id":"evt-003"}}
{"event":{"id":"evt-003","kind":"auth"},"host":{"name":"SRV-APP-09"},"@timestamp":"2026-10-20T08:02:00Z"}
{"index":{"_id":"evt-004"}}
{"event":{"id":"evt-004","kind":"network"},"host":{"name":"SRV-OTHER"},"@timestamp":"2026-10-20T08:03:00Z"}
NDJSON

    LLAMA_AGENT_INVESTIGATION_REAL_ENDPOINT="${endpoint}" \
    LLAMA_AGENT_INVESTIGATION_REAL_DIALECT="${dialect}" \
        "${ctest_bin}" --test-dir "${build_dir}" -R '^test-agent-investigation-http$' --output-on-failure
}

if ! command -v "${podman_bin}" >/dev/null || ! command -v "${ctest_bin}" >/dev/null; then
    echo "Podman and CTest are required" >&2
    exit 2
fi
if [[ ! -x "${build_dir}/bin/test-agent-investigation-http" ]]; then
    echo "Build test-agent-investigation-http before running this smoke: ${build_dir}" >&2
    exit 2
fi

run_engine elasticsearch docker.io/library/elasticsearch:8.17.3 19200 "${elasticsearch_name}"
run_engine opensearch docker.io/opensearchproject/opensearch:2.19.2 19201 "${opensearch_name}"
