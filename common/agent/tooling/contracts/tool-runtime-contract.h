#pragma once

#include "resource/resource-contract.h"
#include "agent/dataset-contracts.h"
#include "agent/contracts/agent-learning.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct common_agent_tool_call {
    std::string name;
    std::string arguments_json = "{}";
};

// Safe transport facts collected by a host-owned HTTP connector. URLs are
// retained internally for later redaction at the trace boundary.
struct common_tool_http_metadata {
    std::string provider;
    std::string operation;
    std::string method;
    std::string requested_url;
    std::string final_url;
    std::string resource_ref;
    int status = 0;
    std::string content_type;
    size_t response_bytes = 0;
    int64_t duration_ms = 0;
    int retry_after_seconds = -1;
};

enum class common_tool_failure_class { validation, policy, not_found, timeout, network, execution, limit };

inline const char * common_tool_failure_class_name(common_tool_failure_class value) {
    switch (value) {
        case common_tool_failure_class::validation: return "validation";
        case common_tool_failure_class::policy: return "policy";
        case common_tool_failure_class::not_found: return "not_found";
        case common_tool_failure_class::timeout: return "timeout";
        case common_tool_failure_class::network: return "network";
        case common_tool_failure_class::execution: return "execution";
        case common_tool_failure_class::limit: return "limit";
    }
    return "execution";
}

struct common_tool_execution_result {
    bool ok = false;
    std::string output;
    std::string failure_code;
    common_tool_failure_class failure_class = common_tool_failure_class::execution;
    bool retryable = false;
    std::string safe_summary;
    std::string raw_diagnostic;
    std::string content_summary;
    std::vector<common_runtime_resource_ref> resource_refs;
    std::vector<common_agent_dataset_ref> dataset_refs;
    common_tool_http_metadata http;
    // Host-produced semantic signals are carried with the tool result and
    // attached to the owning turn only after the runtime records its output.
    std::vector<common_learning_signal> learning_signals;

    static common_tool_execution_result success(std::string output) {
        common_tool_execution_result result;
        result.ok = true;
        result.output = std::move(output);
        return result;
    }
    static common_tool_execution_result success(
            std::string output,
            std::string content_summary,
            std::vector<common_runtime_resource_ref> resource_refs = {}) {
        common_tool_execution_result result = success(std::move(output));
        result.content_summary = std::move(content_summary);
        result.resource_refs = std::move(resource_refs);
        return result;
    }
    static common_tool_execution_result failure(std::string code, common_tool_failure_class failure_class, bool retryable,
            std::string safe_summary, std::string raw_diagnostic = {}) {
        common_tool_execution_result result;
        result.failure_code = std::move(code);
        result.failure_class = failure_class;
        result.retryable = retryable;
        result.safe_summary = std::move(safe_summary);
        result.raw_diagnostic = std::move(raw_diagnostic);
        return result;
    }
};
