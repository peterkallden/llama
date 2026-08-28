#pragma once

#include "resource/resource-contract.h"
#include "agent/dataset-contracts.h"

#include <string>
#include <vector>

struct common_agent_tool_call {
    std::string name;
    std::string arguments_json = "{}";
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

    static common_tool_execution_result success(std::string output) {
        return {true, std::move(output), {}, common_tool_failure_class::execution, false, {}, {}, {}, {}, {}};
    }
    static common_tool_execution_result success(
            std::string output,
            std::string content_summary,
            std::vector<common_runtime_resource_ref> resource_refs = {}) {
        common_tool_execution_result result{
            true, std::move(output), {}, common_tool_failure_class::execution, false, {}, {}, {}, {}, {}};
        result.content_summary = std::move(content_summary);
        result.resource_refs = std::move(resource_refs);
        return result;
    }
    static common_tool_execution_result failure(std::string code, common_tool_failure_class failure_class, bool retryable,
            std::string safe_summary, std::string raw_diagnostic = {}) {
        return {false, {}, std::move(code), failure_class, retryable, std::move(safe_summary),
            std::move(raw_diagnostic), {}, {}, {}};
    }
};
