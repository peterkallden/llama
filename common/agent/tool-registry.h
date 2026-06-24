#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

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

    static common_tool_execution_result success(std::string output) { return {true, std::move(output)}; }
    static common_tool_execution_result failure(std::string code, common_tool_failure_class failure_class, bool retryable,
            std::string safe_summary, std::string raw_diagnostic = {}) {
        return {false, {}, std::move(code), failure_class, retryable, std::move(safe_summary), std::move(raw_diagnostic)};
    }
};

struct common_registered_tool {
    std::string name;
    uint32_t version = 1;
    std::string executor_id = "native.unnamed";
    std::string arguments_schema;
    bool read_only = true;
    // A policy-gated proposal may write only through a native policy callback.
    // It is not a general write capability.
    bool policy_gated = false;
    std::function<common_tool_execution_result(const std::string &)> handler;
};

struct common_registered_tool_call { std::string name; std::string arguments_json = "{}"; };

class common_tool_registry {
public:
    bool register_tool(common_registered_tool tool, std::string & error);
    // Validates registration and the object-shaped JSON contract without
    // invoking the handler. Runtime callers use this to distinguish a bad
    // plan/tool contract from an ordinary handler failure.
    bool validate(const common_registered_tool_call & call, std::string & error) const;
    common_tool_execution_result execute(const common_registered_tool_call & call) const;
    bool contains(const std::string & name) const;
    bool matches_binding(const std::string & name, uint32_t version, const std::string & executor_id) const;
    bool is_read_only(const std::string & name) const;
    bool is_policy_gated(const std::string & name) const;
private:
    std::unordered_map<std::string, common_registered_tool> tools;
};
