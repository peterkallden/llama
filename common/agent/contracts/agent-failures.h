#pragma once

#include <string>

enum class common_agent_failure_class {
    validation, policy, not_found, timeout, network, execution,
    model_format, plan_conflict, limit,
};

inline const char * common_agent_failure_class_name(common_agent_failure_class value) {
    switch (value) {
        case common_agent_failure_class::validation: return "validation";
        case common_agent_failure_class::policy: return "policy";
        case common_agent_failure_class::not_found: return "not_found";
        case common_agent_failure_class::timeout: return "timeout";
        case common_agent_failure_class::network: return "network";
        case common_agent_failure_class::execution: return "execution";
        case common_agent_failure_class::model_format: return "model_format";
        case common_agent_failure_class::plan_conflict: return "plan_conflict";
        case common_agent_failure_class::limit: return "limit";
    }
    return "execution";
}

struct common_agent_failure {
    std::string code;
    common_agent_failure_class classification = common_agent_failure_class::execution;
    std::string stage;
    std::string tool_name;
    std::string step_id;
    std::string evidence_id;
    bool retryable = false;
    std::string safe_summary;
    std::string repair_context_json;
};
