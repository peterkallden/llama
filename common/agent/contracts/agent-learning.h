#pragma once

#include <string>

enum class common_learning_signal_type {
    tool_failure, successful_recovery, reflection_hint, user_correction,
};

inline const char * common_learning_signal_type_name(common_learning_signal_type type) {
    switch (type) {
        case common_learning_signal_type::tool_failure: return "tool_failure";
        case common_learning_signal_type::successful_recovery: return "successful_recovery";
        case common_learning_signal_type::reflection_hint: return "reflection_hint";
        case common_learning_signal_type::user_correction: return "user_correction";
    }
    return "unknown";
}

struct common_agent_user_correction {
    std::string source_turn_id;
    std::string statement;
};

struct common_learning_signal {
    common_learning_signal_type type = common_learning_signal_type::tool_failure;
    std::string plan_id;
    std::string step_id;
    std::string tool_name;
    std::string evidence_id;
    std::string summary;
    // Host-owned canonical metadata. These fields keep learning transport
    // neutral: provider_kind is provenance, while tool_family is the
    // model-facing family used for policy and corpus views.
    std::string tool_family;
    std::string provider_kind;
};
