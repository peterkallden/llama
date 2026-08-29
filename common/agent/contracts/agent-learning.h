#pragma once

#include <string>
#include <utility>

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

// Canonical host-owned metadata for a tool-learning signal. The family is the
// model-facing routing family; provider_kind records where the execution came
// from without making provider naming part of the model contract.
struct common_learning_tool_metadata {
    std::string tool_family;
    std::string provider_kind;
};

struct common_learning_signal {
    common_learning_signal(
            common_learning_signal_type type = common_learning_signal_type::tool_failure,
            std::string plan_id = {},
            std::string step_id = {},
            std::string tool_name = {},
            std::string evidence_id = {},
            std::string summary = {},
            std::string tool_family = {},
            std::string provider_kind = {})
        : type(type), plan_id(std::move(plan_id)), step_id(std::move(step_id)),
          tool_name(std::move(tool_name)), evidence_id(std::move(evidence_id)),
          summary(std::move(summary)), tool_family(std::move(tool_family)),
          provider_kind(std::move(provider_kind)) {}

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
