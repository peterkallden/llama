#include "agent/runtime-json-contracts.h"

using json = nlohmann::ordered_json;

nlohmann::ordered_json common_agent_runtime_user_correction_to_json(
        const std::string & source_turn_id,
        const std::string & statement) {
    return {
        {"source_turn_id", source_turn_id},
        {"statement", statement},
    };
}

nlohmann::ordered_json common_agent_runtime_failure_observation_to_json(
        const common_agent_failure & failure) {
    return {
        {"failure", {
            {"code", failure.code},
            {"class", common_agent_failure_class_name(failure.classification)},
            {"stage", failure.stage},
            {"tool", failure.tool_name},
            {"step_id", failure.step_id},
            {"retryable", failure.retryable},
            {"safe_summary", failure.safe_summary},
            {"evidence_id", failure.evidence_id},
        }},
    };
}

nlohmann::ordered_json common_agent_runtime_reflection_learning_hint_to_json(
        const common_reflection_learning_hint & hint) {
    return {
        {"category", hint.category},
        {"statement", hint.statement},
        {"expected_reuse", hint.expected_reuse},
    };
}

std::string common_agent_runtime_normalize_reasoning_observation_json(
        const std::string & reasoning_text) {
    const auto parsed = json::parse(reasoning_text, nullptr, false);
    if (parsed.is_object()) {
        return parsed.dump();
    }
    return json({
        {"summary", reasoning_text},
        {"format", "unstructured"},
    }).dump();
}
