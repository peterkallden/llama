#include "agent/adaptation/learning-cause-classifier.h"

#include <algorithm>

common_learning_cause common_learning_classify_result(
        const common_agent_result & result,
        const common_learning_cause_classifier_config & config) {
    const auto failure = std::find_if(result.failures.begin(), result.failures.end(),
        [](const auto & value) { return !value.tool_name.empty(); });
    if (failure == result.failures.end()) {
        for (const auto & signal : result.learning_signals) {
            if (signal.type == common_learning_signal_type::user_correction) {
                return common_learning_cause::project_knowledge;
            }
        }
        return common_learning_cause::unknown;
    }

    switch (failure->classification) {
        case common_agent_failure_class::policy:
            return common_learning_cause::policy;
        case common_agent_failure_class::validation:
            return config.stable_model_facing_tools.count(failure->tool_name) != 0
                ? common_learning_cause::model_behavior
                : common_learning_cause::host_contract;
        case common_agent_failure_class::model_format:
            return common_learning_cause::model_behavior;
        case common_agent_failure_class::not_found:
        case common_agent_failure_class::timeout:
        case common_agent_failure_class::network:
        case common_agent_failure_class::execution:
        case common_agent_failure_class::plan_conflict:
        case common_agent_failure_class::limit:
            return common_learning_cause::host_contract;
    }
    return common_learning_cause::unknown;
}

std::string common_learning_recovery_reference(const common_agent_result & result) {
    const auto recovery = std::find_if(result.learning_signals.begin(), result.learning_signals.end(),
        [](const auto & signal) {
            return signal.type == common_learning_signal_type::successful_recovery && !signal.evidence_id.empty();
        });
    return recovery == result.learning_signals.end() ? std::string{} : recovery->evidence_id;
}
