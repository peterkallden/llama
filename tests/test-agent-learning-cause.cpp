#include "agent/adaptation/learning-cause-classifier.h"

#include <cassert>

int main() {
    common_agent_result validation;
    validation.failures.push_back({
        "tool.invalid_arguments", common_agent_failure_class::validation,
        "tool_execution", "data.query", "step_1", "evidence-1", false,
        "invalid arguments", "{}"});
    validation.learning_signals.push_back({common_learning_signal_type::tool_failure,
        "plan-1", "step_1", "data.query", "evidence-1", "invalid arguments"});

    assert(common_learning_classify_result(validation) == common_learning_cause::host_contract);
    common_learning_cause_classifier_config stable;
    stable.stable_model_facing_tools.insert("data.query");
    assert(common_learning_classify_result(validation, stable) == common_learning_cause::model_behavior);
    assert(common_learning_recovery_reference(validation) == "evidence-1");

    common_agent_result network;
    network.failures.push_back({
        "tool.timeout", common_agent_failure_class::timeout,
        "tool_execution", "data.query", "step_1", "evidence-2", true,
        "request timed out", ""});
    assert(common_learning_classify_result(network, stable) == common_learning_cause::host_contract);

    common_agent_result correction;
    correction.learning_signals.push_back({common_learning_signal_type::user_correction,
        "plan-2", "", "", "correction-1", "use the bounded dataset"});
    assert(common_learning_classify_result(correction) == common_learning_cause::project_knowledge);
    return 0;
}
