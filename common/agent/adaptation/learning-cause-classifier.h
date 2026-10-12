#pragma once

#include "agent/adaptation/learning-observation.h"
#include "agent/contracts/agent-result.h"

#include <set>
#include <string>

struct common_learning_cause_classifier_config {
    // Validation failures are host-contract evidence by default. A host may
    // explicitly allow stable model-facing tools to be classified as model
    // behavior after the contract has been verified.
    std::set<std::string> stable_model_facing_tools;
};

common_learning_cause common_learning_classify_result(
        const common_agent_result & result,
        const common_learning_cause_classifier_config & config = {});

std::string common_learning_recovery_reference(const common_agent_result & result);
