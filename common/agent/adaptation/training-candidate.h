#pragma once

#include "agent/adaptation/learning-observation.h"

#include <cstddef>
#include <string>
#include <vector>

enum class common_learning_destination {
    retain,
    memory,
    procedure,
    training_candidate,
    reject,
};

enum class common_training_candidate_status {
    observed,
    eligible,
    approved,
    rejected,
};

const char * common_learning_destination_name(common_learning_destination destination);
const char * common_training_candidate_status_name(common_training_candidate_status status);

struct common_training_candidate {
    int schema_version = 1;
    std::string id;
    std::vector<std::string> transaction_ids;
    common_learning_cause cause = common_learning_cause::unknown;
    std::string hypothesis;
    std::string approved_prompt;
    std::string approved_target;
    size_t observed_occurrences = 0;
    size_t verified_recoveries = 0;
    size_t contradictions = 0;
    float confidence = 0.0f;
    common_training_candidate_status status = common_training_candidate_status::observed;
};

struct common_training_candidate_policy {
    size_t min_occurrences = 3;
    size_t min_verified_recoveries = 2;
    size_t max_transaction_ids = 64;
    size_t max_text_size = 2048;
    float min_confidence = 0.80f;
};

common_learning_destination common_learning_destination_for(
        const common_learning_observation & observation);

bool common_training_candidate_qualifies(
        const common_training_candidate & candidate,
        const common_training_candidate_policy & policy,
        std::string & error);
