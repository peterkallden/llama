#pragma once

#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"
#include "agent/adaptation/training-candidate.h"

#include <vector>

// Resolved inputs from the existing FlyDelta evidence artifacts and generic
// learning ledger. The adapter validates that they describe the same verified
// execution before projecting them into the shared training-candidate path.
struct common_flydelta_training_case_source {
    common_flydelta_teaching_relation relation;
    common_adaptation_evidence evidence;
    common_learning_case learning_case;
    common_learning_transaction transaction;
};

struct common_flydelta_training_admission_policy {
    size_t minimum_independent_relations = 3;
    size_t max_relations = 64;
};

bool common_flydelta_admit_training_cases(
        const std::vector<common_flydelta_training_case_source> & sources,
        const common_flydelta_training_admission_policy & policy,
        std::vector<common_training_candidate> & candidates,
        std::string & error);
