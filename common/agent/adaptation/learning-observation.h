#pragma once

#include "agent/agent-scope.h"
#include "agent/contracts/agent-learning.h"

#include <cstddef>
#include <string>
#include <vector>

// Host-owned classification.  Model output may suggest a correction, but it
// cannot decide that the correction is suitable for model adaptation.
enum class common_learning_cause {
    unknown,
    model_behavior,
    host_contract,
    policy,
    missing_evidence,
    project_knowledge,
};

enum class common_learning_verification {
    unverified,
    host_verified,
    user_confirmed,
};

const char * common_learning_cause_name(common_learning_cause cause);
const char * common_learning_verification_name(common_learning_verification verification);

struct common_learning_observation {
    int schema_version = 1;
    std::string id;
    common_agent_scope scope;
    std::string source_turn_id;
    std::string source_plan_id;
    std::vector<common_learning_signal> signals;
    std::vector<std::string> evidence_ids;
    std::string recovery_of_signal_id;
    common_learning_cause cause = common_learning_cause::unknown;
    common_learning_verification verification = common_learning_verification::unverified;
    std::string idempotency_key;
    std::string content_hash;
    bool collection_allowed = false;
};

// A normal successful turn is metrics, not adaptation evidence.  A turn with
// a user correction, failure, recovery, or bounded reflection hint is a
// candidate for observation, subject to collection policy.
bool common_learning_observation_qualifies(const common_learning_observation & observation);

bool common_learning_observation_validate(
        const common_learning_observation & observation,
        size_t max_evidence,
        std::string & error);

std::string common_learning_observation_canonical(const common_learning_observation & observation);
std::string common_learning_observation_hash(const common_learning_observation & observation);
