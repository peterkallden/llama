#include "agent/adaptation/training-candidate.h"

#include <cassert>

int main() {
    common_learning_observation model_observation;
    model_observation.cause = common_learning_cause::model_behavior;
    model_observation.verification = common_learning_verification::host_verified;
    assert(common_learning_destination_for(model_observation) == common_learning_destination::training_candidate);

    common_learning_observation host_observation = model_observation;
    host_observation.cause = common_learning_cause::host_contract;
    assert(common_learning_destination_for(host_observation) == common_learning_destination::retain);

    common_learning_observation project_observation = model_observation;
    project_observation.cause = common_learning_cause::project_knowledge;
    assert(common_learning_destination_for(project_observation) == common_learning_destination::memory);

    common_training_candidate candidate;
    candidate.id = "learning://candidate/1";
    candidate.transaction_ids = {"learning://transaction/1", "learning://transaction/2", "learning://transaction/3"};
    candidate.cause = common_learning_cause::model_behavior;
    candidate.hypothesis = "The model emits an invalid binding for a stable contract.";
    candidate.approved_prompt = "Use the stable fixture contract.";
    candidate.approved_target = "Emit a valid tool binding.";
    candidate.observed_occurrences = 3;
    candidate.verified_recoveries = 2;
    candidate.confidence = 0.9f;
    std::string error;
    assert(!common_training_candidate_qualifies(candidate, {}, error));
    candidate.status = common_training_candidate_status::approved;
    assert(common_training_candidate_qualifies(candidate, {}, error));

    candidate.contradictions = 1;
    assert(!common_training_candidate_qualifies(candidate, {}, error));
    candidate.contradictions = 0;
    candidate.cause = common_learning_cause::host_contract;
    assert(!common_training_candidate_qualifies(candidate, {}, error));
    return 0;
}
