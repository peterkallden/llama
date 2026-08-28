#include "agent/adaptation/learning-observation.h"

#include <cassert>

static common_learning_observation base_observation() {
    common_learning_observation observation;
    observation.id = "learning://observation/1";
    observation.scope.namespace_id = "local";
    observation.scope.session_id = "session-a";
    observation.scope.project_id = "project-a";
    observation.scope.turn_id = "turn-a";
    observation.source_turn_id = "turn-a";
    observation.source_plan_id = "plan-a";
    observation.signals.push_back({common_learning_signal_type::tool_failure, "plan-a", "step-a",
        "data.inspect", "tool:step-a:failure", "invalid binding"});
    observation.evidence_ids = {"tool:step-a:failure"};
    observation.cause = common_learning_cause::model_behavior;
    observation.verification = common_learning_verification::host_verified;
    observation.idempotency_key = "turn-a:tool:step-a:failure";
    observation.content_hash = "sha256:source";
    observation.collection_allowed = true;
    return observation;
}

int main() {
    auto observation = base_observation();
    std::string error;
    assert(common_learning_observation_qualifies(observation));
    assert(common_learning_observation_validate(observation, 4, error));
    assert(error.empty());
    assert(common_learning_observation_hash(observation) == common_learning_observation_hash(observation));

    auto ordinary = observation;
    ordinary.signals.clear();
    assert(!common_learning_observation_qualifies(ordinary));
    assert(!common_learning_observation_validate(ordinary, 4, error));

    auto denied = observation;
    denied.collection_allowed = false;
    assert(!common_learning_observation_qualifies(denied));
    assert(!common_learning_observation_validate(denied, 4, error));

    auto oversized = observation;
    oversized.evidence_ids = {"a", "b", "c"};
    assert(!common_learning_observation_validate(oversized, 2, error));
    return 0;
}
