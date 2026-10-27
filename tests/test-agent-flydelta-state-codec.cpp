#include "tools/agent/daemon/flydelta/state.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    using namespace agent_daemon_flydelta_internal;
    common_flydelta_experiment_plan plan;
    plan.continuation.region.layer_indices = {2};
    plan.continuation.region.anchor_layer_index = 2;
    plan.continuation.region.total_scale = 0.02f;
    plan.continuation.region.per_layer_scale = 0.02f;
    common_flydelta_utility_history history{2, 1};
    const auto original = daemon_flydelta_orchestration_state_json(plan, history, "resource://bootstrap");
    common_flydelta_experiment_plan restored;
    common_flydelta_utility_history restored_history;
    std::string error;
    CHECK(daemon_flydelta_orchestration_state_from_json(original.dump(), restored, restored_history, error));
    CHECK(daemon_flydelta_orchestration_state_json(restored, restored_history, "resource://bootstrap") == original);
    CHECK(daemon_flydelta_continuation_key(restored) == daemon_flydelta_continuation_key(plan));

    auto legacy = original;
    for (const char * field : {"search_anchor_ref", "search_anchor_round", "search_anchor_pending",
            "search_model_call_limit", "search_model_calls_used"}) legacy.erase(field);
    CHECK(daemon_flydelta_orchestration_state_from_json(legacy.dump(), restored, restored_history, error));
    CHECK(restored.search_anchor_ref.empty() && !restored.search_anchor_pending);
    CHECK(restored.search_model_call_limit == 256 && restored.search_model_calls_used == 0);
    auto invalid = original;
    invalid["phase"] = "invented_phase";
    CHECK(!daemon_flydelta_orchestration_state_from_json(invalid.dump(), restored, restored_history, error));
    invalid = original;
    invalid["search_model_calls_used"] = 257;
    CHECK(!daemon_flydelta_orchestration_state_from_json(invalid.dump(), restored, restored_history, error));

    common_agent_scope scope;
    scope.namespace_id = "local";
    scope.project_id = "one";
    const auto fingerprint = daemon_flydelta_scope_fingerprint(scope);
    scope.project_id = "two";
    CHECK(daemon_flydelta_scope_fingerprint(scope) != fingerprint);
    common_flydelta_semantic_progress_observation observation;
    observation.available = observation.regression_coverage_complete = true;
    observation.objective.assessment_known = observation.objective.regression_known = true;
    observation.objective.objective_id = "objective";
    observation.objective.objective_revision = "revision:1";
    observation.objective.verifier_ref = "verifier:1";
    observation.objective.satisfied_goals = {"goal:a"};
    common_flydelta_semantic_progress_observation decoded;
    CHECK(daemon_flydelta_anchor_observation_from_json(daemon_flydelta_anchor_observation_json(observation), decoded, error));
    CHECK(decoded.objective.satisfied_goals == observation.objective.satisfied_goals);
    observation.objective.regression_known = false;
    CHECK(!daemon_flydelta_anchor_observation_from_json(daemon_flydelta_anchor_observation_json(observation), decoded, error));
    return 0;
}
