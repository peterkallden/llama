#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_search_observation observation(
        common_flydelta_counterfactual_outcome outcome,
        bool signal,
        bool budget = true) {
    common_flydelta_search_observation value;
    value.experiment_id = "flydelta://experiment/lifecycle";
    value.candidate_id = "flydelta://candidate/1";
    value.outcome = outcome;
    value.host_verified = true;
    value.budget_remaining = budget;
    value.diagnostics_available = signal;
    value.diagnostics = {1, 2, 0.5f, 0.2f, 0.1f};
    value.sequence_margin_available = signal;
    value.sequence_margin_delta = signal ? 0.4f : 0.0f;
    return value;
}

static common_flydelta_experiment_champion champion(
        const std::string & id, float score, size_t helped, bool holdout, bool regression) {
    common_flydelta_experiment_champion value;
    value.experiment_id = "flydelta://experiment/lifecycle";
    value.candidate_id = id;
    value.model_profile_id = "sha256:model";
    value.fixture_set_revision = "fixtures:v1";
    value.verifier_revision = "verifier:v1";
    value.objective_score = score;
    value.evaluated_turns = 8;
    value.helped_trials = helped;
    value.host_verified = true;
    value.holdout_passed = holdout;
    value.no_regression = regression;
    return value;
}

int main() {
    std::string error;
    common_flydelta_candidate_lineage lineage;
    lineage.candidate_id = "flydelta://candidate/1";
    lineage.parent_candidate_id = "flydelta://candidate/root";
    lineage.mutation_kind = "scale_refinement";
    lineage.generation = 1;
    lineage.direction_id = "flydelta://direction/raw";
    lineage.layer_indices = {2};
    lineage.scale = 0.08f;
    lineage.intervention_budget = 0.08f;
    CHECK(common_flydelta_candidate_lineage_validate(lineage, 8, error));

    common_flydelta_search_decision decision;
    CHECK(common_flydelta_decide_search_disposition(
        observation(common_flydelta_counterfactual_outcome::unknown, true), decision, error));
    CHECK(decision.disposition == common_flydelta_search_disposition::refine);
    CHECK(decision.artifact_action ==
        common_flydelta_experimental_artifact_action::retain_experimental);
    CHECK(decision.evidence_score == 0.0f && decision.search_priority == 1.0f);

    CHECK(common_flydelta_decide_search_disposition(
        observation(common_flydelta_counterfactual_outcome::neutral, true), decision, error));
    CHECK(decision.disposition == common_flydelta_search_disposition::refine);

    CHECK(common_flydelta_decide_search_disposition(
        observation(common_flydelta_counterfactual_outcome::unknown, false), decision, error));
    CHECK(decision.disposition == common_flydelta_search_disposition::retain);

    CHECK(common_flydelta_decide_search_disposition(
        observation(common_flydelta_counterfactual_outcome::harmed, true), decision, error));
    CHECK(decision.disposition == common_flydelta_search_disposition::reject);
    CHECK(decision.artifact_action == common_flydelta_experimental_artifact_action::reject);
    CHECK(common_flydelta_decide_search_disposition(
        observation(common_flydelta_counterfactual_outcome::helped, true), decision, error));
    CHECK(decision.disposition == common_flydelta_search_disposition::validate_repeatability);
    CHECK(decision.artifact_action ==
        common_flydelta_experimental_artifact_action::review_candidate);

    const auto current = champion("baseline", 0.70f, 0, true, true);
    const auto weaker = champion("challenger-weak", 0.80f, 1, false, true);
    common_flydelta_experiment_champion selected;
    CHECK(common_flydelta_select_experiment_champion(current, weaker, selected, error));
    CHECK(selected.candidate_id == current.candidate_id);

    const auto better = champion("challenger-good", 0.80f, 2, true, true);
    CHECK(common_flydelta_select_experiment_champion(current, better, selected, error));
    CHECK(selected.candidate_id == better.candidate_id);

    auto harmful = champion("challenger-harmful", 0.90f, 3, true, true);
    harmful.harmed_trials = 1;
    CHECK(common_flydelta_select_experiment_champion(current, harmful, selected, error));
    CHECK(selected.candidate_id == current.candidate_id);

    auto other_experiment = champion("other", 0.95f, 4, true, true);
    other_experiment.experiment_id =
        "flydelta://experiment/other";
    CHECK(!common_flydelta_select_experiment_champion(current, other_experiment, selected, error));

    common_learning_in_memory_lifecycle_store lifecycle;
    common_flydelta_lifecycle_event_context context;
    context.event_id = "event:search-1";
    context.idempotency_key = "flydelta:search-1";
    context.source_id = "evidence:search-1";
    context.scope.namespace_id = "local";
    context.scope.project_id = "agent-tests";
    context.scope.session_id = "session-1";
    context.content_hash = "sha256:search-1";
    context.created_at = "2026-09-15T00:00:00Z";
    common_flydelta_search_decision unknown_decision;
    auto unknown_observation = observation(common_flydelta_counterfactual_outcome::unknown, true);
    unknown_observation.search_kind = "alpha";
    unknown_observation.experimental_artifact_id = "flydelta://sideband/experiment-v1";
    CHECK(common_flydelta_decide_search_disposition(
        unknown_observation,
        unknown_decision, error));
    CHECK(common_flydelta_append_search_lifecycle(
        lifecycle, context, unknown_observation,
        unknown_decision, &lineage, error));
    CHECK(common_flydelta_append_search_lifecycle(
        lifecycle, context, unknown_observation,
        unknown_decision, &lineage, error));
    auto records = lifecycle.list(error);
    CHECK(error.empty() && records.size() == 1);
    CHECK(records.front().kind == common_learning_lifecycle_kind::flydelta_result);
    CHECK(records.front().payload_json.find("candidate_search_decision") != std::string::npos);
    CHECK(records.front().payload_json.find("\"artifact_status\":\"experimental\"") !=
        std::string::npos);
    CHECK(records.front().payload_json.find("\"search_kind\":\"alpha\"") !=
        std::string::npos);

    context.event_id = "event:champion-1";
    context.idempotency_key = "flydelta:champion-1";
    context.source_id = "evidence:champion-1";
    CHECK(common_flydelta_append_champion_lifecycle(
        lifecycle, context, current, better, selected, error));
    records = lifecycle.list(error);
    CHECK(error.empty() && records.size() == 2);
    CHECK(records.back().payload_json.find("experiment_champion_selection") != std::string::npos);
    return 0;
}
