#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_direction_candidate direction(int layer, float first, float second) {
    common_flydelta_direction_candidate value;
    value.kind = common_flydelta_direction_kind::raw_repair;
    value.layer_index = layer;
    value.values = {first, second};
    value.source_samples = 1;
    value.retained_samples = 1;
    value.median_alignment = 1.0f;
    return value;
}

static common_flydelta_experiment_fixture fixture() {
    common_flydelta_experiment_fixture value;
    value.id = "flydelta://fixture/coefficient";
    value.task_fingerprint = "sha256:task";
    value.model_profile_fingerprint = "sha256:model";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.template_fingerprint = "sha256:template";
    value.execution_context_fingerprint = "sha256:context";
    value.verifier_revision = "verifier:v1";
    return value;
}

int main() {
    std::string error;
    common_flydelta_low_rank_basis basis;
    CHECK(common_flydelta_build_low_rank_basis(
        2, 2, {direction(2, 1.0f, 0.0f), direction(2, 0.0f, 1.0f)}, basis, error));
    CHECK(basis.vectors.size() == 2);
    CHECK(std::fabs(basis.vectors[0][0] * basis.vectors[1][0] +
        basis.vectors[0][1] * basis.vectors[1][1]) < 0.0001f);

    common_flydelta_decision_margin margin;
    margin.available = true;
    margin.positive_total_logprob = -1.0f;
    margin.negative_total_logprob = -4.0f;
    margin.positive_token_count = 2;
    margin.negative_token_count = 4;
    CHECK(common_flydelta_decision_margin_validate(margin, error));
    CHECK(std::fabs(margin.total_delta() - 3.0f) < 0.0001f);
    CHECK(std::fabs(margin.normalized_delta() - 0.5f) < 0.0001f);

    common_flydelta_coefficient_search_config config;
    config.step = 0.05f;
    config.max_candidates = 5;
    std::vector<std::vector<float>> proposals;
    CHECK(common_flydelta_propose_low_rank_coefficients(config, 2, proposals, error));
    CHECK(proposals.size() == 5 && proposals[0][0] == 0.0f);
    CHECK(std::string(common_flydelta_coefficient_search_strategy_name(
        common_flydelta_coefficient_search_strategy::coordinate)) == "coordinate");
    config.population_size = 0;
    config.iterations = 0;
    CHECK(common_flydelta_coefficient_search_config_validate(config, 2, error));
    config.strategy = common_flydelta_coefficient_search_strategy::tfo_lite;
    config.population_size = 3;
    config.iterations = 2;
    config.seed = 17;
    CHECK(common_flydelta_coefficient_search_config_validate(config, 2, error));

    std::vector<common_flydelta_coefficient_trial> trials;
    common_flydelta_coefficient_selection selection;
    CHECK(common_flydelta_run_low_rank_coefficient_search(
        fixture(), basis, config,
        [](const auto &, const auto &, const auto & coefficients, bool apply,
                auto & trial, auto & result, std::string &) {
            trial.executed = true;
            trial.verifier_known = true;
            trial.evidence_ref = "evidence://coefficient-trial";
            trial.passed = apply && coefficients[0] > 0.0f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            result.available = true;
            result.positive_total_logprob = apply ? -1.0f : -2.0f;
            result.negative_total_logprob = -2.0f;
            result.positive_token_count = 1;
            result.negative_token_count = 1;
            return true;
        }, trials, selection, error));
    CHECK(selection.selected);
    CHECK(trials[selection.trial_index].outcome == common_flydelta_counterfactual_outcome::helped);

    common_flydelta_coefficient_search_config tfo_config;
    tfo_config.strategy = common_flydelta_coefficient_search_strategy::tfo_lite;
    tfo_config.step = 0.05f;
    tfo_config.max_candidates = 6;
    tfo_config.population_size = 3;
    tfo_config.iterations = 2;
    tfo_config.seed = 23;
    std::vector<common_flydelta_coefficient_trial> tfo_trials;
    common_flydelta_coefficient_selection tfo_selection;
    CHECK(common_flydelta_run_low_rank_coefficient_search(
        fixture(), basis, tfo_config,
        [](const auto &, const auto &, const auto & coefficients, bool apply,
                auto & trial, auto & result, std::string &) {
            trial.executed = true;
            trial.verifier_known = true;
            trial.evidence_ref = "evidence://coefficient-tfo-trial";
            trial.passed = apply && coefficients[0] > 0.0f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            result.available = true;
            result.positive_total_logprob = coefficients[0] + coefficients[1];
            result.negative_total_logprob = 0.0f;
            result.positive_token_count = 1;
            result.negative_token_count = 1;
            return true;
        }, tfo_trials, tfo_selection, error));
    CHECK(tfo_trials.size() >= 3 && tfo_trials.size() <= 6);
    CHECK(tfo_selection.selected);
    bool found_mixed_arm = false;
    for (const auto & trial : tfo_trials) {
        found_mixed_arm = found_mixed_arm ||
            (std::fabs(trial.coefficients[0]) > 0.0001f &&
             std::fabs(trial.coefficients[1]) > 0.0001f);
        CHECK(trial.mutation_kind == "initial_coordinate" ||
              trial.mutation_kind == "initial_mixed" ||
              trial.mutation_kind == "forage_anchor" ||
              trial.mutation_kind == "forage_perturbation");
    }
    CHECK(found_mixed_arm);

    common_learning_in_memory_lifecycle_store lifecycle;
    common_flydelta_lifecycle_event_context lifecycle_context;
    lifecycle_context.event_id = "event:coefficient-search";
    lifecycle_context.idempotency_key = "idempotency:coefficient-search";
    lifecycle_context.source_id = "source:coefficient-search";
    lifecycle_context.scope.namespace_id = "local";
    lifecycle_context.scope.project_id = "agent-tests";
    lifecycle_context.scope.session_id = "session-1";
    lifecycle_context.content_hash = "sha256:coefficient-search";
    lifecycle_context.created_at = "2026-09-15T00:00:00Z";
    CHECK(common_flydelta_append_coefficient_search_lifecycle(
        lifecycle, lifecycle_context, fixture(), basis, tfo_config, tfo_trials,
        "flydelta://sideband/coefficient-search", error));
    const auto records = lifecycle.list(error);
    CHECK(error.empty() && records.size() == tfo_trials.size());
    CHECK(records.front().payload_json.find("coefficient-tfo") != std::string::npos);
    CHECK(records.front().payload_json.find("coefficients") != std::string::npos);

    common_flydelta_coefficient_search_config coordinate_config = tfo_config;
    coordinate_config.strategy = common_flydelta_coefficient_search_strategy::coordinate;
    coordinate_config.max_candidates = tfo_trials.size();
    std::vector<std::vector<float>> coordinate_proposals;
    CHECK(common_flydelta_propose_low_rank_coefficients(
        coordinate_config, basis.vectors.size(), coordinate_proposals, error));
    CHECK(!coordinate_proposals.empty() &&
          coordinate_proposals.size() <= coordinate_config.max_candidates);
    // The strategies share the same bounded arm contract, but TFO-lite may
    // use mixed coefficients where the coordinate stencil cannot.
    for (const auto & trial : tfo_trials) {
        CHECK(std::sqrt(trial.coefficients[0] * trial.coefficients[0] +
            trial.coefficients[1] * trial.coefficients[1]) <=
            tfo_config.max_l2_norm + 0.0001f);
    }
    return 0;
}
