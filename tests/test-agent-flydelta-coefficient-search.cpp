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
    return 0;
}
