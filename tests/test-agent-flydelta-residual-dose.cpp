#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"
#include <cmath>
#include <cstdio>
#include <limits>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); return 1; } } while (false)
int main() {
    std::string error;
    common_flydelta_low_rank_basis basis;
    basis.dimension = 2; basis.layer_index = 2;
    basis.vectors = {{1, 0}, {1, 1}};
    float norm = 0;
    CHECK(common_flydelta_intervention_norm(basis, {1, 1}, norm, error));
    CHECK(std::abs(norm - std::sqrt(5.0f)) < 1e-6f);
    CHECK(common_flydelta_intervention_norm(basis, {0, 0}, norm, error) && norm == 0);
    CHECK(!common_flydelta_intervention_norm(basis, {1}, norm, error));
    CHECK(!common_flydelta_intervention_norm(basis,
        {1, std::numeric_limits<float>::quiet_NaN()}, norm, error));
    common_flydelta_hidden_state_capture capture;
    capture.captured = true; capture.model_profile_fingerprint = "model:test";
    capture.capture_layout_revision = "layer-input:v1";
    capture.layer_indices = {2}; capture.n_embd = 2;
    capture.position = common_flydelta_capture_position::generation_boundary;
    capture.token_index = 0; capture.prompt_token_count = 1;
    capture.values = {3, 4};
    auto larger = capture; larger.values = {6, 8};
    common_flydelta_residual_calibration calibration;
    CHECK(common_flydelta_calibrate_residual_dose({capture, larger}, 2,
        "calibration:test", "generation_boundary", calibration, error));
    CHECK(calibration.median_norm == 7.5f && calibration.sample_count == 2);
    larger.capture_layout_revision = "different";
    CHECK(!common_flydelta_calibrate_residual_dose({capture, larger}, 2,
        "calibration:test", "generation_boundary", calibration, error));
    CHECK(common_flydelta_calibrate_residual_dose({capture}, 2,
        "calibration:test", "generation_boundary", calibration, error));
    calibration.tokenizer_fingerprint = "tokenizer:test";
    calibration.template_fingerprint = "template:test";
    common_flydelta_experiment_fixture fixture;
    fixture.tokenizer_fingerprint = calibration.tokenizer_fingerprint;
    fixture.template_fingerprint = calibration.template_fingerprint;
    fixture.model_profile_fingerprint = capture.model_profile_fingerprint;
    common_flydelta_coefficient_search_config config;
    config.experimental_dose_mode = common_flydelta_coefficient_dose_mode::residual_fraction;
    config.residual_calibration = calibration;
    config.injection_capture_layout_revision = capture.capture_layout_revision;
    config.injection_token_selection = "generation_boundary";
    config.max_candidates = 8;
    std::vector<std::vector<float>> proposals;
    std::vector<float> doses;
    CHECK(common_flydelta_propose_calibrated_coefficients(fixture, basis, config,
        proposals, error, &doses));
    CHECK(proposals.size() == 5 && doses.size() == proposals.size());
    CHECK(proposals[0] == std::vector<float>({0, 0}));
    for (size_t i = 1; i < proposals.size(); ++i) {
        CHECK(common_flydelta_intervention_norm(basis, proposals[i], norm, error));
        CHECK(std::abs(norm - 0.1f) < 1e-6f);
    }
    auto again = proposals;
    CHECK(common_flydelta_propose_calibrated_coefficients(fixture, basis, config, proposals, error));
    CHECK(again == proposals);
    config.max_l2_norm = 0.001f;
    CHECK(common_flydelta_propose_calibrated_coefficients(fixture, basis, config, proposals, error, &doses));
    CHECK(doses[1] == config.residual_fraction);
    CHECK(common_flydelta_intervention_norm(basis, proposals[1], norm, error) && norm < 0.1f);
    config.max_l2_norm = 0.32f;
    config.experimental_dose_mode = common_flydelta_coefficient_dose_mode::direction_dose;
    config.max_candidates = 12;
    CHECK(common_flydelta_propose_calibrated_coefficients(fixture, basis, config, proposals, error));
    CHECK(proposals.size() <= 12);
    config.residual_calibration.model_profile_fingerprint = "wrong-model";
    CHECK(!common_flydelta_propose_calibrated_coefficients(fixture, basis, config, proposals, error));
    config.residual_calibration = calibration;
    config.strategy = common_flydelta_coefficient_search_strategy::tfo_lite;
    CHECK(!common_flydelta_propose_calibrated_coefficients(fixture, basis, config, proposals, error));
    config.strategy = common_flydelta_coefficient_search_strategy::coordinate;
    basis.vectors = {{1, 0}, {1, 0}};
    CHECK(common_flydelta_propose_calibrated_coefficients(fixture, basis, config, proposals, error));
    for (size_t i = 0; i < proposals.size(); ++i) for (size_t j = 0; j < i; ++j) {
        std::vector<float> diff = {proposals[i][0]-proposals[j][0], proposals[i][1]-proposals[j][1]};
        CHECK(common_flydelta_intervention_norm(basis, diff, norm, error) && norm > 1e-9f);
    }
    return 0;
}
