#include "agent/adaptation/flydelta/flydelta-training.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_training_example example(
        common_flydelta_training_split split,
        common_flydelta_counterfactual_outcome outcome) {
    common_flydelta_training_example value;
    value.id = "flydelta://training/1";
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.context_fingerprint = "sha256:context";
    value.basis_revision = "flydelta://basis/1";
    value.evidence_ref = "evidence://counterfactual/1";
    value.split = split;
    value.context.expansion_dim = 8;
    value.context.indices = {1, 4};
    value.context.values = {0.8f, 0.2f};
    value.target_coefficients = outcome == common_flydelta_counterfactual_outcome::helped
        ? std::vector<float>{0.25f, -0.1f}
        : std::vector<float>{0.0f, 0.0f};
    value.outcome = outcome;
    value.confidence = 1.0f;
    return value;
}

int main() {
    std::string error;

    common_flydelta_alpha_search_config alpha_config;
    alpha_config.candidates = {0.05f, 0.1f, 0.2f};
    alpha_config.magnitude_penalty = 0.25f;
    std::vector<common_flydelta_alpha_trial> trials = {
        {0.05f, common_flydelta_counterfactual_outcome::helped, 0.60f, true, true, "evidence:small"},
        {0.1f, common_flydelta_counterfactual_outcome::helped, 0.70f, true, true, "evidence:medium"},
        {0.2f, common_flydelta_counterfactual_outcome::harmed, -0.20f, true, true, "evidence:harmed"},
    };
    common_flydelta_alpha_selection selection;
    CHECK(common_flydelta_select_alpha(alpha_config, trials, selection, error));
    CHECK(selection.selected && std::fabs(selection.alpha - 0.1f) < 1e-6f);

    common_flydelta_delta_memory memory({8, 2, 0.5f});
    auto train = example(common_flydelta_training_split::train,
            common_flydelta_counterfactual_outcome::helped);
    CHECK(common_flydelta_training_example_validate(train, memory.config(), error));
    const auto round_trip = common_flydelta_training_example_to_json(train);
    common_flydelta_training_example parsed;
    CHECK(common_flydelta_training_example_from_json(round_trip, parsed, error));
    CHECK(parsed.behavior_key == train.behavior_key && parsed.context.indices == train.context.indices);
    size_t trained = 0;
    CHECK(common_flydelta_train_delta_memory(memory, {train}, 0.5f, 1.0f, trained, error));
    CHECK(trained == 1);
    std::vector<float> prediction;
    CHECK(memory.predict(train.context, prediction, error));
    CHECK(std::fabs(prediction[0]) > 0.0f);

    auto validation = example(common_flydelta_training_split::validation,
            common_flydelta_counterfactual_outcome::helped);
    CHECK(!common_flydelta_train_delta_memory(memory, {validation}, 0.5f, 1.0f, trained, error));
    auto invalid_negative = example(common_flydelta_training_split::train,
            common_flydelta_counterfactual_outcome::harmed);
    invalid_negative.target_coefficients[0] = 0.1f;
    CHECK(!common_flydelta_training_example_validate(invalid_negative, memory.config(), error));
    return 0;
}
