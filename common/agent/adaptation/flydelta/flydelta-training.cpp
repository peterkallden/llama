#include "agent/adaptation/flydelta/flydelta-training.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool finite_unit(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

bool finite_alpha(float value) {
    return std::isfinite(value) && std::fabs(value) <= 4.0f;
}

bool all_zero(const std::vector<float> & values) {
    for (const float value : values) {
        if (std::fabs(value) > std::numeric_limits<float>::epsilon()) return false;
    }
    return true;
}

common_flydelta_counterfactual_outcome parse_outcome(const std::string & value) {
    if (value == "helped") return common_flydelta_counterfactual_outcome::helped;
    if (value == "neutral") return common_flydelta_counterfactual_outcome::neutral;
    if (value == "harmed") return common_flydelta_counterfactual_outcome::harmed;
    return common_flydelta_counterfactual_outcome::unknown;
}

} // namespace

const char * common_flydelta_training_split_name(common_flydelta_training_split split) {
    switch (split) {
        case common_flydelta_training_split::train: return "train";
        case common_flydelta_training_split::validation: return "validation";
        case common_flydelta_training_split::holdout: return "holdout";
    }
    return "holdout";
}

bool common_flydelta_training_split_from_name(
        const std::string & value,
        common_flydelta_training_split & split) {
    if (value == "train") split = common_flydelta_training_split::train;
    else if (value == "validation") split = common_flydelta_training_split::validation;
    else if (value == "holdout") split = common_flydelta_training_split::holdout;
    else return false;
    return true;
}

bool common_flydelta_alpha_search_config_validate(
        const common_flydelta_alpha_search_config & config,
        std::string & error) {
    error.clear();
    if (config.candidates.empty() || config.candidates.size() > config.max_candidates ||
            config.max_candidates == 0 || !std::isfinite(config.magnitude_penalty) ||
            config.magnitude_penalty < 0.0f) {
        error = "FlyDelta alpha search configuration is invalid";
        return false;
    }
    for (const float alpha : config.candidates) {
        if (!finite_alpha(alpha)) {
            error = "FlyDelta alpha candidate is invalid";
            return false;
        }
    }
    return true;
}

bool common_flydelta_alpha_trial_validate(
        const common_flydelta_alpha_trial & trial,
        std::string & error) {
    error.clear();
    if (!finite_alpha(trial.alpha) || !std::isfinite(trial.quality_delta) ||
            trial.quality_delta < -1.0f || trial.quality_delta > 1.0f) {
        error = "FlyDelta alpha trial is invalid";
        return false;
    }
    if (trial.verifier_known && !bounded(trial.evidence_ref)) {
        error = "FlyDelta verified alpha trial requires evidence";
        return false;
    }
    // A verifier may know that both arms failed without that constituting a
    // comparable outcome. Such a trial is retained as known negative
    // evidence, but common_flydelta_select_alpha deliberately skips it.
    return true;
}

bool common_flydelta_select_alpha(
        const common_flydelta_alpha_search_config & config,
        const std::vector<common_flydelta_alpha_trial> & trials,
        common_flydelta_alpha_selection & selection,
        std::string & error) {
    error.clear();
    selection = {};
    if (!common_flydelta_alpha_search_config_validate(config, error) || trials.empty()) {
        if (error.empty()) error = "FlyDelta alpha search requires trials";
        return false;
    }
    float best_score = -std::numeric_limits<float>::infinity();
    float best_magnitude = std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (!common_flydelta_alpha_trial_validate(trial, error)) return false;
        if (!trial.executed || !trial.verifier_known ||
                trial.outcome != common_flydelta_counterfactual_outcome::helped) continue;
        if (std::find(config.candidates.begin(), config.candidates.end(), trial.alpha) ==
                config.candidates.end()) {
            error = "FlyDelta alpha trial is not part of the configured candidate grid";
            return false;
        }
        const float score = trial.quality_delta - config.magnitude_penalty * std::fabs(trial.alpha);
        const float magnitude = std::fabs(trial.alpha);
        if (!selection.selected || score > best_score ||
                (score == best_score && (magnitude < best_magnitude ||
                (magnitude == best_magnitude && trial.alpha < selection.alpha)))) {
            selection.selected = true;
            selection.alpha = trial.alpha;
            selection.score = score;
            selection.trial_index = index;
            best_score = score;
            best_magnitude = magnitude;
        }
    }
    return true;
}

bool common_flydelta_run_alpha_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_alpha_search_config & config,
        const common_flydelta_alpha_runner & runner,
        std::vector<common_flydelta_alpha_trial> & trials,
        common_flydelta_alpha_selection & selection,
        std::string & error) {
    error.clear();
    trials.clear();
    selection = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_alpha_search_config_validate(config, error) || !runner) {
        if (error.empty()) error = "FlyDelta alpha runner configuration is invalid";
        return false;
    }

    common_flydelta_counterfactual_trial baseline;
    if (!runner(fixture, 0.0f, false, baseline, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error)) {
        return false;
    }
    for (const float alpha : config.candidates) {
        common_flydelta_counterfactual_trial candidate;
        if (!runner(fixture, alpha, std::fabs(alpha) > std::numeric_limits<float>::epsilon(),
                candidate, error) ||
                !common_flydelta_counterfactual_trial_validate(candidate, error)) {
            return false;
        }
        common_flydelta_alpha_trial trial;
        trial.alpha = alpha;
        trial.executed = candidate.executed;
        trial.verifier_known = baseline.verifier_known && candidate.verifier_known;
        trial.outcome = common_flydelta_classify_counterfactual(baseline, candidate);
        trial.quality_delta = candidate.quality - baseline.quality;
        trial.evidence_ref = candidate.evidence_ref;
        trials.push_back(std::move(trial));
    }
    return common_flydelta_select_alpha(config, trials, selection, error);
}

bool common_flydelta_training_example_from_alpha_selection(
        const std::string & id,
        const std::string & behavior_key,
        const std::string & context_fingerprint,
        const std::string & basis_revision,
        const common_flydelta_sparse_code & context,
        const std::vector<float> & target_coefficients,
        const std::string & evidence_ref,
        common_flydelta_training_split split,
        float confidence,
        const common_flydelta_memory_config & memory,
        const std::vector<common_flydelta_alpha_trial> & trials,
        const common_flydelta_alpha_selection & selection,
        common_flydelta_training_example & example,
        std::string & error) {
    error.clear();
    example = {};
    if (!selection.selected || selection.trial_index >= trials.size() ||
            trials[selection.trial_index].alpha != selection.alpha ||
            trials[selection.trial_index].outcome != common_flydelta_counterfactual_outcome::helped ||
            !trials[selection.trial_index].executed ||
            !trials[selection.trial_index].verifier_known) {
        error = "FlyDelta training example requires a selected verified HELPED alpha trial";
        return false;
    }
    example.schema_version = 1;
    example.id = id;
    example.behavior_key = behavior_key;
    example.context_fingerprint = context_fingerprint;
    example.basis_revision = basis_revision;
    example.evidence_ref = evidence_ref.empty()
        ? trials[selection.trial_index].evidence_ref : evidence_ref;
    example.split = split;
    example.context = context;
    example.target_coefficients = target_coefficients;
    example.outcome = common_flydelta_counterfactual_outcome::helped;
    example.confidence = confidence;
    return common_flydelta_training_example_validate(example, memory, error);
}

bool common_flydelta_training_example_validate(
        const common_flydelta_training_example & example,
        const common_flydelta_memory_config & memory,
        std::string & error) {
    error.clear();
    if (!bounded(example.id) || !bounded(example.behavior_key) ||
            !bounded(example.context_fingerprint) || !bounded(example.basis_revision) ||
            !bounded(example.evidence_ref) || !common_flydelta_validate_memory_config(memory, error) ||
            example.context.expansion_dim != memory.expansion_dim ||
            example.context.indices.size() != example.context.values.size() ||
            example.context.indices.empty() || example.target_coefficients.size() != memory.target_dim ||
            !finite_unit(example.confidence) || example.confidence == 0.0f ||
            example.outcome == common_flydelta_counterfactual_outcome::unknown) {
        if (error.empty()) error = "FlyDelta training example identity or dimensions are invalid";
        return false;
    }
    for (size_t i = 0; i < example.context.indices.size(); ++i) {
        if (example.context.indices[i] >= memory.expansion_dim ||
                !std::isfinite(example.context.values[i])) {
            error = "FlyDelta training example sparse context is invalid";
            return false;
        }
    }
    for (const float value : example.target_coefficients) {
        if (!std::isfinite(value) || std::fabs(value) > memory.max_abs_weight) {
            error = "FlyDelta training example target exceeds memory bounds";
            return false;
        }
    }
    if (example.outcome != common_flydelta_counterfactual_outcome::helped &&
            !all_zero(example.target_coefficients)) {
        error = "FlyDelta neutral or harmed examples must target no-op";
        return false;
    }
    if (example.outcome == common_flydelta_counterfactual_outcome::helped &&
            all_zero(example.target_coefficients)) {
        error = "FlyDelta helped example requires a non-zero target";
        return false;
    }
    return true;
}

std::string common_flydelta_training_example_to_json(
        const common_flydelta_training_example & example) {
    return json{
        {"schema_version", example.schema_version},
        {"id", example.id},
        {"behavior_key", example.behavior_key},
        {"context_fingerprint", example.context_fingerprint},
        {"basis_revision", example.basis_revision},
        {"evidence_ref", example.evidence_ref},
        {"split", common_flydelta_training_split_name(example.split)},
        {"context", {
            {"expansion_dim", example.context.expansion_dim},
            {"indices", example.context.indices},
            {"values", example.context.values},
        }},
        {"target_coefficients", example.target_coefficients},
        {"outcome", common_flydelta_counterfactual_outcome_name(example.outcome)},
        {"confidence", example.confidence},
    }.dump();
}

bool common_flydelta_training_example_from_json(
        const std::string & text,
        common_flydelta_training_example & example,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        example = {};
        example.schema_version = value.value("schema_version", 0);
        example.id = value.value("id", "");
        example.behavior_key = value.value("behavior_key", "");
        example.context_fingerprint = value.value("context_fingerprint", "");
        example.basis_revision = value.value("basis_revision", "");
        example.evidence_ref = value.value("evidence_ref", "");
        if (!common_flydelta_training_split_from_name(value.value("split", ""), example.split)) {
            error = "FlyDelta training example split is invalid";
            return false;
        }
        const auto context = value.value("context", json::object());
        example.context.expansion_dim = context.value("expansion_dim", 0U);
        example.context.indices = context.value("indices", std::vector<uint32_t>{});
        example.context.values = context.value("values", std::vector<float>{});
        example.target_coefficients = value.value("target_coefficients", std::vector<float>{});
        example.outcome = parse_outcome(value.value("outcome", "unknown"));
        example.confidence = value.value("confidence", 0.0f);
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta training example JSON: ") + exception.what();
        return false;
    }
    return true;
}

bool common_flydelta_train_delta_memory(
        common_flydelta_delta_memory & memory,
        const std::vector<common_flydelta_training_example> & examples,
        float learning_rate,
        float decay,
        size_t & trained_examples,
        std::string & error) {
    error.clear();
    trained_examples = 0;
    if (examples.empty() || !std::isfinite(learning_rate) || learning_rate <= 0.0f ||
            !std::isfinite(decay) || decay < 0.0f || decay > 1.0f) {
        error = "FlyDelta training batch parameters are invalid";
        return false;
    }
    for (const auto & example : examples) {
        if (example.split != common_flydelta_training_split::train ||
                !common_flydelta_training_example_validate(example, memory.config(), error)) {
            if (error.empty()) error = "FlyDelta training batch contains a non-train example";
            return false;
        }
    }
    for (const auto & example : examples) {
        if (!memory.learn(example.context, example.target_coefficients, example.confidence,
                learning_rate, decay, error)) return false;
        ++trained_examples;
    }
    return true;
}
