#pragma once

#include "historical-window.h"

namespace recirculation {

inline std::string alternative_counterfactual_user(const experiment_case & item) {
    if (item.task_type != "object_location" || item.expected.empty() || item.incorrect.empty() ||
            item.expected.rfind("The ", 0) != 0 ||
            item.incorrect.rfind("The ", 0) != 0 || item.expected.back() != '.' ||
            item.incorrect.back() != '.') throw std::invalid_argument("unsupported contrast case");
    const std::string new_location = item.expected.substr(4, item.expected.size() - 5);
    const std::string old_location = item.incorrect.substr(4, item.incorrect.size() - 5);
    const std::string needle = "to the " + new_location + ".";
    const size_t update = item.user.find(" UPDATE:");
    const size_t position = item.user.find(needle, update);
    if (update == std::string::npos || position == std::string::npos ||
            item.user.find(needle, position + 1) != std::string::npos)
        throw std::invalid_argument("contrast update is ambiguous");
    auto counterfactual = item.user;
    counterfactual.replace(position, needle.size(), "to the " + old_location + ".");
    return counterfactual;
}

struct alternative_layer_capture {
    std::vector<float> destination;
    std::vector<float> downstream;
};

inline alternative_layer_capture alternative_capture_layers(llama_model * model,
        const recirculation_config & config, const std::vector<llama_token> & tokens) {
    if (tokens.size() < 6 || tokens.size() > 640) throw std::invalid_argument("invalid contrast token count");
    llama_backend host(model, config, uint32_t(tokens.size() + 1));
    alternative_layer_capture result;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        capture current;
        if (!host.first(tokens[i], int32_t(i), i + 2 == tokens.size(), current))
            throw std::runtime_error("contrast capture failed");
        if (i + 2 == tokens.size()) {
            result.destination = std::move(current.destination);
            result.downstream = std::move(current.source);
        }
        host.clear_hooks();
    }
    if (result.destination.size() != size_t(llama_model_n_embd(model)) ||
            result.downstream.size() != result.destination.size())
        throw std::runtime_error("contrast capture width changed");
    return result;
}

struct alternative_direction_fit {
    std::vector<float> values;
    std::vector<float> downstream_reference;
    std::vector<std::string> training_ids;
    std::vector<std::string> excluded_ids;
    std::vector<double> training_alignments;
    std::vector<std::optional<double>> leave_one_out_alignments;
    std::vector<std::optional<double>> downstream_leave_one_out_alignments;
};

inline std::vector<float> alternative_mean_direction(const std::vector<std::vector<float>> & samples,
        size_t excluded = SIZE_MAX) {
    if (samples.size() < 2 || samples.front().empty()) throw std::invalid_argument("insufficient direction samples");
    std::vector<double> sum(samples.front().size(), 0);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i == excluded) continue;
        if (samples[i].size() != sum.size()) throw std::invalid_argument("direction sample width changed");
        for (size_t j = 0; j < sum.size(); ++j) {
            if (!std::isfinite(samples[i][j])) throw std::invalid_argument("non-finite direction sample");
            sum[j] += samples[i][j];
        }
    }
    double squared = 0;
    for (double value : sum) squared += value * value;
    if (!std::isfinite(squared) || squared <= 1e-16) return {};
    std::vector<float> mean(sum.size());
    for (size_t j = 0; j < sum.size(); ++j) mean[j] = float(sum[j] / std::sqrt(squared));
    return mean;
}

inline std::optional<double> alternative_alignment(const std::vector<float> & sample,
        const std::vector<float> & direction) {
    if (sample.size() != direction.size() || sample.empty()) return std::nullopt;
    double dot = 0, a2 = 0, b2 = 0;
    for (size_t i = 0; i < sample.size(); ++i) {
        if (!std::isfinite(sample[i]) || !std::isfinite(direction[i])) return std::nullopt;
        dot += double(sample[i]) * direction[i];
        a2 += double(sample[i]) * sample[i];
        b2 += double(direction[i]) * direction[i];
    }
    if (a2 <= 1e-20 || b2 <= 1e-20) return std::nullopt;
    return std::clamp(dot / std::sqrt(a2 * b2), -1.0, 1.0);
}

inline alternative_direction_fit alternative_fit_direction(llama_model * model,
        const recirculation_config & config, const std::vector<experiment_case> & development) {
    alternative_direction_fit fit;
    std::vector<std::vector<float>> destination_samples, downstream_samples;
    for (const auto & item : development) {
        if (item.task_type != "object_location") continue;
        const auto counterfactual = alternative_counterfactual_user(item);
        const auto counterfactual_tokens = experiment_tokenize(model,
            render_experiment_prompt(model, item.system, counterfactual));
        if (counterfactual_tokens.size() != item.prompt_tokens.size()) {
            fit.excluded_ids.push_back(item.id);
            continue;
        }
        const auto positive = alternative_capture_layers(model, config, item.prompt_tokens);
        const auto negative = alternative_capture_layers(model, config, counterfactual_tokens);
        const auto normalize_difference = [](const std::vector<float> & a, const std::vector<float> & b) {
            std::vector<float> delta(a.size());
            double squared = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                delta[i] = a[i] - b[i];
                squared += double(delta[i]) * delta[i];
            }
            if (!std::isfinite(squared) || squared <= 1e-16) return std::vector<float>{};
            for (float & value : delta) value = float(value / std::sqrt(squared));
            return delta;
        };
        auto destination_delta = normalize_difference(positive.destination, negative.destination);
        auto downstream_delta = normalize_difference(positive.downstream, negative.downstream);
        if (destination_delta.empty() || downstream_delta.empty()) {
            fit.excluded_ids.push_back(item.id);
            continue;
        }
        destination_samples.push_back(std::move(destination_delta));
        downstream_samples.push_back(std::move(downstream_delta));
        fit.training_ids.push_back(item.id);
    }
    if (fit.training_ids.size() < 2) throw std::runtime_error("insufficient aligned development contrasts");
    fit.values = alternative_mean_direction(destination_samples);
    fit.downstream_reference = alternative_mean_direction(downstream_samples);
    if (fit.values.empty() || fit.downstream_reference.empty())
        throw std::runtime_error("development contrasts have no shared direction");
    for (size_t i = 0; i < destination_samples.size(); ++i) {
        fit.training_alignments.push_back(*alternative_alignment(destination_samples[i], fit.values));
        const auto heldout_destination = alternative_mean_direction(destination_samples, i);
        const auto heldout_downstream = alternative_mean_direction(downstream_samples, i);
        fit.leave_one_out_alignments.push_back(alternative_alignment(destination_samples[i], heldout_destination));
        fit.downstream_leave_one_out_alignments.push_back(alternative_alignment(downstream_samples[i], heldout_downstream));
    }
    return fit;
}

inline experiment_json alternative_direction_case(llama_model * model, const experiment_case & item,
        const recirculation_config & config, const std::vector<float> & direction,
        const std::vector<float> & downstream_reference) {
    if (item.task_type != "object_location" || item.prompt_tokens.size() < 6)
        throw std::invalid_argument("unsupported alternative evaluation case");
    if (direction.size() != size_t(llama_model_n_embd(model)) ||
            downstream_reference.size() != direction.size())
        throw std::invalid_argument("alternative direction width changed");
    double direction_norm = 0;
    for (float value : direction) {
        if (!std::isfinite(value)) throw std::invalid_argument("alternative direction is not finite");
        direction_norm += double(value) * value;
    }
    if (!std::isfinite(direction_norm) || direction_norm <= 1e-20)
        throw std::invalid_argument("alternative direction has zero norm");
    const size_t start = item.prompt_tokens.size() - 6;
    std::vector<llama_token> prefix(item.prompt_tokens.begin(), item.prompt_tokens.begin() + start);
    std::vector<llama_token> window(item.prompt_tokens.begin() + start, item.prompt_tokens.end());
    std::vector<float> shuffled = direction;
    std::rotate(shuffled.begin(), shuffled.begin() + shuffled.size() / 2, shuffled.end());
    std::optional<double> heldout_alignment;
    const auto counterfactual_tokens = experiment_tokenize(model,
        render_experiment_prompt(model, item.system, alternative_counterfactual_user(item)));
    if (counterfactual_tokens.size() == item.prompt_tokens.size()) {
        const auto positive = alternative_capture_layers(model, config, item.prompt_tokens);
        const auto negative = alternative_capture_layers(model, config, counterfactual_tokens);
        double dot = 0, direction_squared = 0, contrast_squared = 0;
        for (size_t i = 0; i < direction.size(); ++i) {
            const double delta = double(positive.destination[i]) - negative.destination[i];
            dot += direction[i] * delta;
            direction_squared += double(direction[i]) * direction[i];
            contrast_squared += delta * delta;
        }
        if (direction_squared > 1e-20 && contrast_squared > 1e-20)
            heldout_alignment = dot / std::sqrt(direction_squared * contrast_squared);
    }
    window_patch_options ordinary_options;
    ordinary_options.downstream_reference = downstream_reference;
    auto alternative_options = ordinary_options;
    alternative_options.initial_direction = direction;
    auto fixed_options = alternative_options;
    fixed_options.update_from_source = false;
    auto matched_options = fixed_options;
    auto shuffled_options = ordinary_options;
    shuffled_options.initial_direction = shuffled;
    const auto baseline = run_window(model, config, prefix, window, 4, 1, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, ordinary_options);
    const auto recurring = run_window(model, config, prefix, window, 4, 2, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, ordinary_options);
    const auto alternative = run_window(model, config, prefix, window, 4, 1, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, alternative_options);
    const auto alternative_recurring = run_window(model, config, prefix, window, 4, 2, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, alternative_options);
    const auto alternative_fixed = run_window(model, config, prefix, window, 4, 2, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, fixed_options);
    matched_options.fixed_second_patch_norm = alternative_recurring.patch_displacement_norms.back();
    const auto alternative_matched = run_window(model, config, prefix, window, 4, 2, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, matched_options);
    if (std::abs(alternative_matched.patch_displacement_norms.back() -
            alternative_recurring.patch_displacement_norms.back()) >
            1e-4 * std::max(1.0, alternative_recurring.patch_displacement_norms.back()))
        throw std::runtime_error("alternative second patch norm control failed");
    const auto shuffled_control = run_window(model, config, prefix, window, 4, 1, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, shuffled_options);
    const auto score = [&](const window_result & result, bool refined) {
        const auto expected = historical_window_continuation_score(model, item, item.expected,
            config, result, item.prompt_tokens.size(), refined);
        const auto incorrect = historical_window_continuation_score(model, item, item.incorrect,
            config, result, item.prompt_tokens.size(), refined);
        return experiment_json{{"margin", expected.total - incorrect.total},
            {"expected_token_logprobs", expected.token_logprobs},
            {"incorrect_token_logprobs", incorrect.token_logprobs},
            {"patch_norms", result.patch_displacement_norms},
            {"patch_cosines", result.patch_direction_cosines},
            {"downstream_projection_deltas", refined ? result.downstream_projection_deltas :
                std::vector<std::optional<double>>{}},
            {"replay_wall_us", result.replay_wall_us},
            {"committed_by_entropy", result.committed}};
    };
    return {{"case_id", item.id}, {"target_position", item.prompt_tokens.size() - 2},
        {"heldout_contrast_alignment", heldout_alignment},
        {"baseline", score(baseline, false)}, {"ordinary_one", score(baseline, true)},
        {"ordinary_two", score(recurring, true)}, {"alternative_one", score(alternative, true)},
        {"alternative_two", score(alternative_recurring, true)},
        {"alternative_fixed_two", score(alternative_fixed, true)},
        {"alternative_matched_two", score(alternative_matched, true)},
        {"shuffled_one", score(shuffled_control, true)}};
}
}
