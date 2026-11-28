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

inline std::vector<float> alternative_capture_destination(llama_model * model,
        const recirculation_config & config, const std::vector<llama_token> & tokens) {
    if (tokens.size() < 6 || tokens.size() > 640) throw std::invalid_argument("invalid contrast token count");
    llama_backend host(model, config, uint32_t(tokens.size() + 1));
    std::vector<float> destination;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        capture current;
        if (!host.first(tokens[i], int32_t(i), i + 2 == tokens.size(), current))
            throw std::runtime_error("contrast capture failed");
        if (i + 2 == tokens.size()) destination = std::move(current.destination);
        host.clear_hooks();
    }
    if (destination.size() != size_t(llama_model_n_embd(model)))
        throw std::runtime_error("contrast capture width changed");
    return destination;
}

struct alternative_direction_fit {
    std::vector<float> values;
    std::vector<std::string> training_ids;
    std::vector<std::string> excluded_ids;
    std::vector<double> training_alignments;
};

inline alternative_direction_fit alternative_fit_direction(llama_model * model,
        const recirculation_config & config, const std::vector<experiment_case> & development) {
    alternative_direction_fit fit;
    std::vector<double> sum(size_t(llama_model_n_embd(model)), 0);
    std::vector<std::vector<float>> samples;
    for (const auto & item : development) {
        if (item.task_type != "object_location") continue;
        const auto counterfactual = alternative_counterfactual_user(item);
        const auto counterfactual_tokens = experiment_tokenize(model,
            render_experiment_prompt(model, item.system, counterfactual));
        if (counterfactual_tokens.size() != item.prompt_tokens.size()) {
            fit.excluded_ids.push_back(item.id);
            continue;
        }
        const auto positive = alternative_capture_destination(model, config, item.prompt_tokens);
        const auto negative = alternative_capture_destination(model, config, counterfactual_tokens);
        std::vector<float> delta(sum.size());
        double norm = 0;
        for (size_t i = 0; i < sum.size(); ++i) {
            delta[i] = positive[i] - negative[i];
            norm += double(delta[i]) * delta[i];
        }
        norm = std::sqrt(norm);
        if (!std::isfinite(norm) || norm <= 1e-8) {
            fit.excluded_ids.push_back(item.id);
            continue;
        }
        for (size_t i = 0; i < sum.size(); ++i) {
            delta[i] = float(delta[i] / norm);
            sum[i] += delta[i];
        }
        samples.push_back(std::move(delta));
        fit.training_ids.push_back(item.id);
    }
    if (fit.training_ids.size() < 2) throw std::runtime_error("insufficient aligned development contrasts");
    fit.values.resize(sum.size());
    double norm = 0;
    for (double value : sum) norm += value * value;
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm <= 1e-8) throw std::runtime_error("development contrasts have no shared direction");
    for (size_t i = 0; i < sum.size(); ++i) fit.values[i] = float(sum[i] / norm);
    for (const auto & sample : samples) {
        double dot = 0;
        for (size_t i = 0; i < sample.size(); ++i) dot += double(sample[i]) * fit.values[i];
        fit.training_alignments.push_back(dot);
    }
    return fit;
}

inline experiment_json alternative_direction_case(llama_model * model, const experiment_case & item,
        const recirculation_config & config, const std::vector<float> & direction) {
    if (item.task_type != "object_location" || item.prompt_tokens.size() < 6)
        throw std::invalid_argument("unsupported alternative evaluation case");
    if (direction.size() != size_t(llama_model_n_embd(model)))
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
        const auto positive = alternative_capture_destination(model, config, item.prompt_tokens);
        const auto negative = alternative_capture_destination(model, config, counterfactual_tokens);
        double dot = 0, direction_squared = 0, contrast_squared = 0;
        for (size_t i = 0; i < direction.size(); ++i) {
            const double delta = double(positive[i]) - negative[i];
            dot += direction[i] * delta;
            direction_squared += double(direction[i]) * direction[i];
            contrast_squared += delta * delta;
        }
        if (direction_squared > 1e-20 && contrast_squared > 1e-20)
            heldout_alignment = dot / std::sqrt(direction_squared * contrast_squared);
    }
    const auto baseline = run_window(model, config, prefix, window, 4, 1);
    const auto recurring = run_window(model, config, prefix, window, 4, 2);
    const auto alternative = run_window(model, config, prefix, window, 4, 1, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, {direction});
    const auto alternative_recurring = run_window(model, config, prefix, window, 4, 2, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, {direction});
    const auto alternative_fixed = run_window(model, config, prefix, window, 4, 2, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, {direction, false});
    const auto alternative_matched = run_window(model, config, prefix, window, 4, 2, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false,
        {direction, false, alternative_recurring.patch_displacement_norms.back()});
    if (std::abs(alternative_matched.patch_displacement_norms.back() -
            alternative_recurring.patch_displacement_norms.back()) >
            1e-4 * std::max(1.0, alternative_recurring.patch_displacement_norms.back()))
        throw std::runtime_error("alternative second patch norm control failed");
    const auto shuffled_control = run_window(model, config, prefix, window, 4, 1, 0.5f,
        window_target_mode::fixed, 0, 0, 1, 1, 1, false, {shuffled});
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
