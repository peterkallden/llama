#pragma once

#include "historical-window.h"
#include <random>

namespace recirculation {

inline std::vector<uint8_t> selective_random_mask(size_t width, size_t count, uint32_t seed) {
    if (!width || !count || count > width) throw std::invalid_argument("invalid mask density");
    std::vector<size_t> indices(width);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);
    std::vector<uint8_t> mask(width, 0);
    for (size_t i = 0; i < count; ++i) mask[indices[i]] = 1;
    return mask;
}

inline std::vector<uint8_t> selective_magnitude_mask(const std::vector<double> & magnitudes, size_t count) {
    if (magnitudes.empty() || !count || count > magnitudes.size()) throw std::invalid_argument("invalid mask density");
    std::vector<size_t> indices(magnitudes.size());
    std::iota(indices.begin(), indices.end(), 0);
    for (double value : magnitudes) if (!std::isfinite(value)) throw std::invalid_argument("invalid source magnitude");
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&](size_t a, size_t b) {
        return magnitudes[a] == magnitudes[b] ? a < b : magnitudes[a] > magnitudes[b];
    });
    std::vector<uint8_t> mask(magnitudes.size(), 0);
    for (size_t i = 0; i < count; ++i) mask[indices[i]] = 1;
    return mask;
}

inline std::pair<std::vector<llama_token>, std::vector<llama_token>> selective_window(const experiment_case & item) {
    if (item.prompt_tokens.size() < 6 || item.prompt_tokens.size() > 640)
        throw std::invalid_argument("invalid selective mask case");
    const size_t start = item.prompt_tokens.size() - 6;
    return {{item.prompt_tokens.begin(), item.prompt_tokens.begin() + start},
        {item.prompt_tokens.begin() + start, item.prompt_tokens.end()}};
}

inline experiment_json selective_mask_case(llama_model * model, const experiment_case & item,
        const recirculation_config & config, const std::vector<uint8_t> & selected,
        const std::vector<std::vector<uint8_t>> & controls) {
    const auto [prefix, window] = selective_window(item);
    const auto ordinary = run_window(model, config, prefix, window, 4, 1);
    const auto score = [&](const window_result & result, bool refined) {
        const auto expected = historical_window_continuation_score(model, item, item.expected,
            config, result, item.prompt_tokens.size(), refined);
        const auto incorrect = historical_window_continuation_score(model, item, item.incorrect,
            config, result, item.prompt_tokens.size(), refined);
        return experiment_json{{"margin", expected.total - incorrect.total},
            {"expected_token_logprobs", expected.token_logprobs},
            {"incorrect_token_logprobs", incorrect.token_logprobs},
            {"patch_norm", refined ? result.patch_displacement_norms.front() : 0},
            {"replay_wall_us", result.replay_wall_us}};
    };
    const double ordinary_norm = ordinary.patch_displacement_norms.front();
    const auto masked = [&](const std::vector<uint8_t> & mask) {
        window_patch_options options;
        options.source_mask = mask;
        const auto result = run_window(model, config, prefix, window, 4, 1, 0.5f,
            window_target_mode::fixed, 0, 0, 1, 1, 1, false, options);
        const double actual = result.patch_displacement_norms.front();
        if (std::abs(actual - ordinary_norm) > 1e-4 * std::max(1.0, ordinary_norm))
            throw std::runtime_error("masked injection norm differs from scalar control");
        return score(result, true);
    };
    experiment_json random = experiment_json::array();
    for (const auto & control : controls) random.push_back(masked(control));
    return {{"case_id", item.id}, {"target_position", item.prompt_tokens.size() - 2},
        {"baseline", score(ordinary, false)}, {"scalar", score(ordinary, true)},
        {"selected_mask", masked(selected)}, {"random_controls", random}};
}

}
