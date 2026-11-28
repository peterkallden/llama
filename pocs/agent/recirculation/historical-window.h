#pragma once

#include "experiment.h"
#include "window.h"

namespace recirculation {

struct historical_window_bounds {
    size_t start = 0;
    size_t target_offset = 2;
    size_t size = 6;
};

inline historical_window_bounds historical_window_for_update(const experiment_case & item) {
    if (item.target_position < 2 ||
            size_t(item.target_position) + 4 > item.prompt_tokens.size())
        throw std::invalid_argument("update position cannot fit a six-token replay window");
    return {size_t(item.target_position - 2), 2, 6};
}

struct historical_continuation_score {
    double total = 0;
    std::vector<double> token_logprobs;
};

inline historical_continuation_score historical_window_continuation_score(llama_model * model,
        const experiment_case & item, const std::string & continuation,
        const recirculation_config & config, const window_result & replay,
        size_t window_end, bool refined) {
    const auto tokens = experiment_tokenize(model, item.prompt + continuation);
    if (tokens.size() <= item.prompt_tokens.size() || tokens.size() > 640 ||
            !std::equal(item.prompt_tokens.begin(), item.prompt_tokens.end(), tokens.begin()) ||
            window_end > item.prompt_tokens.size())
        throw std::runtime_error("historical window continuation tokenization changed");
    llama_backend host(model, config, uint32_t(tokens.size() + 1));
    const auto & state = refined ? replay.refined_sequence_state : replay.baseline_sequence_state;
    if (state.empty() || llama_state_seq_set_data(host.context(), state.data(), state.size(), 0) != state.size() ||
            !host.ready(int32_t(window_end)))
        throw std::runtime_error("historical window state transfer failed");
    // Match the resident A/readout/B contract: the first prediction after
    // replay uses the already exposed Pass A logits. Later positions read the
    // chosen KV history.
    std::vector<float> logits = replay.baseline_logits;
    std::vector<double> logprobs;
    for (size_t position = window_end; position < tokens.size(); ++position) {
        if (position >= item.prompt_tokens.size()) {
            const double value = recirculation_token_log_probability(logits, tokens[position]);
            if (!std::isfinite(value)) throw std::runtime_error("invalid historical window log probability");
            logprobs.push_back(value);
        }
        if (position + 1 < tokens.size()) {
            capture current;
            if (!host.first(tokens[position], int32_t(position), false, current))
                throw std::runtime_error("historical window continuation decode failed");
            logits = std::move(current.logits);
            host.clear_hooks();
        }
    }
    double total = 0;
    if (!recirculation_sum_log_probabilities(logprobs, total))
        throw std::runtime_error("invalid historical window continuation total");
    return {total, std::move(logprobs)};
}

inline experiment_json historical_window_case(llama_model * model, const experiment_case & item,
        const recirculation_config & config, unsigned rounds, std::ostream & diagnostic_trace) {
    const auto bounds = historical_window_for_update(item);
    const size_t window_end = bounds.start + bounds.size;
    std::vector<llama_token> prefix(item.prompt_tokens.begin(), item.prompt_tokens.begin() + bounds.start);
    std::vector<llama_token> window(item.prompt_tokens.begin() + bounds.start,
        item.prompt_tokens.begin() + window_end);
    const auto replay_started = std::chrono::steady_clock::now();
    const auto replay = run_window(model, config, prefix, window, bounds.target_offset, rounds);
    const auto replay_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - replay_started).count();
    const auto old_baseline = score_experiment_case(model, item, config, experiment_arm::baseline, diagnostic_trace);
    const auto old_targeted = score_experiment_case(model, item, config, experiment_arm::targeted, diagnostic_trace);
    const auto score_paths = [&](const window_result & result, bool refined) {
        return std::pair<historical_continuation_score, historical_continuation_score>{
            historical_window_continuation_score(model, item, item.expected, config, result, window_end, refined),
            historical_window_continuation_score(model, item, item.incorrect, config, result, window_end, refined)};
    };
    const auto baseline = score_paths(replay, false);
    const auto refined = score_paths(replay, true);
    const double window_baseline = baseline.first.total - baseline.second.total;
    const double window_refined = refined.first.total - refined.second.total;
    const double policy_margin = replay.committed ? window_refined : window_baseline;
    const double final_patch_norm = replay.patch_displacement_norms.back();
    const double initial_patch_norm = replay.patch_displacement_norms.front();
    const double matched_alpha = std::abs(final_patch_norm - initial_patch_norm) <=
            1e-4 * std::max(1.0, initial_patch_norm) ? double(config.alpha) :
            final_patch_norm / replay.original_destination_norm;
    if (!std::isfinite(matched_alpha) || matched_alpha > std::numeric_limits<float>::max())
        throw std::runtime_error("invalid matched dose");
    auto matched_config = config;
    matched_config.alpha = float(matched_alpha);
    const auto matched_started = std::chrono::steady_clock::now();
    const auto matched_replay = run_window(model, matched_config, prefix, window, bounds.target_offset, 1);
    const auto matched_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - matched_started).count();
    if (std::abs(matched_replay.patch_displacement_norms.back() - final_patch_norm) >
            1e-4 * std::max(1.0, final_patch_norm))
        throw std::runtime_error("matched dose patch norm changed");
    const auto matched = score_paths(matched_replay, true);
    const double matched_margin = matched.first.total - matched.second.total;
    const auto token_deltas = [](const std::vector<double> & before, const std::vector<double> & after) {
        if (before.size() != after.size()) throw std::runtime_error("continuation length changed");
        std::vector<double> delta;
        for (size_t i = 0; i < before.size(); ++i) delta.push_back(after[i] - before[i]);
        return delta;
    };
    return {{"case_id", item.id}, {"update_token_position", item.target_position},
        {"window_start", bounds.start}, {"window_end_exclusive", window_end},
        {"target_offset", bounds.target_offset}, {"rounds", rounds},
        {"completed_rounds", replay.completed_rounds}, {"committed", replay.committed},
        {"baseline_entropy", replay.baseline_entropy}, {"refined_entropy", replay.refined_entropy},
        {"old_baseline_margin", old_baseline.margin}, {"old_targeted_margin", old_targeted.margin},
        {"old_targeted_gain", old_targeted.margin - old_baseline.margin},
        {"window_baseline_margin", window_baseline}, {"window_refined_margin", window_refined},
        {"window_policy_margin", policy_margin}, {"window_forced_gain", window_refined - window_baseline},
        {"window_policy_gain", policy_margin - window_baseline},
        {"matched_dose_alpha", matched_alpha}, {"matched_dose_margin", matched_margin},
        {"matched_dose_gain", matched_margin - window_baseline},
        {"matched_patch_norm_relative_error", std::abs(matched_replay.patch_displacement_norms.back() - final_patch_norm) /
            std::max(1e-20, final_patch_norm)},
        {"window_replay_us", replay_us}, {"matched_replay_us", matched_us},
        {"matched_patch_displacement_norm", matched_replay.patch_displacement_norms.back()},
        {"patch_displacement_norms", replay.patch_displacement_norms},
        {"patch_direction_cosines", replay.patch_direction_cosines},
        {"patch_norm_ratios", replay.patch_norm_ratios},
        {"expected_token_logprobs_baseline", baseline.first.token_logprobs},
        {"expected_token_logprobs_refined", refined.first.token_logprobs},
        {"expected_token_logprobs_matched", matched.first.token_logprobs},
        {"expected_token_logprob_deltas", token_deltas(baseline.first.token_logprobs, refined.first.token_logprobs)},
        {"incorrect_token_logprobs_baseline", baseline.second.token_logprobs},
        {"incorrect_token_logprobs_refined", refined.second.token_logprobs},
        {"incorrect_token_logprobs_matched", matched.second.token_logprobs},
        {"incorrect_token_logprob_deltas", token_deltas(baseline.second.token_logprobs, refined.second.token_logprobs)},
        {"baseline_parity_delta", window_baseline - old_baseline.margin},
        {"preferred_expected_old_baseline", old_baseline.margin > 0},
        {"preferred_expected_window_policy", policy_margin > 0}};
}
}
