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

inline double historical_window_continuation_score(llama_model * model,
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
    return total;
}

inline experiment_json historical_window_case(llama_model * model, const experiment_case & item,
        const recirculation_config & config, unsigned rounds, std::ostream & diagnostic_trace) {
    const auto bounds = historical_window_for_update(item);
    const size_t window_end = bounds.start + bounds.size;
    std::vector<llama_token> prefix(item.prompt_tokens.begin(), item.prompt_tokens.begin() + bounds.start);
    std::vector<llama_token> window(item.prompt_tokens.begin() + bounds.start,
        item.prompt_tokens.begin() + window_end);
    const auto replay = run_window(model, config, prefix, window, bounds.target_offset, rounds);
    const auto old_baseline = score_experiment_case(model, item, config, experiment_arm::baseline, diagnostic_trace);
    const auto old_targeted = score_experiment_case(model, item, config, experiment_arm::targeted, diagnostic_trace);
    const auto score_margin = [&](bool refined) {
        const double expected = historical_window_continuation_score(model, item, item.expected,
            config, replay, window_end, refined);
        const double incorrect = historical_window_continuation_score(model, item, item.incorrect,
            config, replay, window_end, refined);
        return expected - incorrect;
    };
    const double window_baseline = score_margin(false);
    const double window_refined = score_margin(true);
    const double policy_margin = replay.committed ? window_refined : window_baseline;
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
        {"baseline_parity_delta", window_baseline - old_baseline.margin},
        {"preferred_expected_old_baseline", old_baseline.margin > 0},
        {"preferred_expected_window_policy", policy_margin > 0}};
}
}
