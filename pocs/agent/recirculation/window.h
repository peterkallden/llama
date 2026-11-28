#pragma once
#include "llama-backend.h"
#include <numeric>

namespace recirculation {

struct window_result {
    std::vector<float> baseline_logits;
    std::vector<float> refined_logits;
    std::vector<uint8_t> baseline_state;
    std::vector<uint8_t> refined_state;
    std::vector<uint8_t> refined_sequence_state;
    std::vector<float> patch_changes;
    double baseline_entropy = 0;
    double refined_entropy = 0;
    int32_t target_position = -1;
    unsigned completed_rounds = 0;
    bool committed = false;
    double trigger_signal = 0;
};

enum class window_target_mode { fixed, uncertainty, curvature };

inline double window_curvature(const std::vector<float> & before,
        const std::vector<float> & middle, const std::vector<float> & after) {
    if (before.size() != middle.size() || middle.size() != after.size() || before.empty()) return 0;
    double a2 = 0, b2 = 0, dot = 0;
    for (size_t i = 0; i < before.size(); ++i) {
        if (!std::isfinite(before[i]) || !std::isfinite(middle[i]) || !std::isfinite(after[i])) return 0;
        const double a = double(middle[i]) - before[i], b = double(after[i]) - middle[i];
        a2 += a * a; b2 += b * b; dot += a * b;
    }
    if (a2 <= 1e-20 || b2 <= 1e-20) return 0;
    return std::sqrt(std::max(0.0, 2.0 - 2.0 * std::clamp(dot / std::sqrt(a2 * b2), -1.0, 1.0)));
}

inline int32_t window_select_target(const std::vector<double> & entropies,
        const std::vector<std::vector<float>> & destinations, window_target_mode mode,
        double threshold, double & selected_signal) {
    selected_signal = 0;
    if (entropies.size() != destinations.size() || entropies.size() < 4 ||
            !std::isfinite(threshold) || threshold < 0) return -1;
    for (size_t i = 2; i + 1 < entropies.size(); ++i) {
        const double signal = mode == window_target_mode::uncertainty ? entropies[i] - entropies[i - 1] :
            window_curvature(destinations[i - 2], destinations[i - 1], destinations[i]);
        if (std::isfinite(signal) && signal > 0 && signal >= threshold) {
            selected_signal = signal;
            return int32_t(i);
        }
    }
    return -1;
}

inline std::vector<uint8_t> window_state(llama_context * ctx) {
    const size_t size = llama_state_get_size(ctx);
    if (!size || size > 512ULL * 1024 * 1024) throw std::runtime_error("window state exceeds budget");
    std::vector<uint8_t> state(size);
    const size_t copied = llama_state_get_data(ctx, state.data(), state.size());
    if (!copied || copied > state.size()) throw std::runtime_error("window state capture failed");
    state.resize(copied);
    return state;
}

inline std::vector<uint8_t> window_sequence_state(llama_context * ctx) {
    const size_t size = llama_state_seq_get_size(ctx, 0);
    if (!size || size > 512ULL * 1024 * 1024) throw std::runtime_error("window sequence state exceeds budget");
    std::vector<uint8_t> state(size);
    if (llama_state_seq_get_data(ctx, state.data(), state.size(), 0) != size)
        throw std::runtime_error("window sequence state capture failed");
    return state;
}

inline void window_restore(llama_context * ctx, const std::vector<uint8_t> & state) {
    if (state.empty() || llama_state_set_data(ctx, state.data(), state.size()) != state.size())
        throw std::runtime_error("window state restore failed");
}

inline double window_entropy(const std::vector<float> & logits) {
    if (logits.empty()) throw std::runtime_error("missing window logits");
    double top = -std::numeric_limits<double>::infinity();
    for (float value : logits) {
        if (!std::isfinite(value)) throw std::runtime_error("invalid window logits");
        top = std::max(top, double(value));
    }
    double sum = 0, weighted = 0;
    for (float value : logits) {
        const double weight = std::exp(double(value) - top);
        sum += weight;
        weighted += weight * (double(value) - top);
    }
    return (std::log(sum) - weighted / sum) / std::log(double(logits.size()));
}

inline std::vector<float> window_next_patch(const std::vector<float> & previous,
        const std::vector<float> & original_destination, const std::vector<float> & source,
        float alpha, float damping) {
    if (previous.size() != original_destination.size() || source.size() != previous.size() ||
            !std::isfinite(damping) || damping <= 0 || damping > 1) throw std::runtime_error("invalid refinement state");
    std::vector<float> candidate;
    float source_norm = 0, destination_norm = 0;
    if (!recirculation_mix(original_destination.data(), source.data(), source.size(), alpha, 1.0f,
            candidate, source_norm, destination_norm) || destination_norm <= 0) throw std::runtime_error("invalid refinement source");
    double squared = 0;
    for (size_t i = 0; i < candidate.size(); ++i) {
        const double value = (1.0 - damping) * previous[i] + damping * candidate[i];
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
            throw std::runtime_error("invalid refinement patch");
        candidate[i] = float(value);
        squared += value * value;
    }
    if (std::sqrt(squared) > 2.0 * destination_norm) throw std::runtime_error("refinement patch norm exceeded");
    return candidate;
}

// The caller owns fixed tokens. The last window token must follow the patch.
inline window_result run_window(llama_model * model, const recirculation_config & config,
        const std::vector<llama_token> & prefix, const std::vector<llama_token> & window,
        size_t target_offset, unsigned rounds, float damping = 0.5f,
        window_target_mode mode = window_target_mode::fixed, double threshold = 0,
        uint32_t context_capacity = 0, int threads = 1,
        uint32_t batch_capacity = 1, uint32_t ubatch_capacity = 1, bool batch_prefix = false) {
    if (!model || window.size() < 4 || window.size() > 8 ||
            (mode == window_target_mode::fixed && (target_offset < 2 || target_offset + 1 >= window.size())) ||
            rounds < 1 || rounds > 3 ||
            !recirculation_config_valid(config, llama_model_n_layer(model), llama_model_n_embd(model)) ||
            prefix.size() + window.size() > 640) throw std::invalid_argument("invalid window request");
    const uint32_t minimum_capacity = uint32_t(prefix.size() + window.size() + 1);
    const uint32_t capacity = context_capacity ? context_capacity : minimum_capacity;
    if (capacity < minimum_capacity || threads < 1) throw std::invalid_argument("invalid trial context capacity");
    llama_backend baseline(model, config, capacity, threads, batch_capacity, ubatch_capacity);
    capture values;
    if (batch_prefix) {
        if (!baseline.prefill(prefix)) throw std::runtime_error("window batch prefix decode failed");
    } else {
        for (size_t i = 0; i < prefix.size(); ++i)
            if (!baseline.first(prefix[i], int32_t(i), false, values)) throw std::runtime_error("window prefix decode failed");
    }
    const auto prefix_state = window_state(baseline.context());
    window_result result;
    std::vector<double> entropies;
    std::vector<std::vector<float>> destinations;
    capture original;
    for (size_t i = 0; i < window.size(); ++i) {
        const int32_t pos = int32_t(prefix.size() + i);
        capture current;
        if (!baseline.first(window[i], pos, mode != window_target_mode::fixed || i == target_offset, current))
            throw std::runtime_error("window baseline decode failed");
        entropies.push_back(window_entropy(current.logits));
        destinations.push_back(current.destination);
        if (mode == window_target_mode::fixed && i == target_offset) original = current;
        result.baseline_logits = std::move(current.logits);
        baseline.clear_hooks();
    }
    result.baseline_state = window_state(baseline.context());
    result.baseline_entropy = window_entropy(result.baseline_logits);
    if (mode != window_target_mode::fixed) {
        const int32_t selected = window_select_target(entropies, destinations, mode, threshold, result.trigger_signal);
        if (selected < 0) return result;
        target_offset = size_t(selected);
        // Recompute the selected row from the immutable prefix so the capture
        // retains its source vector after the baseline loop has completed.
        llama_backend probe(model, config, capacity, threads, batch_capacity, ubatch_capacity);
        window_restore(probe.context(), prefix_state);
        for (size_t i = 0; i <= target_offset; ++i) {
            capture current;
            if (!probe.first(window[i], int32_t(prefix.size() + i), i == target_offset, current))
                throw std::runtime_error("window target capture failed");
            if (i == target_offset) original = std::move(current);
            probe.clear_hooks();
        }
    }
    result.target_position = int32_t(prefix.size() + target_offset);
    float sn = 0, dn = 0;
    std::vector<float> patch;
    if (original.destination.empty() || original.source.size() != original.destination.size() ||
            !recirculation_mix(original.destination.data(), original.source.data(), original.source.size(),
                config.alpha, 1.0f, patch, sn, dn) || dn <= 0) throw std::runtime_error("window capture invalid");
    for (unsigned round = 0; round < rounds; ++round) {
        llama_backend trial(model, config, capacity, threads, batch_capacity, ubatch_capacity);
        window_restore(trial.context(), prefix_state);
        capture patched;
        for (size_t i = 0; i < window.size(); ++i) {
            const int32_t pos = int32_t(prefix.size() + i);
            capture current;
            if (i == target_offset) {
                bool matched = false;
                if (!trial.patched(window[i], pos, patch, matched, &current) || !matched)
                    throw std::runtime_error("window patch failed");
                patched = current;
            } else if (!trial.first(window[i], pos, false, current)) throw std::runtime_error("window trial decode failed");
            result.refined_logits = std::move(current.logits);
            trial.clear_hooks();
        }
        result.refined_state = window_state(trial.context());
        result.refined_sequence_state = window_sequence_state(trial.context());
        result.refined_entropy = window_entropy(result.refined_logits);
        result.completed_rounds = round + 1;
        if (round + 1 < rounds) {
            auto next = window_next_patch(patch, original.destination, patched.source, config.alpha, damping);
            double change = 0;
            for (size_t i = 0; i < patch.size(); ++i) change += double(next[i] - patch[i]) * (next[i] - patch[i]);
            result.patch_changes.push_back(float(std::sqrt(change)));
            patch = std::move(next);
        }
    }
    result.committed = result.refined_entropy <= result.baseline_entropy - 0.001;
    return result;
}
}
