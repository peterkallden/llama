#pragma once
#include "../recirculation-contract.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace recirculation {
struct observation {
    int32_t position = -1;
    bool first_completed = false;
    bool rewound = false;
    bool patch_matched = false;
    bool second_completed = false;
    bool committed = false;
    float source_norm = 0, destination_norm = 0, delta_norm = 0;
    float source_destination_cosine = 0;
    float novelty = 0;
    int64_t first_us = 0, second_us = 0;
    std::string failure;
};
struct capture {
    std::vector<float> logits, destination, source;
};
enum class patch_policy { every_token, selected_position, host_window, activation_novelty };
struct target_policy {
    patch_policy mode = patch_policy::selected_position;
    int32_t window_start = -1;
    int32_t window_end = -1;
    float novelty_threshold = 0.15f;
};
struct backend {
    virtual ~backend() = default;
    virtual bool ready(int32_t position) = 0;
    virtual bool first(int32_t token, int32_t position, bool capture_inputs, capture & result) = 0;
    virtual bool rewind(int32_t position) = 0;
    virtual bool second(int32_t token, int32_t position, const std::vector<float> & patch, bool & matched) = 0;
    virtual void clear_hooks() noexcept = 0;
    virtual void discard() noexcept = 0;
};
// One instance owns one immutable configuration and a contiguous sequence.
// Sampling is a caller operation over the returned first-pass logits.
class runner {
public:
    runner(backend & host, recirculation_config config, size_t layers, size_t width,
            bool enabled = true, patch_policy policy = patch_policy::every_token,
            int32_t selected_position = -1, std::vector<float> source_override = {}, target_policy target = {})
        : host(host), config(std::move(config)), width(width), enabled(enabled), policy(policy),
          selected_position(selected_position), source_override(std::move(source_override)), target(target) {
        if (!recirculation_config_valid(this->config, layers, width)) throw std::invalid_argument("invalid recirculation config");
        if (policy == patch_policy::selected_position && selected_position < 0) throw std::invalid_argument("invalid selected position");
        if ((policy == patch_policy::host_window || policy == patch_policy::activation_novelty) &&
                (target.mode != policy || target.window_start < 0 || target.window_end < target.window_start ||
                 !std::isfinite(target.novelty_threshold) || target.novelty_threshold < 0 || target.novelty_threshold > 2))
            throw std::invalid_argument("invalid target policy");
        if (!this->source_override.empty() && this->source_override.size() != width) throw std::invalid_argument("invalid control vector width");
    }
    bool step(int32_t token, int32_t position, std::vector<float> & logits, observation & obs) {
        obs = {}; obs.position = position; logits.clear();
        if (failed) { obs.failure = "context_discarded"; return false; }
        struct cleanup { backend & host; ~cleanup() { host.clear_hooks(); } } guard{host};
        auto fail = [&](const char * reason) {
            obs.failure = reason; failed = true; host.clear_hooks(); host.discard(); return false;
        };
        try {
            if (position != next || position == INT32_MAX || !host.ready(position)) return fail("unsupported_memory_or_position");
            const bool eligible = enabled && (policy == patch_policy::every_token ||
                (policy == patch_policy::selected_position && position == selected_position));
            const bool probe = enabled && !intervened && target.window_start <= position && position <= target.window_end &&
                (policy == patch_policy::activation_novelty ||
                 (policy == patch_policy::host_window && position == target.window_end));
            const bool capture_inputs = eligible || probe;
            capture values;
            auto start = std::chrono::steady_clock::now();
            if (!host.first(token, position, capture_inputs, values)) return fail("first_pass_failed");
            obs.first_us = elapsed(start); obs.first_completed = true;
            if (values.logits.empty()) return fail("missing_logits");
            for (float value : values.logits) if (!std::isfinite(value)) return fail("invalid_logits");
            bool patch = eligible;
            if (probe) {
                if (values.destination.size() != width) return fail("missing_capture");
                if (policy == patch_policy::host_window) patch = position == target.window_end;
                else {
                    double dot = 0, current = 0, previous = 0;
                    for (size_t i = 0; i < width; ++i) {
                        if (!std::isfinite(values.destination[i])) return fail("invalid_capture");
                        current += double(values.destination[i]) * values.destination[i];
                        if (!previous_destination.empty()) {
                            dot += double(values.destination[i]) * previous_destination[i];
                            previous += double(previous_destination[i]) * previous_destination[i];
                        }
                    }
                    if (current > 0 && previous > 0) {
                        obs.novelty = float(1.0 - std::clamp(dot / std::sqrt(current * previous), -1.0, 1.0));
                        patch = obs.novelty >= target.novelty_threshold;
                    }
                    previous_destination = values.destination;
                }
            }
            if (patch) {
                if (values.destination.size() != width || values.source.size() != width) return fail("missing_capture");
                const std::vector<float> * source = &values.source;
                std::vector<float> matched_control;
                if (!source_override.empty()) {
                    double norm = 0;
                    for (float value : values.source) {
                        if (!std::isfinite(value)) return fail("invalid_capture");
                        norm += double(value) * value;
                    }
                    if (!recirculation_rescale_norm(source_override, float(std::sqrt(norm)), matched_control)) return fail("invalid_control_vector");
                    source = &matched_control;
                }
                std::vector<float> patch;
                if (!recirculation_mix(values.destination.data(), source->data(), width,
                        config.alpha, config.beta, patch, obs.source_norm, obs.destination_norm)) return fail("invalid_mixture");
                double dot = 0;
                for (size_t i = 0; i < width; ++i) dot += double((*source)[i]) * values.destination[i];
                if (obs.source_norm > 0 && obs.destination_norm > 0) {
                    obs.source_destination_cosine = float(std::clamp(dot / (double(obs.source_norm) * obs.destination_norm), -1.0, 1.0));
                }
                double delta = 0;
                for (size_t i = 0; i < width; ++i) { const double d = double(patch[i]) - values.destination[i]; delta += d*d; }
                if (std::sqrt(delta) > std::numeric_limits<float>::max()) return fail("invalid_delta_norm");
                obs.delta_norm = float(std::sqrt(delta));
                if (!host.rewind(position)) return fail("rewind_failed");
                obs.rewound = true;
                start = std::chrono::steady_clock::now();
                if (!host.second(token, position, patch, obs.patch_matched)) return fail("second_pass_failed");
                obs.second_us = elapsed(start); obs.second_completed = true;
                if (!obs.patch_matched) return fail("patch_not_matched");
                intervened = true;
            }
            logits = std::move(values.logits); obs.committed = true; ++next;
            return true;
        } catch (const std::exception &) { return fail("backend_exception"); }
    }
private:
    static int64_t elapsed(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    }
    backend & host;
    const recirculation_config config;
    const size_t width;
    const bool enabled;
    const patch_policy policy;
    const int32_t selected_position;
    const std::vector<float> source_override;
    int32_t next = 0;
    bool failed = false;
    target_policy target;
    bool intervened = false;
    std::vector<float> previous_destination;
};
}
