#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include <utility>

inline bool recirculation_supported_architecture(const std::string & arch) {
    return arch == "qwen2" || arch == "qwen3" || arch == "phi3";
}

struct recirculation_config {
    std::string model_identity;
    size_t source_layer_input = 0;
    size_t destination_layer_input = 0;
    float alpha = 0.1f;
    float beta = 1.0f;
    unsigned iterations = 1;
    unsigned schema_version = 1;
};

inline bool recirculation_config_valid(const recirculation_config & c, size_t layers, size_t width) {
    return c.schema_version == 1 && !c.model_identity.empty() && width > 0 && layers > 1 &&
        c.source_layer_input < layers && c.destination_layer_input < layers &&
        c.source_layer_input > c.destination_layer_input &&
        std::isfinite(c.alpha) && std::isfinite(c.beta) &&
        c.alpha >= 0.0f && c.beta >= 0.0f && c.iterations == 1;
}

inline bool recirculation_mix(const float * destination, const float * source, size_t width,
        float alpha, float beta, std::vector<float> & output, float & source_norm, float & destination_norm) {
    if (!destination || !source || width == 0 || !std::isfinite(alpha) || !std::isfinite(beta) || alpha < 0 || beta < 0) return false;
    double ds = 0.0, ss = 0.0;
    for (size_t i = 0; i < width; ++i) {
        if (!std::isfinite(destination[i]) || !std::isfinite(source[i])) return false;
        ds += double(destination[i]) * destination[i];
        ss += double(source[i]) * source[i];
    }
    const double dn = std::sqrt(ds), sn = std::sqrt(ss);
    const double limit = std::numeric_limits<float>::max();
    if (!(sn > 0.0) || !std::isfinite(sn) || !std::isfinite(dn) || sn > limit || dn > limit) return false;
    std::vector<float> candidate(width);
    for (size_t i = 0; i < width; ++i) {
        const double value = double(beta) * destination[i] + double(alpha) * (dn / sn) * source[i];
        if (!std::isfinite(value) || std::abs(value) > limit) return false;
        candidate[i] = float(value);
    }
    output = std::move(candidate);
    destination_norm = float(dn);
    source_norm = float(sn);
    return true;
}

inline bool recirculation_rescale_norm(const std::vector<float> & input, float target_norm,
        std::vector<float> & output) {
    if (input.empty() || !std::isfinite(target_norm) || target_norm <= 0.0f) return false;
    double squared = 0.0;
    for (float value : input) {
        if (!std::isfinite(value)) return false;
        squared += double(value) * value;
    }
    const double norm = std::sqrt(squared);
    if (!std::isfinite(norm) || norm <= 0.0) return false;
    std::vector<float> candidate(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const double value = double(input[i]) * target_norm / norm;
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) return false;
        candidate[i] = float(value);
    }
    output = std::move(candidate);
    return true;
}

inline bool recirculation_token_for_span(const std::vector<size_t> & token_byte_lengths,
        size_t byte_start, size_t byte_length, size_t & token_index) {
    if (byte_length == 0 || byte_start > std::numeric_limits<size_t>::max() - byte_length) return false;
    const size_t byte_end = byte_start + byte_length;
    size_t offset = 0;
    for (size_t i = 0; i < token_byte_lengths.size(); ++i) {
        const size_t next = offset + token_byte_lengths[i];
        if (next < offset) return false;
        if (byte_end > offset && byte_end <= next) {
            token_index = i;
            return byte_start < next;
        }
        offset = next;
    }
    return false;
}

inline double recirculation_token_log_probability(const std::vector<float> & logits, int32_t token) {
    if (logits.empty() || token < 0 || size_t(token) >= logits.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double top = -std::numeric_limits<double>::infinity();
    for (float value : logits) {
        if (!std::isfinite(value)) return std::numeric_limits<double>::quiet_NaN();
        top = std::max(top, double(value));
    }
    double total = 0.0;
    for (float value : logits) total += std::exp(double(value) - top);
    return double(logits[size_t(token)]) - top - std::log(total);
}

inline double recirculation_answer_margin(double expected_logprob, double incorrect_logprob) {
    if (!std::isfinite(expected_logprob) || !std::isfinite(incorrect_logprob)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return expected_logprob - incorrect_logprob;
}

inline bool recirculation_sum_log_probabilities(const std::vector<double> & values, double & total) {
    if (values.empty()) return false;
    double sum = 0.0;
    for (double value : values) {
        if (!std::isfinite(value)) return false;
        sum += value;
        if (!std::isfinite(sum)) return false;
    }
    total = sum;
    return true;
}
