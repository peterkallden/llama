#include "agent/adaptation/flydelta/flydelta.h"

#include "hash/hash.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

bool finite_positive(float value) {
    return std::isfinite(value) && value > 0.0f;
}

bool finite_unit(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

std::string hash_text(const std::string & text) {
    return "sha256:" + hash_sha256_hex(text.data(), text.size());
}

json artifact_json_without_hash(const common_flydelta_artifact & artifact) {
    return json{
        {"schema_version", artifact.schema_version},
        {"kind", "flydelta"},
        {"id", artifact.id},
        {"generation", artifact.generation},
        {"encoder", {
            {"seed", artifact.encoder.seed},
            {"input_dim", artifact.encoder.input_dim},
            {"expansion_dim", artifact.encoder.expansion_dim},
            {"fan_in", artifact.encoder.fan_in},
            {"winners", artifact.encoder.winners},
        }},
        {"memory", {
            {"expansion_dim", artifact.memory.expansion_dim},
            {"target_dim", artifact.memory.target_dim},
            {"max_abs_weight", artifact.memory.max_abs_weight},
        }},
        {"compatibility", {
            {"base_model_fingerprint", artifact.compatibility.base_model_fingerprint},
            {"tokenizer_fingerprint", artifact.compatibility.tokenizer_fingerprint},
            {"template_fingerprint", artifact.compatibility.template_fingerprint},
            {"architecture", artifact.compatibility.architecture},
            {"inference_layout_revision", artifact.compatibility.inference_layout_revision},
        }},
        {"weights", artifact.weights},
    };
}

bool same(const std::string & actual, const std::string & expected, const char * name, std::string & error) {
    if (actual != expected) {
        error = std::string("FlyDelta ") + name + " is incompatible";
        return false;
    }
    return true;
}

} // namespace

bool common_flydelta_validate_encoder_config(
        const common_flydelta_encoder_config & config,
        std::string & error) {
    error.clear();
    if (config.input_dim == 0 || config.expansion_dim == 0 || config.fan_in == 0 || config.winners == 0) {
        error = "FlyDelta encoder dimensions must be non-zero";
        return false;
    }
    if (config.fan_in > config.input_dim || config.winners > config.expansion_dim) {
        error = "FlyDelta encoder bounds are invalid";
        return false;
    }
    if (config.input_dim > (1U << 20) || config.expansion_dim > (1U << 22) || config.winners > config.expansion_dim / 2 + 1) {
        error = "FlyDelta encoder exceeds safety bounds";
        return false;
    }
    return true;
}

common_flydelta_encoder::common_flydelta_encoder(common_flydelta_encoder_config config)
    : config_(config) {}

bool common_flydelta_encoder::encode(
        const std::vector<float> & input,
        common_flydelta_sparse_code & code,
        std::string & error) const {
    error.clear();
    if (!common_flydelta_validate_encoder_config(config_, error) || input.size() != config_.input_dim) {
        if (error.empty()) error = "FlyDelta input dimension does not match encoder";
        return false;
    }
    float norm = 0.0f;
    for (const float value : input) {
        if (!std::isfinite(value)) {
            error = "FlyDelta input contains a non-finite value";
            return false;
        }
        norm += value * value;
    }
    norm = std::sqrt(norm);

    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(config_.expansion_dim);
    for (size_t output = 0; output < config_.expansion_dim; ++output) {
        float value = 0.0f;
        for (size_t edge = 0; edge < config_.fan_in; ++edge) {
            const uint64_t mixed = splitmix64(config_.seed ^ (output * 0x632be59bd9b4e019ULL) ^ edge);
            const size_t input_index = static_cast<size_t>(mixed % config_.input_dim);
            const float sign = (mixed & 1U) == 0 ? 1.0f : -1.0f;
            value += sign * input[input_index];
        }
        candidates.emplace_back(norm > std::numeric_limits<float>::epsilon() ? value / norm : 0.0f,
            static_cast<uint32_t>(output));
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto & left, const auto & right) {
        if (left.first != right.first) return left.first > right.first;
        return left.second < right.second;
    });

    code = {};
    code.expansion_dim = config_.expansion_dim;
    code.indices.reserve(config_.winners);
    code.values.reserve(config_.winners);
    for (size_t i = 0; i < config_.winners; ++i) {
        code.values.push_back(candidates[i].first);
        code.indices.push_back(candidates[i].second);
    }
    return true;
}

float common_flydelta_sparse_similarity(
        const common_flydelta_sparse_code & left,
        const common_flydelta_sparse_code & right) {
    if (left.expansion_dim == 0 || left.expansion_dim != right.expansion_dim ||
            left.indices.size() != left.values.size() || right.indices.size() != right.values.size()) {
        return 0.0f;
    }
    float dot = 0.0f;
    float left_norm = 0.0f;
    float right_norm = 0.0f;
    for (size_t i = 0; i < left.indices.size(); ++i) {
        left_norm += left.values[i] * left.values[i];
        for (size_t j = 0; j < right.indices.size(); ++j) {
            if (left.indices[i] == right.indices[j]) dot += left.values[i] * right.values[j];
        }
    }
    for (const float value : right.values) right_norm += value * value;
    const float denominator = std::sqrt(left_norm * right_norm);
    return denominator > std::numeric_limits<float>::epsilon() ? dot / denominator : 0.0f;
}

common_flydelta_recognition_memory::common_flydelta_recognition_memory(size_t max_prototypes)
    : max_prototypes_(max_prototypes) {}

bool common_flydelta_recognition_memory::remember(
        const common_flydelta_sparse_code & code,
        std::string & error) {
    error.clear();
    if (max_prototypes_ == 0 || code.expansion_dim == 0 || code.indices.size() != code.values.size()) {
        error = "FlyDelta recognition code is invalid";
        return false;
    }
    if (prototypes_.size() == max_prototypes_) prototypes_.erase(prototypes_.begin());
    prototypes_.push_back(code);
    return true;
}

float common_flydelta_recognition_memory::familiarity(
        const common_flydelta_sparse_code & code) const {
    float best = 0.0f;
    for (const auto & prototype : prototypes_) best = std::max(best, common_flydelta_sparse_similarity(code, prototype));
    return std::max(0.0f, std::min(1.0f, best));
}

float common_flydelta_recognition_memory::novelty(
        const common_flydelta_sparse_code & code) const {
    return 1.0f - familiarity(code);
}

bool common_flydelta_validate_memory_config(
        const common_flydelta_memory_config & config,
        std::string & error) {
    error.clear();
    if (config.expansion_dim == 0 || config.target_dim == 0 || !finite_positive(config.max_abs_weight)) {
        error = "FlyDelta memory configuration is invalid";
        return false;
    }
    if (config.expansion_dim > (1U << 22) || config.target_dim > (1U << 16)) {
        error = "FlyDelta memory exceeds safety bounds";
        return false;
    }
    return true;
}

common_flydelta_delta_memory::common_flydelta_delta_memory(common_flydelta_memory_config config)
    : config_(config), weights_(config.target_dim * config.expansion_dim, 0.0f) {}

bool common_flydelta_delta_memory::predict(
        const common_flydelta_sparse_code & code,
        std::vector<float> & output,
        std::string & error) const {
    error.clear();
    if (!common_flydelta_validate_memory_config(config_, error) || code.expansion_dim != config_.expansion_dim ||
            code.indices.size() != code.values.size()) {
        if (error.empty()) error = "FlyDelta sparse code does not match memory";
        return false;
    }
    output.assign(config_.target_dim, 0.0f);
    for (size_t i = 0; i < code.indices.size(); ++i) {
        if (code.indices[i] >= config_.expansion_dim || !std::isfinite(code.values[i])) {
            error = "FlyDelta sparse code contains an invalid winner";
            return false;
        }
        for (size_t target = 0; target < config_.target_dim; ++target) {
            output[target] += weights_[target * config_.expansion_dim + code.indices[i]] * code.values[i];
        }
    }
    for (float & value : output) {
        value = std::max(-config_.max_abs_weight, std::min(config_.max_abs_weight, value));
    }
    return true;
}

bool common_flydelta_delta_memory::learn(
        const common_flydelta_sparse_code & code,
        const std::vector<float> & target,
        float confidence,
        float learning_rate,
        float decay,
        std::string & error) {
    error.clear();
    if (!finite_unit(confidence) || !finite_positive(learning_rate) || !finite_unit(decay) || target.size() != config_.target_dim) {
        error = "FlyDelta learning parameters are invalid";
        return false;
    }
    std::vector<float> prediction;
    if (!predict(code, prediction, error)) return false;
    float code_norm = 0.0f;
    for (const float value : code.values) {
        if (!std::isfinite(value)) { error = "FlyDelta code contains a non-finite value"; return false; }
        code_norm += value * value;
    }
    const float denominator = std::max(std::numeric_limits<float>::epsilon(), code_norm);
    for (size_t i = 0; i < code.indices.size(); ++i) {
        if (code.indices[i] >= config_.expansion_dim) { error = "FlyDelta winner exceeds expansion dimension"; return false; }
        const size_t column = code.indices[i];
        for (size_t row = 0; row < config_.target_dim; ++row) {
            const float delta = (target[row] - prediction[row]) * code.values[i] / denominator;
            float & weight = weights_[row * config_.expansion_dim + column];
            weight = decay * weight + learning_rate * confidence * delta;
            weight = std::max(-config_.max_abs_weight, std::min(config_.max_abs_weight, weight));
        }
    }
    return true;
}

bool common_flydelta_delta_memory::set_weights(
        const std::vector<float> & weights,
        std::string & error) {
    error.clear();
    if (weights.size() != weights_.size()) { error = "FlyDelta weight dimensions do not match memory"; return false; }
    for (const float value : weights) {
        if (!std::isfinite(value) || std::fabs(value) > config_.max_abs_weight) {
            error = "FlyDelta weight exceeds bound";
            return false;
        }
    }
    weights_ = weights;
    return true;
}

bool common_flydelta_artifact_validate(
        const common_flydelta_artifact & artifact,
        size_t max_weights,
        size_t max_serialized_bytes,
        std::string & error) {
    error.clear();
    if (artifact.schema_version != 1 || artifact.id.empty()) { error = "FlyDelta artifact identity is invalid"; return false; }
    if (!common_flydelta_validate_encoder_config(artifact.encoder, error) ||
            !common_flydelta_validate_memory_config(artifact.memory, error)) return false;
    if (artifact.encoder.expansion_dim != artifact.memory.expansion_dim || artifact.weights.size() != artifact.memory.target_dim * artifact.memory.expansion_dim) {
        error = "FlyDelta artifact dimensions do not match";
        return false;
    }
    if (artifact.weights.size() > max_weights) { error = "FlyDelta artifact exceeds weight bound"; return false; }
    if (artifact.compatibility.base_model_fingerprint.empty() || artifact.compatibility.tokenizer_fingerprint.empty() ||
            artifact.compatibility.template_fingerprint.empty() || artifact.compatibility.architecture.empty() ||
            artifact.compatibility.inference_layout_revision.empty()) {
        error = "FlyDelta artifact compatibility identity is incomplete";
        return false;
    }
    for (const float value : artifact.weights) {
        if (!std::isfinite(value) || std::fabs(value) > artifact.memory.max_abs_weight) { error = "FlyDelta artifact contains an invalid weight"; return false; }
    }
    if (max_serialized_bytes != 0 && artifact_json_without_hash(artifact).dump().size() > max_serialized_bytes) {
        error = "FlyDelta artifact exceeds serialized byte bound";
        return false;
    }
    return true;
}

std::string common_flydelta_artifact_hash(const common_flydelta_artifact & artifact) {
    return hash_text(artifact_json_without_hash(artifact).dump());
}

std::string common_flydelta_artifact_to_json(const common_flydelta_artifact & artifact) {
    auto value = artifact_json_without_hash(artifact);
    value["content_hash"] = artifact.content_hash.empty() ? common_flydelta_artifact_hash(artifact) : artifact.content_hash;
    return value.dump();
}

bool common_flydelta_artifact_from_json(
        const std::string & text,
        size_t max_weights,
        size_t max_serialized_bytes,
        common_flydelta_artifact & artifact,
        std::string & error) {
    error.clear();
    if (max_serialized_bytes != 0 && text.size() > max_serialized_bytes) { error = "FlyDelta artifact input exceeds byte bound"; return false; }
    const auto value = json::parse(text, nullptr, false);
    if (value.is_discarded() || !value.is_object() || value.value("kind", "") != "flydelta") { error = "invalid FlyDelta artifact JSON"; return false; }
    try {
        artifact = {};
        artifact.schema_version = value.value("schema_version", 0);
        artifact.id = value.value("id", "");
        artifact.generation = value.value("generation", 0ULL);
        const auto & encoder = value.at("encoder");
        artifact.encoder.seed = encoder.value("seed", 0ULL);
        artifact.encoder.input_dim = encoder.value("input_dim", 0U);
        artifact.encoder.expansion_dim = encoder.value("expansion_dim", 0U);
        artifact.encoder.fan_in = encoder.value("fan_in", 0U);
        artifact.encoder.winners = encoder.value("winners", 0U);
        const auto & memory = value.at("memory");
        artifact.memory.expansion_dim = memory.value("expansion_dim", 0U);
        artifact.memory.target_dim = memory.value("target_dim", 0U);
        artifact.memory.max_abs_weight = memory.value("max_abs_weight", 0.0f);
        const auto & compatibility = value.at("compatibility");
        artifact.compatibility.base_model_fingerprint = compatibility.value("base_model_fingerprint", "");
        artifact.compatibility.tokenizer_fingerprint = compatibility.value("tokenizer_fingerprint", "");
        artifact.compatibility.template_fingerprint = compatibility.value("template_fingerprint", "");
        artifact.compatibility.architecture = compatibility.value("architecture", "");
        artifact.compatibility.inference_layout_revision = compatibility.value("inference_layout_revision", "");
        artifact.weights = value.at("weights").get<std::vector<float>>();
        artifact.content_hash = value.value("content_hash", "");
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta artifact fields: ") + exception.what();
        return false;
    }
    if (!common_flydelta_artifact_validate(artifact, max_weights, max_serialized_bytes, error)) return false;
    if (artifact.content_hash != common_flydelta_artifact_hash(artifact)) { error = "FlyDelta artifact hash mismatch"; return false; }
    return true;
}

bool common_flydelta_artifact_matches(
        const common_flydelta_artifact & artifact,
        const common_flydelta_compatibility & expected,
        std::string & error) {
    error.clear();
    return same(artifact.compatibility.base_model_fingerprint, expected.base_model_fingerprint, "base model", error) &&
        same(artifact.compatibility.tokenizer_fingerprint, expected.tokenizer_fingerprint, "tokenizer", error) &&
        same(artifact.compatibility.template_fingerprint, expected.template_fingerprint, "template", error) &&
        same(artifact.compatibility.architecture, expected.architecture, "architecture", error) &&
        same(artifact.compatibility.inference_layout_revision, expected.inference_layout_revision, "inference layout", error);
}
