#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct common_flydelta_encoder_config {
    uint64_t seed = 0x9e3779b97f4a7c15ULL;
    size_t input_dim = 0;
    size_t expansion_dim = 0;
    size_t fan_in = 0;
    size_t winners = 0;
};

struct common_flydelta_sparse_code {
    size_t expansion_dim = 0;
    std::vector<uint32_t> indices;
    std::vector<float> values;
};

bool common_flydelta_validate_encoder_config(
        const common_flydelta_encoder_config & config,
        std::string & error);

class common_flydelta_encoder {
public:
    explicit common_flydelta_encoder(common_flydelta_encoder_config config);

    bool encode(
            const std::vector<float> & input,
            common_flydelta_sparse_code & code,
            std::string & error) const;

    const common_flydelta_encoder_config & config() const { return config_; }

private:
    common_flydelta_encoder_config config_;
};

float common_flydelta_sparse_similarity(
        const common_flydelta_sparse_code & left,
        const common_flydelta_sparse_code & right);

class common_flydelta_recognition_memory {
public:
    explicit common_flydelta_recognition_memory(size_t max_prototypes = 128);

    bool remember(const common_flydelta_sparse_code & code, std::string & error);
    float familiarity(const common_flydelta_sparse_code & code) const;
    float novelty(const common_flydelta_sparse_code & code) const;
    size_t size() const { return prototypes_.size(); }

private:
    size_t max_prototypes_;
    std::vector<common_flydelta_sparse_code> prototypes_;
};

struct common_flydelta_memory_config {
    size_t expansion_dim = 0;
    size_t target_dim = 0;
    float max_abs_weight = 1.0f;
};

bool common_flydelta_validate_memory_config(
        const common_flydelta_memory_config & config,
        std::string & error);

class common_flydelta_delta_memory {
public:
    explicit common_flydelta_delta_memory(common_flydelta_memory_config config);

    bool predict(
            const common_flydelta_sparse_code & code,
            std::vector<float> & output,
            std::string & error) const;

    bool learn(
            const common_flydelta_sparse_code & code,
            const std::vector<float> & target,
            float confidence,
            float learning_rate,
            float decay,
            std::string & error);

    const common_flydelta_memory_config & config() const { return config_; }
    const std::vector<float> & weights() const { return weights_; }
    bool set_weights(const std::vector<float> & weights, std::string & error);

private:
    common_flydelta_memory_config config_;
    std::vector<float> weights_;
};

struct common_flydelta_compatibility {
    std::string base_model_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string architecture;
    std::string inference_layout_revision;
};

// Serialized steering direction owned by a complete FlyDelta v2 artifact.
// Runtime/basis-builder statistics are intentionally not part of this codec.
struct common_flydelta_artifact_direction {
    int32_t layer_index = -1;
    std::vector<float> values;
};

struct common_flydelta_artifact {
    int schema_version = 1;
    std::string id;
    uint64_t generation = 0;
    common_flydelta_encoder_config encoder;
    common_flydelta_memory_config memory;
    common_flydelta_compatibility compatibility;
    std::vector<float> weights;
    size_t model_n_embd = 0;
    size_t model_n_layers = 0;
    int32_t il_start = 1;
    int32_t il_end = 0;
    std::vector<common_flydelta_artifact_direction> steering_basis;
    std::string content_hash;
};

bool common_flydelta_artifact_validate(
        const common_flydelta_artifact & artifact,
        size_t max_weights,
        size_t max_serialized_bytes,
        std::string & error);

std::string common_flydelta_artifact_to_json(
        const common_flydelta_artifact & artifact);

bool common_flydelta_artifact_from_json(
        const std::string & text,
        size_t max_weights,
        size_t max_serialized_bytes,
        common_flydelta_artifact & artifact,
        std::string & error);

std::string common_flydelta_artifact_hash(
        const common_flydelta_artifact & artifact);

bool common_flydelta_artifact_matches(
        const common_flydelta_artifact & artifact,
        const common_flydelta_compatibility & expected,
        std::string & error);
