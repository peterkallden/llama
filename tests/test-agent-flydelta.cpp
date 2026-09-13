#include "agent/adaptation/flydelta/flydelta.h"

#include <cassert>
#include <cmath>

static common_flydelta_encoder_config encoder_config() {
    common_flydelta_encoder_config config;
    config.seed = 42;
    config.input_dim = 4;
    config.expansion_dim = 8;
    config.fan_in = 2;
    config.winners = 2;
    return config;
}

static common_flydelta_artifact artifact() {
    common_flydelta_artifact value;
    value.id = "flydelta:test:v1";
    value.encoder = encoder_config();
    value.memory.expansion_dim = 8;
    value.memory.target_dim = 2;
    value.memory.max_abs_weight = 0.5f;
    value.weights.assign(16, 0.0f);
    value.compatibility.base_model_fingerprint = "sha256:model";
    value.compatibility.tokenizer_fingerprint = "sha256:tokenizer";
    value.compatibility.template_fingerprint = "sha256:template";
    value.compatibility.architecture = "llama";
    value.compatibility.inference_layout_revision = "l_out:v1";
    value.content_hash = common_flydelta_artifact_hash(value);
    return value;
}

int main() {
    std::string error;
    common_flydelta_encoder encoder(encoder_config());
    common_flydelta_sparse_code first;
    common_flydelta_sparse_code second;
    assert(encoder.encode({1.0f, 0.0f, 0.5f, -0.25f}, first, error));
    assert(encoder.encode({1.0f, 0.0f, 0.5f, -0.25f}, second, error));
    assert(first.indices == second.indices);
    assert(first.values == second.values);
    assert(first.indices.size() == 2);
    assert(!encoder.encode({1.0f}, second, error));

    common_flydelta_recognition_memory recognition(1);
    assert(recognition.remember(first, error));
    assert(recognition.familiarity(first) > 0.99f);
    assert(recognition.novelty(first) < 0.01f);

    common_flydelta_delta_memory memory({8, 2, 0.5f});
    std::vector<float> prediction;
    assert(memory.predict(first, prediction, error));
    assert(prediction.size() == 2);
    assert(memory.learn(first, {1.0f, -1.0f}, 1.0f, 1.0f, 1.0f, error));
    assert(memory.predict(first, prediction, error));
    assert(std::fabs(prediction[0]) > 0.0f);
    assert(std::fabs(prediction[0]) <= 0.5f);
    assert(!memory.learn(first, {1.0f}, 1.0f, 1.0f, 1.0f, error));

    auto value = artifact();
    assert(common_flydelta_artifact_validate(value, 32, 65536, error));
    const auto text = common_flydelta_artifact_to_json(value);
    common_flydelta_artifact parsed;
    assert(common_flydelta_artifact_from_json(text, 32, 65536, parsed, error));
    assert(parsed.id == value.id);
    assert(parsed.content_hash == common_flydelta_artifact_hash(parsed));
    auto tampered = text;
    tampered[tampered.find("sha256:model") + 7] = 'X';
    assert(!common_flydelta_artifact_from_json(tampered, 32, 65536, parsed, error));
    auto expected = value.compatibility;
    assert(common_flydelta_artifact_matches(value, expected, error));
    expected.architecture = "qwen";
    assert(!common_flydelta_artifact_matches(value, expected, error));
    return 0;
}
