#include "agent/adaptation/flydelta/flydelta.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

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
    CHECK(encoder.encode({1.0f, 0.0f, 0.5f, -0.25f}, first, error));
    CHECK(encoder.encode({1.0f, 0.0f, 0.5f, -0.25f}, second, error));
    CHECK(first.indices == second.indices);
    CHECK(first.values == second.values);
    CHECK(first.indices.size() == 2);
    CHECK(!encoder.encode({1.0f}, second, error));

    common_flydelta_recognition_memory recognition(1);
    CHECK(recognition.remember(first, error));
    CHECK(recognition.familiarity(first) > 0.99f);
    CHECK(recognition.novelty(first) < 0.01f);

    common_flydelta_delta_memory memory({8, 2, 0.5f});
    std::vector<float> prediction;
    CHECK(memory.predict(first, prediction, error));
    CHECK(prediction.size() == 2);
    CHECK(memory.learn(first, {1.0f, -1.0f}, 1.0f, 1.0f, 1.0f, error));
    CHECK(memory.predict(first, prediction, error));
    CHECK(std::fabs(prediction[0]) > 0.0f);
    CHECK(std::fabs(prediction[0]) <= 0.5f);
    CHECK(!memory.learn(first, {1.0f}, 1.0f, 1.0f, 1.0f, error));

    auto value = artifact();
    CHECK(common_flydelta_artifact_validate(value, 32, 65536, error));
    const auto text = common_flydelta_artifact_to_json(value);
    CHECK(value.content_hash.rfind("sha256:", 0) == 0);
    CHECK(common_flydelta_artifact_hash(value).size() == 71);
    common_flydelta_artifact parsed;
    CHECK(common_flydelta_artifact_from_json(text, 32, 65536, parsed, error));
    CHECK(parsed.id == value.id);
    CHECK(parsed.content_hash == common_flydelta_artifact_hash(parsed));
    auto tampered = text;
    tampered[tampered.find("sha256:model") + 7] = 'X';
    CHECK(!common_flydelta_artifact_from_json(tampered, 32, 65536, parsed, error));
    auto expected = value.compatibility;
    CHECK(common_flydelta_artifact_matches(value, expected, error));
    expected.architecture = "qwen";
    CHECK(!common_flydelta_artifact_matches(value, expected, error));

    auto v2 = value;
    v2.schema_version = 2;
    v2.model_n_embd = 4;
    v2.model_n_layers = 3;
    v2.il_start = 1;
    v2.il_end = 2;
    v2.steering_basis = {
        {1, {1.0f, 0.0f, 0.0f, 0.0f}},
        {2, {0.0f, 1.0f, 0.0f, 0.0f}},
    };
    v2.content_hash = common_flydelta_artifact_hash(v2);
    CHECK(common_flydelta_artifact_validate(v2, 32, 65536, error));
    common_flydelta_artifact parsed_v2;
    CHECK(common_flydelta_artifact_from_json(
        common_flydelta_artifact_to_json(v2), 32, 65536, parsed_v2, error));
    CHECK(parsed_v2.schema_version == 2 && parsed_v2.steering_basis.size() == 2);
    v2.steering_basis.front().values.pop_back();
    CHECK(!common_flydelta_artifact_validate(v2, 32, 65536, error));
    return 0;
}
