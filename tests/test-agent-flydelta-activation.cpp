#include "agent/adaptation/flydelta/flydelta-activation.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    common_flydelta_gate_config gate_config;
    gate_config.enabled = true;
    gate_config.max_scale = 0.5f;
    common_flydelta_activation_request request;
    request.candidate_id = "flydelta://candidate/1";
    request.artifact_id = "flydelta://artifact/1";
    request.model_profile_fingerprint = "model-profile-1";
    request.capture_layout_revision = "layout-1";
    request.model_n_embd = 2;
    request.model_n_layers = 4;
    request.il_start = 1;
    request.il_end = 3;
    common_flydelta_basis_direction direction;
    direction.layer_index = 2;
    direction.values = {1.0f, 2.0f};
    request.directions = {direction};
    request.coefficients = {0.5f};
    request.gate_request.explicit_opt_in = true;
    request.gate_request.candidate_status = common_flydelta_candidate_status::approved;
    request.gate_request.basis_available = true;
    request.gate_request.familiarity = 0.9f;
    request.gate_request.novelty = 0.1f;
    request.gate_request.requested_scale = 0.5f;

    std::string error;
    common_flydelta_activation_result result;
    CHECK(common_flydelta_prepare_activation(gate_config, request, 1024, result, error));
    CHECK(result.gate.apply && result.overlay.enabled);
    CHECK(result.sparse_overlay.enabled);
    CHECK(result.sparse_overlay.layer_indices.size() == 1);
    CHECK(result.sparse_overlay.layer_indices[0] == 2);
    CHECK(result.sparse_overlay.data[0] == 0.25f && result.sparse_overlay.data[1] == 0.5f);
    CHECK(result.overlay.data[2] == 0.25f && result.overlay.data[3] == 0.5f);
    CHECK(common_flydelta_activation_result_validate(result, 2, 4, 1024, error));
    auto inconsistent = result;
    inconsistent.sparse_overlay.data.front() += 0.25f;
    CHECK(!common_flydelta_activation_result_validate(inconsistent, 2, 4, 1024, error));

    request.gate_request.explicit_opt_in = false;
    CHECK(common_flydelta_prepare_activation(gate_config, request, 1024, result, error));
    CHECK(!result.gate.apply && !result.overlay.enabled);
    CHECK(!result.sparse_overlay.enabled);

    request.gate_request.explicit_opt_in = true;
    request.artifact_id.clear();
    CHECK(!common_flydelta_prepare_activation(gate_config, request, 1024, result, error));
    CHECK(!error.empty());

    common_flydelta_artifact artifact;
    artifact.schema_version = 2;
    artifact.id = "flydelta://artifact/v2";
    artifact.generation = 7;
    artifact.encoder = {42, 2, 2, 1, 1};
    artifact.memory = {2, 1, 1.0f};
    artifact.compatibility.base_model_fingerprint = "sha256:model";
    artifact.compatibility.tokenizer_fingerprint = "sha256:tokenizer";
    artifact.compatibility.template_fingerprint = "sha256:template";
    artifact.compatibility.architecture = "qwen2";
    artifact.compatibility.inference_layout_revision = "layout-1";
    artifact.weights = {1.0f, 0.0f};
    artifact.model_n_embd = 2;
    artifact.model_n_layers = 4;
    artifact.il_start = 1;
    artifact.il_end = 3;
    artifact.steering_basis = {{2, {1.0f, 2.0f}}};
    artifact.content_hash = common_flydelta_artifact_hash(artifact);

    common_flydelta_sparse_code code;
    code.expansion_dim = 2;
    code.indices = {0};
    code.values = {1.0f};
    common_flydelta_gate_request artifact_gate = request.gate_request;
    request.artifact_id = "flydelta://artifact/v2";
    request.gate_request = artifact_gate;
    CHECK(common_flydelta_prepare_activation_from_artifact(
            artifact,
            artifact.compatibility,
            "model-profile-1",
            code,
            "flydelta://candidate/v2",
            gate_config,
            artifact_gate,
            32,
            65536,
            1024,
            result,
            error));
    CHECK(result.gate.apply && result.overlay.enabled);
    CHECK(result.sparse_overlay.enabled);
    CHECK(result.sparse_overlay.artifact_id == artifact.id);
    CHECK(result.sparse_overlay.layer_indices.size() == 1);
    CHECK(result.sparse_overlay.data[0] == 0.5f && result.sparse_overlay.data[1] == 1.0f);
    CHECK(result.overlay.artifact_id == artifact.id);
    CHECK(result.overlay.data.size() == 6);
    CHECK(result.overlay.data[2] == 0.5f && result.overlay.data[3] == 1.0f);
    CHECK(common_flydelta_activation_result_validate(result, 2, 4, 1024, error));

    artifact_gate.explicit_opt_in = false;
    CHECK(common_flydelta_prepare_activation_from_artifact(
            artifact,
            artifact.compatibility,
            "model-profile-1",
            code,
            "flydelta://candidate/v2",
            gate_config,
            artifact_gate,
            32,
            65536,
            1024,
            result,
            error));
    CHECK(!result.gate.apply && !result.overlay.enabled);

    artifact_gate.explicit_opt_in = true;
    artifact.schema_version = 1;
    CHECK(!common_flydelta_prepare_activation_from_artifact(
            artifact,
            artifact.compatibility,
            "model-profile-1",
            code,
            "flydelta://candidate/v2",
            gate_config,
            artifact_gate,
            32,
            65536,
            1024,
            result,
            error));
    CHECK(!error.empty());
    return 0;
}
