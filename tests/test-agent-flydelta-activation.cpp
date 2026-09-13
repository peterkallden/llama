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
    CHECK(result.overlay.data[2] == 0.25f && result.overlay.data[3] == 0.5f);

    request.gate_request.explicit_opt_in = false;
    CHECK(common_flydelta_prepare_activation(gate_config, request, 1024, result, error));
    CHECK(!result.gate.apply && !result.overlay.enabled);

    request.gate_request.explicit_opt_in = true;
    request.artifact_id.clear();
    CHECK(!common_flydelta_prepare_activation(gate_config, request, 1024, result, error));
    CHECK(!error.empty());
    return 0;
}
