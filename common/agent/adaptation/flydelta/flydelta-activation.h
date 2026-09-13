#pragma once

#include "agent/adaptation/flydelta/flydelta-overlay.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Host-resolved input for one activation decision. Artifact and profile
// metadata are supplied by the host; this contract never resolves files.
struct common_flydelta_activation_request {
    std::string candidate_id;
    std::string artifact_id;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    size_t model_n_embd = 0;
    size_t model_n_layers = 0;
    int32_t il_start = 1;
    int32_t il_end = 0;
    std::vector<common_flydelta_basis_direction> directions;
    std::vector<float> coefficients;
    common_flydelta_gate_request gate_request;
};

struct common_flydelta_activation_result {
    common_flydelta_gate_decision gate;
    common_flydelta_static_overlay overlay;
};

// Applies the host-owned order of operations: decide the no-op gate first,
// then validate activation metadata and compose a static overlay. A refused
// gate is a successful empty result and does not require artifact metadata.
bool common_flydelta_prepare_activation(
        const common_flydelta_gate_config & gate_config,
        const common_flydelta_activation_request & request,
        size_t max_bytes,
        common_flydelta_activation_result & result,
        std::string & error);
