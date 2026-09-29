#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Internal, request-scoped causal intervention.  This contract intentionally
// has no FlyDelta, alpha, repair or lifecycle vocabulary: callers provide the
// fully prepared vector and the server applies it at one graph site only.
enum class common_agent_residual_patch_operation : uint8_t {
    replace,
    add,
};

enum class common_agent_residual_patch_site : uint8_t {
    layer_input_residual,
};

struct common_agent_residual_patch_request {
    common_agent_residual_patch_operation operation = common_agent_residual_patch_operation::add;
    common_agent_residual_patch_site site = common_agent_residual_patch_site::layer_input_residual;
    uint32_t layer = 0;
    // Absolute prompt position and server sequence identity.  V0 permits one
    // patch point only; the server rejects a request when it cannot apply it
    // exactly rather than broadcasting or falling back.
    int32_t absolute_position = -1;
    int32_t sequence_id = -1;
    std::vector<float> values;
    bool observe_applied_vector = false;
    std::string identity;
};

struct common_agent_residual_patch_observation {
    bool attempted = false;
    bool applied = false;
    uint32_t layer = 0;
    int32_t absolute_position = -1;
    int32_t sequence_id = -1;
    common_agent_residual_patch_site site = common_agent_residual_patch_site::layer_input_residual;
    std::string failure_reason;
};
