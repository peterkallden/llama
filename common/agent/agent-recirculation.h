#pragma once

#include <cstdint>
#include <memory>
#include <string>

// Request-scoped experimental window replay. This is never a FlyDelta
// activation or evidence of a beneficial outcome.
enum class common_agent_recirculation_target : uint8_t {
    fixed,
    uncertainty,
    curvature,
};

struct common_agent_recirculation_request {
    uint32_t source_layer = 0;
    uint32_t destination_layer = 0;
    float alpha = 0.1f;
    uint32_t rounds = 1;
    uint32_t window_tokens = 6;
    common_agent_recirculation_target target = common_agent_recirculation_target::fixed;
    float threshold = 0;
};

struct common_agent_recirculation_observation {
    bool attempted = false;
    bool committed = false;
    int32_t target_position = -1;
    uint32_t completed_rounds = 0;
    double baseline_entropy = 0;
    double refined_entropy = 0;
    std::string failure_reason;
    bool recalled_profile = false;
    std::string memory_error;
};
