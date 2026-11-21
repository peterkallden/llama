#pragma once

#include "agent/agent-inference.h"
#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"

#include <functional>
#include <memory>
#include <string>

struct common_flydelta_two_pass_result {
    common_agent_generation_result capture_generation;
    common_flydelta_activation_result activation;
    common_agent_generation_result final_generation;
};

using common_flydelta_activation_builder = std::function<bool(
        const common_flydelta_hidden_state_capture & capture,
        common_flydelta_activation_result & activation,
        std::string & error)>;

// Runs a host-controlled V1 experiment over two fresh backend generations.
// The first request may capture prompt activations. The builder is the only
// place that may turn the capture into an activation; the second request gets
// a separate immutable activation snapshot and never captures into that same
// context. No learning state is persisted by this helper.
bool common_flydelta_run_two_pass(
        common_agent_inference & inference,
        const common_agent_generation_request & request,
        const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture,
        const common_flydelta_activation_builder & build_activation,
        common_flydelta_two_pass_result & result,
        std::string & error);
