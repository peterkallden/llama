#include "agent/adaptation/flydelta/flydelta-two-pass.h"

bool common_flydelta_run_two_pass(
        common_agent_inference & inference,
        const common_agent_generation_request & request,
        const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & capture,
        const common_flydelta_activation_builder & build_activation,
        common_flydelta_two_pass_result & result,
        std::string & error) {
    result = {};
    error.clear();
    if (!capture || !capture->enabled) {
        error = "FlyDelta two-pass requires an enabled capture request";
        return false;
    }
    if (!build_activation) {
        error = "FlyDelta two-pass requires an activation builder";
        return false;
    }

    common_agent_generation_request capture_request = request;
    capture_request.flydelta_activation.reset();
    capture_request.flydelta_capture = capture;
    if (!inference.generate(capture_request, result.capture_generation) ||
            !common_agent_generation_succeeded(result.capture_generation)) {
        error = result.capture_generation.error_message.empty()
            ? "FlyDelta capture pass failed"
            : result.capture_generation.error_message;
        return false;
    }
    if (!result.capture_generation.flydelta_capture ||
            !result.capture_generation.flydelta_capture->captured) {
        error = result.capture_generation.flydelta_capture &&
                !result.capture_generation.flydelta_capture->failure_reason.empty()
            ? result.capture_generation.flydelta_capture->failure_reason
            : "FlyDelta capture pass returned no capture";
        return false;
    }
    if (!common_flydelta_hidden_state_capture_validate(
            *result.capture_generation.flydelta_capture, capture->max_bytes, error)) {
        return false;
    }
    if (!build_activation(
            *result.capture_generation.flydelta_capture, result.activation, error)) {
        return false;
    }

    auto activation = std::make_shared<const common_flydelta_activation_result>(result.activation);
    common_agent_generation_request final_request = request;
    final_request.flydelta_capture.reset();
    final_request.flydelta_activation = std::move(activation);
    if (!inference.generate(final_request, result.final_generation) ||
            !common_agent_generation_succeeded(result.final_generation)) {
        error = result.final_generation.error_message.empty()
            ? "FlyDelta final pass failed"
            : result.final_generation.error_message;
        return false;
    }
    return true;
}
