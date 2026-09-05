#include "astc-gpu-selector-adapter.h"

#include <algorithm>
#include <cmath>

namespace {

bool valid_trace(const std::vector<double> & values, uint32_t width) {
    return values.empty() || (width != 0 && values.size() % width == 0 &&
        std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        }));
}

size_t delta_size(const astc_gpu_selector_delta_request & request, bool validation) {
    const auto & trace = validation ? request.validation_activations : request.calibration_activations;
    return size_t(trace.size() / request.tensor_width) * request.tensor_height;
}

bool zero_delta(const std::vector<double> & values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
        return value == 0.0;
    });
}

} // namespace

bool astc_gpu_validate_selector_delta_request(
    const astc_gpu_selector_delta_request & request,
    const char * label,
    std::string & error) {
    if (request.tensor_width == 0 || request.tensor_height == 0 ||
        request.source_blocks_x == 0 ||
        !valid_trace(request.calibration_activations, request.tensor_width) ||
        !valid_trace(request.validation_activations, request.tensor_width)) {
        error = std::string("invalid ") + (label ? label : "") +
            " selector delta request";
        return false;
    }
    return true;
}

std::vector<double> astc_gpu_zero_selector_delta(
    const astc_gpu_selector_delta_request & request,
    bool validation) {
    return std::vector<double>(delta_size(request, validation), 0.0);
}

bool astc_gpu_finalize_selector_group(
    std::vector<astc_vulkan_paired_candidate_delta> & group,
    const std::array<uint8_t, 16> & neutral_payload,
    const astc_gpu_selector_delta_request & request,
    const char * label,
    std::string & error) {
    const size_t calibration_size = delta_size(request, false);
    const size_t validation_size = delta_size(request, true);
    if (calibration_size == 0 || group.empty()) {
        error = std::string(label ? label : "selector") +
            (calibration_size == 0 ? " calibration trace is empty" :
                                     " selector group is empty");
        return false;
    }
    for (const auto & candidate : group) {
        if (candidate.calibration_delta.size() != calibration_size ||
            (!request.validation_activations.empty() &&
             candidate.validation_delta.size() != validation_size) ||
            (request.validation_activations.empty() &&
             !candidate.validation_delta.empty())) {
            error = std::string("malformed ") + (label ? label : "") +
                " selector candidate delta";
            return false;
        }
    }
    const auto neutral = std::find_if(group.begin(), group.end(),
        [&](const auto & candidate) { return candidate.payload == neutral_payload; });
    if (neutral == group.end()) {
        error = std::string(label ? label : "selector") +
            " neutral payload not found";
        return false;
    }
    if (!zero_delta(neutral->calibration_delta) ||
        (!request.validation_activations.empty() && !zero_delta(neutral->validation_delta))) {
        error = std::string(label ? label : "selector") +
            " neutral candidate must have zero deltas";
        return false;
    }
    if (neutral != group.begin()) std::iter_swap(group.begin(), neutral);
    return true;
}
