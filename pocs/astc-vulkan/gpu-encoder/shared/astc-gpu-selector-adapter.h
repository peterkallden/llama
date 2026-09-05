#pragma once

// Representation-neutral helpers for the offline selector seam shared by
// D1 and D2.  The frontends still own semantic reconstruction; this module
// only owns trace validation, zero-baseline construction, and candidate-group
// normalization before the generic activation-space selector.

#include "astc-vulkan-paired-selector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct astc_gpu_selector_delta_request {
    uint32_t tensor_width = 0;
    uint32_t tensor_height = 0;
    uint32_t source_blocks_x = 0;
    // Row-major [sample][tensor_width]. Validation may be empty.
    std::vector<double> calibration_activations;
    std::vector<double> validation_activations;
};

// Validates dimensions and finite trace values shared by all semantic
// frontends. `label` is only used to make diagnostics identify the caller.
bool astc_gpu_validate_selector_delta_request(
    const astc_gpu_selector_delta_request & request,
    const char * label,
    std::string & error);

// Creates the all-zero delta required for candidate zero in a block group.
std::vector<double> astc_gpu_zero_selector_delta(
    const astc_gpu_selector_delta_request & request,
    bool validation);

// Verifies delta shapes, moves the candidate matching `neutral_payload` to
// index zero, and enforces the neutral all-zero contract.  D1/D2 call this
// after filling a group using their own semantic decoder.
bool astc_gpu_finalize_selector_group(
    std::vector<astc_vulkan_paired_candidate_delta> & group,
    const std::array<uint8_t, 16> & neutral_payload,
    const astc_gpu_selector_delta_request & request,
    const char * label,
    std::string & error);
