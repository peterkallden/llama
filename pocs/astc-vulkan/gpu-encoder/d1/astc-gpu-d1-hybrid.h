#pragma once

// D1 frontend hand-off for the shared offline hybrid seam. D1 supplies the
// scalar/gauge decoded-weight interpretation; physical proposal, legal finish,
// payload verification, and global selector remain shared contracts.

#include "astc-gpu-d1-candidates.h"
#include "astc-gpu-encoder-finisher.h"
#include "astc-gpu-selector-adapter.h"

#include <cstdint>
#include <string>
#include <vector>

using astc_gpu_d1_selector_delta_request = astc_gpu_selector_delta_request;

// Converts exact decoded D1 candidates to the common global selector format.
// Candidate zero in every physical block is the scalar decoded baseline.
bool astc_gpu_d1_make_selector_candidates(
    const astc_gpu_d1_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    const astc_gpu_d1_selector_delta_request & request,
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> & candidates,
    std::string & error);
