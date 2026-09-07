#pragma once

#include "astc-vulkan-d2-prescreen.h"

#include <string>

// Executes the D2 proxy screen on a generic Vulkan compute device. The CPU
// implementation remains the numerical reference and this function rejects a
// GPU result that does not match it within the fixed tolerance.
bool astc_vulkan_score_d2_prescreen_gpu_default(
    const std::string & spirv_path,
    const std::vector<float> & weights,
    uint32_t rows,
    uint32_t columns,
    const std::vector<float> & column_energy,
    const std::vector<astc_vulkan_d2_prescreen_candidate> & candidates,
    std::vector<astc_vulkan_d2_prescreen_score> & scores,
    std::string & error);
