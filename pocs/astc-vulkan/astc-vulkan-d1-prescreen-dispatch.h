#pragma once

#include "astc-vulkan-d1-prescreen.h"

#include <string>

// One-shot offline Vulkan backend for D1 pre-screening. This opens a compute
// device, evaluates the supplied SPIR-V kernel, and returns CPU-compatible
// score records. It intentionally needs no sampled-ASTC support.
bool astc_vulkan_score_d1_prescreen_gpu_default(
    const std::string & spirv_path,
    const std::vector<float> & weights,
    uint32_t rows,
    uint32_t columns,
    const std::vector<float> & column_energy,
    const std::vector<astc_vulkan_d1_prescreen_candidate> & candidates,
    std::vector<astc_vulkan_d1_prescreen_score> & scores,
    std::string & error);
