#pragma once

// Fixed-function sampled-ASTC decode gate for CPU-finished GPU proposals.
// Kept separate from the generic proposer session because it depends on the
// astcenc-backed finisher result type and uses ASTC image resources.

#include "astc-gpu-encoder-finisher.h"

#include <cstdint>
#include <string>
#include <vector>

bool astc_gpu_encoder_verify_d1_vulkan_decode_default(
    const std::string & validation_spirv_path,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    uint32_t blocks_x, double & mse, float & max_abs, std::string & error);

inline bool astc_gpu_encoder_verify_d1_4x4_vulkan_decode_default(
    const std::string & validation_spirv_path,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    uint32_t blocks_x, double & mse, float & max_abs, std::string & error) {
    return astc_gpu_encoder_verify_d1_vulkan_decode_default(
        validation_spirv_path, finished, blocks_x, mse, max_abs, error);
}
