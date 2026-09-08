#pragma once

#include "astc-gpu-encoder-subset.h"

#include <string>
#include <vector>

// Runs the first exact GPU subset. This is an offline path and emits one
// standard 16-byte ASTC payload per source block in request order.
bool astc_gpu_exact_subset_encode_gpu_default(
    const std::string & spirv_path,
    const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_exact_subset_block> & blocks,
    std::string & error);

