#pragma once

// CPU exact finisher for GPU-proposed physical blocks.
//
// The GPU proposer decides only which source blocks/proposals survive. This
// module creates the legal standard ASTC payload using astcenc, validates its
// block information, and performs exact CPU decode. It is intentionally the
// only payload-producing implementation in GPU-encoder v1.

#include "astc-gpu-encoder.h"

#include <astcenc.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct astc_gpu_encoder_finished_block {
    uint32_t source_block_id = 0;
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k4x4;
    std::array<uint8_t, 16> payload{};
    // Exact astcenc CPU decode in physical RGBA raster order. This is retained
    // only for verification gates; it is not a cache/runtime representation.
    std::vector<float> decoded_rgba;
};

enum class astc_gpu_encoder_finish_mode : uint8_t {
    // Exact reference astcenc encode over the requested source blocks.
    reference,
    // Candidate-bank-guided exact encode. v1 uses the proposal bank to
    // restrict which source blocks are finished; low-level astcenc search
    // remains unchanged and authoritative.
    guided,
};

struct astc_gpu_encoder_finish_options {
    float quality = ASTCENC_PRE_MEDIUM;
    astc_gpu_encoder_finish_mode mode = astc_gpu_encoder_finish_mode::reference;
};

bool astc_gpu_encoder_finish_with_options(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    const astc_gpu_encoder_finish_options & options,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error);

// Representation-neutral exact finisher for any currently declared 2D
// footprint. Source construction and semantic reconstruction stay in D1/D2
// frontends; this function only creates and verifies a legal physical payload.
bool astc_gpu_encoder_finish(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    float quality,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error);

// D1 source-compatible name retained for existing callers. New D2 code must
// use the representation-neutral entry point above.
bool astc_gpu_encoder_finish_d1_scalar(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    float quality,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error);

// Convenience v1 guard for the GPU-proposer path. It makes it impossible for
// a caller to imply that the current 4x4 GPU kernel supports another shape.
bool astc_gpu_encoder_finish_d1_scalar_4x4(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    float quality,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error);
