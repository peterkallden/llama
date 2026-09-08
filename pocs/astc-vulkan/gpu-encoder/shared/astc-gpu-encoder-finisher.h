#pragma once

// CPU exact finisher for GPU-proposed physical blocks.
//
// The GPU proposer decides only which source blocks/proposals survive. This
// module creates the legal standard ASTC payload using astcenc, validates its
// block information, and performs exact CPU decode. It is intentionally the
// only payload-producing implementation in GPU-encoder v1.

#include "astc-gpu-encoder.h"
#include "astc-gpu-encoder-subset.h"

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
    // Independent source blocks can be finished concurrently. Each worker
    // owns one astcenc context; output order remains the proposal order.
    uint32_t worker_count = 1;
};

// Offline encoder policy, not a runtime representation choice. The speed
// policy preserves the existing bounded proposer -> CPU finish behavior. The
// neural-quality policy adds an independently encoded astcenc reference for
// every logical block, so broader GPU/source candidates can never remove the
// reference floor from a selector bank.
enum class astc_gpu_encoder_candidate_profile : uint8_t {
    speed,
    neural_quality,
};

struct astc_gpu_encoder_neural_hybrid_finish_options {
    // Candidate zero/reference is deliberately encoded independently from the
    // exploratory bank. This is normally ASTCENC_PRE_THOROUGH.
    float reference_quality = ASTCENC_PRE_THOROUGH;
    // GPU proposals may be numerous, so their CPU exact finish can use a
    // lower preset while the reference remains mandatory.
    float exploration_quality = ASTCENC_PRE_MEDIUM;
    uint32_t worker_count = 1;
};

bool astc_gpu_encoder_finish_with_options(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    const astc_gpu_encoder_finish_options & options,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error);

// Quality-profile seam shared by D1 and D2. mandatory_reference_source_ids
// identifies one semantically neutral/reference physical source per logical
// block. Those blocks are encoded at reference_quality; retained exploratory
// candidates are encoded separately at exploration_quality. Output contains
// each source id once, with reference payloads taking precedence on overlap.
// It performs no neural ranking or selection.
bool astc_gpu_encoder_finish_neural_hybrid(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    const std::vector<uint32_t> & mandatory_reference_source_ids,
    const astc_gpu_encoder_neural_hybrid_finish_options & options,
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

// Exact decode-only bridge for payloads emitted by the audited GPU subset.
// It deliberately does not invoke astcenc compression: the supplied 16-byte
// payload remains the candidate that D1/D2 rank and later verify in Vulkan.
bool astc_gpu_exact_subset_finish_payloads(
    astc_vulkan_footprint footprint,
    const std::vector<astc_gpu_exact_subset_block> & payloads,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error);
