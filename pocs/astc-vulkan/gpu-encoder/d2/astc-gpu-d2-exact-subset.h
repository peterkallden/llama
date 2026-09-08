#pragma once

// D2-owned exact-subset candidate planning.
//
// The shared exact encoder only sees physical RGBA texels and one selected
// ASTC mode per dispatch. D2-LA needs two semantic candidates for the same
// source block: a one-plane L+A control and an Alpha dual-plane alternative.
// This module creates those two independent physical dispatch requests. It
// does not rank them, decode them, or decide a cache artifact.

#include "astc-gpu-encoder.h"
#include "astc-gpu-d2-candidates.h"

#include <vector>

struct astc_gpu_d2_exact_subset_bank {
    // Shared 8x5 L+A mode; its binary grid is fit from both semantic lanes.
    astc_gpu_encoder_request one_plane;
    // Same legal mode, but grid fitting is biased to a single semantic lane.
    astc_gpu_encoder_request one_plane_luminance_weights;
    astc_gpu_encoder_request one_plane_alpha_weights;
    astc_gpu_encoder_request one_plane_refined;
    astc_gpu_encoder_request one_plane_mean_refined;
    astc_gpu_encoder_request one_plane_quantile_refined;
    astc_gpu_encoder_request alpha_dual_plane;
};

// The semantic companion to the seven physical requests. Candidate zero is
// balanced one-plane L+A and therefore an exact decoded fallback; all
// remaining records retain the same D2 pair-map/layout metadata.
struct astc_gpu_d2_exact_subset_candidate_bank {
    astc_gpu_d2_candidate_bank semantic_bank;
    astc_gpu_d2_exact_subset_bank physical_bank;
};

// Builds the narrow D2-LA 8x5 exact candidate bank. Each source block must
// already use the D2-LA physical convention R=G=B=L and A=the second semantic
// lane, normalized to finite UNORM values. Source IDs are preserved so later
// decode/ranking code can join results with D2 metadata.
bool astc_gpu_d2_build_luminance_alpha_exact_subset_bank(
    const std::vector<astc_gpu_encoder_source_block> & source_blocks,
    uint32_t max_blocks_per_batch,
    astc_gpu_d2_exact_subset_bank & bank);

// Builds both layers for the D2 selector path. The provided source has the
// D2-LA physical convention, while layouts/pairings describe logical output
// reconstruction. No candidate is selected here.
bool astc_gpu_d2_build_luminance_alpha_exact_subset_candidate_bank(
    const std::vector<astc_gpu_encoder_source_block> & source_blocks,
    const std::vector<astc_vulkan_paired_layout> & layouts,
    const std::vector<astc_vulkan_d2_pairing> & pairings,
    uint32_t max_blocks_per_batch,
    astc_gpu_d2_exact_subset_candidate_bank & bank);
