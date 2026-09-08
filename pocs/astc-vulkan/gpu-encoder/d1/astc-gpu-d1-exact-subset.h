#pragma once

// D1 semantic bridge for the audited exact 6x6 GPU subset. Both physical
// source sets reconstruct as scalar at runtime; their distinct IDs retain the
// baseline/refined choice for exact decoded activation ranking.

#include "astc-gpu-d1-candidates.h"

struct astc_gpu_d1_exact_subset_candidate_bank {
    astc_gpu_d1_candidate_bank semantic_bank;
    astc_gpu_encoder_request binary_request;
    astc_gpu_encoder_request refined_request;
    astc_gpu_encoder_request mean_refined_request;
    astc_gpu_encoder_request quantile_refined_request;
};

// Builds scalar candidate zero plus three bounded local-fitting alternatives
// for each 6x6 logical block. Every alternative uses the same standard ASTC
// physical mode; only its endpoint/weight initialization differs. The caller
// dispatches each request through its matching GPU shader, CPU-decodes the
// legal payloads, then feeds the combined results to the D1 ranker/selector.
bool astc_gpu_d1_build_refined_6x6_exact_subset_candidate_bank(
    const std::vector<astc_gpu_encoder_source_block> & scalar_blocks,
    uint32_t max_blocks_per_batch,
    astc_gpu_d1_exact_subset_candidate_bank & bank);
