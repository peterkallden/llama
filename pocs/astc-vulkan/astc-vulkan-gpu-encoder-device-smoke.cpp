#include "astc-gpu-d1-source.h"
#include "astc-gpu-d1-candidates.h"
#include "astc-gpu-d1-hybrid.h"
#include "astc-gpu-d1-neural-rank.h"
#include "astc-gpu-d2-source.h"
#include "astc-gpu-d2-candidates.h"
#include "astc-gpu-d2-hybrid.h"
#include "astc-gpu-d2-neural-rank.h"
#include "astc-gpu-encoder-dispatch.h"
#include "astc-gpu-encoder-finisher.h"
#include "astc-gpu-encoder-vulkan-verify.h"

#include <astcenc.h>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 9) return 2;
    const std::vector<float> weights{
        -1.0f, -0.5f, 0.0f, 0.5f, 1.0f,
         0.2f,  0.3f, 0.4f, 0.5f, 0.6f,
         0.7f,  0.8f, 0.9f, 1.0f, 0.1f};
    std::vector<astc_gpu_encoder_source_block> blocks;
    if (!astc_gpu_d1_build_scalar_source_blocks(astc_vulkan_footprint::k4x4,
                                                  weights, 3, 5, blocks)) return 1;
    // Non-contiguous IDs and a one-block batch force the reusable session to
    // exercise its source-ID indirection and multi-dispatch path.
    blocks[0].source_block_id = 41;
    blocks[1].source_block_id = 99;
    astc_gpu_encoder_request request;
    request.footprint = astc_vulkan_footprint::k4x4;
    request.max_blocks_per_batch = 1;
    request.blocks = blocks;
    std::vector<astc_gpu_encoder_proposal> expected;
    std::vector<astc_gpu_encoder_proposal> actual;
    std::string error;
    if (!astc_gpu_encoder_propose_cpu_reference(request, expected) ||
        !astc_gpu_encoder_propose_gpu_default(argv[1], request, actual, error)) {
        std::fprintf(stderr, "GPU D1 proposer smoke failed: %s\n", error.c_str());
        return 77;
    }
    if (actual.size() != expected.size()) return 1;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].source_block_id != expected[index].source_block_id ||
            actual[index].weight_grid_x != 4 || actual[index].weight_grid_y != 4 ||
            std::fabs(actual[index].approximate_error - expected[index].approximate_error) > 1e-6f) return 1;
        for (uint32_t channel = 0; channel < 4; ++channel) if (
            std::fabs(actual[index].endpoint_low[channel] - expected[index].endpoint_low[channel]) > 1e-6f ||
            std::fabs(actual[index].endpoint_high[channel] - expected[index].endpoint_high[channel]) > 1e-6f) return 1;
    }
    std::vector<astc_gpu_encoder_finished_block> finished;
    if (!astc_gpu_encoder_finish_d1_scalar_4x4(request, actual, ASTCENC_PRE_MEDIUM,
                                                 finished, error) ||
        finished.size() != actual.size()) {
        std::fprintf(stderr, "GPU D1 CPU finisher smoke failed: %s\n", error.c_str());
        return 1;
    }
    for (const auto & block : finished) for (const float value : block.decoded_rgba) {
        if (!std::isfinite(value)) return 1;
    }
    std::vector<astc_gpu_encoder_finished_block> guided_finished;
    const astc_gpu_encoder_finish_options guided_options{
        static_cast<float>(ASTCENC_PRE_MEDIUM), astc_gpu_encoder_finish_mode::guided, 2};
    if (!astc_gpu_encoder_finish_with_options(request, actual, guided_options,
                                               guided_finished, error) ||
        guided_finished.size() != finished.size()) return 1;
    for (size_t index = 0; index < finished.size(); ++index) {
        if (guided_finished[index].source_block_id != finished[index].source_block_id ||
            guided_finished[index].payload != finished[index].payload) return 1;
    }
    double decode_mse = 0.0;
    float decode_max_abs = 0.0f;
    if (!astc_gpu_encoder_verify_d1_4x4_vulkan_decode_default(
            argv[2], finished, 2, decode_mse, decode_max_abs, error)) {
        std::fprintf(stderr, "GPU D1 fixed-function decode gate failed: %s\n", error.c_str());
        return 1;
    }
    // GPU-proposed D1 candidate bank: scalar stays mandatory and gauge-L+A
    // is an independently finished CPU ASTC candidate for each logical block.
    std::vector<float> gauge_delta(weights.size());
    for (size_t index = 0; index < gauge_delta.size(); ++index) {
        gauge_delta[index] = (index & 1u) == 0 ? 0.025f : -0.025f;
    }
    std::vector<astc_gpu_encoder_source_block> gauge_blocks;
    // The earlier transport check intentionally uses sparse source IDs. The
    // logical D1 selector contract is raster-indexed, so use a compact copy
    // for this independent hand-off test.
    auto bank_scalar_blocks = blocks;
    for (size_t index = 0; index < bank_scalar_blocks.size(); ++index)
        bank_scalar_blocks[index].source_block_id = static_cast<uint32_t>(index);
    astc_gpu_d1_candidate_bank bank;
    astc_gpu_encoder_request bank_request;
    if (!astc_gpu_d1_build_gauge_la_source_blocks(astc_vulkan_footprint::k4x4,
                                                    weights, gauge_delta, 3, 5, gauge_blocks) ||
        !astc_gpu_d1_build_candidate_bank(astc_vulkan_footprint::k4x4, bank_scalar_blocks,
            {{astc_gpu_d1_candidate_family::gauge_la, gauge_blocks}}, bank) ||
        !astc_gpu_d1_candidate_bank_request(bank, static_cast<uint32_t>(bank.candidate_blocks.size()), bank_request)) return 1;
    std::vector<astc_gpu_encoder_proposal> bank_cpu;
    std::vector<astc_gpu_encoder_proposal> bank_gpu;
    std::vector<astc_gpu_encoder_proposal> bank_selected;
    std::vector<astc_gpu_encoder_finished_block> bank_finished;
    if (!astc_gpu_encoder_propose_cpu_reference(bank_request, bank_cpu) ||
        !astc_gpu_encoder_propose_gpu_default(argv[1], bank_request, bank_gpu, error) ||
        bank_gpu.size() != bank_cpu.size()) return 1;
    for (size_t index = 0; index < bank_gpu.size(); ++index) if (
        bank_gpu[index].source_block_id != bank_cpu[index].source_block_id ||
        std::fabs(bank_gpu[index].approximate_error - bank_cpu[index].approximate_error) > 1e-6f) return 1;
    if (!astc_gpu_d1_select_candidate_bank_proposals(bank, bank_gpu, 2, bank_selected) ||
        bank_selected.size() != bank_gpu.size() ||
        !astc_gpu_encoder_finish_d1_scalar_4x4(bank_request, bank_selected,
                                                 ASTCENC_PRE_FAST, bank_finished, error) ||
        bank_finished.size() != bank_selected.size()) return 1;
    const astc_gpu_d1_activation_rank_request rank_request{5, 3, 2,
        {1, 2, 3, 4, 5, 0.5f, 1, 1.5f, 2, 2.5f}};
    std::vector<astc_gpu_encoder_finished_block> neural_selected;
    std::vector<astc_gpu_d1_finished_candidate_score> neural_scores;
    if (!astc_gpu_d1_rank_finished_candidates_activation(
            bank, bank_finished, rank_request, 2, neural_selected, neural_scores) ||
        neural_selected.size() != bank_finished.size() || neural_scores.size() != bank_finished.size()) return 1;
    astc_gpu_d1_selector_delta_request d1_selector_request;
    d1_selector_request.tensor_width = rank_request.tensor_width;
    d1_selector_request.tensor_height = rank_request.tensor_height;
    d1_selector_request.source_blocks_x = rank_request.source_blocks_x;
    d1_selector_request.calibration_activations.assign(
        rank_request.activations.begin(), rank_request.activations.end());
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> d1_selector_candidates;
    if (!astc_gpu_d1_make_selector_candidates(
            bank, bank_finished, d1_selector_request, d1_selector_candidates, error) ||
        d1_selector_candidates.size() != blocks.size() ||
        d1_selector_candidates[0].size() != 2) return 1;
    astc_vulkan_paired_selection_result d1_selection;
    if (!astc_vulkan_select_paired_candidates(
            {3, 2, 0}, std::vector<double>(6, 0.1), {},
            d1_selector_candidates, d1_selection) ||
        d1_selection.calibration_selected_candidates.size() != blocks.size()) return 1;
    astc_gpu_encoder_benchmark_result benchmark;
    if (!astc_gpu_encoder_benchmark_gpu_default(argv[1], astc_vulkan_footprint::k4x4,
                                                  request, 2, 5, benchmark, error) ||
        benchmark.block_count != blocks.size() || benchmark.blocks_per_second <= 0.0) {
        std::fprintf(stderr, "GPU D1 proposer session benchmark failed: %s\n", error.c_str());
        return 1;
    }
    // Every physical D1 footprint gets an explicit SPIR-V ABI. The CPU
    // finisher and fixed-function decoder checks remain footprint-generic.
    std::vector<astc_gpu_encoder_source_block> blocks_6x6;
    if (!astc_gpu_d1_build_scalar_source_blocks(astc_vulkan_footprint::k6x6,
                                                  weights, 3, 5, blocks_6x6)) return 1;
    astc_gpu_encoder_request request_6x6;
    request_6x6.footprint = astc_vulkan_footprint::k6x6;
    request_6x6.max_blocks_per_batch = 1;
    request_6x6.blocks = blocks_6x6;
    std::vector<astc_gpu_encoder_proposal> proposed_6x6;
    std::vector<astc_gpu_encoder_finished_block> finished_6x6;
    std::vector<astc_gpu_encoder_proposal> gpu_proposed_6x6;
    if (!astc_gpu_encoder_propose_cpu_reference(request_6x6, proposed_6x6) ||
        !astc_gpu_encoder_propose_gpu_for_footprint_default(
            argv[3], astc_vulkan_footprint::k6x6, request_6x6, gpu_proposed_6x6, error) ||
        !astc_gpu_encoder_finish_d1_scalar(request_6x6, gpu_proposed_6x6,
                                            ASTCENC_PRE_FAST, finished_6x6, error) ||
        finished_6x6.size() != 1 || finished_6x6[0].decoded_rgba.size() != 6 * 6 * 4) return 1;
    if (gpu_proposed_6x6.size() != proposed_6x6.size() ||
        std::fabs(gpu_proposed_6x6[0].approximate_error - proposed_6x6[0].approximate_error) > 1e-6f ||
        gpu_proposed_6x6[0].weight_grid_x != 6 || gpu_proposed_6x6[0].weight_grid_y != 6) return 1;
    if (!astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[2], finished_6x6, 1, decode_mse, decode_max_abs, error)) return 1;

    // Real GPU-proposer candidate-bank path for 6x6.  Four physical source
    // candidates are proposed, but only K=2 (scalar plus the best alternative)
    // reach the exact CPU/libastc finisher.  This is intentionally an opt-in
    // offline smoke; it does not alter the production cache path yet.
    std::vector<float> gauge_delta_a(weights.size());
    std::vector<float> gauge_delta_b(weights.size());
    std::vector<float> gauge_delta_c(weights.size());
    for (size_t index = 0; index < weights.size(); ++index) {
        gauge_delta_a[index] = (index & 1u) ? -0.025f : 0.025f;
        gauge_delta_b[index] = (index % 3u == 0) ? 0.04f : -0.015f;
        gauge_delta_c[index] = (index % 4u < 2u) ? -0.035f : 0.02f;
    }
    std::vector<astc_gpu_encoder_source_block> gauge_a_6x6;
    std::vector<astc_gpu_encoder_source_block> gauge_b_6x6;
    std::vector<astc_gpu_encoder_source_block> gauge_c_6x6;
    astc_gpu_d1_candidate_bank bank_6x6;
    astc_gpu_encoder_request bank_request_6x6;
    if (!astc_gpu_d1_build_gauge_la_source_blocks(
            astc_vulkan_footprint::k6x6, weights, gauge_delta_a, 3, 5, gauge_a_6x6) ||
        !astc_gpu_d1_build_gauge_la_source_blocks(
            astc_vulkan_footprint::k6x6, weights, gauge_delta_b, 3, 5, gauge_b_6x6) ||
        !astc_gpu_d1_build_gauge_la_source_blocks(
            astc_vulkan_footprint::k6x6, weights, gauge_delta_c, 3, 5, gauge_c_6x6) ||
        !astc_gpu_d1_build_candidate_bank(
            astc_vulkan_footprint::k6x6, blocks_6x6,
            {{astc_gpu_d1_candidate_family::gauge_la, gauge_a_6x6},
             {astc_gpu_d1_candidate_family::gauge_la, gauge_b_6x6},
             {astc_gpu_d1_candidate_family::gauge_la, gauge_c_6x6}},
            bank_6x6) ||
        !astc_gpu_d1_candidate_bank_request(bank_6x6, 4, bank_request_6x6)) return 1;
    std::vector<astc_gpu_encoder_proposal> bank_proposals_6x6;
    if (!astc_gpu_encoder_propose_gpu_for_footprint_default(
            argv[3], astc_vulkan_footprint::k6x6, bank_request_6x6,
            bank_proposals_6x6, error) || bank_proposals_6x6.size() != 4) return 1;
    astc_gpu_d1_hybrid_result bank_result_6x6;
    if (!astc_gpu_d1_finish_candidate_bank(
            bank_6x6, bank_proposals_6x6,
            astc_gpu_d1_hybrid_options{2, 2, ASTCENC_PRE_FAST},
            bank_result_6x6, error) ||
        bank_result_6x6.retained_proposals.size() != 2 ||
        bank_result_6x6.finished_blocks.size() != 2) return 1;
    for (const auto & block : bank_result_6x6.finished_blocks) {
        if (block.decoded_rgba.size() != 6 * 6 * 4) return 1;
        for (float value : block.decoded_rgba) if (!std::isfinite(value)) return 1;
    }
    std::printf("GPU D1 6x6 candidate-bank: proposed=%zu retained=%zu finished=%zu K=2\n",
                bank_proposals_6x6.size(), bank_result_6x6.retained_proposals.size(),
                bank_result_6x6.finished_blocks.size());

    // Multi-block timing gate.  The one-block smoke above verifies wiring;
    // this bounded 48x96 fixture measures the actual batch behaviour before
    // the bank can become the default GPU path.
    constexpr uint32_t benchmark_rows = 48;
    constexpr uint32_t benchmark_columns = 96;
    std::vector<float> benchmark_weights(size_t(benchmark_rows) * benchmark_columns);
    for (uint32_t row = 0; row < benchmark_rows; ++row) {
        for (uint32_t column = 0; column < benchmark_columns; ++column) {
            benchmark_weights[size_t(row) * benchmark_columns + column] =
                0.15f + 0.70f * (static_cast<float>((row * 17u + column * 13u) % 97u) / 96.0f);
        }
    }
    std::vector<astc_gpu_encoder_source_block> benchmark_scalar;
    if (!astc_gpu_d1_build_scalar_source_blocks(
            astc_vulkan_footprint::k6x6, benchmark_weights,
            benchmark_rows, benchmark_columns, benchmark_scalar)) return 1;
    const size_t benchmark_logical_blocks = benchmark_scalar.size();
    std::vector<float> benchmark_delta_a(benchmark_weights.size());
    std::vector<float> benchmark_delta_b(benchmark_weights.size());
    std::vector<float> benchmark_delta_c(benchmark_weights.size());
    for (size_t index = 0; index < benchmark_weights.size(); ++index) {
        benchmark_delta_a[index] = (index & 1u) ? -0.02f : 0.02f;
        benchmark_delta_b[index] = (index % 3u == 0) ? 0.03f : -0.01f;
        benchmark_delta_c[index] = (index % 5u < 2u) ? -0.025f : 0.015f;
    }
    std::vector<astc_gpu_encoder_source_block> benchmark_a;
    std::vector<astc_gpu_encoder_source_block> benchmark_b;
    std::vector<astc_gpu_encoder_source_block> benchmark_c;
    astc_gpu_d1_candidate_bank benchmark_bank;
    astc_gpu_encoder_request benchmark_bank_request;
    if (!astc_gpu_d1_build_gauge_la_source_blocks(
            astc_vulkan_footprint::k6x6, benchmark_weights,
            benchmark_delta_a, benchmark_rows, benchmark_columns, benchmark_a) ||
        !astc_gpu_d1_build_gauge_la_source_blocks(
            astc_vulkan_footprint::k6x6, benchmark_weights,
            benchmark_delta_b, benchmark_rows, benchmark_columns, benchmark_b) ||
        !astc_gpu_d1_build_gauge_la_source_blocks(
            astc_vulkan_footprint::k6x6, benchmark_weights,
            benchmark_delta_c, benchmark_rows, benchmark_columns, benchmark_c) ||
        !astc_gpu_d1_build_candidate_bank(
            astc_vulkan_footprint::k6x6, benchmark_scalar,
            {{astc_gpu_d1_candidate_family::gauge_la, benchmark_a},
             {astc_gpu_d1_candidate_family::gauge_la, benchmark_b},
             {astc_gpu_d1_candidate_family::gauge_la, benchmark_c}},
            benchmark_bank) ||
        !astc_gpu_d1_candidate_bank_request(
            benchmark_bank, 256, benchmark_bank_request)) return 1;
    std::vector<astc_gpu_encoder_proposal> benchmark_proposals;
    const auto proposal_begin = std::chrono::steady_clock::now();
    if (!astc_gpu_encoder_propose_gpu_for_footprint_default(
            argv[3], astc_vulkan_footprint::k6x6,
            benchmark_bank_request, benchmark_proposals, error)) return 1;
    const auto proposal_end = std::chrono::steady_clock::now();
    if (benchmark_proposals.size() != benchmark_bank.candidate_blocks.size()) return 1;
    astc_gpu_d1_hybrid_result benchmark_result;
    const auto finish_begin = std::chrono::steady_clock::now();
    if (!astc_gpu_d1_finish_candidate_bank(
            benchmark_bank, benchmark_proposals,
            astc_gpu_d1_hybrid_options{2, 2, ASTCENC_PRE_FAST},
            benchmark_result, error)) return 1;
    const auto finish_end = std::chrono::steady_clock::now();
    if (benchmark_result.retained_proposals.size() != benchmark_logical_blocks * 2u ||
        benchmark_result.finished_blocks.size() != benchmark_logical_blocks * 2u) return 1;
    for (const auto & block : benchmark_result.finished_blocks) {
        if (block.decoded_rgba.size() != 6 * 6 * 4) return 1;
        for (float value : block.decoded_rgba) if (!std::isfinite(value)) return 1;
    }
    const double proposal_ms = std::chrono::duration<double, std::milli>(proposal_end - proposal_begin).count();
    const double finish_ms = std::chrono::duration<double, std::milli>(finish_end - finish_begin).count();

    astc_gpu_encoder_request benchmark_scalar_request;
    benchmark_scalar_request.mode = astc_gpu_encode_mode::propose;
    benchmark_scalar_request.footprint = astc_vulkan_footprint::k6x6;
    benchmark_scalar_request.max_blocks_per_batch = 256;
    benchmark_scalar_request.blocks = benchmark_scalar;
    std::vector<astc_gpu_encoder_proposal> benchmark_scalar_proposals;
    const auto scalar_proposal_begin = std::chrono::steady_clock::now();
    if (!astc_gpu_encoder_propose_gpu_for_footprint_default(
            argv[3], astc_vulkan_footprint::k6x6, benchmark_scalar_request,
            benchmark_scalar_proposals, error) ||
        benchmark_scalar_proposals.size() != benchmark_logical_blocks) return 1;
    const auto scalar_proposal_end = std::chrono::steady_clock::now();
    astc_gpu_d1_candidate_bank scalar_bank;
    if (!astc_gpu_d1_build_candidate_bank(
            astc_vulkan_footprint::k6x6, benchmark_scalar, {}, scalar_bank)) return 1;
    astc_gpu_d1_hybrid_result scalar_result;
    const auto scalar_finish_begin = std::chrono::steady_clock::now();
    if (!astc_gpu_d1_finish_candidate_bank(
            scalar_bank, benchmark_scalar_proposals,
            astc_gpu_d1_hybrid_options{1, 1, ASTCENC_PRE_FAST},
            scalar_result, error) ||
        scalar_result.finished_blocks.size() != benchmark_logical_blocks) return 1;
    const auto scalar_finish_end = std::chrono::steady_clock::now();
    const double scalar_proposal_ms = std::chrono::duration<double, std::milli>(
        scalar_proposal_end - scalar_proposal_begin).count();
    const double scalar_finish_ms = std::chrono::duration<double, std::milli>(
        scalar_finish_end - scalar_finish_begin).count();
    std::printf("GPU D1 6x6 bank benchmark: logical=%zu proposed=%zu retained=%zu "
                "proposal_ms=%.3f finish_ms=%.3f retain_ratio=%.3f\n",
                benchmark_logical_blocks, benchmark_proposals.size(),
                benchmark_result.finished_blocks.size(), proposal_ms, finish_ms,
                static_cast<double>(benchmark_result.finished_blocks.size()) /
                    static_cast<double>(benchmark_proposals.size()));
    std::printf("GPU D1 6x6 scalar baseline: logical=%zu proposed=%zu finished=%zu "
                "proposal_ms=%.3f finish_ms=%.3f total_ms=%.3f\n",
                benchmark_logical_blocks, benchmark_scalar_proposals.size(),
                scalar_result.finished_blocks.size(), scalar_proposal_ms, scalar_finish_ms,
                scalar_proposal_ms + scalar_finish_ms);
    std::vector<astc_gpu_encoder_source_block> blocks_8x6;
    if (!astc_gpu_d1_build_scalar_source_blocks(astc_vulkan_footprint::k8x6,
                                                  weights, 3, 5, blocks_8x6)) return 1;
    astc_gpu_encoder_request request_8x6;
    request_8x6.footprint = astc_vulkan_footprint::k8x6;
    request_8x6.max_blocks_per_batch = 1;
    request_8x6.blocks = blocks_8x6;
    std::vector<astc_gpu_encoder_proposal> cpu_proposed_8x6;
    std::vector<astc_gpu_encoder_proposal> gpu_proposed_8x6;
    std::vector<astc_gpu_encoder_finished_block> finished_8x6;
    if (!astc_gpu_encoder_propose_cpu_reference(request_8x6, cpu_proposed_8x6) ||
        !astc_gpu_encoder_propose_gpu_for_footprint_default(
            argv[5], astc_vulkan_footprint::k8x6, request_8x6, gpu_proposed_8x6, error) ||
        !astc_gpu_encoder_finish_d1_scalar(request_8x6, gpu_proposed_8x6,
                                            ASTCENC_PRE_FAST, finished_8x6, error) ||
        gpu_proposed_8x6.size() != 1 || finished_8x6.size() != 1 ||
        finished_8x6[0].decoded_rgba.size() != 8 * 6 * 4 ||
        gpu_proposed_8x6[0].weight_grid_x != 8 || gpu_proposed_8x6[0].weight_grid_y != 6 ||
        std::fabs(gpu_proposed_8x6[0].approximate_error - cpu_proposed_8x6[0].approximate_error) > 1e-6f ||
        !astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[2], finished_8x6, 1, decode_mse, decode_max_abs, error)) return 1;
    std::vector<astc_gpu_encoder_source_block> blocks_5x5;
    if (!astc_gpu_d1_build_scalar_source_blocks(astc_vulkan_footprint::k5x5,
                                                  weights, 3, 5, blocks_5x5)) return 1;
    astc_gpu_encoder_request request_5x5;
    request_5x5.footprint = astc_vulkan_footprint::k5x5;
    request_5x5.max_blocks_per_batch = 1;
    request_5x5.blocks = blocks_5x5;
    std::vector<astc_gpu_encoder_proposal> cpu_proposed_5x5;
    std::vector<astc_gpu_encoder_proposal> gpu_proposed_5x5;
    std::vector<astc_gpu_encoder_finished_block> finished_5x5;
    if (!astc_gpu_encoder_propose_cpu_reference(request_5x5, cpu_proposed_5x5) ||
        !astc_gpu_encoder_propose_gpu_for_footprint_default(
            argv[4], astc_vulkan_footprint::k5x5, request_5x5, gpu_proposed_5x5, error) ||
        !astc_gpu_encoder_finish_d1_scalar(request_5x5, gpu_proposed_5x5,
                                            ASTCENC_PRE_FAST, finished_5x5, error) ||
        gpu_proposed_5x5.size() != 1 || finished_5x5.size() != 1 ||
        gpu_proposed_5x5[0].weight_grid_x != 5 || gpu_proposed_5x5[0].weight_grid_y != 5 ||
        std::fabs(gpu_proposed_5x5[0].approximate_error - cpu_proposed_5x5[0].approximate_error) > 1e-6f ||
        !astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[2], finished_5x5, 1, decode_mse, decode_max_abs, error)) return 1;
    // D2 owns paired rows and the layout map, but the physical proposal,
    // legal payload finish and fixed-function decode are shared unchanged.
    const std::vector<float> paired_weights{
        0.10f, 0.20f, 0.30f, 0.40f, 0.50f,
        0.60f, 0.70f, 0.80f, 0.90f, 0.10f,
        0.25f, 0.35f, 0.45f, 0.55f, 0.65f};
    const auto run_d2 = [&](astc_vulkan_footprint footprint, const char * shader,
                            uint32_t expected_width, uint32_t expected_height) {
        std::vector<astc_vulkan_paired_layout> layouts;
        std::vector<astc_gpu_encoder_source_block> blocks;
        if (!astc_gpu_d2_make_uniform_layout_map(footprint, 3, 5,
                astc_vulkan_paired_layout::rg_b, layouts) ||
            !astc_gpu_d2_build_paired_source_blocks(footprint, paired_weights, 3, 5,
                layouts, {}, astc_vulkan_paired_semantic::direct_rgb, blocks)) return false;
        std::vector<astc_gpu_encoder_source_block> la_blocks;
        if (!astc_gpu_d2_build_paired_source_blocks(footprint, paired_weights, 3, 5,
                layouts, {}, astc_vulkan_paired_semantic::luminance_alpha, la_blocks)) return false;
        // A source-derived steering alternative exercises the third D2
        // candidate family. Alpha stays runtime-semantic-free; it exists only
        // to perturb the offline ASTC candidate space before exact ranking.
        const std::vector<float> steering_texels{
            0.20f, 0.35f, 0.50f, 0.65f, 0.80f,
            0.75f, 0.60f, 0.45f, 0.30f, 0.15f};
        std::vector<astc_gpu_encoder_source_block> steered_blocks;
        if (!astc_gpu_d2_build_paired_source_blocks(footprint, paired_weights, 3, 5,
                layouts, steering_texels, astc_vulkan_paired_semantic::direct_rgb,
                steered_blocks)) return false;
        astc_gpu_d2_candidate_bank bank;
        if (!astc_gpu_d2_build_candidate_bank(footprint, blocks, layouts,
                {{astc_gpu_d2_candidate_family::luminance_alpha,
                  astc_vulkan_paired_semantic::luminance_alpha, la_blocks, layouts},
                 {astc_gpu_d2_candidate_family::direct_steered,
                  astc_vulkan_paired_semantic::direct_rgb, steered_blocks, layouts}}, bank)) return false;
        astc_gpu_encoder_request request;
        std::vector<astc_gpu_encoder_proposal> cpu, gpu;
        if (!astc_gpu_d2_candidate_bank_request(bank, 3, request) ||
            !astc_gpu_encoder_propose_cpu_reference(request, cpu) ||
            !astc_gpu_encoder_propose_gpu_for_footprint_default(shader, footprint, request, gpu, error) ||
            cpu.size() != 3 || gpu.size() != 3 ||
            gpu[0].weight_grid_x != expected_width || gpu[0].weight_grid_y != expected_height ||
            std::fabs(gpu[0].approximate_error - cpu[0].approximate_error) > 1e-6f ||
            std::fabs(gpu[1].approximate_error - cpu[1].approximate_error) > 1e-6f ||
            std::fabs(gpu[2].approximate_error - cpu[2].approximate_error) > 1e-6f) return false;
        astc_gpu_d2_activation_rank_request rank_request;
        rank_request.tensor_width = 5;
        rank_request.tensor_height = 3;
        rank_request.source_blocks_x = 1;
        rank_request.activations = {
            1.0f, 0.5f, 1.5f, 2.0f, 0.25f,
            0.75f, 1.25f, 0.5f, 1.0f, 1.75f};
        astc_gpu_d2_hybrid_result hybrid;
        const astc_gpu_d2_hybrid_options hybrid_options{3, 2, ASTCENC_PRE_FAST};
        if (!astc_gpu_d2_finish_and_rank(bank, gpu, rank_request, hybrid_options, hybrid, error) ||
            hybrid.retained_proposals.size() != 3 || hybrid.finished_blocks.size() != 3 ||
            hybrid.ranked_blocks.size() != 2 || hybrid.activation_scores.size() != 3 ||
            hybrid.finished_blocks[0].decoded_rgba.size() != size_t(expected_width) * expected_height * 4 ||
            !astc_gpu_encoder_verify_d1_vulkan_decode_default(
                argv[2], hybrid.finished_blocks, 3, decode_mse, decode_max_abs, error)) return false;
        astc_gpu_d2_selector_delta_request selector_request;
        selector_request.tensor_width = rank_request.tensor_width;
        selector_request.tensor_height = rank_request.tensor_height;
        selector_request.source_blocks_x = rank_request.source_blocks_x;
        selector_request.calibration_activations.assign(
            rank_request.activations.begin(), rank_request.activations.end());
        std::vector<std::vector<astc_vulkan_paired_candidate_delta>> selector_candidates;
        if (!astc_gpu_d2_make_selector_candidates(
                bank, hybrid.finished_blocks, selector_request,
                selector_candidates, error) || selector_candidates.size() != 1 ||
            selector_candidates[0].size() != 3) return false;
        astc_vulkan_paired_selection_result selection;
        if (!astc_vulkan_select_paired_candidates(
                {3, 2, 0}, std::vector<double>(6, 0.1), {},
                selector_candidates, selection) ||
            selection.validation_prefix != selection.commits.size() ||
            selection.calibration_selected_candidates.size() != 1) return false;

        // H5 is the deployed paired D2 geometry. Exercise the complete GPU
        // proposer -> CPU finisher -> exact rank/selector seam with a real
        // serialized non-adjacent pair map and an inverse-restored Givens
        // alternative. The proposer itself only sees physical RGBA, which is
        // exactly why this test has to cross the D2 frontend boundary.
        if (footprint == astc_vulkan_footprint::k8x5) {
            std::vector<float> transformed_weights(10 * 8);
            for (uint32_t row = 0; row < 10; ++row) {
                for (uint32_t column = 0; column < 8; ++column) {
                    transformed_weights[row * 8 + column] =
                        0.05f + 0.9f * float((row * 3 + column * 5) % 17) / 16.0f;
                }
            }
            const std::vector<uint8_t> pair_map{1, 0, 3, 2, 5, 4, 7, 6, 9, 8};
            std::vector<astc_vulkan_d2_pairing> pairings;
            std::vector<astc_vulkan_paired_layout> transformed_layouts;
            if (!astc_gpu_d2_expand_pair_map(footprint, 10, 8, pair_map, pairings) ||
                pairings.size() != 1 ||
                !astc_gpu_d2_make_uniform_layout_map(
                    footprint, 10, 8, astc_vulkan_paired_layout::rg_b,
                    transformed_layouts)) return false;
            std::vector<astc_gpu_encoder_source_block> neutral_blocks;
            std::vector<astc_gpu_encoder_source_block> rotated_blocks;
            const astc_vulkan_d2_givens_transform givens{0.35f};
            if (!astc_gpu_d2_build_paired_source_blocks(
                    footprint, transformed_weights, 10, 8, transformed_layouts, {},
                    astc_vulkan_paired_semantic::direct_rgb, neutral_blocks, pairings) ||
                !astc_gpu_d2_build_paired_source_blocks(
                    footprint, transformed_weights, 10, 8, transformed_layouts, {},
                    astc_vulkan_paired_semantic::direct_rgb, rotated_blocks, pairings, givens)) return false;
            astc_gpu_d2_candidate_bank transformed_bank;
            if (!astc_gpu_d2_build_candidate_bank(
                    footprint, neutral_blocks, transformed_layouts,
                    {{astc_gpu_d2_candidate_family::direct_steered,
                      astc_vulkan_paired_semantic::direct_rgb, rotated_blocks,
                      transformed_layouts, givens}}, transformed_bank, pairings)) return false;
            astc_gpu_encoder_request transformed_request;
            std::vector<astc_gpu_encoder_proposal> transformed_gpu;
            if (!astc_gpu_d2_candidate_bank_request(transformed_bank, 2, transformed_request) ||
                !astc_gpu_encoder_propose_gpu_for_footprint_default(
                    shader, footprint, transformed_request, transformed_gpu, error) ||
                transformed_gpu.size() != 2) return false;
            astc_gpu_d2_activation_rank_request transformed_rank;
            transformed_rank.tensor_width = 8;
            transformed_rank.tensor_height = 10;
            transformed_rank.source_blocks_x = 1;
            transformed_rank.activations.resize(16);
            for (size_t index = 0; index < transformed_rank.activations.size(); ++index)
                transformed_rank.activations[index] = 0.25f + 0.125f * float(index % 7);
            astc_gpu_d2_hybrid_result transformed_hybrid;
            if (!astc_gpu_d2_finish_and_rank(
                    transformed_bank, transformed_gpu, transformed_rank,
                    {2, 2, ASTCENC_PRE_FAST}, transformed_hybrid, error) ||
                transformed_hybrid.ranked_blocks.size() != 2 ||
                transformed_hybrid.activation_scores.size() != 2) return false;
            astc_gpu_d2_selector_delta_request transformed_selector;
            transformed_selector.tensor_width = 8;
            transformed_selector.tensor_height = 10;
            transformed_selector.source_blocks_x = 1;
            transformed_selector.calibration_activations.assign(
                transformed_rank.activations.begin(), transformed_rank.activations.end());
            std::vector<std::vector<astc_vulkan_paired_candidate_delta>> transformed_candidates;
            if (!astc_gpu_d2_make_selector_candidates(
                    transformed_bank, transformed_hybrid.finished_blocks,
                    transformed_selector, transformed_candidates, error) ||
                transformed_candidates.size() != 1 ||
                transformed_candidates[0].size() != 2) return false;
        }
        return true;
    };
    if (!run_d2(astc_vulkan_footprint::k8x5, argv[6], 8, 5) ||
        !run_d2(astc_vulkan_footprint::k6x5, argv[7], 6, 5) ||
        !run_d2(astc_vulkan_footprint::k10x5, argv[8], 10, 5)) return 1;
    std::printf("GPU ASTC proposer smoke passed (%zu D1 blocks plus D2 6x5/8x5/10x5, decode MSE %.9g, max %.9g, %.3f blocks/s)\n",
                actual.size(), decode_mse, decode_max_abs, benchmark.blocks_per_second);
    return 0;
}
