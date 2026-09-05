#include "astc-gpu-d1-source.h"
#include "astc-gpu-d1-candidates.h"
#include "astc-gpu-d1-neural-rank.h"
#include "astc-gpu-d1-hybrid.h"
#include "astc-gpu-d2-source.h"
#include "astc-gpu-d2-candidates.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

int main() {
    const auto require = [](bool condition) {
        if (!condition) std::abort();
    };
    const std::vector<float> weights{
        -1.0f, -0.5f, 0.0f, 0.5f, 1.0f,
         0.2f,  0.3f, 0.4f, 0.5f, 0.6f,
         0.7f,  0.8f, 0.9f, 1.0f, 0.1f};
    std::vector<astc_gpu_encoder_source_block> blocks;
    require(astc_gpu_d1_build_scalar_source_blocks(
        astc_vulkan_footprint::k4x4, weights, 3, 5, blocks));
    require(blocks.size() == 2);
    require(blocks[0].texels.size() == 16 && blocks[1].texels.size() == 16);
    require(blocks[0].texels[0].rgba[0] == -1.0f);
    require(blocks[0].texels[0].rgba[0] == blocks[0].texels[0].rgba[2]);
    require(blocks[1].texels[1].rgba[0] == 0.0f); // deterministic right-edge pad
    require(blocks[1].texels[1].rgba[3] == 1.0f);

    const std::vector<float> gauge{0.10f, -0.10f, 0.05f, -0.05f, 0.0f,
                                    0.02f, -0.02f, 0.03f, -0.03f, 0.0f,
                                    0.04f, -0.04f, 0.01f, -0.01f, 0.0f};
    std::vector<astc_gpu_encoder_source_block> gauge_blocks;
    require(astc_gpu_d1_build_gauge_la_source_blocks(
        astc_vulkan_footprint::k6x6, weights, gauge, 3, 5, gauge_blocks));
    require(gauge_blocks.size() == 1 && gauge_blocks[0].texels.size() == 36);
    for (uint32_t row = 0; row < 3; ++row) for (uint32_t column = 0; column < 5; ++column) {
        const auto & texel = gauge_blocks[0].texels[row * 6 + column];
        require(std::fabs((texel.rgba[0] + texel.rgba[3]) * 0.5f -
                          weights[row * 5 + column]) < 1e-6f);
    }
    require(gauge_blocks[0].texels[5].rgba[0] == 0.0f &&
            gauge_blocks[0].texels[5].rgba[3] == 0.0f);

    std::vector<astc_gpu_encoder_source_block> gauge_blocks_4x4;
    require(astc_gpu_d1_build_gauge_la_source_blocks(
        astc_vulkan_footprint::k4x4, weights, gauge, 3, 5, gauge_blocks_4x4));
    astc_gpu_d1_candidate_bank bank;
    require(astc_gpu_d1_build_candidate_bank(
        astc_vulkan_footprint::k4x4, blocks,
        {{astc_gpu_d1_candidate_family::gauge_la, gauge_blocks_4x4}}, bank));
    require(bank.candidate_blocks.size() == 4 && bank.records.size() == 4);
    astc_gpu_encoder_request bank_request;
    require(astc_gpu_d1_candidate_bank_request(bank, 4, bank_request));
    require(bank_request.candidate_metadata.size() == bank_request.blocks.size());
    require(bank_request.candidate_metadata[0].source_block_id == bank_request.blocks[0].source_block_id);
    require(bank_request.candidate_metadata[0].logical_source_block_id == blocks[0].source_block_id);
    std::vector<astc_gpu_encoder_proposal> bank_proposals;
    require(astc_gpu_encoder_propose_cpu_reference(bank_request, bank_proposals));
    std::vector<astc_gpu_encoder_proposal> selected_scalar;
    require(astc_gpu_d1_select_candidate_bank_proposals(bank, bank_proposals, 1, selected_scalar));
    require(selected_scalar.size() == 2);
    require(bank.records[selected_scalar[0].source_block_id].family == astc_gpu_d1_candidate_family::scalar);
    require(bank.records[selected_scalar[1].source_block_id].family == astc_gpu_d1_candidate_family::scalar);
    std::vector<astc_gpu_encoder_proposal> selected_pair;
    require(astc_gpu_d1_select_candidate_bank_proposals(bank, bank_proposals, 2, selected_pair));
    require(selected_pair.size() == 4);

    // Synthetic exact-decode fixtures: deliberately worsen scalar decode so
    // activation ranking must retain the gauge alternative after scalar.
    std::vector<astc_gpu_encoder_finished_block> finished;
    for (size_t index = 0; index < bank.records.size(); ++index) {
        astc_gpu_encoder_finished_block block;
        block.source_block_id = bank.records[index].candidate_source_block_id;
        block.footprint = astc_vulkan_footprint::k4x4;
        block.decoded_rgba.resize(4 * 4 * 4);
        for (uint32_t texel = 0; texel < 16; ++texel) {
            const auto & source = bank.candidate_blocks[index].texels[texel].rgba;
            for (uint32_t channel = 0; channel < 4; ++channel) block.decoded_rgba[texel * 4 + channel] = source[channel];
            if (bank.records[index].family == astc_gpu_d1_candidate_family::scalar) {
                // The first two real output rows receive equal-and-opposite
                // errors. A broken block-wide sum before the square would let
                // them cancel; the activation objective must score rows
                // independently.
                const uint32_t row = texel / 4;
                const float offset = row == 0 ? 0.1f : (row == 1 ? -0.1f : 0.0f);
                block.decoded_rgba[texel * 4 + 0] += offset;
                block.decoded_rgba[texel * 4 + 1] += offset;
                block.decoded_rgba[texel * 4 + 2] += offset;
            }
        }
        finished.push_back(std::move(block));
    }
    const astc_gpu_d1_activation_rank_request rank_request{5, 3, 2,
        {1, 1, 1, 1, 1}};
    std::vector<astc_gpu_encoder_finished_block> activation_selected;
    std::vector<astc_gpu_d1_finished_candidate_score> activation_scores;
    require(astc_gpu_d1_rank_finished_candidates_activation(
        bank, finished, rank_request, 2, activation_selected, activation_scores));
    require(activation_selected.size() == 4 && activation_scores.size() == 4);
    require(activation_selected[1].source_block_id == bank.records[1].candidate_source_block_id);
    const auto first_scalar = std::find_if(activation_scores.begin(), activation_scores.end(), [](const auto & score) {
        return score.logical_source_block_id == 0 && score.family == astc_gpu_d1_candidate_family::scalar;
    });
    require(first_scalar != activation_scores.end() && first_scalar->activation_error > 0.1);
    require(astc_gpu_d1_rank_finished_candidates_activation(
        bank, finished, rank_request, 1, activation_selected, activation_scores));
    require(activation_selected.size() == 2);

    // The selector adapter must use logical block IDs for geometry, not the
    // globally unique physical candidate IDs used by the proposer/finisher.
    // This fixture deliberately makes those IDs non-contiguous.
    astc_gpu_d1_candidate_bank selector_bank = bank;
    std::vector<astc_gpu_encoder_finished_block> selector_finished = finished;
    for (size_t index = 0; index < selector_bank.records.size(); ++index) {
        const uint32_t physical_id = 100u + static_cast<uint32_t>(index) * 17u;
        selector_bank.records[index].candidate_source_block_id = physical_id;
        selector_finished[index].source_block_id = physical_id;
        selector_finished[index].payload[0] = static_cast<uint8_t>(index + 1);
    }
    astc_gpu_d1_selector_delta_request selector_request;
    selector_request.tensor_width = 5;
    selector_request.tensor_height = 3;
    selector_request.source_blocks_x = 1;
    selector_request.calibration_activations.assign(5, 1.0);
    selector_request.validation_activations.assign(5, 1.0);
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> selector_candidates;
    std::string selector_error;
    require(astc_gpu_d1_make_selector_candidates(
        selector_bank, selector_finished, selector_request,
        selector_candidates, selector_error));
    require(selector_candidates.size() == 2);
    for (const auto & group : selector_candidates) {
        require(!group.empty());
        require(std::all_of(group.front().calibration_delta.begin(),
                            group.front().calibration_delta.end(),
                            [](double value) { return value == 0.0; }));
        require(std::all_of(group.front().validation_delta.begin(),
                            group.front().validation_delta.end(),
                            [](double value) { return value == 0.0; }));
    }

    // D2 keeps paired geometry/layout metadata entirely in its frontend. The
    // shared physical proposer sees only these 8x5 RGBA source blocks.
    const std::vector<float> paired_weights{
        0.10f, 0.20f, 0.30f, 0.40f, 0.50f,
        0.60f, 0.70f, 0.80f, 0.90f, 0.10f,
        0.25f, 0.35f, 0.45f, 0.55f, 0.65f};
    std::vector<astc_vulkan_paired_layout> paired_layouts;
    require(astc_gpu_d2_make_uniform_layout_map(astc_vulkan_footprint::k8x5,
        3, 5, astc_vulkan_paired_layout::rg_b, paired_layouts));
    std::vector<astc_gpu_encoder_source_block> paired_neutral;
    require(astc_gpu_d2_build_paired_source_blocks(astc_vulkan_footprint::k8x5,
        paired_weights, 3, 5, paired_layouts, {},
        astc_vulkan_paired_semantic::direct_rgb, paired_neutral));
    require(paired_neutral.size() == 1 && paired_neutral[0].texels.size() == 40);
    require(std::fabs(paired_neutral[0].texels[0].rgba[0] - 0.10f) < 1e-6f &&
            std::fabs(paired_neutral[0].texels[0].rgba[1] - 0.10f) < 1e-6f &&
            std::fabs(paired_neutral[0].texels[0].rgba[2] - 0.60f) < 1e-6f &&
            std::fabs(paired_neutral[0].texels[0].rgba[3] - 0.50f) < 1e-6f);
    std::vector<astc_gpu_encoder_source_block> paired_la;
    require(astc_gpu_d2_build_paired_source_blocks(astc_vulkan_footprint::k8x5,
        paired_weights, 3, 5, paired_layouts, {},
        astc_vulkan_paired_semantic::luminance_alpha, paired_la));
    require(std::fabs(paired_la[0].texels[0].rgba[0] - 0.10f) < 1e-6f &&
            std::fabs(paired_la[0].texels[0].rgba[3] - 0.60f) < 1e-6f);
    std::vector<float> paired_steering(10, 0.25f);
    std::vector<astc_gpu_encoder_source_block> paired_steered;
    require(astc_gpu_d2_build_paired_source_blocks(astc_vulkan_footprint::k8x5,
        paired_weights, 3, 5, paired_layouts, paired_steering,
        astc_vulkan_paired_semantic::direct_rgb, paired_steered));
    astc_gpu_d2_candidate_bank paired_bank;
    require(astc_gpu_d2_build_candidate_bank(astc_vulkan_footprint::k8x5,
        paired_neutral, paired_layouts,
        {{astc_gpu_d2_candidate_family::direct_steered,
          astc_vulkan_paired_semantic::direct_rgb, paired_steered, paired_layouts}}, paired_bank));
    astc_gpu_encoder_request paired_request;
    require(astc_gpu_d2_candidate_bank_request(paired_bank, 2, paired_request));
    require(paired_request.candidate_metadata.size() == paired_request.blocks.size());
    require(paired_request.candidate_metadata[0].family_id ==
            static_cast<uint32_t>(astc_gpu_d2_candidate_family::direct_neutral));
    std::vector<astc_gpu_encoder_proposal> paired_proposals;
    std::vector<astc_gpu_encoder_proposal> paired_selected;
    require(astc_gpu_encoder_propose_cpu_reference(paired_request, paired_proposals) &&
            astc_gpu_d2_select_candidate_bank_proposals(paired_bank, paired_proposals, 2, paired_selected));
    require(paired_selected.size() == 2 && paired_bank.records[paired_selected[0].source_block_id].family ==
            astc_gpu_d2_candidate_family::direct_neutral);

    astc_gpu_encoder_request request;
    request.footprint = astc_vulkan_footprint::k4x4;
    request.max_blocks_per_batch = 1;
    request.blocks = blocks;
    request.blocks[0].source_block_id = 41;
    request.blocks[1].source_block_id = 99;
    std::vector<astc_gpu_encoder_batch> batches;
    require(astc_gpu_encoder_plan_batches(request, batches));
    require(batches.size() == 2 && batches[0].block_count == 1);

    std::vector<astc_gpu_encoder_proposal> proposals;
    require(astc_gpu_encoder_propose_cpu_reference(request, proposals));
    require(proposals.size() == 2);
    require(proposals[0].source_block_id == 41);
    require(std::fabs(proposals[0].endpoint_low[0] + 1.0f) < 1e-6f);
    require(std::fabs(proposals[0].endpoint_high[0] - 1.0f) < 1e-6f);
    require(proposals[0].approximate_error > 0.0f);

    request.blocks[0].texels.pop_back();
    require(!astc_gpu_encoder_plan_batches(request, batches));
    return 0;
}
