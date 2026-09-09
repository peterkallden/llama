#include "astc-gpu-d1-source.h"
#include "astc-gpu-d1-candidates.h"
#include "astc-gpu-d1-exact-subset.h"
#include "astc-gpu-d1-neural-rank.h"
#include "astc-gpu-d1-hybrid.h"
#include "astc-gpu-d2-source.h"
#include "astc-gpu-d2-candidates.h"
#include "astc-gpu-d2-exact-subset.h"
#include "astc-gpu-d2-neural-rank.h"
#include "astc-gpu-d2-hybrid.h"
#include "astc-gpu-encoder-subset.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

int main() {
    const auto require = [](bool condition) {
        if (!condition) std::abort();
    };
    // Exact-subset variants must be pinned to the same audited physical mode
    // on CPU and GPU. Refinement changes candidate fitting only.
    const auto * d1_mode = astc_gpu_exact_subset_audited_mode(
        astc_gpu_exact_subset_kind::d1_luminance_binary_refined_6x6);
    const auto * d2_mode = astc_gpu_exact_subset_audited_mode(
        astc_gpu_exact_subset_kind::luminance_alpha_dual_binary_8x5);
    require(d1_mode != nullptr && d1_mode->block_mode == 0x104u &&
            d1_mode->footprint == astc_vulkan_footprint::k6x6);
    require(d2_mode != nullptr && d2_mode->block_mode == 0x4c1u &&
            d2_mode->dual_plane && d2_mode->dual_plane_component == 3u);
    astc_gpu_encoder_request audited_request;
    audited_request.mode = astc_gpu_encode_mode::exact_subset;
    audited_request.exact_subset = astc_gpu_exact_subset_kind::d1_luminance_binary_6x6;
    audited_request.footprint = astc_vulkan_footprint::k6x6;
    require(astc_gpu_exact_subset_matches_audited_mode(audited_request));
    audited_request.footprint = astc_vulkan_footprint::k5x5;
    require(!astc_gpu_exact_subset_matches_audited_mode(audited_request));

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

    // The refined exact 6x6 path is a normal D1 candidate family, not a
    // separate runtime representation. Its source IDs remain distinct so the
    // existing ranker/selector can compare exact decoded payloads.
    std::vector<float> exact_weights(36);
    for (uint32_t index = 0; index < exact_weights.size(); ++index) {
        exact_weights[index] = 0.05f + 0.90f * static_cast<float>((index * 7u) % 19u) / 18.0f;
    }
    std::vector<astc_gpu_encoder_source_block> scalar_blocks_6x6;
    require(astc_gpu_d1_build_scalar_source_blocks(
        astc_vulkan_footprint::k6x6, exact_weights, 6, 6, scalar_blocks_6x6));
    astc_gpu_d1_exact_subset_candidate_bank refined_bank;
    require(astc_gpu_d1_build_refined_6x6_exact_subset_candidate_bank(
        scalar_blocks_6x6, 8, refined_bank));
    require(refined_bank.semantic_bank.records.size() == 4 &&
            refined_bank.binary_request.blocks.size() == 1 &&
            refined_bank.refined_request.blocks.size() == 1 &&
            refined_bank.mean_refined_request.blocks.size() == 1 &&
            refined_bank.quantile_refined_request.blocks.size() == 1);
    require(refined_bank.semantic_bank.records[0].family == astc_gpu_d1_candidate_family::scalar &&
            refined_bank.semantic_bank.records[1].family == astc_gpu_d1_candidate_family::scalar_refined &&
            refined_bank.semantic_bank.records[2].family == astc_gpu_d1_candidate_family::scalar_mean_refined &&
            refined_bank.semantic_bank.records[3].family == astc_gpu_d1_candidate_family::scalar_quantile_refined);
    std::vector<astc_gpu_exact_subset_block> binary_payloads, refined_payloads, mean_payloads, quantile_payloads;
    require(astc_gpu_exact_subset_encode_cpu(refined_bank.binary_request, binary_payloads) &&
            astc_gpu_exact_subset_encode_cpu(refined_bank.refined_request, refined_payloads) &&
            astc_gpu_exact_subset_encode_cpu(refined_bank.mean_refined_request, mean_payloads) &&
            astc_gpu_exact_subset_encode_cpu(refined_bank.quantile_refined_request, quantile_payloads));
    std::vector<astc_gpu_encoder_finished_block> binary_finished, refined_finished, mean_finished, quantile_finished, combined_finished;
    std::string exact_subset_error;
    require(astc_gpu_exact_subset_finish_payloads(astc_vulkan_footprint::k6x6,
                                                   binary_payloads, binary_finished, exact_subset_error) &&
            astc_gpu_exact_subset_finish_payloads(astc_vulkan_footprint::k6x6,
                                                   refined_payloads, refined_finished, exact_subset_error) &&
            astc_gpu_exact_subset_finish_payloads(astc_vulkan_footprint::k6x6,
                                                   mean_payloads, mean_finished, exact_subset_error) &&
            astc_gpu_exact_subset_finish_payloads(astc_vulkan_footprint::k6x6,
                                                   quantile_payloads, quantile_finished, exact_subset_error));
    combined_finished = binary_finished;
    combined_finished.insert(combined_finished.end(), refined_finished.begin(), refined_finished.end());
    combined_finished.insert(combined_finished.end(), mean_finished.begin(), mean_finished.end());
    combined_finished.insert(combined_finished.end(), quantile_finished.begin(), quantile_finished.end());
    std::vector<astc_gpu_encoder_finished_block> refined_ranked;
    std::vector<astc_gpu_d1_finished_candidate_score> refined_scores;
    const astc_gpu_d1_activation_rank_request refined_rank_request{6, 6, 1,
                                                                     std::vector<float>(6, 1.0f)};
    require(astc_gpu_d1_rank_finished_candidates_activation(
        refined_bank.semantic_bank, combined_finished, refined_rank_request, 4,
        refined_ranked, refined_scores));
    require(refined_ranked.size() == 4 && refined_scores.size() == 4);
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
    astc_gpu_d1_hybrid_result hybrid_result;
    std::string hybrid_error;
    require(astc_gpu_d1_finish_and_rank(
            bank, bank_proposals, rank_request,
            astc_gpu_d1_hybrid_options{2, 2, ASTCENC_PRE_FAST},
            hybrid_result, hybrid_error));
    require(hybrid_result.retained_proposals.size() == selected_pair.size());
    require(hybrid_result.finished_blocks.size() == finished.size());
    require(hybrid_result.ranked_blocks.size() == finished.size());
    require(hybrid_result.activation_scores.size() == finished.size());
    astc_gpu_d1_hybrid_result bank_only_result;
    require(astc_gpu_d1_finish_candidate_bank(
        bank, bank_proposals, astc_gpu_d1_hybrid_options{2, 2, ASTCENC_PRE_FAST},
        bank_only_result, hybrid_error));
    require(bank_only_result.retained_proposals.size() == 4);
    require(bank_only_result.finished_blocks.size() == 4);
    // Neural-quality is an offline candidate policy: it preserves an
    // independently thorough scalar reference for each logical block even if
    // GPU proposal retention also selected that source. The selector still
    // sees the normal D1 candidate-bank contract afterwards.
    astc_gpu_d1_hybrid_options quality_options;
    quality_options.max_finish_candidates_per_logical = 2;
    quality_options.max_ranked_candidates_per_logical = 2;
    quality_options.astcenc_quality = ASTCENC_PRE_FAST;
    quality_options.profile = astc_gpu_encoder_candidate_profile::neural_quality;
    quality_options.reference_astcenc_quality = ASTCENC_PRE_THOROUGH;
    quality_options.worker_count = 2;
    astc_gpu_d1_hybrid_result quality_result;
    require(astc_gpu_d1_finish_and_rank(
        bank, bank_proposals, rank_request, quality_options, quality_result, hybrid_error));
    require(quality_result.finished_blocks.size() == 4);
    require(quality_result.finished_blocks[0].source_block_id == bank.records[0].candidate_source_block_id);
    require(quality_result.finished_blocks[1].source_block_id == bank.records[2].candidate_source_block_id);
    require(quality_result.ranked_blocks.size() == 4);
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
    astc_gpu_d2_hybrid_options paired_quality_options;
    paired_quality_options.max_finish_candidates_per_logical = 2;
    paired_quality_options.max_ranked_candidates_per_logical = 2;
    paired_quality_options.astcenc_quality = ASTCENC_PRE_FAST;
    paired_quality_options.profile = astc_gpu_encoder_candidate_profile::neural_quality;
    paired_quality_options.reference_astcenc_quality = ASTCENC_PRE_THOROUGH;
    paired_quality_options.worker_count = 2;
    astc_gpu_d2_hybrid_result paired_quality_result;
    const astc_gpu_d2_activation_rank_request paired_rank_request{5, 3, 1,
                                                                     std::vector<float>(5, 1.0f)};
    require(astc_gpu_d2_finish_and_rank(
        paired_bank, paired_proposals, paired_rank_request, paired_quality_options,
        paired_quality_result, hybrid_error));
    require(paired_quality_result.finished_blocks.size() == 2);
    require(paired_quality_result.finished_blocks.front().source_block_id ==
            paired_bank.records.front().candidate_source_block_id);

    // Pair-map and Givens alternatives must rank in original logical-row
    // space. In particular, a candidate may use a different RGB layout than
    // the neutral source without changing the exact source target.
    std::vector<float> transformed_weights(10u * 8u);
    for (size_t index = 0; index < transformed_weights.size(); ++index) {
        transformed_weights[index] = 0.05f + 0.9f * static_cast<float>(index % 17u) / 16.0f;
    }
    const std::vector<uint8_t> pair_map{1, 0, 3, 2, 5, 4, 7, 6, 9, 8};
    std::vector<astc_vulkan_d2_pairing> transformed_pairings;
    require(astc_gpu_d2_expand_pair_map(astc_vulkan_footprint::k8x5, 10, 8,
                                         pair_map, transformed_pairings));
    require(transformed_pairings.size() == 1);
    std::vector<astc_vulkan_paired_layout> transformed_layouts;
    require(astc_gpu_d2_make_uniform_layout_map(astc_vulkan_footprint::k8x5,
                                                 10, 8,
                                                 astc_vulkan_paired_layout::rg_b,
                                                 transformed_layouts));
    std::vector<astc_gpu_encoder_source_block> transformed_neutral;
    require(astc_gpu_d2_build_paired_source_blocks(
        astc_vulkan_footprint::k8x5, transformed_weights, 10, 8,
        transformed_layouts, {}, astc_vulkan_paired_semantic::direct_rgb,
        transformed_neutral, transformed_pairings));
    std::vector<astc_vulkan_paired_layout> transformed_alternative_layouts = transformed_layouts;
    transformed_alternative_layouts[0] = astc_vulkan_paired_layout::r_gb;
    std::vector<astc_gpu_encoder_source_block> transformed_alternative;
    const astc_vulkan_d2_givens_transform givens{0.35f};
    require(astc_gpu_d2_build_paired_source_blocks(
        astc_vulkan_footprint::k8x5, transformed_weights, 10, 8,
        transformed_alternative_layouts, {}, astc_vulkan_paired_semantic::direct_rgb,
        transformed_alternative, transformed_pairings, givens));
    astc_gpu_d2_candidate_bank transformed_bank;
    require(astc_gpu_d2_build_candidate_bank(
        astc_vulkan_footprint::k8x5, transformed_neutral, transformed_layouts,
        {{astc_gpu_d2_candidate_family::direct_steered,
          astc_vulkan_paired_semantic::direct_rgb, transformed_alternative,
          transformed_alternative_layouts, givens}}, transformed_bank, transformed_pairings));
    std::vector<astc_gpu_encoder_finished_block> transformed_finished;
    for (size_t index = 0; index < transformed_bank.records.size(); ++index) {
        astc_gpu_encoder_finished_block block;
        block.source_block_id = transformed_bank.records[index].candidate_source_block_id;
        block.footprint = astc_vulkan_footprint::k8x5;
        block.payload[0] = static_cast<uint8_t>(index + 1u);
        for (const auto & texel : transformed_bank.candidate_blocks[index].texels) {
            block.decoded_rgba.insert(block.decoded_rgba.end(), texel.rgba.begin(), texel.rgba.end());
        }
        transformed_finished.push_back(std::move(block));
    }
    const astc_gpu_d2_activation_rank_request transformed_rank{8, 10, 1,
                                                                 std::vector<float>(8, 1.0f)};
    std::vector<astc_gpu_encoder_finished_block> transformed_ranked;
    std::vector<astc_gpu_d2_finished_candidate_score> transformed_scores;
    require(astc_gpu_d2_rank_finished_candidates_activation(
        transformed_bank, transformed_finished, transformed_rank, 2,
        transformed_ranked, transformed_scores));
    require(transformed_ranked.size() == 2 && transformed_scores.size() == 2);
    for (const auto & score : transformed_scores) require(score.activation_error < 1e-9);
    astc_gpu_d2_selector_delta_request transformed_selector_request;
    transformed_selector_request.tensor_width = 8;
    transformed_selector_request.tensor_height = 10;
    transformed_selector_request.source_blocks_x = 1;
    transformed_selector_request.calibration_activations.assign(8, 1.0);
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> transformed_selector;
    require(astc_gpu_d2_make_selector_candidates(
        transformed_bank, transformed_finished, transformed_selector_request,
        transformed_selector, selector_error));
    require(transformed_selector.size() == 1 && transformed_selector[0].size() == 2);
    for (const double delta : transformed_selector[0][1].calibration_delta) {
        require(std::fabs(delta) < 1e-6);
    }

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

    // The exact subset starts with a legal constant-color ASTC payload. It
    // is separate from the proposer and accepts only normalized sources.
    const std::vector<float> subset_weights{
        0.20f, 0.40f, 0.60f, 0.80f,
        0.10f, 0.30f, 0.50f, 0.70f,
        0.25f, 0.45f, 0.65f, 0.85f,
        0.05f, 0.15f, 0.35f, 0.55f};
    std::vector<astc_gpu_encoder_source_block> subset_source;
    require(astc_gpu_d1_build_scalar_source_blocks(
        astc_vulkan_footprint::k4x4, subset_weights, 4, 4, subset_source));
    astc_gpu_encoder_request subset_request;
    subset_request.mode = astc_gpu_encode_mode::exact_subset;
    subset_request.footprint = astc_vulkan_footprint::k4x4;
    subset_request.max_blocks_per_batch = 1;
    subset_request.blocks = subset_source;
    std::vector<astc_gpu_exact_subset_block> subset_blocks;
    require(astc_gpu_exact_subset_encode_cpu(subset_request, subset_blocks));
    require(subset_blocks.size() == 1);
    require(subset_blocks[0].payload[0] == 0xFC && subset_blocks[0].payload[1] == 0xFD);
    require(subset_blocks[0].unorm16_rgba[0] == subset_blocks[0].unorm16_rgba[1]);
    require(subset_blocks[0].unorm16_rgba[1] == subset_blocks[0].unorm16_rgba[2]);
    require(subset_blocks[0].unorm16_rgba[3] == 65535u);
    auto invalid_subset = subset_request;
    invalid_subset.blocks[0].texels[0].rgba[0] = -0.1f;
    require(!astc_gpu_exact_subset_encode_cpu(invalid_subset, subset_blocks));

    std::vector<astc_gpu_encoder_source_block> binary_source;
    require(astc_gpu_d1_build_scalar_source_blocks(
        astc_vulkan_footprint::k6x6,
        std::vector<float>(36, 0.25f), 6, 6, binary_source));
    for (uint32_t index = 0; index < binary_source[0].texels.size(); ++index) {
        binary_source[0].texels[index].rgba[0] = index & 1u ? 0.80f : 0.20f;
        binary_source[0].texels[index].rgba[1] = binary_source[0].texels[index].rgba[0];
        binary_source[0].texels[index].rgba[2] = binary_source[0].texels[index].rgba[0];
    }
    auto binary_request = subset_request;
    binary_request.footprint = astc_vulkan_footprint::k6x6;
    binary_request.exact_subset = astc_gpu_exact_subset_kind::d1_luminance_binary_6x6;
    binary_request.blocks = binary_source;
    require(astc_gpu_exact_subset_encode_cpu(binary_request, subset_blocks));
    require(subset_blocks.size() == 1);
    require(subset_blocks[0].payload[0] == 0x04u && subset_blocks[0].payload[1] == 0x01u);
    require(subset_blocks[0].payload[0] != 0xFCu);

    std::vector<float> subset_d2_weights(10 * 8, 0.5f);
    for (uint32_t row = 0; row < 10; ++row) for (uint32_t column = 0; column < 8; ++column) {
        subset_d2_weights[row * 8 + column] = 0.05f + 0.90f * float((row * 5u + column * 3u) % 17u) / 16.0f;
    }
    std::vector<astc_vulkan_paired_layout> subset_d2_layouts;
    std::vector<astc_gpu_encoder_source_block> subset_d2_source;
    require(astc_gpu_d2_make_uniform_layout_map(
        astc_vulkan_footprint::k8x5, 10, 8, astc_vulkan_paired_layout::rg_b, subset_d2_layouts));
    require(astc_gpu_d2_build_paired_source_blocks(
        astc_vulkan_footprint::k8x5, subset_d2_weights, 10, 8, subset_d2_layouts, {},
        astc_vulkan_paired_semantic::luminance_alpha, subset_d2_source));
    auto subset_d2_request = subset_request;
    subset_d2_request.footprint = astc_vulkan_footprint::k8x5;
    subset_d2_request.blocks = subset_d2_source;
    subset_d2_request.exact_subset = astc_gpu_exact_subset_kind::luminance_alpha_binary_8x5;
    require(astc_gpu_exact_subset_encode_cpu(subset_d2_request, subset_blocks));
    require(subset_blocks.size() == 1 && subset_blocks[0].payload[0] == 0x65u);
    subset_d2_request.exact_subset = astc_gpu_exact_subset_kind::luminance_alpha_dual_binary_8x5;
    require(astc_gpu_exact_subset_encode_cpu(subset_d2_request, subset_blocks));
    require(subset_blocks.size() == 1 && subset_blocks[0].payload[0] == 0xc1u &&
            (subset_blocks[0].payload[1] & 0x04u) != 0u);

    astc_gpu_d2_exact_subset_bank subset_d2_bank;
    require(astc_gpu_d2_build_luminance_alpha_exact_subset_bank(
        subset_d2_source, 3, subset_d2_bank));
    require(subset_d2_bank.one_plane.blocks.size() == subset_d2_source.size());
    require(subset_d2_bank.one_plane.blocks[0].source_block_id == subset_d2_source[0].source_block_id);
    require(subset_d2_bank.one_plane.exact_subset ==
            astc_gpu_exact_subset_kind::luminance_alpha_binary_8x5 &&
            subset_d2_bank.one_plane_luminance_weights.exact_subset ==
            astc_gpu_exact_subset_kind::luminance_alpha_binary_luminance_weights_8x5 &&
            subset_d2_bank.one_plane_alpha_weights.exact_subset ==
            astc_gpu_exact_subset_kind::luminance_alpha_binary_alpha_weights_8x5 &&
            subset_d2_bank.one_plane_refined.exact_subset ==
            astc_gpu_exact_subset_kind::luminance_alpha_binary_refined_8x5 &&
            subset_d2_bank.one_plane_mean_refined.exact_subset ==
            astc_gpu_exact_subset_kind::luminance_alpha_binary_mean_refined_8x5 &&
            subset_d2_bank.one_plane_quantile_refined.exact_subset ==
            astc_gpu_exact_subset_kind::luminance_alpha_binary_quantile_refined_8x5 &&
            subset_d2_bank.alpha_dual_plane.exact_subset ==
            astc_gpu_exact_subset_kind::luminance_alpha_dual_binary_8x5);
    std::vector<astc_gpu_exact_subset_block> balanced_d2_blocks;
    std::vector<astc_gpu_exact_subset_block> luminance_d2_blocks;
    std::vector<astc_gpu_exact_subset_block> alpha_d2_blocks;
    require(astc_gpu_exact_subset_encode_cpu(subset_d2_bank.one_plane, balanced_d2_blocks));
    require(astc_gpu_exact_subset_encode_cpu(
        subset_d2_bank.one_plane_luminance_weights, luminance_d2_blocks));
    require(astc_gpu_exact_subset_encode_cpu(
        subset_d2_bank.one_plane_alpha_weights, alpha_d2_blocks));
    require(astc_gpu_exact_subset_encode_cpu(
        subset_d2_bank.one_plane_refined, subset_blocks));
    require(balanced_d2_blocks.size() == 1 && luminance_d2_blocks.size() == 1 &&
            alpha_d2_blocks.size() == 1);
    require(balanced_d2_blocks[0].payload != luminance_d2_blocks[0].payload ||
            balanced_d2_blocks[0].payload != alpha_d2_blocks[0].payload ||
            luminance_d2_blocks[0].payload != alpha_d2_blocks[0].payload);
    std::vector<astc_gpu_encoder_finished_block> decoded_d2_blocks;
    std::string decoded_d2_error;
    require(astc_gpu_exact_subset_finish_payloads(
        astc_vulkan_footprint::k8x5, balanced_d2_blocks,
        decoded_d2_blocks, decoded_d2_error));
    require(decoded_d2_blocks.size() == balanced_d2_blocks.size() &&
            decoded_d2_blocks[0].payload == balanced_d2_blocks[0].payload &&
            decoded_d2_blocks[0].decoded_rgba.size() == 8u * 5u * 4u);
    require(astc_gpu_exact_subset_encode_cpu(subset_d2_bank.alpha_dual_plane, subset_blocks));

    astc_gpu_d2_exact_subset_candidate_bank selector_d2_bank;
    require(astc_gpu_d2_build_luminance_alpha_exact_subset_candidate_bank(
        subset_d2_source, subset_d2_layouts, {}, 3, selector_d2_bank));
    require(selector_d2_bank.semantic_bank.records.size() == 5 &&
            selector_d2_bank.physical_bank.one_plane.blocks[0].source_block_id == 0 &&
            selector_d2_bank.physical_bank.one_plane_luminance_weights.blocks[0].source_block_id == 1 &&
            selector_d2_bank.physical_bank.one_plane_alpha_weights.blocks[0].source_block_id == 2 &&
            selector_d2_bank.physical_bank.one_plane_refined.blocks[0].source_block_id == 3 &&
            selector_d2_bank.physical_bank.alpha_dual_plane.blocks[0].source_block_id == 4);
    std::vector<astc_gpu_encoder_finished_block> selector_d2_finished;
    const auto append_exact_finished = [&](const astc_gpu_encoder_request & request) {
        std::vector<astc_gpu_exact_subset_block> payloads;
        std::vector<astc_gpu_encoder_finished_block> decoded;
        return astc_gpu_exact_subset_encode_cpu(request, payloads) &&
            astc_gpu_exact_subset_finish_payloads(request.footprint, payloads, decoded, decoded_d2_error) &&
            (selector_d2_finished.insert(selector_d2_finished.end(), decoded.begin(), decoded.end()), true);
    };
    require(append_exact_finished(selector_d2_bank.physical_bank.one_plane) &&
            append_exact_finished(selector_d2_bank.physical_bank.one_plane_luminance_weights) &&
            append_exact_finished(selector_d2_bank.physical_bank.one_plane_alpha_weights) &&
            append_exact_finished(selector_d2_bank.physical_bank.one_plane_refined) &&
            append_exact_finished(selector_d2_bank.physical_bank.alpha_dual_plane));
    astc_gpu_d2_selector_delta_request exact_d2_selector_request;
    exact_d2_selector_request.tensor_width = 8;
    exact_d2_selector_request.tensor_height = 10;
    exact_d2_selector_request.source_blocks_x = 1;
    exact_d2_selector_request.calibration_activations.assign(8, 1.0);
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> exact_d2_selector_candidates;
    require(astc_gpu_d2_make_selector_candidates(
        selector_d2_bank.semantic_bank, selector_d2_finished,
        exact_d2_selector_request, exact_d2_selector_candidates, decoded_d2_error));
    require(exact_d2_selector_candidates.size() == 1 &&
            exact_d2_selector_candidates[0].size() == 5 &&
            std::all_of(exact_d2_selector_candidates[0][0].calibration_delta.begin(),
                        exact_d2_selector_candidates[0][0].calibration_delta.end(),
                        [](double value) { return value == 0.0; }));

    request.blocks[0].texels.pop_back();
    require(!astc_gpu_encoder_plan_batches(request, batches));
    return 0;
}
