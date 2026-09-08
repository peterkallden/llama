#include "astc-gpu-d2-exact-subset.h"
#include "astc-gpu-d2-source.h"
#include "astc-gpu-encoder-exact-dispatch.h"
#include "astc-gpu-encoder-finisher.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct Stats {
    double mse = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    float max_abs = 0.0f;
    size_t unique_payloads = 0;
    size_t best_blocks = 0;
};

std::string payload_key(const std::array<uint8_t, 16> & payload) {
    return std::string(reinterpret_cast<const char *>(payload.data()), payload.size());
}

bool load_normalized(const std::string & path, uint32_t rows, uint32_t columns,
                     std::vector<float> & values, float & source_min, float & source_max) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    values.resize(size_t(rows) * columns);
    if (!input.read(reinterpret_cast<char *>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float)))) return false;
    source_min = std::numeric_limits<float>::infinity();
    source_max = -std::numeric_limits<float>::infinity();
    for (const float value : values) {
        if (!std::isfinite(value)) return false;
        source_min = std::min(source_min, value);
        source_max = std::max(source_max, value);
    }
    const float range = source_max - source_min;
    for (float & value : values) value = range > 0.0f ? (value - source_min) / range : 0.5f;
    return true;
}

bool measure_candidate(const astc_gpu_encoder_request & request,
                       const std::vector<astc_gpu_exact_subset_block> & payloads,
                       const std::vector<astc_gpu_encoder_finished_block> & decoded,
                       Stats & stats) {
    if (payloads.size() != request.blocks.size() || decoded.size() != request.blocks.size()) return false;
    std::vector<double> block_errors;
    block_errors.reserve(decoded.size());
    double total = 0.0;
    float max_abs = 0.0f;
    std::unordered_set<std::string> unique;
    for (size_t block = 0; block < decoded.size(); ++block) {
        unique.insert(payload_key(payloads[block].payload));
        double block_error = 0.0;
        const auto & source = request.blocks[block].texels;
        if (decoded[block].decoded_rgba.size() != source.size() * 4u) return false;
        for (size_t texel = 0; texel < source.size(); ++texel) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                const float delta = decoded[block].decoded_rgba[texel * 4u + channel] -
                    source[texel].rgba[channel];
                const double squared = double(delta) * delta;
                total += squared;
                block_error += squared;
                max_abs = std::max(max_abs, std::fabs(delta));
            }
        }
        block_errors.push_back(block_error / double(source.size() * 4u));
    }
    std::sort(block_errors.begin(), block_errors.end());
    stats.mse = total / double(decoded.size() * request.blocks[0].texels.size() * 4u);
    stats.p50 = block_errors[block_errors.size() / 2u];
    stats.p95 = block_errors[(block_errors.size() * 95u) / 100u];
    stats.max_abs = max_abs;
    stats.unique_payloads = unique.size();
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 9) {
        std::cerr << "usage: fixture-smoke weights.f32 rows columns balanced.spv luminance.spv alpha.spv refined.spv dual.spv\n";
        return 2;
    }
    const uint32_t rows = static_cast<uint32_t>(std::stoul(argv[2]));
    const uint32_t columns = static_cast<uint32_t>(std::stoul(argv[3]));
    if (rows == 0 || columns == 0 || rows % 10u != 0 || columns % 8u != 0) {
        std::cerr << "rows must be a multiple of 10 and columns a multiple of 8\n";
        return 2;
    }
    std::vector<float> weights;
    float source_min = 0.0f, source_max = 0.0f;
    if (!load_normalized(argv[1], rows, columns, weights, source_min, source_max)) {
        std::cerr << "failed to load finite F32 fixture\n";
        return 1;
    }
    std::vector<astc_vulkan_paired_layout> layouts;
    std::vector<astc_gpu_encoder_source_block> source;
    if (!astc_gpu_d2_make_uniform_layout_map(
            astc_vulkan_footprint::k8x5, rows, columns,
            astc_vulkan_paired_layout::rg_b, layouts) ||
        !astc_gpu_d2_build_paired_source_blocks(
            astc_vulkan_footprint::k8x5, weights, rows, columns, layouts, {},
            astc_vulkan_paired_semantic::luminance_alpha, source)) {
        std::cerr << "failed to build D2-LA source fixture\n";
        return 1;
    }
    astc_gpu_d2_exact_subset_bank bank;
    if (!astc_gpu_d2_build_luminance_alpha_exact_subset_bank(source, 2048, bank)) {
        std::cerr << "failed to build D2 exact candidate bank\n";
        return 1;
    }
    const std::array<const char *, 5> shaders{argv[4], argv[5], argv[6], argv[7], argv[8]};
    const std::array<const astc_gpu_encoder_request *, 5> requests{
        &bank.one_plane, &bank.one_plane_luminance_weights,
        &bank.one_plane_alpha_weights, &bank.one_plane_refined, &bank.alpha_dual_plane};
    const std::array<const char *, 6> names{
        "balanced", "luminance", "alpha", "refined", "dual", "cpu_astcenc_thorough"};
    std::array<Stats, 6> stats{};
    std::vector<std::vector<astc_gpu_exact_subset_block>> payloads(6);
    std::vector<std::vector<astc_gpu_encoder_finished_block>> decoded(6);
    for (size_t candidate = 0; candidate < requests.size(); ++candidate) {
        std::string error;
        if (!astc_gpu_exact_subset_encode_gpu_default(
                shaders[candidate], *requests[candidate], payloads[candidate], error) ||
            !astc_gpu_exact_subset_finish_payloads(
                astc_vulkan_footprint::k8x5, payloads[candidate], decoded[candidate], error) ||
            !measure_candidate(*requests[candidate], payloads[candidate], decoded[candidate], stats[candidate])) {
            std::cerr << names[candidate] << " failed: " << error << "\n";
            return 1;
        }
    }
    // CPU astcenc is the quality reference for the same physical source.
    astc_gpu_encoder_request cpu_request;
    cpu_request.mode = astc_gpu_encode_mode::propose;
    cpu_request.footprint = astc_vulkan_footprint::k8x5;
    cpu_request.max_blocks_per_batch = 2048;
    cpu_request.blocks = source;
    std::vector<astc_gpu_encoder_proposal> cpu_proposals;
    std::vector<astc_gpu_encoder_finished_block> cpu_finished;
    std::string cpu_error;
    if (!astc_gpu_encoder_propose_cpu_reference(cpu_request, cpu_proposals) ||
        !astc_gpu_encoder_finish_with_options(
            cpu_request, cpu_proposals,
            astc_gpu_encoder_finish_options{
                ASTCENC_PRE_THOROUGH, astc_gpu_encoder_finish_mode::reference, 4},
            cpu_finished, cpu_error) || cpu_finished.size() != source.size()) {
        std::cerr << "CPU astcenc reference failed: " << cpu_error << "\n";
        return 1;
    }
    payloads[5].reserve(cpu_finished.size());
    for (const auto & block : cpu_finished) {
        astc_gpu_exact_subset_block payload;
        payload.source_block_id = block.source_block_id;
        payload.payload = block.payload;
        payloads[5].push_back(payload);
    }
    decoded[5] = std::move(cpu_finished);
    if (!measure_candidate(cpu_request, payloads[5], decoded[5], stats[5])) {
        std::cerr << "CPU astcenc reference error measurement failed\n";
        return 1;
    }
    const std::array<const astc_gpu_encoder_request *, 6> all_requests{
        requests[0], requests[1], requests[2], requests[3], requests[4], &cpu_request};
    std::vector<size_t> winners(source.size(), 0);
    std::array<size_t, 5> bank_wins{};
    double bank_total_error = 0.0;
    for (size_t block = 0; block < source.size(); ++block) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t candidate = 0; candidate < stats.size(); ++candidate) {
            double error = 0.0;
            const auto & texels = all_requests[candidate]->blocks[block].texels;
            for (size_t texel = 0; texel < texels.size(); ++texel) for (uint32_t channel = 0; channel < 4; ++channel) {
                const float delta = decoded[candidate][block].decoded_rgba[texel * 4u + channel] -
                    texels[texel].rgba[channel];
                error += double(delta) * delta;
            }
            if (error < best) { best = error; winners[block] = candidate; }
        }
        ++stats[winners[block]].best_blocks;
        double bank_best = std::numeric_limits<double>::infinity();
        size_t bank_winner = 0;
        for (size_t candidate = 0; candidate < requests.size(); ++candidate) {
            double error = 0.0;
            const auto & texels = all_requests[candidate]->blocks[block].texels;
            for (size_t texel = 0; texel < texels.size(); ++texel) for (uint32_t channel = 0; channel < 4; ++channel) {
                const float delta = decoded[candidate][block].decoded_rgba[texel * 4u + channel] - texels[texel].rgba[channel];
                error += double(delta) * delta;
            }
            if (error < bank_best) { bank_best = error; bank_winner = candidate; }
        }
        ++bank_wins[bank_winner];
        bank_total_error += bank_best;
    }
    std::cout << std::fixed << std::setprecision(8)
              << "source_range=" << source_min << "," << source_max
              << " rows=" << rows << " columns=" << columns
              << " blocks=" << source.size() << "\n";
    for (size_t candidate = 0; candidate < stats.size(); ++candidate) {
        std::cout << names[candidate]
                  << " mse=" << stats[candidate].mse
                  << " block_p50=" << stats[candidate].p50
                  << " block_p95=" << stats[candidate].p95
                  << " max_abs=" << stats[candidate].max_abs
                  << " unique_payloads=" << stats[candidate].unique_payloads
                  << " best_blocks=" << stats[candidate].best_blocks << "\n";
    }
    std::cout << "speed_bank_source_oracle_mse=" << bank_total_error / double(source.size() * 40u * 4u)
              << " wins=balanced:" << bank_wins[0] << ",luminance:" << bank_wins[1]
              << ",alpha:" << bank_wins[2] << ",refined:" << bank_wins[3]
              << ",dual:" << bank_wins[4] << "\n";
    return 0;
}
