#include "astc-gpu-d1-source.h"
#include "astc-gpu-encoder-exact-dispatch.h"
#include "astc-gpu-encoder-finisher.h"
#include "astc-gpu-encoder-subset.h"

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

bool load_normalized_prefix(const std::string & path, uint32_t rows, uint32_t columns,
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

bool measure_source(const std::vector<astc_gpu_encoder_source_block> & source,
                    const std::vector<astc_gpu_encoder_finished_block> & decoded,
                    Stats & stats) {
    if (source.size() != decoded.size() || source.empty()) return false;
    std::vector<double> block_errors;
    block_errors.reserve(source.size());
    std::unordered_set<std::string> unique_payloads;
    double total = 0.0;
    float max_abs = 0.0f;
    for (size_t block = 0; block < source.size(); ++block) {
        if (decoded[block].decoded_rgba.size() != source[block].texels.size() * 4u) return false;
        unique_payloads.insert(payload_key(decoded[block].payload));
        double block_error = 0.0;
        for (size_t texel = 0; texel < source[block].texels.size(); ++texel) {
            const float delta = decoded[block].decoded_rgba[texel * 4u] -
                                source[block].texels[texel].rgba[0];
            block_error += double(delta) * delta;
            total += double(delta) * delta;
            max_abs = std::max(max_abs, std::fabs(delta));
        }
        block_errors.push_back(block_error / double(source[block].texels.size()));
    }
    std::sort(block_errors.begin(), block_errors.end());
    stats.mse = total / double(source.size() * source.front().texels.size());
    stats.p50 = block_errors[block_errors.size() / 2u];
    stats.p95 = block_errors[(block_errors.size() * 95u) / 100u];
    stats.max_abs = max_abs;
    stats.unique_payloads = unique_payloads.size();
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::cerr << "usage: d1-fixture-smoke weights.f32 rows columns minmax.spv mean.spv quantile.spv max-blocks\n";
        return 2;
    }
    const uint32_t rows = static_cast<uint32_t>(std::stoul(argv[2]));
    const uint32_t columns = static_cast<uint32_t>(std::stoul(argv[3]));
    const size_t max_blocks = static_cast<size_t>(std::stoul(argv[7]));
    if (rows == 0 || columns == 0 || max_blocks == 0) return 2;

    std::vector<float> values;
    float source_min = 0.0f, source_max = 0.0f;
    if (!load_normalized_prefix(argv[1], rows, columns, values, source_min, source_max)) {
        std::cerr << "failed to load finite F32 fixture prefix\n";
        return 1;
    }
    std::vector<astc_gpu_encoder_source_block> source;
    if (!astc_gpu_d1_build_scalar_source_blocks(
            astc_vulkan_footprint::k6x6, values, rows, columns, source)) {
        std::cerr << "failed to build D1 scalar source\n";
        return 1;
    }
    if (source.size() > max_blocks) source.resize(max_blocks);

    astc_gpu_encoder_request request;
    request.mode = astc_gpu_encode_mode::propose;
    request.footprint = astc_vulkan_footprint::k6x6;
    request.max_blocks_per_batch = 256;
    request.blocks = source;

    std::vector<astc_gpu_encoder_proposal> proposals;
    std::vector<astc_gpu_encoder_finished_block> thorough;
    std::string error;
    if (!astc_gpu_encoder_propose_cpu_reference(request, proposals) ||
        !astc_gpu_encoder_finish_with_options(
            request, proposals,
            {ASTCENC_PRE_THOROUGH, astc_gpu_encoder_finish_mode::reference, 4},
            thorough, error)) {
        std::cerr << "CPU astcenc thorough failed: " << error << "\n";
        return 1;
    }

    astc_gpu_encoder_request baseline_request = request;
    baseline_request.mode = astc_gpu_encode_mode::exact_subset;
    baseline_request.exact_subset = astc_gpu_exact_subset_kind::d1_luminance_binary_6x6;
    std::vector<astc_gpu_exact_subset_block> baseline_cpu;
    if (!astc_gpu_exact_subset_encode_cpu(baseline_request, baseline_cpu)) {
        std::cerr << "baseline exact subset failed\n";
        return 1;
    }
    std::vector<astc_gpu_encoder_finished_block> baseline;
    if (!astc_gpu_exact_subset_finish_payloads(
            astc_vulkan_footprint::k6x6, baseline_cpu, baseline, error)) {
        std::cerr << "baseline exact decode failed: " << error << "\n";
        return 1;
    }

    const auto run_gpu_variant = [&](astc_gpu_exact_subset_kind kind, const char * shader,
                                     std::vector<astc_gpu_encoder_finished_block> & decoded) {
        astc_gpu_encoder_request variant_request = request;
        variant_request.mode = astc_gpu_encode_mode::exact_subset;
        variant_request.exact_subset = kind;
        std::vector<astc_gpu_exact_subset_block> cpu, gpu;
        if (!astc_gpu_exact_subset_encode_cpu(variant_request, cpu) ||
            !astc_gpu_exact_subset_encode_gpu_default(shader, variant_request, gpu, error) ||
            cpu.size() != gpu.size()) return false;
        for (size_t index = 0; index < cpu.size(); ++index) if (cpu[index].payload != gpu[index].payload) return false;
        return astc_gpu_exact_subset_finish_payloads(
            astc_vulkan_footprint::k6x6, gpu, decoded, error);
    };
    std::vector<astc_gpu_encoder_finished_block> refined, mean_refined, quantile_refined;
    if (!run_gpu_variant(astc_gpu_exact_subset_kind::d1_luminance_binary_refined_6x6,
                         argv[4], refined) ||
        !run_gpu_variant(astc_gpu_exact_subset_kind::d1_luminance_binary_mean_refined_6x6,
                         argv[5], mean_refined) ||
        !run_gpu_variant(astc_gpu_exact_subset_kind::d1_luminance_binary_quantile_refined_6x6,
                         argv[6], quantile_refined)) {
        std::cerr << "refined GPU subset failed: " << error << "\n";
        return 1;
    }

    Stats baseline_stats, refined_stats, mean_stats, quantile_stats, thorough_stats;
    if (!measure_source(source, baseline, baseline_stats) ||
        !measure_source(source, refined, refined_stats) ||
        !measure_source(source, mean_refined, mean_stats) ||
        !measure_source(source, quantile_refined, quantile_stats) ||
        !measure_source(source, thorough, thorough_stats)) {
        std::cerr << "source error measurement failed\n";
        return 1;
    }
    const std::array<const std::vector<astc_gpu_encoder_finished_block> *, 4> speed_bank{
        &baseline, &refined, &mean_refined, &quantile_refined};
    std::array<size_t, 4> speed_wins{};
    double speed_bank_total = 0.0;
    for (size_t block = 0; block < source.size(); ++block) {
        double best_error = std::numeric_limits<double>::infinity();
        size_t best_candidate = 0;
        for (size_t candidate = 0; candidate < speed_bank.size(); ++candidate) {
            double candidate_error = 0.0;
            for (size_t texel = 0; texel < source[block].texels.size(); ++texel) {
                const float delta = (*speed_bank[candidate])[block].decoded_rgba[texel * 4u] -
                                    source[block].texels[texel].rgba[0];
                candidate_error += double(delta) * delta;
            }
            if (candidate_error < best_error) {
                best_error = candidate_error;
                best_candidate = candidate;
            }
        }
        speed_bank_total += best_error;
        ++speed_wins[best_candidate];
    }
    const double speed_bank_mse = speed_bank_total / double(source.size() * source.front().texels.size());
    std::cout << std::fixed << std::setprecision(8)
              << "d1-fixture footprint=6x6 rows=" << rows << " columns=" << columns
              << " blocks=" << source.size() << " source_range=" << source_min << "," << source_max << "\n"
              << "binary_cpu legal=1 mse=" << baseline_stats.mse
              << " block_p50=" << baseline_stats.p50 << " block_p95=" << baseline_stats.p95
              << " max_abs=" << baseline_stats.max_abs
              << " unique_payloads=" << baseline_stats.unique_payloads << "\n"
              << "refined_gpu legal=1 cpu_gpu_payload_equal=1 mse=" << refined_stats.mse
              << " block_p50=" << refined_stats.p50 << " block_p95=" << refined_stats.p95
              << " max_abs=" << refined_stats.max_abs
              << " unique_payloads=" << refined_stats.unique_payloads << "\n"
              << "mean_refined_gpu legal=1 cpu_gpu_payload_equal=1 mse=" << mean_stats.mse
              << " block_p50=" << mean_stats.p50 << " block_p95=" << mean_stats.p95
              << " max_abs=" << mean_stats.max_abs
              << " unique_payloads=" << mean_stats.unique_payloads << "\n"
              << "quantile_refined_gpu legal=1 cpu_gpu_payload_equal=1 mse=" << quantile_stats.mse
              << " block_p50=" << quantile_stats.p50 << " block_p95=" << quantile_stats.p95
              << " max_abs=" << quantile_stats.max_abs
              << " unique_payloads=" << quantile_stats.unique_payloads << "\n"
              << "astcenc_thorough legal=1 mse=" << thorough_stats.mse
              << " block_p50=" << thorough_stats.p50 << " block_p95=" << thorough_stats.p95
              << " max_abs=" << thorough_stats.max_abs
              << " unique_payloads=" << thorough_stats.unique_payloads << "\n"
              << "refined_vs_thorough_mse_ratio="
              << (thorough_stats.mse > 0.0 ? refined_stats.mse / thorough_stats.mse : 0.0)
              << " refined_vs_binary_mse_ratio="
              << (baseline_stats.mse > 0.0 ? refined_stats.mse / baseline_stats.mse : 0.0)
              << " refined_not_worse=" << (refined_stats.mse <= thorough_stats.mse + 1e-12 ? 1 : 0) << "\n"
              << "speed_bank_source_oracle_mse=" << speed_bank_mse
              << " wins=binary:" << speed_wins[0]
              << ",minmax:" << speed_wins[1]
              << ",mean:" << speed_wins[2]
              << ",quantile:" << speed_wins[3]
              << "\n";
    return 0;
}
