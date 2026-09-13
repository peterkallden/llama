// Bounded real-source discovery for token-local E1/E2 embedding layouts.
//
// This tool deliberately remains offline-only. It loads token_embd.weight via
// the public llama model API, encodes a deterministic sample of token rows
// with the CPU ASTC oracle, decodes the exact payloads again, and reports the
// resulting embedding error and storage geometry. With a cache directory it
// writes a validated token-local E1 annex; it is still not a runtime
// get_rows provider and does not modify the binary ASTC manifest.

#include "astc-vulkan-embedding-layout.h"

#include <astcenc.h>

#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct profile_runner {
    astc_vulkan_embedding_layout layout;
    astcenc_context * context = nullptr;
    astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                              ASTCENC_SWZ_B, ASTCENC_SWZ_A };
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t lanes = 0;
};

struct profile_stats {
    uint64_t rows = 0;
    uint64_t blocks = 0;
    uint64_t legal_blocks = 0;
    double squared_error = 0.0;
    double reference_energy = 0.0;
    float max_error = 0.0f;
    float min_scale = std::numeric_limits<float>::infinity();
    float max_scale = 0.0f;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s MODEL.gguf [sample_tokens] [cache_dir]\n"
        "  sample_tokens defaults to 64; use 0 for all vocabulary rows\n"
        "  cache_dir writes a complete token-local E1 10x5 payload into that cache\n",
        argv0);
}

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

bool make_runner(astc_vulkan_embedding_representation representation,
                 astc_vulkan_footprint footprint,
                 profile_runner & runner,
                 std::string & error) {
    if (!astc_vulkan_embedding_make_profile(representation, footprint,
                                             runner.layout.profile, error)) {
        return false;
    }
    runner.layout.dimensions = 1; // Replaced after model dimensions are known.
    runner.lanes = runner.layout.profile.lanes_per_texel;
    runner.width = astc_vulkan_format(footprint).block_width;
    runner.height = astc_vulkan_format(footprint).block_height;

    astcenc_config config{};
    const astcenc_error configured = astcenc_config_init(
        ASTCENC_PRF_LDR, runner.width, runner.height, 1,
        ASTCENC_PRE_FAST, 0, &config);
    if (configured != ASTCENC_SUCCESS) {
        error = std::string("astcenc config failed: ") +
                astcenc_get_error_string(configured);
        return false;
    }
    // E1 duplicates its semantic lane in RGB(A). E2 uses RGB as lane 0 and
    // alpha as lane 1, so the aggregate image-side weight is balanced.
    if (runner.lanes == 2) {
        config.cw_r_weight = 1.0f / 3.0f;
        config.cw_g_weight = 1.0f / 3.0f;
        config.cw_b_weight = 1.0f / 3.0f;
        config.cw_a_weight = 1.0f;
    }
    // Keep the oracle context single-threaded.  The fast ASTC search is
    // intentionally the same deterministic quality profile used by the
    // bounded discovery gate; a future sharded implementation must use one
    // independent context per shard rather than changing this oracle's
    // quality characteristics.
    const astcenc_error allocated = astcenc_context_alloc(&config, 1, &runner.context);
    if (allocated != ASTCENC_SUCCESS) {
        error = std::string("astcenc context allocation failed: ") +
                astcenc_get_error_string(allocated);
        return false;
    }
    return true;
}

void destroy_runner(profile_runner & runner) {
    if (runner.context != nullptr) {
        astcenc_context_free(runner.context);
        runner.context = nullptr;
    }
}

bool encode_row(const profile_runner & runner,
                const float * source_row,
                uint32_t dimensions,
                profile_stats & stats,
                std::string & error,
                std::ofstream * payload_output = nullptr,
                std::ofstream * affine_output = nullptr) {
    astc_vulkan_embedding_affine affine;
    std::vector<float> row(source_row, source_row + dimensions);
    affine = astc_vulkan_embedding_affine_from_row(row);
    stats.min_scale = std::min(stats.min_scale, affine.scale);
    stats.max_scale = std::max(stats.max_scale, affine.scale);

    astc_vulkan_embedding_layout layout = runner.layout;
    layout.dimensions = dimensions;
    layout.logical_to_physical.resize(dimensions);
    for (uint32_t i = 0; i < dimensions; ++i) layout.logical_to_physical[i] = i;
    if (!astc_vulkan_embedding_validate_layout(layout, error)) return false;

    std::vector<float> image_values(
        static_cast<size_t>(astc_vulkan_embedding_tile_count(layout)) * runner.width *
            runner.height * 4, 0.0f);
    const uint32_t tile_count = astc_vulkan_embedding_tile_count(layout);
    const uint32_t values_per_tile = astc_vulkan_embedding_values_per_tile(layout.profile);
    for (uint32_t logical = 0; logical < dimensions; ++logical) {
        const uint32_t physical = layout.logical_to_physical[logical];
        const uint32_t in_tile = physical % values_per_tile;
        const uint32_t tile = physical / values_per_tile;
        const uint32_t texel = in_tile / runner.lanes;
        const uint32_t lane = in_tile % runner.lanes;
        const uint32_t x = texel % runner.width;
        const uint32_t y = texel / runner.width;
        const float normalized = clamp01(0.5f + 0.5f *
            ((row[logical] - affine.bias) / affine.scale));
        const size_t pixel = (static_cast<size_t>(y) * tile_count * runner.width +
                              static_cast<size_t>(tile) * runner.width + x) * 4;
        if (runner.lanes == 1) {
            image_values[pixel + 0] = normalized;
            image_values[pixel + 1] = normalized;
            image_values[pixel + 2] = normalized;
            image_values[pixel + 3] = normalized;
        } else if (lane == 0) {
            image_values[pixel + 0] = normalized;
            image_values[pixel + 1] = normalized;
            image_values[pixel + 2] = normalized;
        } else {
            image_values[pixel + 3] = normalized;
        }
    }

    void * input_slice = image_values.data();
    astcenc_image input{ tile_count * runner.width, runner.height, 1,
                         ASTCENC_TYPE_F32, &input_slice };
    std::vector<uint8_t> payload(static_cast<size_t>(tile_count) * 16);
    const astcenc_error encoded = astcenc_compress_image(
        runner.context, &input, &runner.swizzle, payload.data(), payload.size(), 0);
    if (encoded != ASTCENC_SUCCESS) {
        error = std::string("ASTC embedding compression failed: ") +
                astcenc_get_error_string(encoded);
        return false;
    }
    std::vector<float> decoded(image_values.size(), 0.0f);
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{ tile_count * runner.width, runner.height, 1,
                                 ASTCENC_TYPE_F32, &decoded_slice };
    const astcenc_error decompressed = astcenc_decompress_image(
        runner.context, payload.data(), payload.size(), &decoded_image,
        &runner.swizzle, 0);
    if (decompressed != ASTCENC_SUCCESS) {
        error = std::string("ASTC embedding decompression failed: ") +
                astcenc_get_error_string(decompressed);
        return false;
    }

    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        astcenc_block_info info{};
        const astcenc_error inspected = astcenc_get_block_info(
            runner.context, payload.data() + static_cast<size_t>(tile) * 16, &info);
        if (inspected != ASTCENC_SUCCESS) {
            error = std::string("ASTC embedding block inspection failed: ") +
                    astcenc_get_error_string(inspected);
            return false;
        }
        ++stats.legal_blocks;
    }

    for (uint32_t logical = 0; logical < dimensions; ++logical) {
        const uint32_t physical = layout.logical_to_physical[logical];
        const uint32_t in_tile = physical % values_per_tile;
        const uint32_t tile = physical / values_per_tile;
        const uint32_t texel = in_tile / runner.lanes;
        const uint32_t lane = in_tile % runner.lanes;
        const uint32_t x = texel % runner.width;
        const uint32_t y = texel / runner.width;
        const size_t pixel = (static_cast<size_t>(y) * tile_count * runner.width +
                              static_cast<size_t>(tile) * runner.width + x) * 4;
        const float decoded_normalized = runner.lanes == 1 || lane == 0
            ? decoded[pixel + 0] : decoded[pixel + 3];
        const float reconstructed = affine.bias + affine.scale *
            (2.0f * decoded_normalized - 1.0f);
        const float delta = row[logical] - reconstructed;
        stats.squared_error += static_cast<double>(delta) * delta;
        stats.reference_energy += static_cast<double>(row[logical]) * row[logical];
        stats.max_error = std::max(stats.max_error, std::fabs(delta));
    }
    ++stats.rows;
    stats.blocks += tile_count;
    if (payload_output != nullptr) {
        payload_output->write(reinterpret_cast<const char *>(payload.data()),
                              static_cast<std::streamsize>(payload.size()));
        if (affine_output != nullptr) {
            affine_output->write(reinterpret_cast<const char *>(&affine.bias), sizeof(affine.bias));
            affine_output->write(reinterpret_cast<const char *>(&affine.scale), sizeof(affine.scale));
        }
        if (!payload_output->good() || (affine_output != nullptr && !affine_output->good())) {
            error = "cannot write embedding cache payload";
            return false;
        }
    }
    return true;
}

void print_stats(const char * name, const profile_runner & runner,
                 const profile_stats & stats, uint32_t dimensions) {
    const double values = static_cast<double>(std::max<uint64_t>(1, stats.rows)) * dimensions;
    const double mse = stats.squared_error / values;
    const double relative = stats.squared_error / std::max(stats.reference_energy, 1e-12);
    const double bpw = 128.0 / (static_cast<double>(runner.width) * runner.height *
                                runner.layout.profile.lanes_per_texel);
    std::printf("%s rows=%llu blocks=%llu legal=%llu/%llu mse=%.8g "
                "relative_mse=%.8g max_error=%.8g scale=[%.8g,%.8g] "
                "payload_bytes_per_token=%llu nominal_bpw=%.4f\n",
        name, static_cast<unsigned long long>(stats.rows),
        static_cast<unsigned long long>(stats.blocks),
        static_cast<unsigned long long>(stats.legal_blocks),
        static_cast<unsigned long long>(stats.blocks), mse, relative,
        stats.max_error, stats.min_scale, stats.max_scale,
        static_cast<unsigned long long>(astc_vulkan_embedding_payload_bytes_per_token(runner.layout)),
        bpw);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 4) {
        usage(argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    uint32_t requested_tokens = 64;
    if (argc == 3) {
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
            usage(argv[0]);
            return 2;
        }
        requested_tokens = static_cast<uint32_t>(parsed);
    }
    const std::string cache_dir = argc == 4 ? argv[3] : std::string();

    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, params);
    if (model == nullptr) {
        std::fprintf(stderr, "embedding discovery: failed to load %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    const uint32_t dimensions = static_cast<uint32_t>(llama_model_n_embd(model));
    const uint32_t values_count = llama_model_get_tok_embd(model, nullptr);
    if (dimensions == 0 || values_count == 0 || values_count % dimensions != 0) {
        std::fprintf(stderr, "embedding discovery: invalid token_embd shape (%u values, %u dims)\n",
                     values_count, dimensions);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const uint32_t vocabulary = values_count / dimensions;
    std::vector<float> embeddings(values_count);
    if (llama_model_get_tok_embd(model, embeddings.data()) != values_count) {
        std::fprintf(stderr, "embedding discovery: failed to read token_embd.weight\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    profile_runner e1;
    profile_runner e2;
    std::string error;
    if (!make_runner(astc_vulkan_embedding_representation::kE1Local,
                     astc_vulkan_footprint::k10x5, e1, error) ||
        !make_runner(astc_vulkan_embedding_representation::kE2LocalLA,
                     astc_vulkan_footprint::k8x5, e2, error)) {
        std::fprintf(stderr, "embedding discovery: %s\n", error.c_str());
        destroy_runner(e1);
        destroy_runner(e2);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    e1.layout.dimensions = dimensions;
    e1.layout.logical_to_physical.resize(dimensions);
    e2.layout.dimensions = dimensions;
    e2.layout.logical_to_physical.resize(dimensions);
    for (uint32_t i = 0; i < dimensions; ++i) {
        e1.layout.logical_to_physical[i] = i;
        e2.layout.logical_to_physical[i] = i;
    }

    profile_stats e1_stats;
    profile_stats e2_stats;
    std::ofstream payload_output;
    std::ofstream affine_output;
    if (!cache_dir.empty()) {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(cache_dir, ec);
        if (ec) {
            std::fprintf(stderr, "embedding discovery: cannot create cache directory %s\n", cache_dir.c_str());
            destroy_runner(e1); destroy_runner(e2); llama_model_free(model); llama_backend_free(); return 1;
        }
        const fs::path payload_path = fs::path(cache_dir) / "embedding-10x5.astcpack";
        const fs::path affine_path = fs::path(cache_dir) / "embedding-10x5-affine.bin";
        if (fs::exists(payload_path, ec) || fs::exists(affine_path, ec)) {
            std::fprintf(stderr, "embedding discovery: refusing to overwrite existing embedding cache files\n");
            destroy_runner(e1); destroy_runner(e2); llama_model_free(model); llama_backend_free(); return 1;
        }
        payload_output.open(payload_path, std::ios::binary);
        affine_output.open(affine_path, std::ios::binary);
        if (!payload_output || !affine_output) {
            std::fprintf(stderr, "embedding discovery: cannot open cache output files\n");
            destroy_runner(e1); destroy_runner(e2); llama_model_free(model); llama_backend_free(); return 1;
        }
        requested_tokens = 0; // cache generation always covers the full vocabulary
    }
    const uint32_t sample_count = requested_tokens == 0
        ? vocabulary : std::min(requested_tokens, vocabulary);
    std::printf("embedding discovery model=%s vocab=%u dimensions=%u sample_tokens=%u source=token_embd.weight\n",
                model_path, vocabulary, dimensions, sample_count);
    std::printf("E1 geometry tiles/token=%u bytes/token=%llu\n",
                astc_vulkan_embedding_tile_count(e1.layout),
                static_cast<unsigned long long>(astc_vulkan_embedding_payload_bytes_per_token(e1.layout)));
    std::printf("E2 geometry tiles/token=%u bytes/token=%llu\n",
                astc_vulkan_embedding_tile_count(e2.layout),
                static_cast<unsigned long long>(astc_vulkan_embedding_payload_bytes_per_token(e2.layout)));

    for (uint32_t sample = 0; sample < sample_count; ++sample) {
        const uint32_t token = sample_count == vocabulary ? sample :
            (sample_count == 1 ? 0 : static_cast<uint32_t>(
                (static_cast<uint64_t>(sample) * (vocabulary - 1)) / (sample_count - 1)));
        const float * row = embeddings.data() + static_cast<size_t>(token) * dimensions;
        if (!encode_row(e1, row, dimensions, e1_stats, error,
                        cache_dir.empty() ? nullptr : &payload_output,
                        cache_dir.empty() ? nullptr : &affine_output) ||
            !encode_row(e2, row, dimensions, e2_stats, error)) {
            std::fprintf(stderr, "embedding discovery token %u: %s\n", token, error.c_str());
            destroy_runner(e1);
            destroy_runner(e2);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }
    print_stats("E1 token-local 10x5", e1, e1_stats, dimensions);
    if (!cache_dir.empty()) {
        payload_output.close();
        affine_output.close();
        namespace fs = std::filesystem;
        std::ofstream metadata(fs::path(cache_dir) / "embedding-10x5.astce",
                               std::ios::binary | std::ios::trunc);
        metadata << "version=1\n"
                 << "tensor=token_embd.weight\n"
                 << "representation=e1-local\n"
                 << "footprint=10x5\n"
                 << "dimensions=" << dimensions << "\n"
                 << "vocab=" << vocabulary << "\n"
                 << "payload=embedding-10x5.astcpack\n"
                 << "affine=embedding-10x5-affine.bin\n";
        if (!metadata.good()) {
            std::fprintf(stderr, "embedding discovery: cannot write embedding metadata\n");
            destroy_runner(e1); destroy_runner(e2); llama_model_free(model); llama_backend_free(); return 1;
        }
        std::printf("embedding cache output=%s payload=%s affine=%s\n", cache_dir.c_str(),
                    (fs::path(cache_dir) / "embedding-10x5.astcpack").c_str(),
                    (fs::path(cache_dir) / "embedding-10x5-affine.bin").c_str());
    }
    print_stats("E2 token-local LA 8x5", e2, e2_stats, dimensions);
    std::printf("embedding discovery passed (real Qwen source, exact CPU ASTC encode/decode; %s)\n",
                cache_dir.empty() ? "bounded sample" : "full vocabulary cache generation");

    destroy_runner(e1);
    destroy_runner(e2);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
