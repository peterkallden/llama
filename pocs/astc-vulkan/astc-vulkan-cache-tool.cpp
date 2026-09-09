#include "astc-vulkan-cache.h"
#include "astc-vulkan-artifact-policy.h"
#include "astc-vulkan-format.h"
#include "astc-vulkan-compiled-catalog.h"
#include "astc-vulkan-provenance.h"
#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
#include "astc-vulkan-model-cache.h"
#include "astc-vulkan-discovery.h"
#include "astc-vulkan-input.h"
#include "astc-vulkan-d1-prescreen.h"
#include "astc-vulkan-d2-prescreen.h"
#endif
#ifdef ASTC_VULKAN_D2_PRESCREEN_AVAILABLE
#include "astc-vulkan-d2-prescreen-dispatch.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void print_paths(const astc_vulkan_cache_paths & paths) {
    std::printf("astc-cache root=%s\n", paths.root.c_str());
    std::printf("astc-cache manifest=%s\n", paths.manifest.c_str());
    std::printf("astc-cache payload=%s\n", paths.payload.c_str());
    if (!paths.row_scales.empty()) std::printf("astc-cache row-scales=%s\n", paths.row_scales.c_str());
}

struct profile {
    const char * name;
    astc_vulkan_footprint footprint;
    astc_vulkan_representation representation;
    bool experimental;
    const char * purpose;
};

constexpr std::array<profile, 11> kProfiles = {{
    {"d1-4x4",  astc_vulkan_footprint::k4x4,  astc_vulkan_representation::kScalar,       false, "high-fidelity D1"},
    {"d1-5x5",  astc_vulkan_footprint::k5x5,  astc_vulkan_representation::kScalar,       false, "balanced D1"},
    {"d1-6x6",  astc_vulkan_footprint::k6x6,  astc_vulkan_representation::kScalar,       false, "main D1 scalar"},
    {"d1-6x6-gauge", astc_vulkan_footprint::k6x6, astc_vulkan_representation::kGaugeLumaAlpha, true, "D1 L+A gauge research"},
    {"d1-8x6",  astc_vulkan_footprint::k8x6,  astc_vulkan_representation::kGaugeLumaAlpha, true,  "low-rate D1"},
    {"d1-10x6", astc_vulkan_footprint::k10x6, astc_vulkan_representation::kGaugeLumaAlpha, true,  "low-rate D1"},
    {"d1-8x8",  astc_vulkan_footprint::k8x8,  astc_vulkan_representation::kGaugeLumaAlpha, true,  "extreme-rate D1"},
    {"d1-10x8", astc_vulkan_footprint::k10x8, astc_vulkan_representation::kGaugeLumaAlpha, true,  "extreme-rate D1"},
    {"d2-6x5",  astc_vulkan_footprint::k6x5,  astc_vulkan_representation::kPairedD2,     true,  "paired D2 higher-rate"},
    {"d2-8x5",  astc_vulkan_footprint::k8x5,  astc_vulkan_representation::kPairedD2,     true,  "paired D2 iso-rate"},
    {"d2-10x5", astc_vulkan_footprint::k10x5, astc_vulkan_representation::kPairedD2,     true,  "paired D2 low-rate"},
}};

const char * representation_name(astc_vulkan_representation representation) {
    switch (representation) {
        case astc_vulkan_representation::kScalar: return "scalar";
        case astc_vulkan_representation::kGaugeLumaAlpha: return "gauge-la";
        case astc_vulkan_representation::kCDelta: return "c-delta";
        case astc_vulkan_representation::kPairedD2: return "paired-d2";
    }
    return "unknown";
}

double bits_per_weight(astc_vulkan_footprint footprint,
                       astc_vulkan_representation representation) {
    const astc_vulkan_format_info format = astc_vulkan_format(footprint);
    const unsigned density = representation == astc_vulkan_representation::kPairedD2 ? 2 : 1;
    return format.block_width == 0 || format.block_height == 0 ? 0.0 :
        128.0 / (format.block_width * format.block_height * density);
}

const profile * find_profile(const std::string & name) {
    for (const profile & candidate : kProfiles) {
        if (name == candidate.name) return &candidate;
    }
    return nullptr;
}

std::string footprint_name(astc_vulkan_footprint footprint) {
    const auto format = astc_vulkan_format(footprint);
    if (format.block_width == 0 || format.block_height == 0) return {};
    return std::to_string(format.block_width) + "x" +
           std::to_string(format.block_height);
}

bool profile_accepts_manifest(const profile & selected, const std::string & manifest_path,
                              std::string & error) {
    astc_vulkan_manifest manifest;
    if (!astc_vulkan_read_manifest(manifest_path, manifest, error)) return false;
    if (manifest.tensors.empty() && manifest.artifacts.empty()) {
        error = "ASTC profile validation requires at least one tensor";
        return false;
    }
    const auto accepts = [&](const astc_vulkan_tensor_record & tensor) {
        const bool d1_profile = selected.representation != astc_vulkan_representation::kPairedD2;
        const bool d1_tensor = tensor.representation == astc_vulkan_representation::kScalar ||
                               tensor.representation == astc_vulkan_representation::kGaugeLumaAlpha;
        if (tensor.footprint != selected.footprint || (d1_profile ? !d1_tensor :
            tensor.representation != astc_vulkan_representation::kPairedD2)) {
            return false;
        }
        return true;
    };
    for (const auto & tensor : manifest.tensors) {
        if (!accepts(tensor)) {
            error = "artifact does not match the selected ASTC profile";
            return false;
        }
    }
    for (const auto & artifact : manifest.artifacts) {
        if (!accepts(artifact.storage)) {
            error = "artifact does not match the selected ASTC profile";
            return false;
        }
    }
    error.clear();
    return true;
}

void print_profiles() {
    std::printf("ASTC cache profiles:\n");
    for (const profile & entry : kProfiles) {
        const astc_vulkan_format_info format = astc_vulkan_format(entry.footprint);
        std::printf("  %-9s ASTC-%ux%u %-9s %5.3f b/w  %s  %s\n", entry.name,
                    format.block_width, format.block_height, representation_name(entry.representation),
                    bits_per_weight(entry.footprint, entry.representation),
                    entry.experimental ? "experimental" : "standard", entry.purpose);
    }
}

void print_help(const char * executable) {
    std::printf(
        "Vanligt flöde (offline-cache; ingen JIT-encoding):\n"
        "  %s profiles\n"
        "  %s build --model model.gguf --tensor name --trace activations.bin [--profile quality|balanced|compact|speed|auto]\n"
            "            [--footprint 4x4|5x5|6x6|6x5|8x5|10x5 ...] [--representation scalar|paired-d2] [--cache path|auto]\n"
            "            [--backend hybrid|gpu-exact|cpu] [--workers N] [--no-publish 1]\n"
            "            [--gpu-proposer-shader path] [--gpu-exact-shader path]\n"
#ifdef ASTC_VULKAN_D2_PRESCREEN_AVAILABLE
        "  %s d2-prescreen --model model.gguf --tensor name --trace activations.bin\n"
            "            --footprint 6x5|8x5|10x5 --rows N --columns N\n"
            "            [--d2-prescreen cpu|gpu] [--d2-prescreen-top-k N]\n"
#endif
        "  %s inspect --model model.gguf [--cache path|auto]\n"
        "            [--tree 1]  (show source tensor names grouped with semantic paths)\n"
        "  %s catalog --model model.gguf [--cache path|auto] --catalog-output model.astcc\n"
        "            (write a metadata-only compiled tensor catalog; runtime remains unchanged)\n"
        "  %s catalog-inspect --catalog model.astcc\n"
        "            (read back the metadata-only catalog without loading GGUF or payloads)\n"
        "  %s verify --model model.gguf [--cache path|auto]\n"
#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
        "  %s rank --model model.gguf [--cache path|auto] [--policy quality|balanced|compact|speed|auto]\n"
        "            [--shortlist N] [--max-p90-loss X] [--max-worst-loss X] [--p90-weight X]\n"
#endif
#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
        "  %s discover --source-model model.gguf --usage usage.txt|auto --output discovery.tsv\n"
        "            [--profile quality|balanced|compact|speed|auto]\n"
        "            [--footprint 4x4|5x5|6x6|8x5|10x5 ...]\n"
        "            [--representation scalar|gauge-la|paired-d2]\n"
        "            [--min-source-bytes N] [--max-cache-bytes N] [--max-tensors N]\n"
        "            [--quality-trace-map tensor-traces.tsv] [--candidate-plan plan.tsv]\n"
        "  %s build-model --source-model model.gguf --fragment-dir fragments/\n"
            "            --staging build-state/ [--tensor-list tensors.txt] [--cache path|auto]\n"
        "            [--gpu-proposer-shader shader.spv] [--backend hybrid|cpu]\n"
        "            [--row-pairing optimized|adjacent] [--row-transform identity]\n"
        "  %s plan --model runtime.gguf --usage usage.txt [--source-model source-f16.gguf]\n"
        "            [--cache path|auto]\n"
        "            [--profile quality|balanced|compact|speed|auto]\n"
        "            [--policy quality|balanced|compact|speed|auto]\n"
        "            [--device-budget bytes] [--host-budget bytes] [--page-bytes bytes]\n"
            "            [--require-usage 0|1] [--require-benefit 0|1]\n"
            "            [--allow-experimental 0|1] [--allow-unverified 0|1]\n"
#endif
        "\nF16/BF16-cache ovanpå annan GGUF-familj (avancerat, kräver separat model-replay):\n"
        "  %s bind --source-model source-f16.gguf --runtime-model runtime.gguf\n"
            "            --family q4_k_m|q3_k_m|tq2_0|tq1_0|... [--cache source-cache|auto]\n"
        "  %s admit-base --source-model source-f16.gguf --model runtime.gguf\n"
            "            --family <runtime-family> [--cache source-cache|auto]  (alias)\n"
        "\nDetaljerade build-argument (valfria):\n"
        "  --gpu-proposer-shader shader.spv --gpu-exact-shader shader.spv --preset thorough|medium|fast\n"
        "  --artifact-dir dir --source-family fp16|bf16|q4_k_m|q3_k_m|tq2_0|tq1_0|...\n"
        "  --rows N --columns N (D2-shape; alias för crop-gränser i D1)\n"
        "            [--paired-semantic direct|la] [--channel-weights legacy|balanced-a025]\n"
        "            [--source-derived-alpha 0|1] [--row-scale none|absmax]\n"
        "            [--row-pairing optimized|adjacent] [--row-transform identity]\n"
        "            [--workers N] [--d2-prescreen cpu|gpu] [--d2-prescreen-top-k N]\n"
            "            (gpu-exact D2: L+A 6x5, 8x5 eller 10x5, utan absmax eller Givens)\n"
        "\nAvancerat/artifact-packning (för reproducerbara scripts):\n"
        "  %s publish --model model.gguf --artifact-dir artifact-dir [--storage-profile name] [--cache path|auto]\n"
        "  %s install --model model.gguf --artifact-dir artifact-dir [--profile name] [--cache path|auto]\n"
        "  %s create --model model.gguf --manifest artifact.manifest --payload payload.bin\n"
        "            [--layout layout.bin --row-scales scales.bin --pair-map pair-map.bin --provenance provenance.txt --storage-profile name --cache path|auto]\n"
        "\n"
        "The runtime family is a provenance label, not a hardcoded Q4-only switch; any validated GGUF\n"
        "family (for example Q4_K_M, Q3_K_M, TQ2_0 or TQ1_0) still needs its own replay gates before\n"
        "automatic scheduling. publish expects manifest.astcv, payload.astcpack, optional layout-map.bin/row-scales.bin, and optional\n"
        "provenance.txt in artifact-dir. It publishes only already-generated offline artifacts;\n"
        "it never performs just-in-time encoding during model loading. Profiles marked experimental\n"
        "require an explicit experimental runtime selection. For publish/install, use\n"
        "`--storage-profile`; their legacy `--profile d1-...|d2-...` spelling remains accepted.\n"
        "For `build`, --profile selects the default footprint/representation: quality=D1 4x4,\n"
        "balanced/auto/speed=D1 6x6, compact=D2 8x5. Explicit --footprint and\n"
        "--representation override that default. `build-model` remains intentionally explicit\n"
        "per tensor through its tensor-list.\n"
        "build is a bounded D1/D2 orchestrator: it runs the existing latent exporter, packs a\n"
        "v4 artifact and publishes it atomically. It does not perform JIT encoding and\n"
        "defaults model/vulkan evidence gates to false until replay has passed.\n",
        executable, executable,
#ifdef ASTC_VULKAN_D2_PRESCREEN_AVAILABLE
        executable,
#endif
        executable, executable, executable, executable,
#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
        executable, executable, executable, executable,
#endif
        executable, executable, executable, executable, executable);
}

bool parse_u64_argument(const std::string & text, uint64_t & result) {
    if (text.empty()) return false;
    try {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(text, &consumed, 10);
        if (consumed != text.size()) return false;
        result = static_cast<uint64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_finite_float_argument(const std::string & text, float & result) {
    try {
        size_t consumed = 0;
        const float value = std::stof(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value)) return false;
        result = value;
        return true;
    } catch (...) {
        return false;
    }
}

#if defined(ASTC_VULKAN_MODEL_CACHE_AVAILABLE) && defined(ASTC_VULKAN_D2_PRESCREEN_AVAILABLE)
const char * d2_prescreen_semantic_name(astc_vulkan_d2_prescreen_semantic value) {
    return value == astc_vulkan_d2_prescreen_semantic::luminance_alpha ? "la" : "direct";
}

const char * d2_prescreen_normalization_name(astc_vulkan_d2_prescreen_normalization value) {
    return value == astc_vulkan_d2_prescreen_normalization::row_absmax ? "absmax" : "none";
}

bool parse_d2_prescreen_footprint(const std::string & value, astc_vulkan_footprint & footprint) {
    if (value == "6x5") footprint = astc_vulkan_footprint::k6x5;
    else if (value == "8x5") footprint = astc_vulkan_footprint::k8x5;
    else if (value == "10x5") footprint = astc_vulkan_footprint::k10x5;
    else return false;
    return true;
}

bool run_d2_prescreen(const std::string & model, const std::string & tensor,
                      const std::string & trace_path, const std::string & footprint_text,
                      const std::string & rows_text, const std::string & columns_text,
                      const std::string & backend, const std::string & top_k_text,
                      std::string & error) {
    if (model.empty() || tensor.empty() || trace_path.empty() || rows_text.empty() || columns_text.empty()) {
        error = "d2-prescreen requires --model, --tensor, --trace, --rows and --columns";
        return false;
    }
    if (backend != "cpu" && backend != "gpu") {
        error = "--d2-prescreen must be cpu or gpu";
        return false;
    }
    astc_vulkan_footprint footprint;
    if (!parse_d2_prescreen_footprint(footprint_text, footprint)) {
        error = "d2-prescreen supports only 6x5, 8x5 and 10x5";
        return false;
    }
    uint32_t rows = 0, columns = 0, top_k = 3;
    try {
        rows = static_cast<uint32_t>(std::stoul(rows_text));
        columns = static_cast<uint32_t>(std::stoul(columns_text));
        if (!top_k_text.empty()) top_k = static_cast<uint32_t>(std::stoul(top_k_text));
    } catch (...) {
        error = "D2 prescreen dimensions and top-k must be unsigned integers";
        return false;
    }
    if (rows == 0 || columns == 0 || (rows & 1u) != 0 || top_k == 0) {
        error = "D2 prescreen requires non-zero even rows, non-zero columns and top-k";
        return false;
    }

    ggml_vk_astc_loaded_matrix source;
    ggml_vk_astc_activation_trace trace;
    if (!ggml_vk_astc_load_gguf_matrix(model, tensor, source, error) ||
        !ggml_vk_astc_load_activation_trace(trace_path, trace, error)) return false;
    if (rows > source.rows || columns > source.columns || columns > trace.columns || trace.samples == 0) {
        error = "D2 prescreen crop exceeds the source tensor or trace dimensions";
        return false;
    }
    std::vector<float> weights(static_cast<size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row) {
        std::copy_n(source.values.begin() + static_cast<size_t>(row) * source.columns,
                    columns, weights.begin() + static_cast<size_t>(row) * columns);
    }
    std::vector<float> trace_crop(static_cast<size_t>(trace.samples) * columns);
    for (uint32_t sample = 0; sample < trace.samples; ++sample) {
        std::copy_n(trace.values.begin() + static_cast<size_t>(sample) * trace.columns,
                    columns, trace_crop.begin() + static_cast<size_t>(sample) * columns);
    }
    // The profile bank deliberately spans representation and normalization.
    // Exact ASTC construction remains the later oracle, so this screen must
    // not collapse the bank to a single image-space proxy winner.
    const std::vector<astc_vulkan_d2_prescreen_candidate> candidates{
        {footprint, astc_vulkan_d2_prescreen_semantic::luminance_alpha,
         astc_vulkan_d2_prescreen_normalization::none, false, 16},
        {footprint, astc_vulkan_d2_prescreen_semantic::luminance_alpha,
         astc_vulkan_d2_prescreen_normalization::none, true, 16},
        {footprint, astc_vulkan_d2_prescreen_semantic::luminance_alpha,
         astc_vulkan_d2_prescreen_normalization::row_absmax, true, 16},
        {footprint, astc_vulkan_d2_prescreen_semantic::direct,
         astc_vulkan_d2_prescreen_normalization::none, false, 16},
    };
    std::vector<float> energy;
    const uint32_t calibration_samples = std::max(1u, std::min(8u, trace.samples));
    if (!astc_vulkan_d2_prescreen_calibration_energy(trace_crop, trace.samples, columns,
                                                     calibration_samples, energy)) {
        error = "cannot construct the calibration-only D2 sensitivity proxy";
        return false;
    }
    std::vector<astc_vulkan_d2_prescreen_score> scores;
    bool gpu_used = false;
    if (backend == "gpu") {
#ifdef ASTC_VULKAN_D2_PRESCREEN_SHADER_PATH
        gpu_used = astc_vulkan_score_d2_prescreen_gpu_default(
            ASTC_VULKAN_D2_PRESCREEN_SHADER_PATH, weights, rows, columns, energy, candidates,
            scores, error);
        if (!gpu_used) {
            std::printf("astc-cache d2-prescreen gpu-status=fallback-cpu reason=%s\n", error.c_str());
            error.clear();
        }
#endif
    }
    if (!gpu_used && !astc_vulkan_score_d2_prescreen_cpu(
                         weights, rows, columns, energy, candidates, scores)) {
        error = "D2 CPU prescreen failed";
        return false;
    }
    std::vector<astc_vulkan_d2_prescreen_score> shortlist;
    if (!astc_vulkan_select_d2_prescreen_shortlist(scores, std::min<uint32_t>(top_k, scores.size()),
                                                   0.0, shortlist)) {
        error = "D2 prescreen shortlist failed";
        return false;
    }
    std::printf("astc-cache d2-prescreen backend=%s candidates=%zu shortlist=%zu calibration-samples=%u\n",
                gpu_used ? "gpu" : "cpu", scores.size(), shortlist.size(), calibration_samples);
    for (size_t index = 0; index < shortlist.size(); ++index) {
        const auto & score = shortlist[index];
        std::printf("astc-cache d2-prescreen-choice rank=%zu footprint=%s semantic=%s normalization=%s "
                    "source-alpha=%s proxy-error=%.9g rate=%.6g\n", index + 1,
                    footprint_text.c_str(), d2_prescreen_semantic_name(score.candidate.semantic),
                    d2_prescreen_normalization_name(score.candidate.normalization),
                    score.candidate.source_derived_alpha ? "true" : "false",
                    score.normalized_error, score.bits_per_weight);
    }
    return true;
}
#endif

#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
bool parse_discovery_footprint(const std::string & value,
                               astc_vulkan_footprint & footprint) {
    static constexpr std::pair<const char *, astc_vulkan_footprint> kNames[] = {
        {"4x4", astc_vulkan_footprint::k4x4}, {"5x5", astc_vulkan_footprint::k5x5},
        {"6x6", astc_vulkan_footprint::k6x6}, {"6x5", astc_vulkan_footprint::k6x5},
        {"8x5", astc_vulkan_footprint::k8x5}, {"10x5", astc_vulkan_footprint::k10x5},
        {"8x6", astc_vulkan_footprint::k8x6}, {"10x6", astc_vulkan_footprint::k10x6},
        {"8x8", astc_vulkan_footprint::k8x8}, {"10x8", astc_vulkan_footprint::k10x8},
    };
    for (const auto & name : kNames) {
        if (value == name.first) { footprint = name.second; return true; }
    }
    return false;
}

bool parse_discovery_representation(const std::string & value,
                                    astc_vulkan_representation & representation) {
    if (value == "scalar") representation = astc_vulkan_representation::kScalar;
    else if (value == "gauge-la") representation = astc_vulkan_representation::kGaugeLumaAlpha;
    else if (value == "paired-d2") representation = astc_vulkan_representation::kPairedD2;
    else return false;
    return true;
}

bool read_quality_trace_map(const std::string & path,
                            std::unordered_map<std::string, std::string> & result,
                            std::string & error) {
    result.clear();
    if (path.empty()) return true;
    std::ifstream input(path);
    if (!input) {
        error = "cannot open discovery quality-trace map: " + path;
        return false;
    }
    std::string line;
    uint32_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        const size_t separator = line.find('\t');
        if (separator == std::string::npos || separator == 0 || separator + 1 == line.size()) {
            error = "quality-trace map requires <tensor><tab><trace> at line " +
                std::to_string(line_number);
            return false;
        }
        const std::string tensor = line.substr(0, separator);
        const std::string trace = line.substr(separator + 1);
        if (!result.emplace(tensor, trace).second) {
            error = "quality-trace map has duplicate tensor: " + tensor;
            return false;
        }
    }
    return true;
}

bool make_quality_probe(const std::string & model,
                        const astc_vulkan_discovery_entry & entry,
                        const std::string & trace_path,
                        uint32_t calibration_samples,
                        astc_vulkan_discovery_quality_probe & result,
                        std::string & error) {
    result = {};
    result.tensor_name = entry.tensor_name;
    ggml_vk_astc_loaded_matrix matrix;
    ggml_vk_astc_activation_trace trace;
    if (!ggml_vk_astc_load_gguf_matrix(model, entry.tensor_name, matrix, error) ||
        !ggml_vk_astc_load_activation_trace(trace_path, trace, error)) return false;
    if (matrix.rows < entry.rows || matrix.columns < entry.columns ||
        trace.samples == 0 || trace.columns < entry.columns) {
        error = "quality trace or source dimensions are incompatible with tensor " + entry.tensor_name;
        return false;
    }
    const uint32_t rows = entry.rows & ~1u; // D2 requires full logical row pairs.
    if (rows == 0) {
        error = "paired-D2 quality probe requires at least two tensor rows";
        return false;
    }
    std::vector<float> weights(static_cast<size_t>(rows) * entry.columns);
    for (uint32_t row = 0; row < rows; ++row) {
        std::copy_n(matrix.values.begin() + static_cast<size_t>(row) * matrix.columns,
                    entry.columns, weights.begin() + static_cast<size_t>(row) * entry.columns);
    }
    std::vector<float> trace_crop(static_cast<size_t>(trace.samples) * entry.columns);
    for (uint32_t sample = 0; sample < trace.samples; ++sample) {
        std::copy_n(trace.values.begin() + static_cast<size_t>(sample) * trace.columns,
                    entry.columns, trace_crop.begin() + static_cast<size_t>(sample) * entry.columns);
    }
    std::vector<float> energy;
    const uint32_t samples = std::min(std::max(1u, calibration_samples), trace.samples);
    if (!astc_vulkan_d2_prescreen_calibration_energy(trace_crop, trace.samples, entry.columns,
                                                     samples, energy)) {
        error = "cannot construct calibration energy for " + entry.tensor_name;
        return false;
    }
    std::vector<float> row_absmax(rows, 0.0f);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < entry.columns; ++column) {
            row_absmax[row] = std::max(row_absmax[row],
                std::abs(weights[static_cast<size_t>(row) * entry.columns + column]));
        }
    }
    std::sort(row_absmax.begin(), row_absmax.end());
    const size_t p10 = row_absmax.size() <= 1 ? 0 : (row_absmax.size() - 1) / 10;
    const size_t p90 = row_absmax.size() <= 1 ? 0 : (row_absmax.size() - 1) * 9 / 10;
    result.row_absmax_spread = static_cast<double>(row_absmax[p90]) /
        std::max(1e-12, static_cast<double>(row_absmax[p10]));

    std::vector<astc_vulkan_d1_prescreen_score> d1_scores;
    std::vector<astc_vulkan_d2_prescreen_score> d2_scores;
    const std::vector<astc_vulkan_d1_prescreen_candidate> d1_candidates{
        {astc_vulkan_footprint::k10x8, 16},
    };
    const std::vector<astc_vulkan_d2_prescreen_candidate> d2_candidates{
        {astc_vulkan_footprint::k8x5, astc_vulkan_d2_prescreen_semantic::luminance_alpha,
         astc_vulkan_d2_prescreen_normalization::none, false, 16},
    };
    if (!astc_vulkan_score_d1_prescreen_cpu(weights, rows, entry.columns, energy,
                                            d1_candidates, d1_scores) ||
        !astc_vulkan_score_d2_prescreen_cpu(weights, rows, entry.columns, energy,
                                            d2_candidates, d2_scores) ||
        d1_scores.empty() || d2_scores.empty()) {
        error = "cannot score discovery quality probe for " + entry.tensor_name;
        return false;
    }
    result.available = true;
    result.d1_proxy_error = d1_scores.front().normalized_error;
    result.d2_proxy_error = d2_scores.front().normalized_error;
    result.calibration_trace_hash = std::to_string(astc_vulkan_payload_hash64(
        reinterpret_cast<const uint8_t *>(trace_crop.data()), trace_crop.size() * sizeof(float)));
    return true;
}
#endif

std::string shell_quote(const std::string & value) {
    std::string quoted("'");
    for (const char character : value) {
        if (character == '\'') quoted += "'\\''";
        else quoted += character;
    }
    quoted += '\'';
    return quoted;
}

bool run_tool(const std::filesystem::path & executable,
              const std::vector<std::string> & arguments, std::string & error) {
    std::string command = shell_quote(executable.string());
    for (const std::string & argument : arguments) command += " " + shell_quote(argument);
    const int status = std::system(command.c_str());
    if (status != 0) {
        error = "offline ASTC tool failed (status " + std::to_string(status) + ")";
        return false;
    }
    return true;
}

bool read_metadata_value(const std::filesystem::path & path, const char * key,
                         std::string & value) {
    std::ifstream input(path);
    if (!input) return false;
    const std::string prefix = std::string(key) + "=";
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind(prefix, 0) == 0) {
            value = line.substr(prefix.size());
            return !value.empty();
        }
    }
    return false;
}

bool read_metadata_uint(const std::filesystem::path & path, const char * key,
                        uint32_t & value) {
    std::string text;
    if (!read_metadata_value(path, key, text)) return false;
    try {
        const unsigned long parsed = std::stoul(text);
        if (parsed == 0 || parsed > UINT32_MAX) return false;
        value = static_cast<uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

std::filesystem::path sibling_tool(const char * argv0, const char * name) {
    const std::filesystem::path invoked = std::filesystem::absolute(argv0);
    const std::filesystem::path sibling = invoked.parent_path() / name;
    std::error_code ec;
    if (std::filesystem::is_regular_file(sibling, ec)) return sibling;
    return std::filesystem::path(name);
}

std::string sanitized_tensor_name(const std::string & tensor) {
    std::string result;
    result.reserve(tensor.size());
    for (const char character : tensor) {
        if ((character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') || character == '_' || character == '-') {
            result += character;
        } else {
            result += '_';
        }
    }
    return result;
}

std::string sanitized_id(const std::string & tensor, const std::string & footprint) {
    return sanitized_tensor_name(tensor) + "-d1-" + footprint + "-scalar";
}

bool build_d1_cache(const char * argv0, const std::string & model,
                    const std::string & tensor, const std::string & trace,
                    const std::string & footprint, const std::string & cache,
                    const std::string & backend, const std::string & shader,
                    const std::string & exact_shader,
                    const std::string & preset, const std::string & artifact_dir,
                    const std::string & source_family, const std::string & max_rows,
                    const std::string & max_columns, const std::string & workers,
                    bool publish,
                    astc_vulkan_cache_paths & paths,
                    std::string & error) {
    if (model.empty() || tensor.empty() || trace.empty() || footprint.empty()) {
        error = "build requires --model, --tensor, --trace and --footprint";
        return false;
    }
    if (!publish && artifact_dir.empty()) {
        error = "--no-publish requires --artifact-dir so the fragment is retained";
        return false;
    }
    if (find_profile("d1-" + footprint) == nullptr) {
        error = "the first build command supports D1 footprints only (use publish/create for D2 artifacts)";
        return false;
    }
    std::string fingerprint, trace_hash;
    if (!astc_vulkan_sha256_file_hex(model, fingerprint, error) ||
        !astc_vulkan_sha256_file_hex(trace, trace_hash, error)) return false;

    std::filesystem::path output_dir;
    bool remove_output = false;
    if (!artifact_dir.empty()) {
        output_dir = artifact_dir;
        std::error_code ec;
        if (std::filesystem::exists(output_dir, ec)) {
            error = "--artifact-dir already exists; refusing to overwrite it";
            return false;
        }
    } else {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        output_dir = std::filesystem::temp_directory_path() /
                     ("astc-vulkan-build-" + std::to_string(stamp));
        remove_output = true;
    }
    std::error_code ec;
    if (!std::filesystem::create_directories(output_dir, ec) || ec) {
        error = "cannot create build artifact staging directory";
        return false;
    }
    std::printf("astc-cache build stage=prepare tensor=%s footprint=%s source-family=%s backend=%s\n",
                tensor.c_str(), footprint.c_str(), source_family.c_str(), backend.c_str());
    std::fflush(stdout);
    const auto cleanup = [&]() {
        if (remove_output) {
            std::error_code ignored;
            std::filesystem::remove_all(output_dir, ignored);
        }
    };
    const std::filesystem::path exported_astc = output_dir / "generated.astcpack";
    const std::filesystem::path exported_reference = output_dir / "generated.reference.bin";
    const std::filesystem::path exported_metadata = output_dir / "generated.metadata.txt";
    std::vector<std::string> generator_args{
        "--model", model, "--tensor", tensor, "--trace", trace,
        "--footprint", footprint, "--preset", preset,
        "--export-mode", "scalar", "--export-astc", exported_astc.string(),
        "--export-reference", exported_reference.string(),
        "--export-metadata", exported_metadata.string(), "--export-only"};
    if (!max_rows.empty()) { generator_args.push_back("--max-rows"); generator_args.push_back(max_rows); }
    if (!max_columns.empty()) { generator_args.push_back("--max-columns"); generator_args.push_back(max_columns); }
    if (!workers.empty()) { generator_args.push_back("--candidate-threads"); generator_args.push_back(workers); }
    if (backend == "cpu") {
        generator_args.push_back("--backend"); generator_args.push_back("cpu");
    } else if (backend == "gpu-exact") {
        if (exact_shader.empty()) {
            error = "--backend gpu-exact requires --gpu-exact-shader";
            cleanup();
            return false;
        }
        generator_args.push_back("--backend"); generator_args.push_back("gpu-exact");
        generator_args.push_back("--gpu-exact-shader"); generator_args.push_back(exact_shader);
    } else {
        generator_args.push_back("--backend"); generator_args.push_back("hybrid");
        if (!shader.empty()) {
            generator_args.push_back("--gpu-proposer-shader");
            generator_args.push_back(shader);
        }
    }
    std::printf("astc-cache build stage=latent-export status=running\n");
    std::fflush(stdout);
    if (!run_tool(sibling_tool(argv0, "astc-vulkan-latent-smoke"), generator_args, error)) {
        cleanup();
        return false;
    }
    std::printf("astc-cache build stage=latent-export status=done\n");
    std::fflush(stdout);
    uint32_t width = 0, height = 0;
    if (!read_metadata_uint(exported_metadata, "columns", width) ||
        !read_metadata_uint(exported_metadata, "rows", height)) {
        error = "latent exporter metadata did not contain tensor dimensions";
        cleanup();
        return false;
    }
    const std::filesystem::path manifest = output_dir / "manifest.astcv";
    const std::filesystem::path payload = output_dir / "payload.astcpack";
    const std::filesystem::path provenance = output_dir / "provenance.txt";
    const std::string encoder_profile = backend == "gpu-exact" ? "gpu-exact-subset" :
        (backend == "cpu" || shader.empty() ? "cpu-astcenc" : "gpu-proposer-cpu-finisher");
    const std::vector<std::string> pack_args{
        "--input", exported_astc.string(), "--metadata", exported_metadata.string(),
        "--manifest", manifest.string(), "--payload", payload.string(),
        "--tensor", tensor, "--model-fingerprint", fingerprint,
        "--provenance", provenance.string(), "--source-family", source_family,
        "--calibration-hash", trace_hash, "--validation-hash", trace_hash,
        "--holdout-hash", "unspecified", "--selector-config", "build-d1-scalar",
        "--validation-prefix", "0", "--commit-order-hash", "none",
        "--padding-contract", "deterministic-clamp-v1", "--width", std::to_string(width),
        "--height", std::to_string(height), "--footprint", footprint,
        "--representation", "scalar", "--artifact-id", sanitized_id(tensor, footprint),
        "--encoder-profile", encoder_profile, "--variant", "neutral",
        "--normalization", "none", "--model-gate", "false", "--vulkan-gate", "false"};
    std::printf("astc-cache build stage=artifact-pack status=running\n");
    std::fflush(stdout);
    if (!run_tool(sibling_tool(argv0, "astc-vulkan-artifact-pack"), pack_args, error)) {
        cleanup();
        return false;
    }
    std::printf("astc-cache build stage=artifact-pack status=done\n");
    std::fflush(stdout);
    if (!publish) {
        std::printf("astc-cache build stage=publish status=skipped artifact-dir=%s\n",
                    output_dir.string().c_str());
        return true;
    }
    std::printf("astc-cache build stage=publish status=running (hashing source GGUF)\n");
    std::fflush(stdout);
    if (!astc_vulkan_cache_create_with_row_scales(
            model, manifest.string(), payload.string(), {}, {}, provenance.string(),
            cache, paths, error)) {
        cleanup();
        return false;
    }
    std::printf("astc-cache build stage=publish status=done\n");
    std::fflush(stdout);
    std::printf("astc-cache build tensor=%s footprint=%s backend=%s artifact=%s gates=model:false/vulkan:false\n",
                tensor.c_str(), footprint.c_str(), encoder_profile.c_str(),
                sanitized_id(tensor, footprint).c_str());
    if (remove_output) cleanup();
    return true;
}

bool build_d2_cache(const char * argv0, const std::string & model,
                    const std::string & tensor, const std::string & trace,
                    const std::string & footprint, const std::string & cache,
                    const std::string & artifact_dir, const std::string & source_family,
                    const std::string & backend, const std::string & preset,
                    const std::string & exact_shader,
                    const std::string & paired_semantic, const std::string & channel_weights,
                    const std::string & source_alpha, const std::string & row_scale,
                    const std::string & row_pairing, const std::string & row_transform,
                    const std::string & rows_text, const std::string & columns_text,
                    const std::string & calibration_samples,
                    const std::string & validation_samples,
                    const std::string & workers, bool publish,
                    astc_vulkan_cache_paths & paths, std::string & error) {
    if (model.empty() || tensor.empty() || trace.empty() || footprint.empty() ||
        rows_text.empty() || columns_text.empty()) {
        error = "paired-D2 build requires --model, --tensor, --trace, --footprint, --rows and --columns";
        return false;
    }
    if (!publish && artifact_dir.empty()) {
        error = "--no-publish requires --artifact-dir so the fragment is retained";
        return false;
    }
    if (find_profile("d2-" + footprint) == nullptr) {
        error = "paired-D2 build supports only 6x5, 8x5 and 10x5 footprints";
        return false;
    }
    if (paired_semantic != "direct" && paired_semantic != "la") {
        error = "--paired-semantic must be direct or la";
        return false;
    }
    if (channel_weights != "legacy" && channel_weights != "balanced-a025" && channel_weights != "balanced-a050") {
        error = "--channel-weights must be legacy, balanced-a025 or balanced-a050";
        return false;
    }
    if (row_scale != "none" && row_scale != "absmax") {
        error = "--row-scale must be none or absmax";
        return false;
    }
    if (row_pairing != "adjacent" && row_pairing != "optimized") {
        error = "--row-pairing must be adjacent or optimized";
        return false;
    }
    if (row_transform != "identity") {
        error = "D2 Givens export requires a versioned transform-map artifact; use the pairing smoke until that runtime contract is available";
        return false;
    }
    uint32_t rows = 0, columns = 0, calibration = 0, validation = 0;
    try {
        rows = static_cast<uint32_t>(std::stoul(rows_text));
        columns = static_cast<uint32_t>(std::stoul(columns_text));
        calibration = static_cast<uint32_t>(std::stoul(calibration_samples));
        validation = static_cast<uint32_t>(std::stoul(validation_samples));
    } catch (...) {
        error = "D2 dimensions and trace split must be unsigned integers";
        return false;
    }
    if (rows == 0 || columns == 0 || calibration == 0 || validation == 0) {
        error = "D2 dimensions and trace split must be non-zero";
        return false;
    }
    if (paired_semantic != "la" && row_scale != "none") {
        error = "row scaling currently requires the D2-LA semantic";
        return false;
    }
    std::string fingerprint, trace_hash;
    if (!astc_vulkan_sha256_file_hex(model, fingerprint, error) ||
        !astc_vulkan_sha256_file_hex(trace, trace_hash, error)) return false;

    std::filesystem::path output_dir;
    bool remove_output = false;
    if (!artifact_dir.empty()) {
        output_dir = artifact_dir;
        std::error_code ec;
        if (std::filesystem::exists(output_dir, ec)) {
            error = "--artifact-dir already exists; refusing to overwrite it";
            return false;
        }
    } else {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        output_dir = std::filesystem::temp_directory_path() /
                     ("astc-vulkan-build-d2-" + std::to_string(stamp));
        remove_output = true;
    }
    std::error_code ec;
    if (!std::filesystem::create_directories(output_dir, ec) || ec) {
        error = "cannot create D2 build artifact staging directory";
        return false;
    }
    const auto cleanup = [&]() {
        if (remove_output) {
            std::error_code ignored;
            std::filesystem::remove_all(output_dir, ignored);
        }
    };
    const std::filesystem::path generated_payload = output_dir / "generated.astcpack";
    const std::filesystem::path generated_layout = output_dir / "generated.layout.bin";
    const std::filesystem::path generated_scales = output_dir / "generated.row-scales.bin";
    const std::filesystem::path generated_pair_map = output_dir / "generated.pair-map.bin";
    const std::filesystem::path report = output_dir / "generated.report.txt";
    // CPU is the portable/reference paired-D2 finisher. Hybrid keeps the
    // neural proposer path; selecting this explicitly makes large pilot
    // caches usable without paying neural table-init cost.
    std::string executable_name = backend == "cpu" || backend == "gpu-exact"
        ? "astc-vulkan-paired-selection-smoke"
        : "astc-vulkan-paired-selection-smoke-neural";
    if (footprint == "6x5") executable_name += "-6x5";
    else if (footprint == "10x5") executable_name += "-10x5";
    std::vector<std::string> generator_args{
        "--model", model, "--tensor", tensor, "--trace", trace,
        "--rows", std::to_string(rows), "--columns", std::to_string(columns),
        "--calibration-samples", std::to_string(calibration),
        "--validation-samples", std::to_string(validation), "--report", report.string(),
        "--export-payload", generated_payload.string(), "--export-layout", generated_layout.string(),
        "--preset", preset,
        "--channel-weights", channel_weights, "--source-derived-alpha", source_alpha,
        "--paired-semantic", paired_semantic, "--paired-basis", "direct",
        "--row-strip-chunked", "1", "--row-pairing", row_pairing,
        "--row-transform", row_transform};
    if (backend == "gpu-exact") {
        if (exact_shader.empty()) {
            error = "D2 --backend gpu-exact requires --gpu-exact-shader";
            cleanup();
            return false;
        }
        if ((footprint != "6x5" && footprint != "8x5" && footprint != "10x5") ||
            paired_semantic != "la" || row_scale != "none" || row_transform != "identity") {
            error = "D2 GPU exact candidates currently require H5 D2-LA (6x5, 8x5 or 10x5) without row-scale or row transform "
                "(got footprint=" + footprint + ", semantic=" + paired_semantic +
                ", row-scale=" + row_scale + ", row-transform=" + row_transform + ")";
            cleanup();
            return false;
        }
        generator_args.push_back("--gpu-exact-shader");
        generator_args.push_back(exact_shader);
    }
    if (!workers.empty()) {
        generator_args.push_back("--workers");
        generator_args.push_back(workers);
    }
    if (row_scale == "absmax") {
        generator_args.push_back("--row-scale");
        generator_args.push_back("absmax");
        generator_args.push_back("--export-row-scales");
        generator_args.push_back(generated_scales.string());
    }
    if (row_pairing == "optimized") {
        generator_args.push_back("--export-pair-map");
        generator_args.push_back(generated_pair_map.string());
    }
    if (!run_tool(sibling_tool(argv0, executable_name.c_str()), generator_args, error)) {
        cleanup();
        return false;
    }
    const std::filesystem::path manifest = output_dir / "manifest.astcv";
    const std::filesystem::path payload = output_dir / "payload.astcpack";
    const std::filesystem::path layout_payload = output_dir / "layout-map.bin";
    const std::filesystem::path row_scales_payload = output_dir / "row-scales.bin";
    const std::filesystem::path provenance = output_dir / "provenance.txt";
    const std::string artifact_id = sanitized_tensor_name(tensor) + "-d2-" + footprint + "-la-selected";
    std::vector<std::string> pack_args{
        "--input", generated_payload.string(), "--layout-input", generated_layout.string(),
        "--metadata", report.string(), "--manifest", manifest.string(), "--payload", payload.string(),
        "--layout-payload", layout_payload.string(), "--tensor", tensor,
        "--model-fingerprint", fingerprint, "--provenance", provenance.string(),
        "--source-family", source_family, "--calibration-hash", trace_hash,
        "--validation-hash", trace_hash, "--holdout-hash", "unspecified",
        "--selector-config", "d2-row-strip-chunked", "--validation-prefix", "selected",
        "--commit-order-hash", "unspecified", "--padding-contract", "deterministic-clamp-v1",
        "--width", std::to_string(columns), "--height", std::to_string(rows),
        "--footprint", footprint, "--representation", "paired-d2", "--artifact-id", artifact_id,
        "--encoder-profile", "d2-neural-selection", "--paired-semantic", paired_semantic,
        "--variant", "selected", "--normalization", row_scale,
        "--model-gate", "false", "--vulkan-gate", "false"};
    if (row_scale == "absmax") {
        pack_args.push_back("--row-scales-input");
        pack_args.push_back(generated_scales.string());
        pack_args.push_back("--row-scales-payload");
        pack_args.push_back(row_scales_payload.string());
    }
    if (row_pairing == "optimized") {
        pack_args.push_back("--pair-map-input");
        pack_args.push_back(generated_pair_map.string());
        pack_args.push_back("--pair-map-payload");
        pack_args.push_back((output_dir / "pair-map.bin").string());
    }
    if (!run_tool(sibling_tool(argv0, "astc-vulkan-artifact-pack"), pack_args, error)) {
        cleanup();
        return false;
    }
    if (!publish) {
        std::printf("astc-cache build stage=publish status=skipped artifact-dir=%s\n",
                    output_dir.string().c_str());
        return true;
    }
    if (!astc_vulkan_cache_create_with_metadata(
            model, manifest.string(), payload.string(), layout_payload.string(),
            row_scale == "absmax" ? row_scales_payload.string() : std::string(),
            row_pairing == "optimized" ? (output_dir / "pair-map.bin").string() : std::string(),
            provenance.string(), cache, paths, error)) {
        cleanup();
        return false;
    }
    std::printf("astc-cache build tensor=%s footprint=%s representation=paired-d2 semantic=%s artifact=%s gates=model:false/vulkan:false\n",
                tensor.c_str(), footprint.c_str(), paired_semantic.c_str(), artifact_id.c_str());
    if (remove_output) cleanup();
    return true;
}

bool artifact_directory_paths(const std::string & root, std::string & manifest,
                              std::string & payload, std::string & layout,
                              std::string & row_scales, std::string & pair_map,
                              std::string & provenance) {
    if (root.empty()) return false;
    const std::filesystem::path directory(root);
    manifest = (directory / "manifest.astcv").string();
    payload = (directory / "payload.astcpack").string();
    layout = (directory / "layout-map.bin").string();
    row_scales = (directory / "row-scales.bin").string();
    pair_map = (directory / "pair-map.bin").string();
    provenance = (directory / "provenance.txt").string();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(layout, ec)) layout.clear();
    ec.clear();
    if (!std::filesystem::is_regular_file(row_scales, ec)) row_scales.clear();
    ec.clear();
    if (!std::filesystem::is_regular_file(pair_map, ec)) pair_map.clear();
    ec.clear();
    if (!std::filesystem::is_regular_file(provenance, ec)) provenance.clear();
    return true;
}

#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
bool collect_model_fragments(
    const std::string & fragment_root,
    std::vector<astc_vulkan_model_cache_fragment> & fragments,
    std::vector<std::string> & tensor_names,
    std::string & error) {
    fragments.clear();
    tensor_names.clear();
    std::error_code ec;
    const std::filesystem::path root(fragment_root);
    if (!std::filesystem::is_directory(root, ec)) {
        error = "--fragment-dir is not a directory";
        return false;
    }
    std::vector<std::filesystem::path> directories;
    for (const auto & entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) break;
        if (entry.is_directory(ec)) directories.push_back(entry.path());
    }
    if (ec) {
        error = "cannot enumerate model cache fragments";
        return false;
    }
    std::sort(directories.begin(), directories.end());
    for (const auto & directory : directories) {
        const auto manifest_path = directory / "manifest.astcv";
        const auto payload_path = directory / "payload.astcpack";
        if (!std::filesystem::is_regular_file(manifest_path, ec) ||
            !std::filesystem::is_regular_file(payload_path, ec)) continue;
        astc_vulkan_manifest manifest;
        if (!astc_vulkan_read_manifest(manifest_path.string(), manifest, error)) return false;
        if ((manifest.version < 4 || manifest.version > 7) || manifest.artifacts.empty()) {
            error = "model fragment must contain a non-empty v4-v7 manifest: " + directory.string();
            return false;
        }
        for (const auto & artifact : manifest.artifacts) tensor_names.push_back(artifact.storage.name);
        const auto layout_path = directory / "layout-map.bin";
        const auto row_scales_path = directory / "row-scales.bin";
        const auto pair_map_path = directory / "pair-map.bin";
        astc_vulkan_model_cache_fragment fragment;
        fragment.manifest_path = manifest_path.string();
        fragment.payload_path = payload_path.string();
        if (std::filesystem::is_regular_file(layout_path, ec)) fragment.layout_path = layout_path.string();
        if (std::filesystem::is_regular_file(row_scales_path, ec)) fragment.row_scales_path = row_scales_path.string();
        if (std::filesystem::is_regular_file(pair_map_path, ec)) fragment.pair_map_path = pair_map_path.string();
        fragments.push_back(std::move(fragment));
    }
    if (fragments.empty()) {
        error = "no v4 artifact fragments found under --fragment-dir";
        return false;
    }
    std::sort(tensor_names.begin(), tensor_names.end());
    tensor_names.erase(std::unique(tensor_names.begin(), tensor_names.end()), tensor_names.end());
    error.clear();
    return true;
}

struct model_build_job {
    std::string tensor;
    std::string trace;
    std::string footprint;
    std::string representation = "scalar";
    std::string rows;
    std::string columns;
};

bool read_model_build_jobs(const std::string & path,
                           std::vector<model_build_job> & jobs,
                           std::string & error) {
    jobs.clear();
    std::ifstream input(path);
    if (!input) {
        error = "cannot open --tensor-list";
        return false;
    }
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const size_t comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        std::istringstream tokens(line);
        model_build_job job;
        if (!(tokens >> job.tensor >> job.trace >> job.footprint)) continue;
        tokens >> job.representation;
        if (job.representation == "paired-d2") {
            if (!(tokens >> job.rows >> job.columns)) {
                error = "D2 tensor-list entry needs rows and columns at line " +
                    std::to_string(line_number);
                return false;
            }
        }
        jobs.push_back(std::move(job));
    }
    if (jobs.empty()) {
        error = "--tensor-list contains no tensor entries";
        return false;
    }
    error.clear();
    return true;
}

bool build_model_cache(const char * argv0,
                       const std::string & source_model,
                       const std::string & fragment_root,
                       const std::string & staging_root,
                       const std::string & cache,
                       const std::string & tensor_list,
                       const std::string & backend,
                       const std::string & preset,
                       const std::string & source_family,
                       const std::string & workers,
                       const std::string & shader,
                       const std::string & exact_shader,
                       const std::string & row_pairing,
                       const std::string & row_transform,
                       const std::string & calibration_samples,
                       const std::string & validation_samples,
                       astc_vulkan_cache_paths & paths,
                       std::string & error) {
    if (source_model.empty() || fragment_root.empty() || staging_root.empty()) {
        error = "build-model requires --source-model, --fragment-dir and --staging";
        return false;
    }
    if (!tensor_list.empty()) {
        std::vector<model_build_job> jobs;
        if (!read_model_build_jobs(tensor_list, jobs, error)) return false;
        std::error_code ec;
        if (!std::filesystem::create_directories(fragment_root, ec) && ec) {
            error = "cannot create --fragment-dir";
            return false;
        }
        for (const auto & job : jobs) {
            const std::filesystem::path fragment =
                std::filesystem::path(fragment_root) /
                (sanitized_tensor_name(job.tensor) + "-" + job.footprint + "-" +
                 sanitized_tensor_name(job.representation));
            const auto manifest = fragment / "manifest.astcv";
            const auto payload = fragment / "payload.astcpack";
            if (std::filesystem::is_regular_file(manifest, ec) &&
                std::filesystem::is_regular_file(payload, ec)) continue;
            if (std::filesystem::exists(fragment, ec)) {
                error = "incomplete tensor fragment exists; remove or repair: " + fragment.string();
                return false;
            }
            std::vector<std::string> args{
                "build", "--model", source_model, "--tensor", job.tensor,
                "--trace", job.trace, "--footprint", job.footprint,
                "--representation", job.representation, "--artifact-dir", fragment.string(),
                "--backend", backend, "--preset", preset, "--source-family", source_family,
                "--no-publish", "1"};
            if (!calibration_samples.empty()) {
                args.push_back("--calibration-samples"); args.push_back(calibration_samples);
            }
            if (!validation_samples.empty()) {
                args.push_back("--validation-samples"); args.push_back(validation_samples);
            }
            if (!workers.empty()) { args.push_back("--workers"); args.push_back(workers); }
            if (!shader.empty()) {
                args.push_back("--gpu-proposer-shader");
                args.push_back(shader);
            }
            if (!exact_shader.empty()) {
                args.push_back("--gpu-exact-shader");
                args.push_back(exact_shader);
            }
            if (job.representation == "paired-d2") {
                args.push_back("--rows"); args.push_back(job.rows);
                args.push_back("--columns"); args.push_back(job.columns);
                args.push_back("--paired-semantic"); args.push_back("la");
                args.push_back("--channel-weights"); args.push_back("balanced-a025");
                args.push_back("--source-derived-alpha"); args.push_back("1");
                args.push_back("--row-pairing"); args.push_back(row_pairing);
                args.push_back("--row-transform"); args.push_back(row_transform);
            }
            std::printf("astc-cache build-model tensor=%s status=running\n", job.tensor.c_str());
            if (!run_tool(sibling_tool(argv0, "astc-vulkan-cache"), args, error)) return false;
        }
    }
    std::vector<astc_vulkan_model_cache_fragment> fragments;
    std::vector<std::string> tensor_names;
    if (!collect_model_fragments(fragment_root, fragments, tensor_names, error)) return false;
    std::filesystem::create_directories(staging_root);
    const std::filesystem::path state_path =
        std::filesystem::path(staging_root) / "build-state.txt";
    astc_vulkan_model_cache_build_state state;
    if (std::filesystem::is_regular_file(state_path)) {
        if (!astc_vulkan_model_cache_read_build_state(state_path.string(), state, error)) return false;
        if (state.source_model != source_model || state.output_root != staging_root) {
            error = "model cache build-state belongs to a different source or staging root";
            return false;
        }
        std::printf("astc-cache build-model resume completed=%zu\n", state.completed_tensors.size());
    } else {
        state.source_model = source_model;
        state.output_root = staging_root;
    }
    state.completed_tensors = tensor_names;
    if (!astc_vulkan_model_cache_write_build_state(state_path.string(), state, error)) return false;
    if (!astc_vulkan_model_cache_publish_fragments(
            source_model, fragments, staging_root, cache, paths, error)) return false;
    std::printf("astc-cache build-model tensors=%zu fragments=%zu\n",
                tensor_names.size(), fragments.size());
    return true;
}
#endif

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { print_help(argv[0]); return 2; }
    const std::string command = argv[1];
    if (command == "help" || command == "--help" || command == "-h") {
        print_help(argv[0]);
        return 0;
    }
    if (command == "profiles") {
        print_profiles();
        return 0;
    }
    std::string model, source_model, manifest, payload, layout, row_scales, pair_map, provenance, cache = "auto", artifact_dir, profile_name;
    std::string user_profile_name;
    std::string fragment_dir, staging_root, tensor_list;
    std::string tensor, trace, footprint, backend = "hybrid", shader, exact_shader,
        preset = "thorough", source_family = "fp16";
    std::string runtime_family = "unspecified";
    std::string usage_path, discovery_output, candidate_plan_output, quality_trace_map_path,
                policy_name = "balanced", device_budget, host_budget, page_bytes;
    std::string compiled_catalog_input, compiled_catalog_output;
    std::string min_source_bytes, max_cache_bytes, max_tensors;
    std::string shortlist_count = "2", max_p90_loss, max_worst_loss, p90_weight;
    std::string max_rows, max_columns, workers, representation = "scalar", paired_semantic = "la";
    std::string channel_weights = "balanced-a025", source_alpha = "1", row_scale = "none";
    // D2's calibration-selected 10-row pairing is the normal cache path.
    // Adjacent pairing remains available as an explicit regression control.
    std::string row_pairing = "optimized", row_transform = "identity";
    std::string rows, columns, calibration_samples = "8", validation_samples = "7";
    std::string d2_prescreen_backend = "cpu", d2_prescreen_top_k = "3";
    bool no_publish = false, require_usage = false, require_benefit = false;
    bool inspect_tree = false;
    bool representation_explicit = false, footprint_explicit = false, policy_explicit = false;
    bool allow_experimental = false, allow_unverified = false;
    for (int index = 2; index < argc; index += 2) {
        if (index + 1 >= argc) return 2;
        const std::string option = argv[index];
        const std::string value = argv[index + 1];
        if (option == "--model" || option == "--runtime-model") model = value;
        else if (option == "--source-model") source_model = value;
        else if (option == "--manifest") manifest = value;
        else if (option == "--payload") payload = value;
        else if (option == "--layout") layout = value;
        else if (option == "--row-scales") row_scales = value;
        else if (option == "--pair-map") pair_map = value;
        else if (option == "--provenance") provenance = value;
        else if (option == "--cache") cache = value;
        else if (option == "--artifact-dir") artifact_dir = value;
        else if (option == "--fragment-dir") fragment_dir = value;
        else if (option == "--staging") staging_root = value;
        else if (option == "--tensor-list") tensor_list = value;
        else if (option == "--profile") { profile_name = value; user_profile_name = value; }
        else if (option == "--storage-profile") profile_name = value;
        else if (option == "--tensor") tensor = value;
        else if (option == "--trace") trace = value;
        else if (option == "--footprint") { footprint = value; footprint_explicit = true; }
        else if (option == "--backend") backend = value;
        else if (option == "--gpu-proposer-shader") shader = value;
        else if (option == "--gpu-exact-shader") exact_shader = value;
        else if (option == "--preset") preset = value;
        else if (option == "--source-family") source_family = value;
        else if (option == "--family") runtime_family = value;
        else if (option == "--usage") usage_path = value;
        else if (option == "--output") discovery_output = value;
        else if (option == "--catalog-output") compiled_catalog_output = value;
        else if (option == "--catalog") compiled_catalog_input = value;
        else if (option == "--candidate-plan") candidate_plan_output = value;
        else if (option == "--quality-trace-map") quality_trace_map_path = value;
        else if (option == "--policy") { policy_name = value; policy_explicit = true; }
        else if (option == "--device-budget") device_budget = value;
        else if (option == "--host-budget") host_budget = value;
        else if (option == "--page-bytes") page_bytes = value;
        else if (option == "--min-source-bytes") min_source_bytes = value;
        else if (option == "--max-cache-bytes") max_cache_bytes = value;
        else if (option == "--max-tensors") max_tensors = value;
        else if (option == "--shortlist") shortlist_count = value;
        else if (option == "--max-p90-loss") max_p90_loss = value;
        else if (option == "--max-worst-loss") max_worst_loss = value;
        else if (option == "--p90-weight") p90_weight = value;
        else if (option == "--require-usage") require_usage = value == "1" || value == "true";
        else if (option == "--require-benefit") require_benefit = value == "1" || value == "true";
        else if (option == "--allow-experimental") allow_experimental = value == "1" || value == "true";
        else if (option == "--allow-unverified") allow_unverified = value == "1" || value == "true";
        else if (option == "--tree") inspect_tree = value == "1" || value == "true";
        else if (option == "--max-rows") max_rows = value;
        else if (option == "--max-columns") max_columns = value;
        else if (option == "--workers") workers = value;
        else if (option == "--representation") { representation = value; representation_explicit = true; }
        else if (option == "--paired-semantic") paired_semantic = value;
        else if (option == "--channel-weights") channel_weights = value;
        else if (option == "--source-derived-alpha") source_alpha = value;
        else if (option == "--row-scale") row_scale = value;
        else if (option == "--row-pairing") row_pairing = value;
        else if (option == "--row-transform") row_transform = value;
        else if (option == "--rows") rows = value;
        else if (option == "--columns") columns = value;
        else if (option == "--calibration-samples") calibration_samples = value;
        else if (option == "--validation-samples") validation_samples = value;
        else if (option == "--d2-prescreen") d2_prescreen_backend = value;
        else if (option == "--d2-prescreen-top-k") d2_prescreen_top_k = value;
        else if (option == "--no-publish") no_publish = value == "1" || value == "true";
        else return 2;
    }
    std::string error;
    // `--profile` is a user-facing build shortcut. It deliberately chooses a
    // physical baseline only; explicit representation/footprint flags remain
    // authoritative for reproducible research commands. The later planner
    // still selects among evidence-backed artifacts per tensor.
    if (command == "build" && !user_profile_name.empty()) {
        astc_vulkan_user_profile_defaults profile_defaults;
        if (!astc_vulkan_resolve_user_profile(user_profile_name, profile_defaults)) {
            std::fprintf(stderr, "astc-cache %s failed: unknown profile '%s'\n",
                         command.c_str(), user_profile_name.c_str());
            return 2;
        }
        if (!footprint_explicit) footprint = footprint_name(profile_defaults.footprint);
        if (!representation_explicit) representation = representation_name(profile_defaults.representation);
    }
    if (command == "bind" || command == "admit-base") {
        astc_vulkan_cache_runtime_base binding;
        if (source_model.empty() || model.empty() ||
            !astc_vulkan_cache_admit_runtime_base(source_model, model, cache, runtime_family,
                                                   binding, error)) {
            std::fprintf(stderr, "astc-cache %s failed: %s\n", command.c_str(), error.c_str());
            return 1;
        }
        std::printf("astc-cache bind source=%s runtime=%s family=%s schema=%s gates=model:false/vulkan:false\n",
                    source_model.c_str(), model.c_str(), binding.family.c_str(),
                    binding.schema_sha256.c_str());
        std::printf("astc-cache bind note=structural-only; run a runtime-specific replay gate before production scheduling\n");
        return 0;
    }
    if (command == "catalog") {
        if (model.empty() || compiled_catalog_output.empty()) {
            std::fprintf(stderr, "astc-cache catalog requires --model, --cache and --catalog-output\n");
            return 2;
        }
        astc_vulkan_cache_validation validation;
        if (!astc_vulkan_cache_validate(model, cache, validation, error)) {
            std::fprintf(stderr, "astc-cache catalog failed: %s\n", error.c_str());
            return 1;
        }
        astc_vulkan_compiled_catalog catalog;
        if (!astc_vulkan_compiled_catalog_from_manifest(validation.manifest, catalog, error) ||
            !astc_vulkan_write_compiled_catalog(compiled_catalog_output, catalog, error)) {
            std::fprintf(stderr, "astc-cache catalog failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("astc-cache catalog output=%s tensors=%zu model=%s\n",
                    compiled_catalog_output.c_str(), catalog.tensors.size(),
                    catalog.logical_model_id.c_str());
        for (const auto & entry : catalog.tensors) {
            std::printf("astc-cache catalog-entry tensor=%s role=%s path=%s class=%s artifact=%s\n",
                        entry.logical_name.c_str(), entry.semantic_role.c_str(),
                        entry.canonical_path.c_str(), entry.storage_class.c_str(),
                        entry.artifact_id.c_str());
        }
        return 0;
    }
    if (command == "catalog-inspect") {
        if (compiled_catalog_input.empty()) {
            std::fprintf(stderr, "astc-cache catalog-inspect requires --catalog\n");
            return 2;
        }
        astc_vulkan_compiled_catalog catalog;
        if (!astc_vulkan_read_compiled_catalog(compiled_catalog_input, catalog, error)) {
            std::fprintf(stderr, "astc-cache catalog-inspect failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("astc-cache catalog-inspect path=%s tensors=%zu model=%s source=%s\n",
                    compiled_catalog_input.c_str(), catalog.tensors.size(),
                    catalog.logical_model_id.c_str(), catalog.source_model_fingerprint.c_str());
        for (const auto & entry : catalog.tensors) {
            std::printf("astc-cache catalog-entry tensor=%s role=%s path=%s class=%s artifact=%s "
                        "payload=%llu+%llu layout=%llu+%llu scales=%llu+%llu pair-map=%llu+%llu\n",
                        entry.logical_name.c_str(), entry.semantic_role.c_str(),
                        entry.canonical_path.c_str(), entry.storage_class.c_str(),
                        entry.artifact_id.c_str(),
                        static_cast<unsigned long long>(entry.payload_offset),
                        static_cast<unsigned long long>(entry.payload_size),
                        static_cast<unsigned long long>(entry.layout_offset),
                        static_cast<unsigned long long>(entry.layout_size),
                        static_cast<unsigned long long>(entry.row_scale_offset),
                        static_cast<unsigned long long>(entry.row_scale_size),
                        static_cast<unsigned long long>(entry.pair_map_offset),
                        static_cast<unsigned long long>(entry.pair_map_size));
        }
        return 0;
    }
#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
    if (command == "d2-prescreen") {
#ifdef ASTC_VULKAN_D2_PRESCREEN_AVAILABLE
        if (!run_d2_prescreen(model, tensor, trace, footprint, rows, columns,
                              d2_prescreen_backend, d2_prescreen_top_k, error)) {
            std::fprintf(stderr, "astc-cache d2-prescreen failed: %s\n", error.c_str());
            return 1;
        }
        return 0;
#else
        std::fprintf(stderr, "astc-cache d2-prescreen unavailable: this build has no Vulkan D2 prescreen backend\n");
        return 1;
#endif
    }
    if (command == "build-model") {
        astc_vulkan_cache_paths paths;
        const std::string build_source = source_model.empty() ? model : source_model;
        if (!build_model_cache(argv[0], build_source, fragment_dir, staging_root, cache,
                               tensor_list, backend, preset, source_family, workers, shader, exact_shader,
                               row_pairing, row_transform,
                               calibration_samples, validation_samples,
                               paths, error)) {
            std::fprintf(stderr, "astc-cache build-model failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("astc-cache build-model published root=%s\n", paths.root.c_str());
        print_paths(paths);
        return 0;
    }
    if (command == "discover") {
        const std::string inventory_model = source_model.empty() ? model : source_model;
        if (inventory_model.empty() || usage_path.empty() || discovery_output.empty()) {
            std::fprintf(stderr, "astc-cache discover requires --source-model/--model, --usage and --output\n");
            return 2;
        }
        const std::string selected_profile_name = user_profile_name.empty() ? "balanced" : user_profile_name;
        astc_vulkan_user_profile_defaults profile_defaults;
        if (!astc_vulkan_resolve_user_profile(selected_profile_name, profile_defaults)) {
            std::fprintf(stderr, "astc-cache discover failed: unknown profile '%s'\n",
                         selected_profile_name.c_str());
            return 2;
        }
        astc_vulkan_footprint selected_footprint;
        astc_vulkan_representation selected_representation;
        if (footprint.empty()) {
            selected_footprint = profile_defaults.footprint;
        } else if (!parse_discovery_footprint(footprint, selected_footprint)) {
            std::fprintf(stderr, "astc-cache discover failed: invalid footprint\n");
            return 2;
        }
        if (!representation_explicit) {
            selected_representation = profile_defaults.representation;
        } else if (!parse_discovery_representation(representation, selected_representation)) {
            std::fprintf(stderr, "astc-cache discover failed: invalid footprint or representation\n");
            return 2;
        }
        std::vector<ggml_vk_astc_tensor_info> inventory;
        if (!ggml_vk_astc_list_gguf_tensors(inventory_model, inventory, error)) {
            std::fprintf(stderr, "astc-cache discover failed: %s\n", error.c_str());
            return 1;
        }
        std::vector<astc_vulkan_tensor_usage_metrics> usage;
        if (usage_path == "auto") {
            uint32_t order = 0;
            for (const auto & tensor_info : inventory) {
                if (!tensor_info.rank2) continue;
                astc_vulkan_tensor_usage_metrics metric;
                metric.tensor_name = tensor_info.name;
                metric.invocations = 1;
                metric.tokens_seen = 1;
                metric.native_bytes_read = static_cast<uint64_t>(tensor_info.bytes);
                metric.path_probability = 1.0;
                metric.execution_order = order++;
                usage.push_back(std::move(metric));
            }
        } else if (!astc_vulkan_read_tensor_usage_metrics(usage_path, usage, error)) {
            std::fprintf(stderr, "astc-cache discover failed: %s\n", error.c_str());
            return 1;
        }
        astc_vulkan_discovery_options discovery_options;
        discovery_options.footprint = selected_footprint;
        discovery_options.representation = selected_representation;
        if (!min_source_bytes.empty() && !parse_u64_argument(min_source_bytes,
                                                               discovery_options.min_source_bytes)) {
            std::fprintf(stderr, "astc-cache discover failed: invalid --min-source-bytes\n");
            return 2;
        }
        if (!max_cache_bytes.empty() && !parse_u64_argument(max_cache_bytes,
                                                              discovery_options.max_cache_bytes)) {
            std::fprintf(stderr, "astc-cache discover failed: invalid --max-cache-bytes\n");
            return 2;
        }
        uint64_t max_tensor_count = 0;
        if (!max_tensors.empty() && (!parse_u64_argument(max_tensors, max_tensor_count) ||
                                     max_tensor_count > std::numeric_limits<size_t>::max())) {
            std::fprintf(stderr, "astc-cache discover failed: invalid --max-tensors\n");
            return 2;
        }
        discovery_options.max_tensors = static_cast<size_t>(max_tensor_count);
        std::vector<astc_vulkan_discovery_tensor_input> inputs;
        inputs.reserve(inventory.size());
        for (const auto & tensor_info : inventory) {
            inputs.push_back({tensor_info.name, tensor_info.columns, tensor_info.rows,
                              static_cast<uint64_t>(tensor_info.bytes), tensor_info.rank2});
        }
        std::vector<astc_vulkan_discovery_entry> entries;
        if (!astc_vulkan_discover_cache_candidates(inputs, usage, discovery_options,
                                                   entries, error) ||
            !astc_vulkan_write_discovery_report(discovery_output, discovery_options,
                                                entries, error)) {
            std::fprintf(stderr, "astc-cache discover failed: %s\n", error.c_str());
            return 1;
        }
        // For D2 8x5, discovery also emits the bounded pre-build bank. Trace
        // entries are optional but tensor-specific: only a matching map entry
        // enables a calibration-only proxy; absence remains explicit in the
        // plan and can never masquerade as a quality pass.
        std::vector<astc_vulkan_discovery_candidate_plan_entry> candidate_plan;
        std::string published_candidate_plan;
        if (selected_representation == astc_vulkan_representation::kPairedD2 &&
            selected_footprint == astc_vulkan_footprint::k8x5) {
            std::unordered_map<std::string, std::string> trace_map;
            if (!read_quality_trace_map(quality_trace_map_path, trace_map, error)) {
                std::fprintf(stderr, "astc-cache discover failed: %s\n", error.c_str());
                return 1;
            }
            uint64_t requested_calibration = 0;
            if (!parse_u64_argument(calibration_samples, requested_calibration) ||
                requested_calibration == 0 || requested_calibration > UINT32_MAX) {
                std::fprintf(stderr, "astc-cache discover failed: invalid --calibration-samples\n");
                return 2;
            }
            std::vector<astc_vulkan_discovery_quality_probe> probes;
            for (const auto & entry : entries) {
                if (!entry.selected) continue;
                const auto found = trace_map.find(entry.tensor_name);
                if (found == trace_map.end()) {
                    probes.push_back({entry.tensor_name});
                    continue;
                }
                astc_vulkan_discovery_quality_probe probe;
                if (!make_quality_probe(inventory_model, entry, found->second,
                                        static_cast<uint32_t>(requested_calibration), probe, error)) {
                    std::fprintf(stderr, "astc-cache discover failed: %s\n", error.c_str());
                    return 1;
                }
                probes.push_back(std::move(probe));
            }
            if (!astc_vulkan_make_discovery_candidate_plan(
                    entries, probes, discovery_options, {}, candidate_plan, error)) {
                std::fprintf(stderr, "astc-cache discover failed: %s\n", error.c_str());
                return 1;
            }
            published_candidate_plan = candidate_plan_output.empty() ?
                discovery_output + ".candidates.tsv" : candidate_plan_output;
            if (!astc_vulkan_write_discovery_candidate_plan(
                    published_candidate_plan, candidate_plan, error)) {
                std::fprintf(stderr, "astc-cache discover failed: %s\n", error.c_str());
                return 1;
            }
        } else if (!candidate_plan_output.empty() || !quality_trace_map_path.empty()) {
            std::fprintf(stderr, "astc-cache discover failed: quality candidate-plan v1 requires paired-d2 8x5\n");
            return 2;
        }
        size_t selected_count = 0;
        uint64_t selected_bytes = 0;
        for (const auto & entry : entries) {
            if (!entry.selected) continue;
            ++selected_count;
            selected_bytes += entry.estimated_astc_bytes + entry.estimated_layout_bytes;
        }
        const auto selected_format = astc_vulkan_format(selected_footprint);
        std::printf("astc-cache discover profile=%s ASTC-%ux%u representation=%s tensors=%zu selected=%zu estimated-cache-bytes=%llu report=%s\n",
                    selected_profile_name.c_str(), selected_format.block_width,
                    selected_format.block_height, representation_name(selected_representation),
                    entries.size(), selected_count,
                    static_cast<unsigned long long>(selected_bytes), discovery_output.c_str());
        if (!published_candidate_plan.empty()) {
            std::printf("astc-cache discover candidate-plan=%s candidates=%zu trace-probes=%s\n",
                        published_candidate_plan.c_str(), candidate_plan.size(),
                        quality_trace_map_path.empty() ? "none" : "mapped");
        }
        for (const auto & entry : entries) {
            std::printf("astc-cache discover-entry tensor=%s role=%s path=%s selected=%s priority=%.6g source-bytes=%llu astc-bytes=%llu quality-probe=required\n",
                        entry.tensor_name.c_str(), astc_vulkan_tensor_semantic_role(entry.tensor_name).c_str(),
                        astc_vulkan_tensor_canonical_path(entry.tensor_name).c_str(),
                        entry.selected ? "true" : "false",
                        entry.priority_score,
                        static_cast<unsigned long long>(entry.source_bytes),
                        static_cast<unsigned long long>(entry.estimated_astc_bytes +
                                                         entry.estimated_layout_bytes));
        }
        return 0;
    }
    if (command == "rank") {
        if (model.empty()) {
            std::fprintf(stderr, "astc-cache rank requires --model\n");
            return 2;
        }
        if (!policy_explicit && !user_profile_name.empty()) {
            astc_vulkan_user_profile_defaults profile_defaults;
            if (!astc_vulkan_resolve_user_profile(user_profile_name, profile_defaults)) {
                std::fprintf(stderr, "astc-cache rank failed: unknown profile '%s'\n",
                             user_profile_name.c_str());
                return 2;
            }
            policy_name = astc_vulkan_quality_policy_name(profile_defaults.policy);
        }
        astc_vulkan_quality_policy policy;
        if (!astc_vulkan_parse_quality_policy(policy_name, policy)) {
            std::fprintf(stderr, "astc-cache rank failed: unknown policy '%s'\n", policy_name.c_str());
            return 2;
        }
        uint64_t requested_count = 0;
        if (!parse_u64_argument(shortlist_count, requested_count) || requested_count == 0 ||
            requested_count > std::numeric_limits<size_t>::max()) {
            std::fprintf(stderr, "astc-cache rank failed: invalid --shortlist\n");
            return 2;
        }
        astc_vulkan_artifact_selection_rules rules;
        if ((!max_p90_loss.empty() && !parse_finite_float_argument(max_p90_loss, rules.max_p90_loss_delta)) ||
            (!max_worst_loss.empty() && !parse_finite_float_argument(max_worst_loss, rules.max_worst_loss_delta)) ||
            (!p90_weight.empty() && !parse_finite_float_argument(p90_weight, rules.p90_loss_weight)) ||
            rules.max_p90_loss_delta < 0.0f || rules.max_worst_loss_delta < 0.0f ||
            rules.p90_loss_weight < 0.0f) {
            std::fprintf(stderr, "astc-cache rank failed: invalid robust-loss option\n");
            return 2;
        }
        astc_vulkan_cache_validation validation;
        if (!astc_vulkan_cache_validate(model, cache, validation, error)) {
            std::fprintf(stderr, "astc-cache rank failed: %s\n", error.c_str());
            return 1;
        }
        if (validation.manifest.version < 4) {
            std::fprintf(stderr, "astc-cache rank failed: legacy cache has no multi-artifact candidates\n");
            return 1;
        }
        std::unordered_map<std::string, std::vector<astc_vulkan_artifact_candidate>> by_tensor;
        for (const auto & artifact : validation.manifest.artifacts) {
            astc_vulkan_artifact_candidate candidate;
            candidate.tensor = &artifact.storage;
            candidate.variant = artifact.variant;
            candidate.normalization = artifact.normalization;
            candidate.evidence = artifact.evidence;
            candidate.rate_bpw = astc_vulkan_artifact_storage_bpw(artifact);
            candidate.artifact_id = artifact.id;
            by_tensor[artifact.storage.name].push_back(std::move(candidate));
        }
        std::vector<std::string> names;
        names.reserve(by_tensor.size());
        for (const auto & item : by_tensor) names.push_back(item.first);
        std::sort(names.begin(), names.end());
        std::printf("astc-cache rank policy=%s tensors=%zu shortlist=%llu p90-weight=%.6g\n",
                    astc_vulkan_quality_policy_name(policy), names.size(),
                    static_cast<unsigned long long>(requested_count), rules.p90_loss_weight);
        for (const auto & name : names) {
            std::vector<astc_vulkan_artifact_candidate> shortlist;
            if (!astc_vulkan_rank_tensor_artifact_shortlist(
                    by_tensor[name], policy, rules, static_cast<size_t>(requested_count),
                    shortlist, error)) {
                std::fprintf(stderr, "astc-cache rank failed for %s: %s\n", name.c_str(), error.c_str());
                return 1;
            }
            if (shortlist.empty()) {
                std::printf("astc-cache rank-entry tensor=%s native-fallback=true reason=no-evidence-backed-artifact\n",
                            name.c_str());
                continue;
            }
            for (size_t index = 0; index < shortlist.size(); ++index) {
                const auto & entry = shortlist[index];
                std::printf("astc-cache rank-entry tensor=%s role=%s path=%s rank=%zu artifact=%s bpw=%.6g "
                            "median-loss=%.6g p90-loss=%.6g worst-loss=%.6g top1-worst=%.6g "
                            "variant=%u normalization=%u\n",
                            name.c_str(), astc_vulkan_tensor_semantic_role(name).c_str(),
                            astc_vulkan_tensor_canonical_path(name).c_str(), index + 1,
                            entry.artifact_id.c_str(), entry.rate_bpw,
                            astc_vulkan_artifact_median_loss_delta(entry.evidence),
                            astc_vulkan_artifact_p90_loss_delta(entry.evidence),
                            astc_vulkan_artifact_worst_loss_delta(entry.evidence),
                            astc_vulkan_artifact_worst_top1_agreement(entry.evidence),
                            static_cast<unsigned>(entry.variant),
                            static_cast<unsigned>(entry.normalization));
            }
        }
        return 0;
    }
    if (command == "plan") {
        if (model.empty() || usage_path.empty()) {
            std::fprintf(stderr, "astc-cache plan requires --model and --usage\n");
            return 2;
        }
        if (!policy_explicit && !user_profile_name.empty()) {
            astc_vulkan_user_profile_defaults profile_defaults;
            if (!astc_vulkan_resolve_user_profile(user_profile_name, profile_defaults)) {
                std::fprintf(stderr, "astc-cache plan failed: unknown profile '%s'\n",
                             user_profile_name.c_str());
                return 2;
            }
            policy_name = astc_vulkan_quality_policy_name(profile_defaults.policy);
        }
        astc_vulkan_quality_policy policy;
        if (!astc_vulkan_parse_quality_policy(policy_name, policy)) {
            std::fprintf(stderr, "astc-cache plan failed: unknown policy '%s'\n", policy_name.c_str());
            return 2;
        }
        astc_vulkan_model_cache_plan_options plan_options;
        plan_options.policy = policy;
        plan_options.allow_experimental = allow_experimental;
        plan_options.allow_unverified = allow_unverified;
        astc_vulkan_model_cache_catalog catalog;
        const bool has_distinct_source = !source_model.empty() && source_model != model;
        const bool loaded = has_distinct_source ?
            astc_vulkan_model_cache_load_catalog_for_runtime(
                source_model, model, cache, plan_options, catalog, error) :
            astc_vulkan_model_cache_load_catalog(model, cache, plan_options, catalog, error);
        if (!loaded) {
            std::fprintf(stderr, "astc-cache plan failed: %s\n", error.c_str());
            return 1;
        }
        std::vector<astc_vulkan_tensor_usage_metrics> usage;
        if (!astc_vulkan_read_tensor_usage_metrics(usage_path, usage, error)) {
            std::fprintf(stderr, "astc-cache plan failed: %s\n", error.c_str());
            return 1;
        }
        astc_vulkan_memory_budget budget;
        budget.effective_device_limit_bytes = std::numeric_limits<uint64_t>::max();
        if (!device_budget.empty() && !parse_u64_argument(device_budget,
                                                            budget.effective_device_limit_bytes)) {
            std::fprintf(stderr, "astc-cache plan failed: invalid --device-budget\n");
            return 2;
        }
        if (!host_budget.empty() && !parse_u64_argument(host_budget, budget.host_limit_bytes)) {
            std::fprintf(stderr, "astc-cache plan failed: invalid --host-budget\n");
            return 2;
        }
        astc_vulkan_model_cache_usage_options usage_options;
        usage_options.require_usage_metrics = require_usage;
        usage_options.require_positive_benefit = require_benefit;
        astc_vulkan_model_cache_plan planned;
        if (!astc_vulkan_model_cache_plan_usage(
                catalog.plan, usage, usage_options, budget, planned, error)) {
            std::fprintf(stderr, "astc-cache plan failed: %s\n", error.c_str());
            return 1;
        }
        uint64_t max_page_payload = 0;
        if (!page_bytes.empty() && !parse_u64_argument(page_bytes, max_page_payload)) {
            std::fprintf(stderr, "astc-cache plan failed: invalid --page-bytes\n");
            return 2;
        }
        std::vector<astc_vulkan_model_cache_storage_page> pages;
        if (!astc_vulkan_model_cache_make_storage_pages(
                planned, max_page_payload, pages, error)) {
            std::fprintf(stderr, "astc-cache plan failed: %s\n", error.c_str());
            return 1;
        }
        size_t resident_count = planned.residency.resident_items.size();
        size_t fallback_count = 0;
        for (const auto & entry : planned.entries) fallback_count += entry.use_native_fallback ? 1 : 0;
        std::printf("astc-cache plan policy=%s entries=%zu resident=%zu fallback=%zu pages=%zu "
                    "device-bytes=%llu host-bytes=%llu streaming=%s\n",
                    astc_vulkan_quality_policy_name(policy), planned.entries.size(), resident_count,
                    fallback_count, pages.size(),
                    static_cast<unsigned long long>(planned.residency.device_bytes),
                    static_cast<unsigned long long>(planned.residency.host_bytes),
                    planned.residency.requires_streaming ? "true" : "false");
        for (const auto & entry : planned.entries) {
            std::printf("astc-cache plan-entry tensor=%s role=%s path=%s artifact=%s fallback=%s usage=%s heat=%.6g benefit=%.6g "
                        "time-saved-ns=%.6g\n", entry.tensor_name.c_str(),
                        astc_vulkan_tensor_semantic_role(entry.tensor_name).c_str(),
                        astc_vulkan_tensor_canonical_path(entry.tensor_name).c_str(),
                        entry.artifact_id.c_str(),
                        entry.use_native_fallback ? "true" : "false",
                        entry.usage_available ? "true" : "false", entry.heat_score,
                        entry.benefit_score, entry.expected_gpu_time_saved_ns);
        }
        for (size_t index = 0; index < pages.size(); ++index) {
            const auto & page = pages[index];
            const auto format = astc_vulkan_format(page.key.footprint);
            std::printf("astc-cache page=%zu ASTC-%ux%u representation=%s semantic=%u normalization=%u "
                        "row-scales=%s entries=%zu payload-bytes=%llu host-bytes=%llu\n", index,
                        format.block_width, format.block_height,
                        representation_name(page.key.representation),
                        static_cast<unsigned>(page.key.paired_semantic),
                        static_cast<unsigned>(page.key.normalization),
                        page.key.has_row_scales ? "true" : "false", page.entry_indices.size(),
                        static_cast<unsigned long long>(page.payload_bytes),
                        static_cast<unsigned long long>(page.host_bytes));
        }
        return 0;
    }
#endif
    if (command == "build") {
        if (representation != "scalar" && representation != "paired-d2") {
            std::fprintf(stderr, "astc-cache build failed: --representation must be scalar or paired-d2\n");
            return 2;
        }
        if (backend != "hybrid" && backend != "gpu-exact" && backend != "cpu") {
            std::fprintf(stderr, "astc-cache build failed: --backend must be hybrid, gpu-exact or cpu\n");
            return 2;
        }
        if (preset != "thorough" && preset != "medium" && preset != "fast") {
            std::fprintf(stderr, "astc-cache build failed: --preset must be thorough, medium or fast\n");
            return 2;
        }
        astc_vulkan_cache_paths paths;
        const bool d2 = representation == "paired-d2";
        // For D1, accept the familiar --rows/--columns spelling as aliases
        // for crop limits. D2 uses these options for its physical shape.
        if (!d2) {
            if (max_rows.empty()) max_rows = rows;
            if (max_columns.empty()) max_columns = columns;
        }
        const bool built = d2 ? build_d2_cache(
            argv[0], model, tensor, trace, footprint, cache, artifact_dir, source_family,
            backend, preset, exact_shader,
            paired_semantic, channel_weights, source_alpha, row_scale, row_pairing, row_transform, rows, columns,
            calibration_samples, validation_samples, workers, !no_publish, paths, error) : build_d1_cache(
            argv[0], model, tensor, trace, footprint, cache, backend, shader, exact_shader, preset,
            artifact_dir, source_family, max_rows, max_columns, workers, !no_publish, paths, error);
        if (!built) {
            std::fprintf(stderr, "astc-cache build failed: %s\n", error.c_str());
            return 1;
        }
        if (no_publish) {
            std::printf("astc-cache build staged only; artifact-dir=%s\n", artifact_dir.c_str());
        } else {
            std::printf("astc-cache build published root=%s\n", paths.root.c_str());
            print_paths(paths);
        }
        return 0;
    }
    if (command == "inspect" || command == "verify") {
        astc_vulkan_cache_validation validation;
        if (model.empty() || !astc_vulkan_cache_validate(model, cache, validation, error)) {
            std::fprintf(stderr, "astc-cache miss: %s\n", error.c_str());
            return 1;
        }
        if (command == "verify") {
            const size_t records = validation.manifest.version >= 4 ? validation.manifest.artifacts.size() :
                validation.manifest.tensors.size();
            std::printf("astc-cache verify ok records=%zu paired-d2=%s row-scales=%s pair-map=%s\n",
                        records, validation.has_paired_d2 ? "true" : "false",
                        validation.has_row_scales ? "true" : "false",
                        validation.has_pair_map ? "true" : "false");
            return 0;
        }
        const size_t records = validation.manifest.version >= 4 ? validation.manifest.artifacts.size() :
            validation.manifest.tensors.size();
        std::printf("astc-cache hit records=%zu paired-d2=%s row-scales=%s\n", records,
                    validation.has_paired_d2 ? "true" : "false",
                    validation.has_row_scales ? "true" : "false");
        for (const astc_vulkan_tensor_record & tensor : validation.manifest.tensors) {
            const astc_vulkan_format_info format = astc_vulkan_format(tensor.footprint);
            const bool experimental = astc_vulkan_footprint_is_experimental(tensor.footprint) ||
                                      tensor.representation == astc_vulkan_representation::kPairedD2;
            std::printf("astc-cache tensor=%s role=%s path=%s ASTC-%ux%u representation=%s nominal-bpw=%.3f status=%s\n",
                        tensor.name.c_str(), tensor.semantic_role.c_str(), tensor.canonical_path.c_str(),
                        format.block_width, format.block_height,
                        representation_name(tensor.representation),
                        bits_per_weight(tensor.footprint, tensor.representation),
                        experimental ? "experimental" : "standard");
        }
        for (const auto & entry : validation.manifest.artifacts) {
            const auto & tensor = entry.storage;
            const astc_vulkan_format_info format = astc_vulkan_format(tensor.footprint);
            std::printf("astc-cache artifact=%s tensor=%s role=%s path=%s ASTC-%ux%u representation=%s semantic=%u normalization=%u variant=%u loss=%.6g gates=%s/%s\n",
                        entry.id.c_str(), tensor.name.c_str(), tensor.semantic_role.c_str(),
                        tensor.canonical_path.c_str(), format.block_width, format.block_height,
                        representation_name(tensor.representation), static_cast<unsigned>(entry.paired_semantic),
                        static_cast<unsigned>(entry.normalization), static_cast<unsigned>(entry.variant),
                        entry.evidence.loss_delta, entry.evidence.model_gate_passed ? "model" : "no-model",
                        entry.evidence.vulkan_gate_passed ? "vulkan" : "no-vulkan");
        }
        if (inspect_tree) {
            std::printf("astc-cache tree:\n");
            if (validation.manifest.version < 4) {
                for (const auto & tensor : validation.manifest.tensors) {
                    std::printf("  %s  role=%s  path=%s\n", tensor.name.c_str(),
                                tensor.semantic_role.c_str(), tensor.canonical_path.c_str());
                }
            } else {
                for (const auto & entry : validation.manifest.artifacts) {
                    const auto & tensor = entry.storage;
                    std::printf("  %s  artifact=%s  role=%s  path=%s\n", tensor.name.c_str(),
                                entry.id.c_str(), tensor.semantic_role.c_str(),
                                tensor.canonical_path.c_str());
                }
            }
        }
        print_paths(validation.paths);
        return 0;
    }
    if (command == "install" || command == "publish") {
        if (!artifact_directory_paths(artifact_dir, manifest, payload, layout, row_scales, pair_map, provenance)) return 2;
    } else if (command != "create") {
        std::fprintf(stderr, "unknown astc-cache command: %s\n", command.c_str());
        return 2;
    }
    if (!profile_name.empty()) {
        const profile * selected = find_profile(profile_name);
        if (selected == nullptr || !profile_accepts_manifest(*selected, manifest, error)) {
            std::fprintf(stderr, "astc-cache profile failed: %s\n",
                         error.empty() ? "unknown profile" : error.c_str());
            return 1;
        }
    }
    if (command == "create" || command == "install" || command == "publish") {
        astc_vulkan_cache_paths paths;
        if (model.empty() || manifest.empty() || payload.empty() ||
            !astc_vulkan_cache_create_with_metadata(model, manifest, payload, layout, row_scales, pair_map,
                                                    provenance, cache, paths, error)) {
            std::fprintf(stderr, "astc-cache %s failed: %s\n", command.c_str(), error.c_str());
            return 1;
        }
        std::printf("astc-cache %s source=%s%s%s\n", command.c_str(), model.c_str(),
                    profile_name.empty() ? "" : " profile=", profile_name.empty() ? "" : profile_name.c_str());
        print_paths(paths);
        return 0;
    }
    std::fprintf(stderr, "invalid astc-cache command\n");
    return 2;
}
