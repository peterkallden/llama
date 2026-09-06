#include "astc-vulkan-cache.h"
#include "astc-vulkan-format.h"
#include "astc-vulkan-provenance.h"
#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
#include "astc-vulkan-model-cache.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
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

constexpr std::array<profile, 10> kProfiles = {{
    {"d1-4x4",  astc_vulkan_footprint::k4x4,  astc_vulkan_representation::kScalar,       false, "high-fidelity D1"},
    {"d1-5x5",  astc_vulkan_footprint::k5x5,  astc_vulkan_representation::kScalar,       false, "balanced D1"},
    {"d1-6x6",  astc_vulkan_footprint::k6x6,  astc_vulkan_representation::kGaugeLumaAlpha, false, "main D1 gauge"},
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
        "  %s build --model model.gguf --tensor name --trace activations.bin --footprint 6x6\n"
            "            [--representation scalar|paired-d2] [--cache path|auto]\n"
            "            [--backend hybrid|cpu] [--workers N] [--no-publish 1]\n"
        "  %s inspect --model model.gguf [--cache path|auto]\n"
        "  %s verify --model model.gguf [--cache path|auto]\n"
#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
        "  %s build-model --source-model model.gguf --fragment-dir fragments/\n"
            "            --staging build-state/ [--tensor-list tensors.txt] [--cache path|auto]\n"
#endif
        "\nF16-cache ovanpå Q4/Q3 (avancerat, kräver separat model-replay):\n"
        "  %s bind --source-model source-f16.gguf --runtime-model runtime-q4.gguf\n"
            "            [--family q4_k_m] [--cache source-cache|auto]\n"
        "  %s admit-base --source-model source-f16.gguf --model runtime-q4.gguf\n"
            "            [--family q4_k_m] [--cache source-cache|auto]  (alias)\n"
        "\nDetaljerade build-argument (valfria):\n"
        "  --gpu-proposer-shader shader.spv --preset thorough|medium|fast\n"
        "  --artifact-dir dir --source-family fp16|bf16|q4_k_m|q3_k_m\n"
        "  --rows N --columns N (D2-shape; alias för crop-gränser i D1)\n"
        "            [--paired-semantic direct|la] [--channel-weights legacy|balanced-a025]\n"
        "            [--source-derived-alpha 0|1] [--row-scale none|absmax]\n"
        "            [--workers N]\n"
        "\nAvancerat/artifact-packning (för reproducerbara scripts):\n"
        "  %s publish --model model.gguf --artifact-dir artifact-dir [--storage-profile name] [--cache path|auto]\n"
        "  %s install --model model.gguf --artifact-dir artifact-dir [--profile name] [--cache path|auto]\n"
        "  %s create --model model.gguf --manifest artifact.manifest --payload payload.bin\n"
        "            [--layout layout.bin --row-scales scales.bin --provenance provenance.txt --storage-profile name --cache path|auto]\n"
        "\n"
        "publish expects manifest.astcv, payload.astcpack, optional layout-map.bin/row-scales.bin, and optional\n"
        "provenance.txt in artifact-dir. It publishes only already-generated offline artifacts;\n"
        "it never performs just-in-time encoding during model loading. Profiles marked experimental\n"
        "require an explicit experimental runtime selection. `install` and `--profile` remain\n"
        "compatibility aliases; use `publish` and `--storage-profile` for new scripts.\n"
        "build is a bounded D1 orchestrator: it runs the existing latent exporter, packs a\n"
        "v4 artifact and publishes it atomically. It does not perform JIT encoding and\n"
        "defaults model/vulkan evidence gates to false until replay has passed.\n",
        executable, executable, executable, executable,
#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
        executable,
#endif
        executable, executable, executable, executable, executable);
}

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
    const std::string encoder_profile = backend == "cpu" || shader.empty() ?
        "cpu-astcenc" : "gpu-proposer-cpu-finisher";
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
                    const std::string & paired_semantic, const std::string & channel_weights,
                    const std::string & source_alpha, const std::string & row_scale,
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
    const std::filesystem::path report = output_dir / "generated.report.txt";
    std::string executable_name = "astc-vulkan-paired-selection-smoke-neural";
    if (footprint == "6x5") executable_name += "-6x5";
    else if (footprint == "10x5") executable_name += "-10x5";
    std::vector<std::string> generator_args{
        "--model", model, "--tensor", tensor, "--trace", trace,
        "--rows", std::to_string(rows), "--columns", std::to_string(columns),
        "--calibration-samples", std::to_string(calibration),
        "--validation-samples", std::to_string(validation), "--report", report.string(),
        "--export-payload", generated_payload.string(), "--export-layout", generated_layout.string(),
        "--channel-weights", channel_weights, "--source-derived-alpha", source_alpha,
        "--paired-semantic", paired_semantic, "--paired-basis", "direct",
        "--row-strip-chunked", "1"};
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
    if (!run_tool(sibling_tool(argv0, "astc-vulkan-artifact-pack"), pack_args, error)) {
        cleanup();
        return false;
    }
    if (!publish) {
        std::printf("astc-cache build stage=publish status=skipped artifact-dir=%s\n",
                    output_dir.string().c_str());
        return true;
    }
    if (!astc_vulkan_cache_create_with_row_scales(
            model, manifest.string(), payload.string(), layout_payload.string(),
            row_scale == "absmax" ? row_scales_payload.string() : std::string(),
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
                              std::string & row_scales, std::string & provenance) {
    if (root.empty()) return false;
    const std::filesystem::path directory(root);
    manifest = (directory / "manifest.astcv").string();
    payload = (directory / "payload.astcpack").string();
    layout = (directory / "layout-map.bin").string();
    row_scales = (directory / "row-scales.bin").string();
    provenance = (directory / "provenance.txt").string();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(layout, ec)) layout.clear();
    ec.clear();
    if (!std::filesystem::is_regular_file(row_scales, ec)) row_scales.clear();
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
        if (manifest.version != 4 || manifest.artifacts.empty()) {
            error = "model fragment must contain a non-empty v4 manifest: " + directory.string();
            return false;
        }
        for (const auto & artifact : manifest.artifacts) tensor_names.push_back(artifact.storage.name);
        const auto layout_path = directory / "layout-map.bin";
        const auto row_scales_path = directory / "row-scales.bin";
        astc_vulkan_model_cache_fragment fragment;
        fragment.manifest_path = manifest_path.string();
        fragment.payload_path = payload_path.string();
        if (std::filesystem::is_regular_file(layout_path, ec)) fragment.layout_path = layout_path.string();
        if (std::filesystem::is_regular_file(row_scales_path, ec)) fragment.row_scales_path = row_scales_path.string();
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
                (sanitized_tensor_name(job.tensor) + "-" + job.footprint);
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
            if (!workers.empty()) { args.push_back("--workers"); args.push_back(workers); }
            if (job.representation == "paired-d2") {
                args.push_back("--rows"); args.push_back(job.rows);
                args.push_back("--columns"); args.push_back(job.columns);
                args.push_back("--paired-semantic"); args.push_back("la");
                args.push_back("--channel-weights"); args.push_back("balanced-a025");
                args.push_back("--source-derived-alpha"); args.push_back("1");
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
    std::string model, source_model, manifest, payload, layout, row_scales, provenance, cache = "auto", artifact_dir, profile_name;
    std::string fragment_dir, staging_root, tensor_list;
    std::string tensor, trace, footprint, backend = "hybrid", shader, preset = "thorough", source_family = "fp16";
    std::string runtime_family = "unspecified";
    std::string max_rows, max_columns, workers, representation = "scalar", paired_semantic = "la";
    std::string channel_weights = "balanced-a025", source_alpha = "1", row_scale = "none";
    std::string rows, columns, calibration_samples = "8", validation_samples = "7";
    bool no_publish = false;
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
        else if (option == "--provenance") provenance = value;
        else if (option == "--cache") cache = value;
        else if (option == "--artifact-dir") artifact_dir = value;
        else if (option == "--fragment-dir") fragment_dir = value;
        else if (option == "--staging") staging_root = value;
        else if (option == "--tensor-list") tensor_list = value;
        else if (option == "--profile" || option == "--storage-profile") profile_name = value;
        else if (option == "--tensor") tensor = value;
        else if (option == "--trace") trace = value;
        else if (option == "--footprint") footprint = value;
        else if (option == "--backend") backend = value;
        else if (option == "--gpu-proposer-shader") shader = value;
        else if (option == "--preset") preset = value;
        else if (option == "--source-family") source_family = value;
        else if (option == "--family") runtime_family = value;
        else if (option == "--max-rows") max_rows = value;
        else if (option == "--max-columns") max_columns = value;
        else if (option == "--workers") workers = value;
        else if (option == "--representation") representation = value;
        else if (option == "--paired-semantic") paired_semantic = value;
        else if (option == "--channel-weights") channel_weights = value;
        else if (option == "--source-derived-alpha") source_alpha = value;
        else if (option == "--row-scale") row_scale = value;
        else if (option == "--rows") rows = value;
        else if (option == "--columns") columns = value;
        else if (option == "--calibration-samples") calibration_samples = value;
        else if (option == "--validation-samples") validation_samples = value;
        else if (option == "--no-publish") no_publish = value == "1" || value == "true";
        else return 2;
    }
    std::string error;
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
#ifdef ASTC_VULKAN_MODEL_CACHE_AVAILABLE
    if (command == "build-model") {
        astc_vulkan_cache_paths paths;
        const std::string build_source = source_model.empty() ? model : source_model;
        if (!build_model_cache(argv[0], build_source, fragment_dir, staging_root, cache,
                               tensor_list, backend, preset, source_family, workers,
                               paths, error)) {
            std::fprintf(stderr, "astc-cache build-model failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("astc-cache build-model published root=%s\n", paths.root.c_str());
        print_paths(paths);
        return 0;
    }
#endif
    if (command == "build") {
        if (representation != "scalar" && representation != "paired-d2") {
            std::fprintf(stderr, "astc-cache build failed: --representation must be scalar or paired-d2\n");
            return 2;
        }
        if (backend != "hybrid" && backend != "cpu") {
            std::fprintf(stderr, "astc-cache build failed: --backend must be hybrid or cpu\n");
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
            paired_semantic, channel_weights, source_alpha, row_scale, rows, columns,
            calibration_samples, validation_samples, workers, !no_publish, paths, error) : build_d1_cache(
            argv[0], model, tensor, trace, footprint, cache, backend, shader, preset,
            artifact_dir, source_family, max_rows, max_columns, workers, !no_publish, paths, error);
        if (!built) {
            std::fprintf(stderr, "astc-cache build failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("astc-cache build published root=%s\n", paths.root.c_str());
        print_paths(paths);
        return 0;
    }
    if (command == "inspect" || command == "verify") {
        astc_vulkan_cache_validation validation;
        if (model.empty() || !astc_vulkan_cache_validate(model, cache, validation, error)) {
            std::fprintf(stderr, "astc-cache miss: %s\n", error.c_str());
            return 1;
        }
        if (command == "verify") {
            const size_t records = validation.manifest.version == 4 ? validation.manifest.artifacts.size() :
                validation.manifest.tensors.size();
            std::printf("astc-cache verify ok records=%zu paired-d2=%s row-scales=%s\n",
                        records, validation.has_paired_d2 ? "true" : "false",
                        validation.has_row_scales ? "true" : "false");
            return 0;
        }
        const size_t records = validation.manifest.version == 4 ? validation.manifest.artifacts.size() :
            validation.manifest.tensors.size();
        std::printf("astc-cache hit records=%zu paired-d2=%s row-scales=%s\n", records,
                    validation.has_paired_d2 ? "true" : "false",
                    validation.has_row_scales ? "true" : "false");
        for (const astc_vulkan_tensor_record & tensor : validation.manifest.tensors) {
            const astc_vulkan_format_info format = astc_vulkan_format(tensor.footprint);
            const bool experimental = astc_vulkan_footprint_is_experimental(tensor.footprint) ||
                                      tensor.representation == astc_vulkan_representation::kPairedD2;
            std::printf("astc-cache tensor=%s ASTC-%ux%u representation=%s nominal-bpw=%.3f status=%s\n",
                        tensor.name.c_str(), format.block_width, format.block_height,
                        representation_name(tensor.representation),
                        bits_per_weight(tensor.footprint, tensor.representation),
                        experimental ? "experimental" : "standard");
        }
        for (const auto & entry : validation.manifest.artifacts) {
            const auto & tensor = entry.storage;
            const astc_vulkan_format_info format = astc_vulkan_format(tensor.footprint);
            std::printf("astc-cache artifact=%s tensor=%s ASTC-%ux%u representation=%s semantic=%u normalization=%u variant=%u loss=%.6g gates=%s/%s\n",
                        entry.id.c_str(), tensor.name.c_str(), format.block_width, format.block_height,
                        representation_name(tensor.representation), static_cast<unsigned>(entry.paired_semantic),
                        static_cast<unsigned>(entry.normalization), static_cast<unsigned>(entry.variant),
                        entry.evidence.loss_delta, entry.evidence.model_gate_passed ? "model" : "no-model",
                        entry.evidence.vulkan_gate_passed ? "vulkan" : "no-vulkan");
        }
        print_paths(validation.paths);
        return 0;
    }
    if (command == "install" || command == "publish") {
        if (!artifact_directory_paths(artifact_dir, manifest, payload, layout, row_scales, provenance)) return 2;
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
            !astc_vulkan_cache_create_with_row_scales(model, manifest, payload, layout, row_scales,
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
