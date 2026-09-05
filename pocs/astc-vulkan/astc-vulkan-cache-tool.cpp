#include "astc-vulkan-cache.h"
#include "astc-vulkan-format.h"
#include "astc-vulkan-provenance.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
        "usage:\n"
        "  %s profiles\n"
        "  %s inspect --model model.gguf [--cache path|auto]\n"
        "  %s verify --model model.gguf [--cache path|auto]\n"
        "  %s build --model model.gguf --tensor name --trace activations.bin --footprint 6x6\n"
        "            [--cache path|auto] [--backend hybrid|cpu] [--gpu-proposer-shader shader.spv]\n"
        "            [--preset thorough|medium|fast] [--artifact-dir dir] [--source-family name]\n"
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
        executable, executable, executable, executable, executable, executable, executable);
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

std::string sanitized_id(const std::string & tensor, const std::string & footprint) {
    std::string result;
    result.reserve(tensor.size() + footprint.size() + 8);
    for (const char character : tensor) {
        if ((character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') || character == '_' || character == '-') {
            result += character;
        } else {
            result += '_';
        }
    }
    result += "-d1-" + footprint + "-scalar";
    return result;
}

bool build_d1_cache(const char * argv0, const std::string & model,
                    const std::string & tensor, const std::string & trace,
                    const std::string & footprint, const std::string & cache,
                    const std::string & backend, const std::string & shader,
                    const std::string & preset, const std::string & artifact_dir,
                    const std::string & source_family, const std::string & max_rows,
                    const std::string & max_columns, astc_vulkan_cache_paths & paths,
                    std::string & error) {
    if (model.empty() || tensor.empty() || trace.empty() || footprint.empty()) {
        error = "build requires --model, --tensor, --trace and --footprint";
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
    if (backend == "cpu") {
        generator_args.push_back("--backend"); generator_args.push_back("cpu");
    } else {
        generator_args.push_back("--backend"); generator_args.push_back("hybrid");
        if (!shader.empty()) {
            generator_args.push_back("--gpu-proposer-shader");
            generator_args.push_back(shader);
        }
    }
    if (!run_tool(sibling_tool(argv0, "astc-vulkan-latent-smoke"), generator_args, error)) {
        cleanup();
        return false;
    }
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
    if (!run_tool(sibling_tool(argv0, "astc-vulkan-artifact-pack"), pack_args, error)) {
        cleanup();
        return false;
    }
    if (!astc_vulkan_cache_create_with_row_scales(
            model, manifest.string(), payload.string(), {}, {}, provenance.string(),
            cache, paths, error)) {
        cleanup();
        return false;
    }
    std::printf("astc-cache build tensor=%s footprint=%s backend=%s artifact=%s gates=model:false/vulkan:false\n",
                tensor.c_str(), footprint.c_str(), encoder_profile.c_str(),
                sanitized_id(tensor, footprint).c_str());
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
    std::string model, manifest, payload, layout, row_scales, provenance, cache = "auto", artifact_dir, profile_name;
    std::string tensor, trace, footprint, backend = "hybrid", shader, preset = "thorough", source_family = "fp16";
    std::string max_rows, max_columns;
    for (int index = 2; index < argc; index += 2) {
        if (index + 1 >= argc) return 2;
        const std::string option = argv[index];
        const std::string value = argv[index + 1];
        if (option == "--model") model = value;
        else if (option == "--manifest") manifest = value;
        else if (option == "--payload") payload = value;
        else if (option == "--layout") layout = value;
        else if (option == "--row-scales") row_scales = value;
        else if (option == "--provenance") provenance = value;
        else if (option == "--cache") cache = value;
        else if (option == "--artifact-dir") artifact_dir = value;
        else if (option == "--profile" || option == "--storage-profile") profile_name = value;
        else if (option == "--tensor") tensor = value;
        else if (option == "--trace") trace = value;
        else if (option == "--footprint") footprint = value;
        else if (option == "--backend") backend = value;
        else if (option == "--gpu-proposer-shader") shader = value;
        else if (option == "--preset") preset = value;
        else if (option == "--source-family") source_family = value;
        else if (option == "--max-rows") max_rows = value;
        else if (option == "--max-columns") max_columns = value;
        else return 2;
    }
    std::string error;
    if (command == "build") {
        if (backend != "hybrid" && backend != "cpu") {
            std::fprintf(stderr, "astc-cache build failed: --backend must be hybrid or cpu\n");
            return 2;
        }
        if (preset != "thorough" && preset != "medium" && preset != "fast") {
            std::fprintf(stderr, "astc-cache build failed: --preset must be thorough, medium or fast\n");
            return 2;
        }
        astc_vulkan_cache_paths paths;
        if (!build_d1_cache(argv[0], model, tensor, trace, footprint,
                            cache, backend, shader, preset, artifact_dir, source_family,
                            max_rows, max_columns, paths, error)) {
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
