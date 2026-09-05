#include "astc-vulkan-cache.h"
#include "astc-vulkan-format.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <string>

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
        "  %s publish --model model.gguf --artifact-dir artifact-dir [--storage-profile name] [--cache path|auto]\n"
        "  %s install --model model.gguf --artifact-dir artifact-dir [--profile name] [--cache path|auto]\n"
        "  %s create --model model.gguf --manifest artifact.manifest --payload payload.bin\n"
        "            [--layout layout.bin --row-scales scales.bin --provenance provenance.txt --storage-profile name --cache path|auto]\n"
        "\n"
        "publish expects manifest.astcv, payload.astcpack, optional layout-map.bin/row-scales.bin, and optional\n"
        "provenance.txt in artifact-dir. It publishes only already-generated offline artifacts;\n"
        "it never performs just-in-time encoding during model loading. Profiles marked experimental\n"
        "require an explicit experimental runtime selection. `install` and `--profile` remain\n"
        "compatibility aliases; use `publish` and `--storage-profile` for new scripts.\n",
        executable, executable, executable, executable, executable, executable);
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
        else return 2;
    }
    std::string error;
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
