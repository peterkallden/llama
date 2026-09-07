#include "astc-vulkan-cache.h"

#include "astc-vulkan-hash.h"
#include "astc-vulkan-provenance.h"

#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr const char * kManifestFile = "manifest.astcv";
constexpr const char * kPayloadFile = "payload.astcpack";
constexpr const char * kLayoutFile = "layout-map.bin";
constexpr const char * kRowScalesFile = "row-scales.bin";
constexpr const char * kPairMapFile = "pair-map.bin";
constexpr const char * kProvenanceFile = "provenance.txt";
constexpr const char * kSourceHashFile = "source.gguf.sha256";
constexpr const char * kManifestHashFile = "manifest.sha256";
constexpr const char * kPayloadHashFile = "payload.sha256";
constexpr const char * kLayoutHashFile = "layout-map.sha256";
constexpr const char * kRowScalesHashFile = "row-scales.sha256";
constexpr const char * kPairMapHashFile = "pair-map.sha256";
constexpr const char * kCompatibleBasesDirectory = "compatible-bases";
constexpr const char * kRuntimeBaseExtension = ".astcbase";

bool valid_binding_value(const std::string & value) {
    return !value.empty() && value.find_first_of("\r\n=") == std::string::npos;
}

bool read_hash(const std::string & path, std::string & hash) {
    std::ifstream file(path, std::ios::binary);
    if (!file || !std::getline(file, hash) || hash.size() != 64) return false;
    for (const unsigned char value : hash) {
        if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) return false;
    }
    return true;
}

bool write_hash(const std::string & path, const std::string & hash) {
    std::ofstream file(path, std::ios::binary);
    file << hash << '\n';
    return file.good();
}

bool read_binding(const std::string & path, astc_vulkan_cache_runtime_base & binding) {
    binding = {};
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::string version;
    std::string line;
    while (std::getline(file, line)) {
        const size_t separator = line.find('=');
        if (separator == std::string::npos) return false;
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        if (key == "version") version = value;
        else if (key == "source_sha256") binding.source_sha256 = value;
        else if (key == "runtime_sha256") binding.runtime_sha256 = value;
        else if (key == "schema_sha256") binding.schema_sha256 = value;
        else if (key == "family") binding.family = value;
        else if (key == "model_gate") binding.model_gate_passed = value == "1";
        else if (key == "vulkan_gate") binding.vulkan_gate_passed = value == "1";
        else return false;
    }
    const auto valid_hash = [](const std::string & hash) {
        if (hash.size() != 64) return false;
        for (const unsigned char value : hash) {
            if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) return false;
        }
        return true;
    };
    binding.admitted = version == "1" && valid_hash(binding.source_sha256) &&
        valid_hash(binding.runtime_sha256) && valid_hash(binding.schema_sha256) &&
        valid_binding_value(binding.family);
    return binding.admitted;
}

bool write_binding(const std::string & path, const astc_vulkan_cache_runtime_base & binding,
                   std::string & error) {
    if (!binding.admitted || !valid_binding_value(binding.family)) {
        error = "invalid ASTC runtime-base binding";
        return false;
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !(file << "version=1\n"
                       << "source_sha256=" << binding.source_sha256 << '\n'
                       << "runtime_sha256=" << binding.runtime_sha256 << '\n'
                       << "schema_sha256=" << binding.schema_sha256 << '\n'
                       << "family=" << binding.family << '\n'
                       << "model_gate=" << (binding.model_gate_passed ? 1 : 0) << '\n'
                       << "vulkan_gate=" << (binding.vulkan_gate_passed ? 1 : 0) << '\n')) {
        error = "cannot write ASTC runtime-base binding";
        return false;
    }
    error.clear();
    return true;
}

bool gguf_schema_components(const std::string & path, std::string & architecture,
                            std::vector<std::string> & tensor_lines, std::string & error) {
    architecture.clear();
    tensor_lines.clear();
    gguf_init_params params{true, nullptr};
    gguf_context * context = gguf_init_from_file(path.c_str(), params);
    if (context == nullptr) {
        error = "cannot read GGUF schema: " + path;
        return false;
    }
    const int64_t architecture_key = gguf_find_key(context, "general.architecture");
    if (architecture_key < 0 || gguf_get_kv_type(context, architecture_key) != GGUF_TYPE_STRING) {
        gguf_free(context);
        error = "GGUF has no general.architecture string";
        return false;
    }
    architecture = gguf_get_val_str(context, architecture_key);
    tensor_lines.reserve(static_cast<size_t>(gguf_get_n_tensors(context)));
    for (int64_t tensor_id = 0; tensor_id < gguf_get_n_tensors(context); ++tensor_id) {
        const char * name = gguf_get_tensor_name(context, tensor_id);
        const int64_t * dimensions = gguf_get_tensor_ne(context, tensor_id);
        if (name == nullptr || dimensions == nullptr) {
            gguf_free(context);
            error = "GGUF tensor schema is incomplete";
            return false;
        }
        std::string line(name);
        for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            line += ':' + std::to_string(dimensions[dimension]);
        }
        tensor_lines.push_back(std::move(line));
    }
    gguf_free(context);
    std::sort(tensor_lines.begin(), tensor_lines.end());
    error.clear();
    return true;
}

bool gguf_schema_sha256(const std::string & path, std::string & hash, std::string & error) {
    std::string architecture;
    std::vector<std::string> tensor_lines;
    if (!gguf_schema_components(path, architecture, tensor_lines, error)) return false;
    std::string canonical = "astc-runtime-schema-v1\narchitecture=" + architecture +
        "\ntensors=" + std::to_string(tensor_lines.size()) + '\n';
    for (const std::string & line : tensor_lines) canonical += line + '\n';
    hash = astc_vulkan_sha256_hex(canonical.data(), canonical.size());
    error.clear();
    return true;
}

bool gguf_schema_matches(const std::string & source_path, const std::string & runtime_path,
                         std::string & error) {
    std::string source_architecture;
    std::string runtime_architecture;
    std::vector<std::string> source_tensors;
    std::vector<std::string> runtime_tensors;
    if (!gguf_schema_components(source_path, source_architecture, source_tensors, error) ||
        !gguf_schema_components(runtime_path, runtime_architecture, runtime_tensors, error)) {
        return false;
    }
    if (source_architecture != runtime_architecture) {
        error = "GGUF architecture metadata differs";
        return false;
    }
    if (source_tensors.size() != runtime_tensors.size()) {
        error = "GGUF tensor count differs: source=" + std::to_string(source_tensors.size()) +
            " runtime=" + std::to_string(runtime_tensors.size());
        return false;
    }
    for (size_t index = 0; index < source_tensors.size(); ++index) {
        if (source_tensors[index] == runtime_tensors[index]) continue;
        error = "GGUF tensor set/shape differs near " + source_tensors[index];
        return false;
    }
    error.clear();
    return true;
}

bool has_paired_d2(const astc_vulkan_manifest & manifest) {
    if (manifest.version == 4) {
        for (const auto & artifact : manifest.artifacts) {
            if (artifact.storage.representation == astc_vulkan_representation::kPairedD2) return true;
        }
        return false;
    }
    for (const auto & tensor : manifest.tensors) {
        if (tensor.representation == astc_vulkan_representation::kPairedD2) return true;
    }
    return false;
}

bool has_row_scales(const astc_vulkan_manifest & manifest) {
    if (manifest.version < 4) return false;
    for (const auto & artifact : manifest.artifacts) {
        if (artifact.normalization == astc_vulkan_normalization::per_row_absmax) return true;
    }
    return false;
}

bool has_pair_map(const astc_vulkan_manifest & manifest) {
    if (manifest.version < 5) return false;
    for (const auto & artifact : manifest.artifacts) {
        if (artifact.pair_map_byte_size != 0) return true;
    }
    return false;
}

bool file_size(const std::string & path, uint64_t & size) {
    std::error_code ec;
    size = fs::file_size(path, ec);
    return !ec;
}

bool range_hash64(const std::string & path, uint64_t offset, uint64_t size,
                  uint64_t & hash, std::vector<uint8_t> & buffer) {
    std::ifstream file(path, std::ios::binary);
    if (!file || offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;
    file.seekg(static_cast<std::streamoff>(offset));
    if (!file) return false;
    if (buffer.empty()) buffer.resize(1u << 20);
    uint64_t remaining = size;
    hash = 1469598103934665603ULL;
    while (remaining != 0) {
        const std::streamsize count = static_cast<std::streamsize>(std::min<uint64_t>(remaining, buffer.size()));
        file.read(reinterpret_cast<char *>(buffer.data()), count);
        if (file.gcount() != count) return false;
        hash = astc_vulkan_fnv1a64_update(hash, buffer.data(), static_cast<size_t>(count));
        remaining -= static_cast<uint64_t>(count);
    }
    return true;
}

bool read_range(const std::string & path, uint64_t offset, uint64_t size,
                std::vector<uint8_t> & output) {
    if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    output.resize(static_cast<size_t>(size));
    file.seekg(static_cast<std::streamoff>(offset));
    if (!file) return false;
    if (size != 0) file.read(reinterpret_cast<char *>(output.data()), static_cast<std::streamsize>(size));
    return file.good();
}

bool validate_tensor_payloads(const astc_vulkan_manifest & manifest,
                              uint64_t payload_size, const std::string & payload_path,
                              uint64_t layout_size, const std::string & layout_path,
                              uint64_t row_scale_size, const std::string & row_scale_path,
                              uint64_t pair_map_size, const std::string & pair_map_path,
                              std::string & error) {
    if (!astc_vulkan_validate_payload_blob(manifest, payload_size, error)) return false;
    const bool paired = has_paired_d2(manifest);
    if (paired && !astc_vulkan_validate_layout_blob(manifest, layout_size, error)) return false;
    std::vector<uint8_t> buffer;
    const auto validate_storage = [&](const astc_vulkan_tensor_record & tensor) {
        uint64_t payload_hash = 0;
        if (!range_hash64(payload_path, tensor.byte_offset, tensor.byte_size, payload_hash, buffer) ||
            (tensor.payload_hash64 != 0 && payload_hash != tensor.payload_hash64)) {
            error = "ASTC cache tensor payload checksum mismatch";
            return false;
        }
        if (tensor.representation != astc_vulkan_representation::kPairedD2) return true;
        uint64_t layout_hash = 0;
        if (!range_hash64(layout_path, tensor.layout_byte_offset, tensor.layout_byte_size, layout_hash, buffer) ||
            (tensor.layout_hash64 != 0 && layout_hash != tensor.layout_hash64)) {
            error = "ASTC cache paired layout checksum mismatch";
            return false;
        }
        return true;
    };
    if (manifest.version >= 4) {
        for (const auto & artifact : manifest.artifacts) {
            if (!validate_storage(artifact.storage)) return false;
            if (artifact.normalization == astc_vulkan_normalization::per_row_absmax) {
                uint64_t scale_hash = 0;
                if (artifact.row_scale_byte_offset > row_scale_size ||
                    artifact.row_scale_byte_size > row_scale_size - artifact.row_scale_byte_offset ||
                    !range_hash64(row_scale_path, artifact.row_scale_byte_offset,
                                  artifact.row_scale_byte_size, scale_hash, buffer) ||
                    (artifact.row_scale_hash64 != 0 && scale_hash != artifact.row_scale_hash64)) {
                    error = "ASTC cache artifact row-scale checksum mismatch";
                    return false;
                }
            }
            if (artifact.pair_map_byte_size != 0) {
                uint64_t pair_map_hash = 0;
                if (artifact.pair_map_byte_offset > pair_map_size ||
                    artifact.pair_map_byte_size > pair_map_size - artifact.pair_map_byte_offset ||
                    !range_hash64(pair_map_path, artifact.pair_map_byte_offset,
                                  artifact.pair_map_byte_size, pair_map_hash, buffer) ||
                    (artifact.pair_map_hash64 != 0 && pair_map_hash != artifact.pair_map_hash64)) {
                    error = "ASTC cache artifact pair-map checksum mismatch";
                    return false;
                }
                std::vector<uint8_t> pair_map;
                if (!read_range(pair_map_path, artifact.pair_map_byte_offset,
                                artifact.pair_map_byte_size, pair_map) ||
                    !astc_vulkan_validate_pair_map(artifact, pair_map.data(), pair_map.size(), error)) {
                    if (error.empty()) error = "ASTC cache artifact pair-map cannot be read";
                    return false;
                }
            }
        }
    } else {
        for (const auto & tensor : manifest.tensors) {
            if (!validate_storage(tensor)) return false;
        }
    }
    error.clear();
    return true;
}

bool verify_hash(const std::string & path, const std::string & hash_path,
                 const char * label, std::string & error) {
    std::string expected, actual;
    if (!read_hash(hash_path, expected)) {
        error = std::string("missing or invalid ") + label + " SHA-256 record";
        return false;
    }
    if (!astc_vulkan_sha256_file_hex(path, actual, error)) return false;
    if (actual != expected) {
        error = std::string(label) + " SHA-256 mismatch";
        return false;
    }
    return true;
}

bool copy_file(const std::string & input, const std::string & output, std::string & error) {
    std::error_code ec;
    if (!fs::copy_file(input, output, fs::copy_options::none, ec)) {
        error = "cannot copy ASTC cache file: " + input;
        return false;
    }
    return true;
}

} // namespace

bool astc_vulkan_cache_resolve(const std::string & model_path,
                               const std::string & requested_cache_path,
                               astc_vulkan_cache_paths & paths,
                               std::string & error) {
    if (model_path.empty()) {
        error = "ASTC cache requires a source GGUF path";
        return false;
    }
    fs::path root;
    if (requested_cache_path.empty() || requested_cache_path == "auto") {
        root = fs::path(model_path).string() + ".astc-vulkan";
    } else {
        const fs::path requested(requested_cache_path);
        root = requested.filename() == kManifestFile ? requested.parent_path() : requested;
    }
    if (root.empty()) {
        error = "ASTC cache path is empty";
        return false;
    }
    paths.root = root.string();
    paths.manifest = (root / kManifestFile).string();
    paths.payload = (root / kPayloadFile).string();
    paths.layout = (root / kLayoutFile).string();
    paths.row_scales = (root / kRowScalesFile).string();
    paths.pair_map = (root / kPairMapFile).string();
    paths.provenance = (root / kProvenanceFile).string();
    paths.source_sha256 = (root / kSourceHashFile).string();
    paths.manifest_sha256 = (root / kManifestHashFile).string();
    paths.payload_sha256 = (root / kPayloadHashFile).string();
    paths.layout_sha256 = (root / kLayoutHashFile).string();
    paths.row_scales_sha256 = (root / kRowScalesHashFile).string();
    paths.pair_map_sha256 = (root / kPairMapHashFile).string();
    paths.compatible_bases = (root / kCompatibleBasesDirectory).string();
    error.clear();
    return true;
}

bool astc_vulkan_cache_validate(const std::string & model_path,
                                const std::string & requested_cache_path,
                                astc_vulkan_cache_validation & result,
                                std::string & error) {
    result = {};
    if (!astc_vulkan_cache_resolve(model_path, requested_cache_path, result.paths, error)) return false;
    std::error_code ec;
    if (!fs::is_regular_file(model_path, ec) || !fs::is_directory(result.paths.root, ec)) {
        error = "ASTC cache or source GGUF is unavailable";
        return false;
    }
    if (!verify_hash(model_path, result.paths.source_sha256, "source GGUF", error) ||
        !verify_hash(result.paths.manifest, result.paths.manifest_sha256, "manifest", error) ||
        !verify_hash(result.paths.payload, result.paths.payload_sha256, "payload", error) ||
        !astc_vulkan_read_manifest(result.paths.manifest, result.manifest, error)) return false;
    if (!read_hash(result.paths.source_sha256, result.runtime_base.source_sha256)) {
        error = "missing or invalid source GGUF SHA-256 record";
        return false;
    }
    result.runtime_base.is_source = true;
    result.runtime_base.admitted = true;
    result.runtime_base.model_gate_passed = true;
    result.runtime_base.vulkan_gate_passed = true;
    result.runtime_base.runtime_sha256 = result.runtime_base.source_sha256;
    result.runtime_base.family = "source";

    result.has_paired_d2 = has_paired_d2(result.manifest);
    result.has_row_scales = has_row_scales(result.manifest);
    result.has_pair_map = has_pair_map(result.manifest);
    uint64_t payload_size = 0;
    uint64_t layout_size = 0;
    uint64_t row_scale_size = 0;
    uint64_t pair_map_size = 0;
    if (!file_size(result.paths.payload, payload_size)) {
        error = "cannot determine ASTC cache payload size";
        return false;
    }
    if (result.has_paired_d2) {
        if (!verify_hash(result.paths.layout, result.paths.layout_sha256, "layout map", error)) return false;
        if (!file_size(result.paths.layout, layout_size)) {
            error = "cannot determine ASTC cache layout size";
            return false;
        }
    }
    if (result.has_row_scales) {
        if (!verify_hash(result.paths.row_scales, result.paths.row_scales_sha256, "row-scales", error) ||
            !file_size(result.paths.row_scales, row_scale_size)) {
            if (error.empty()) error = "cannot determine ASTC cache row-scale size";
            return false;
        }
    }
    if (result.has_pair_map) {
        if (!verify_hash(result.paths.pair_map, result.paths.pair_map_sha256, "pair map", error) ||
            !file_size(result.paths.pair_map, pair_map_size)) {
            if (error.empty()) error = "cannot determine ASTC cache pair-map size";
            return false;
        }
    }
    if (payload_size == 0 || (result.has_paired_d2 && layout_size == 0) ||
        (result.has_row_scales && row_scale_size == 0) || (result.has_pair_map && pair_map_size == 0) ||
        !validate_tensor_payloads(result.manifest, payload_size, result.paths.payload,
                                  layout_size, result.paths.layout,
                                  row_scale_size, result.paths.row_scales,
                                  pair_map_size, result.paths.pair_map, error)) return false;
    error.clear();
    return true;
}

bool astc_vulkan_cache_admit_runtime_base(
        const std::string & source_model_path, const std::string & runtime_model_path,
        const std::string & requested_cache_path, const std::string & family,
        astc_vulkan_cache_runtime_base & binding, std::string & error) {
    binding = {};
    if (!valid_binding_value(family)) {
        error = "ASTC runtime-base family must be a non-empty single-line value";
        return false;
    }
    // Admission starts from the original strict validation: the cache must
    // still be proven against its exact F16/BF16 source model.
    astc_vulkan_cache_validation source_cache;
    if (!astc_vulkan_cache_validate(source_model_path, requested_cache_path, source_cache, error)) {
        return false;
    }
    std::string source_schema;
    std::string runtime_schema;
    std::string source_hash;
    std::string runtime_hash;
    if (!gguf_schema_sha256(source_model_path, source_schema, error) ||
        !gguf_schema_sha256(runtime_model_path, runtime_schema, error) ||
        !astc_vulkan_sha256_file_hex(source_model_path, source_hash, error) ||
        !astc_vulkan_sha256_file_hex(runtime_model_path, runtime_hash, error)) return false;
    if (source_hash != source_cache.runtime_base.runtime_sha256) {
        error = "ASTC cache source hash changed during runtime-base admission";
        return false;
    }
    if (source_schema != runtime_schema) {
        std::string schema_error;
        if (!gguf_schema_matches(source_model_path, runtime_model_path, schema_error)) {
            error = "runtime GGUF schema does not match the ASTC source model: " + schema_error;
        } else {
            error = "runtime GGUF schema fingerprint differs from the ASTC source model";
        }
        return false;
    }
    if (source_hash == runtime_hash) {
        error = "runtime GGUF is already the ASTC cache source model";
        return false;
    }
    std::error_code ec;
    const fs::path directory(source_cache.paths.compatible_bases);
    if (!fs::create_directories(directory, ec) && ec) {
        error = "cannot create ASTC compatible-base directory";
        return false;
    }
    const fs::path final_path = directory / (runtime_hash + kRuntimeBaseExtension);
    if (fs::exists(final_path, ec)) {
        error = "runtime GGUF is already admitted for this ASTC cache";
        return false;
    }
    binding.admitted = true;
    binding.source_sha256 = source_hash;
    binding.runtime_sha256 = runtime_hash;
    binding.schema_sha256 = source_schema;
    binding.family = family;
    // Runtime quality gates are deliberately false at registration. A later
    // replay/publish flow must attach measured evidence before auto-selection.
    const fs::path temporary = final_path.string() + ".partial";
    if (!write_binding(temporary.string(), binding, error)) return false;
    fs::rename(temporary, final_path, ec);
    if (ec) {
        fs::remove(temporary, ec);
        error = "cannot atomically publish ASTC runtime-base binding";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_cache_validate_runtime_base(
        const std::string & source_model_path, const std::string & runtime_model_path,
        const std::string & requested_cache_path, astc_vulkan_cache_runtime_base & binding,
        std::string & error) {
    binding = {};
    astc_vulkan_cache_validation source_cache;
    if (!astc_vulkan_cache_validate(source_model_path, requested_cache_path, source_cache, error)) {
        return false;
    }
    std::string runtime_hash;
    std::string runtime_schema;
    if (!astc_vulkan_sha256_file_hex(runtime_model_path, runtime_hash, error) ||
        !gguf_schema_sha256(runtime_model_path, runtime_schema, error)) return false;
    const fs::path binding_path = fs::path(source_cache.paths.compatible_bases) /
        (runtime_hash + kRuntimeBaseExtension);
    if (!read_binding(binding_path.string(), binding)) {
        error = "runtime GGUF has no ASTC admission record";
        return false;
    }
    if (binding.source_sha256 != source_cache.runtime_base.runtime_sha256 ||
        binding.runtime_sha256 != runtime_hash || binding.schema_sha256 != runtime_schema) {
        error = "ASTC runtime-base admission record does not match the supplied GGUFs";
        binding = {};
        return false;
    }
    binding.is_source = false;
    error.clear();
    return true;
}

bool astc_vulkan_cache_create(const std::string & model_path,
                              const std::string & manifest_input,
                              const std::string & payload_input,
                              const std::string & layout_input,
                              const std::string & provenance_input,
                              const std::string & requested_cache_path,
                              astc_vulkan_cache_paths & paths,
                              std::string & error) {
    return astc_vulkan_cache_create_with_row_scales(
        model_path, manifest_input, payload_input, layout_input, {}, provenance_input,
        requested_cache_path, paths, error);
}

bool astc_vulkan_cache_create_with_row_scales(const std::string & model_path,
                                              const std::string & manifest_input,
                                              const std::string & payload_input,
                                              const std::string & layout_input,
                                              const std::string & row_scales_input,
                                              const std::string & provenance_input,
                                              const std::string & requested_cache_path,
                                              astc_vulkan_cache_paths & paths,
                                              std::string & error) {
    return astc_vulkan_cache_create_with_metadata(
        model_path, manifest_input, payload_input, layout_input, row_scales_input, {}, provenance_input,
        requested_cache_path, paths, error);
}

bool astc_vulkan_cache_create_with_metadata(const std::string & model_path,
                                            const std::string & manifest_input,
                                            const std::string & payload_input,
                                            const std::string & layout_input,
                                            const std::string & row_scales_input,
                                            const std::string & pair_map_input,
                                            const std::string & provenance_input,
                                            const std::string & requested_cache_path,
                                            astc_vulkan_cache_paths & paths,
                                            std::string & error) {
    if (!astc_vulkan_cache_resolve(model_path, requested_cache_path, paths, error)) return false;
    std::error_code ec;
    if (!fs::is_regular_file(model_path, ec) || !fs::is_regular_file(manifest_input, ec) ||
        !fs::is_regular_file(payload_input, ec)) {
        error = "ASTC cache creation requires a GGUF, manifest and payload file";
        return false;
    }
    if (fs::exists(paths.root, ec)) {
        error = "ASTC cache already exists; refuse to overwrite it";
        return false;
    }
    astc_vulkan_manifest manifest;
    if (!astc_vulkan_read_manifest(manifest_input, manifest, error)) return false;
    const bool paired = has_paired_d2(manifest);
    const bool scaled = has_row_scales(manifest);
    const bool paired_rows = has_pair_map(manifest);
    if (paired && (!fs::is_regular_file(layout_input, ec))) {
        error = "paired-D2 cache creation requires a layout map";
        return false;
    }
    if (scaled && (!fs::is_regular_file(row_scales_input, ec))) {
        error = "row-scaled artifact cache creation requires a row-scale blob";
        return false;
    }
    if (paired_rows && (!fs::is_regular_file(pair_map_input, ec))) {
        error = "pair-optimized artifact cache creation requires a pair map";
        return false;
    }
    uint64_t payload_size = 0;
    uint64_t layout_size = 0;
    uint64_t row_scale_size = 0;
    uint64_t pair_map_size = 0;
    if (!file_size(payload_input, payload_size) || payload_size == 0 ||
        (paired && (!file_size(layout_input, layout_size) || layout_size == 0)) ||
        (scaled && (!file_size(row_scales_input, row_scale_size) || row_scale_size == 0)) ||
        (paired_rows && (!file_size(pair_map_input, pair_map_size) || pair_map_size == 0)) ||
        !validate_tensor_payloads(manifest, payload_size, payload_input, layout_size, layout_input,
                                  row_scale_size, row_scales_input,
                                  pair_map_size, pair_map_input, error)) return false;

    std::string source_hash, manifest_hash, payload_hash, layout_hash, row_scale_hash, pair_map_hash;
    if (!astc_vulkan_sha256_file_hex(model_path, source_hash, error) ||
        !astc_vulkan_sha256_file_hex(manifest_input, manifest_hash, error) ||
        !astc_vulkan_sha256_file_hex(payload_input, payload_hash, error) ||
        (paired && !astc_vulkan_sha256_file_hex(layout_input, layout_hash, error)) ||
        (scaled && !astc_vulkan_sha256_file_hex(row_scales_input, row_scale_hash, error)) ||
        (paired_rows && !astc_vulkan_sha256_file_hex(pair_map_input, pair_map_hash, error))) return false;

    const fs::path partial = paths.root + ".partial-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto fail = [&](const std::string & reason) {
        std::error_code remove_error;
        fs::remove_all(partial, remove_error);
        error = reason;
        return false;
    };
    if (!fs::create_directories(partial, ec) || ec) return fail("cannot create ASTC cache staging directory");
    const astc_vulkan_cache_paths staging = [&]() {
        astc_vulkan_cache_paths result;
        std::string ignored;
        astc_vulkan_cache_resolve(model_path, partial.string(), result, ignored);
        return result;
    }();
    if (!copy_file(manifest_input, staging.manifest, error) ||
        !copy_file(payload_input, staging.payload, error) ||
        (paired && !copy_file(layout_input, staging.layout, error)) ||
        (scaled && !copy_file(row_scales_input, staging.row_scales, error)) ||
        (paired_rows && !copy_file(pair_map_input, staging.pair_map, error)) ||
        (!provenance_input.empty() && !copy_file(provenance_input, staging.provenance, error)) ||
        !write_hash(staging.source_sha256, source_hash) ||
        !write_hash(staging.manifest_sha256, manifest_hash) ||
        !write_hash(staging.payload_sha256, payload_hash) ||
        (paired && !write_hash(staging.layout_sha256, layout_hash)) ||
        (scaled && !write_hash(staging.row_scales_sha256, row_scale_hash)) ||
        (paired_rows && !write_hash(staging.pair_map_sha256, pair_map_hash))) {
        return fail(error.empty() ? "cannot write ASTC cache staging files" : error);
    }
    astc_vulkan_cache_validation validation;
    if (!astc_vulkan_cache_validate(model_path, partial.string(), validation, error)) return fail(error);
    fs::rename(partial, paths.root, ec);
    if (ec) return fail("cannot atomically publish ASTC cache");
    error.clear();
    return true;
}
