#include "astc-vulkan-compiled-model.h"

#include "astc-vulkan-cache.h"
#include "astc-vulkan-compiled-catalog.h"
#include "astc-vulkan-hash.h"
#include "astc-vulkan-manifest.h"
#include "astc-vulkan-native-tensor-table.h"
#include "astc-vulkan-artifact-policy.h"

#include "gguf.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_set>

namespace {

constexpr std::array<char, 8> kMagic = {'A', 'S', 'T', 'C', 'M', '0', '0', '1'};
constexpr uint32_t kV1SectionCount = 8;
constexpr uint32_t kV2SectionCount = 13;
constexpr uint64_t kMaxSectionBytes = 16ull * 1024ull * 1024ull * 1024ull;

struct section_header {
    uint64_t size = 0;
    uint64_t hash = 0;
};

bool write_u32(std::ofstream & file, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        file.put(static_cast<char>((value >> shift) & 0xffu));
    }
    return file.good();
}

bool write_u64(std::ofstream & file, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        file.put(static_cast<char>((value >> shift) & 0xffu));
    }
    return file.good();
}

bool read_u32(std::ifstream & file, uint32_t & value) {
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        char byte = 0;
        if (!file.get(byte)) return false;
        value |= static_cast<uint32_t>(static_cast<unsigned char>(byte)) << shift;
    }
    return true;
}

bool read_u64(std::ifstream & file, uint64_t & value) {
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        char byte = 0;
        if (!file.get(byte)) return false;
        value |= static_cast<uint64_t>(static_cast<unsigned char>(byte)) << shift;
    }
    return true;
}

bool read_file(const std::string & path, std::vector<uint8_t> & bytes, std::string & error,
               bool required = true) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (!required) {
            bytes.clear();
            return true;
        }
        error = "cannot open compiled-model section: " + path;
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff end = file.tellg();
    if (end < 0 || static_cast<uint64_t>(end) > kMaxSectionBytes) {
        error = "compiled-model section is too large: " + path;
        return false;
    }
    file.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(end));
    if (!bytes.empty()) file.read(reinterpret_cast<char *>(bytes.data()), end);
    if (!file.good()) {
        error = "cannot read compiled-model section: " + path;
        return false;
    }
    return true;
}

bool read_file_range(const std::string & path, uint64_t offset, uint64_t size,
                     std::vector<uint8_t> & bytes, std::string & error) {
    if (size > kMaxSectionBytes || offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        error = "compiled-model source range exceeds safety limit";
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) { error = "cannot open source GGUF: " + path; return false; }
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    bytes.resize(static_cast<size_t>(size));
    if (size && !file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size))) {
        error = "cannot read source GGUF tensor bytes"; return false;
    }
    return true;
}

uint64_t hash_bytes(const std::vector<uint8_t> & bytes) {
    return astc_vulkan_payload_hash64(bytes.data(), bytes.size());
}

bool write_section(std::ofstream & file, const std::vector<uint8_t> & bytes) {
    return bytes.empty() ||
        (file.write(reinterpret_cast<const char *>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size())), file.good());
}

bool read_section(std::ifstream & file, const section_header & header,
                  std::vector<uint8_t> & bytes, std::string & error) {
    if (header.size > kMaxSectionBytes || header.size >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = "compiled-model section exceeds safety limit";
        return false;
    }
    bytes.resize(static_cast<size_t>(header.size));
    if (!bytes.empty() && !file.read(reinterpret_cast<char *>(bytes.data()),
                                     static_cast<std::streamsize>(bytes.size()))) {
        error = "truncated compiled-model section";
        return false;
    }
    if (hash_bytes(bytes) != header.hash) {
        error = "compiled-model section checksum mismatch";
        return false;
    }
    return true;
}

std::vector<const std::vector<uint8_t> *> sections(const astc_vulkan_compiled_model & model) {
    return {&model.gguf, &model.manifest, &model.payload, &model.layout,
            &model.row_scales, &model.pair_map, &model.provenance, &model.catalog,
            &model.native_table, &model.native_payload, &model.embedding_metadata,
            &model.embedding_payload, &model.embedding_affine};
}

std::vector<std::vector<uint8_t> *> mutable_sections(astc_vulkan_compiled_model & model) {
    return {&model.gguf, &model.manifest, &model.payload, &model.layout,
            &model.row_scales, &model.pair_map, &model.provenance, &model.catalog,
            &model.native_table, &model.native_payload, &model.embedding_metadata,
            &model.embedding_payload, &model.embedding_affine};
}

uint32_t section_count(const astc_vulkan_compiled_model & model) {
    return model.version >= 2 ? kV2SectionCount : kV1SectionCount;
}

bool valid_storage_mode(astc_vulkan_compiled_storage_mode mode) {
    return mode == astc_vulkan_compiled_storage_mode::hybrid ||
           mode == astc_vulkan_compiled_storage_mode::strict;
}

bool read_embedded_manifest(const std::vector<uint8_t> & bytes,
                            astc_vulkan_manifest & manifest,
                            std::string & error) {
    const auto nonce = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&bytes));
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("astc-vulkan-compiled-model-manifest-" + std::to_string(nonce) + ".astcv");
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file || !write_section(file, bytes)) {
            error = "cannot stage embedded compiled-model manifest";
            file.close();
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            return false;
        }
    }
    const bool ok = astc_vulkan_read_manifest(path.string(), manifest, error);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return ok;
}

// Keep compiled-model rewrites self-contained while reusing the canonical
// manifest serializer.  The temporary file is only an implementation detail;
// the resulting bytes are embedded in the ASTCCM section.
bool write_embedded_manifest(const astc_vulkan_manifest & manifest,
                             std::vector<uint8_t> & bytes,
                             std::string & error) {
    const auto nonce = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&manifest));
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("astc-vulkan-compiled-model-manifest-write-" + std::to_string(nonce) + ".astcv");
    if (!astc_vulkan_write_manifest(path.string(), manifest, error)) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        return false;
    }
    const bool ok = read_file(path.string(), bytes, error);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return ok;
}

bool strict_manifest_filter(astc_vulkan_manifest & manifest, std::string & error) {
    // Legacy single-artifact manifests have no split evidence fields. Keep
    // their established strict behavior; v4+ artifact tables can be filtered
    // safely because each entry carries independent model/Vulkan gates.
    if (manifest.version < 4) return true;
    // Some early v2 containers carried a v7 header but only the legacy
    // tensor table. Those records have no independent evidence and therefore
    // must remain native-only in a strict container.
    if (manifest.artifacts.empty()) {
        // A v4+ manifest is required to contain at least one artifact.  A
        // strict native-only container is nevertheless a valid result when
        // every candidate was rejected by the evidence/rollout policy.  Use
        // the established legacy empty-table representation instead of
        // serializing an invalid v4+ header; the native tensor table remains
        // the authoritative fallback for every logical tensor.
        manifest.version = 3;
        manifest.tensors.clear();
        return astc_vulkan_validate_manifest(manifest, error);
    }
    std::vector<astc_vulkan_artifact_record> approved;
    approved.reserve(manifest.artifacts.size());
    for (const auto & artifact : manifest.artifacts) {
        astc_vulkan_artifact_candidate candidate;
        candidate.tensor = &artifact.storage;
        candidate.variant = artifact.variant;
        candidate.normalization = artifact.normalization;
        candidate.evidence = artifact.evidence;
        candidate.rate_bpw = astc_vulkan_artifact_storage_bpw(artifact);
        candidate.artifact_id = artifact.id;
        // Match the production metadata predicate (including robust replay
        // thresholds), while allowing any device/memory during packaging.
        const bool runtime_representation_supported =
            artifact.storage.representation == astc_vulkan_representation::kScalar ||
            artifact.storage.representation == astc_vulkan_representation::kGaugeLumaAlpha ||
            artifact.storage.representation == astc_vulkan_representation::kPairedD2;
        const bool runtime_shape_supported =
            runtime_representation_supported &&
            astc_vulkan_footprint_is_valid(artifact.storage.footprint) &&
            !astc_vulkan_footprint_is_experimental(artifact.storage.footprint);
        const bool approved_for_production = runtime_shape_supported &&
            astc_vulkan_artifact_is_eligible(candidate, true, true, {});
        if (approved_for_production) {
            approved.push_back(artifact);
        }
    }
    manifest.artifacts = std::move(approved);
    if (manifest.artifacts.empty()) {
        // See the native-only case above.  Do this after filtering so a
        // strict conversion can safely represent a cache whose D1/D2 entries
        // are all experimental or lack model/Vulkan evidence.
        manifest.version = 3;
        manifest.tensors.clear();
    }
    return astc_vulkan_validate_manifest(manifest, error);
}

std::unordered_set<std::string> manifest_astc_tensor_names(const astc_vulkan_manifest & manifest) {
    std::unordered_set<std::string> names;
    if (!manifest.artifacts.empty()) {
        for (const auto & artifact : manifest.artifacts) names.insert(artifact.storage.name);
    } else {
        for (const auto & tensor : manifest.tensors) names.insert(tensor.name);
    }
    return names;
}

bool validate_bootstrap(const astc_vulkan_compiled_model & model, std::string & error) {
    if (model.gguf.empty() || model.native_table.empty()) {
        error = "bootstrap compiled-model is missing GGUF metadata or native table";
        return false;
    }
    const bool has_embedding = !model.embedding_metadata.empty() || !model.embedding_payload.empty() ||
        !model.embedding_affine.empty();
    if (has_embedding && (model.embedding_metadata.empty() || model.embedding_payload.empty() ||
                          model.embedding_affine.empty())) {
        error = "bootstrap embedding annex is incomplete";
        return false;
    }
    gguf_init_params params{true, nullptr};
    gguf_context * ctx = gguf_init_from_buffer(model.gguf.data(), model.gguf.size(), params);
    if (!ctx) { error = "bootstrap GGUF metadata cannot be parsed"; return false; }
    const size_t offset = gguf_get_data_offset(ctx);
    const int64_t tensor_count = gguf_get_n_tensors(ctx);
    gguf_free(ctx);
    if (offset != model.gguf.size() || tensor_count <= 0) {
        error = "bootstrap GGUF does not end exactly at its tensor-data offset";
        return false;
    }
    if (!valid_storage_mode(model.storage_mode)) {
        error = "bootstrap compiled-model has an invalid storage mode";
        return false;
    }
    astc_vulkan_native_tensor_table table;
    if (!astc_vulkan_decode_native_tensor_table(model.native_table, table, error)) return false;
    if (table.tensors.size() != static_cast<size_t>(tensor_count)) {
        error = "bootstrap native table does not cover every GGUF tensor";
        return false;
    }
    for (const auto & record : table.tensors) {
        if (record.storage == astc_vulkan_native_tensor_storage::native &&
            (record.native_offset > model.native_payload.size() ||
             record.native_size > model.native_payload.size() - record.native_offset ||
             astc_vulkan_payload_hash64(model.native_payload.data() + record.native_offset,
                                        record.native_size) != record.native_hash64)) {
            error = "bootstrap native payload range or checksum is invalid";
            return false;
        }
    }
    return true;
}

} // namespace

bool astc_vulkan_compiled_model_validate(
    const astc_vulkan_compiled_model & model,
    std::string & error) {
    if ((model.version != 1 && model.version != astc_vulkan_compiled_model::kCurrentVersion) ||
        model.gguf.empty() || model.manifest.empty() || model.payload.empty() ||
        model.source_model_fingerprint.empty()) {
        error = "compiled-model is missing required GGUF, manifest, payload or fingerprint";
        return false;
    }
    for (const auto * section : sections(model)) {
        if (section->size() > kMaxSectionBytes) {
            error = "compiled-model section exceeds safety limit";
            return false;
        }
    }
    if (model.version >= 2 && !validate_bootstrap(model, error)) return false;

    // Parse the embedded manifest as a final structural gate.  Payload and
    // per-blob checksums were already verified when the cache was packed; this
    // check ensures the container cannot contain arbitrary required bytes.
    astc_vulkan_manifest manifest;
    const bool ok = read_embedded_manifest(model.manifest, manifest, error);
    if (!ok || !astc_vulkan_validate_manifest(manifest, error) ||
        !astc_vulkan_validate_payload_blob(manifest, model.payload.size(), error) ||
        !astc_vulkan_validate_layout_blob(manifest, model.layout.size(), error)) {
        return false;
    }
    for (const auto & artifact : manifest.artifacts) {
        if (artifact.row_scale_byte_offset > model.row_scales.size() ||
            artifact.row_scale_byte_size > model.row_scales.size() - artifact.row_scale_byte_offset ||
            artifact.pair_map_byte_offset > model.pair_map.size() ||
            artifact.pair_map_byte_size > model.pair_map.size() - artifact.pair_map_byte_offset) {
            error = "compiled-model metadata range exceeds its section";
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_compiled_model_write(
    const std::string & output_path,
    const astc_vulkan_compiled_model & model,
    std::string & error) {
    if (!astc_vulkan_compiled_model_validate(model, error)) return false;
    if (model.source_model_fingerprint.size() > (1u << 20)) {
        error = "compiled-model fingerprint is too long";
        return false;
    }

    const std::filesystem::path output = output_path;
    if (std::filesystem::exists(output)) {
        error = "compiled-model output already exists: " + output_path;
        return false;
    }
    const std::filesystem::path staging = output.string() + ".tmp";
    std::error_code ignored;
    std::filesystem::remove(staging, ignored);
    std::ofstream file(staging, std::ios::binary | std::ios::trunc);
    if (!file) {
        error = "cannot open compiled-model output: " + staging.string();
        return false;
    }
    file.write(kMagic.data(), kMagic.size());
    if (!file.good() || !write_u32(file, model.version) || !write_u32(file, section_count(model)) ||
        !write_u32(file, static_cast<uint32_t>(model.source_model_fingerprint.size())) ||
        !write_u32(file, model.version >= 2 ? static_cast<uint32_t>(model.storage_mode) : 0) ||
        (!model.source_model_fingerprint.empty() &&
         (file.write(model.source_model_fingerprint.data(),
                     static_cast<std::streamsize>(model.source_model_fingerprint.size())),
          !file.good()))) {
        error = "cannot write compiled-model header";
        file.close();
        std::filesystem::remove(staging, ignored);
        return false;
    }
    const auto all_sections = sections(model);
    for (size_t index = 0; index < section_count(model); ++index) {
        const auto * section = all_sections[index];
        if (!write_u64(file, section->size()) || !write_u64(file, hash_bytes(*section))) {
            error = "cannot write compiled-model section header";
            file.close();
            std::filesystem::remove(staging, ignored);
            return false;
        }
    }
    for (size_t index = 0; index < section_count(model); ++index) {
        const auto * section = all_sections[index];
        if (!write_section(file, *section)) {
            error = "cannot write compiled-model section";
            file.close();
            std::filesystem::remove(staging, ignored);
            return false;
        }
    }
    const bool write_ok = file.good();
    file.close();
    if (!write_ok) {
        error = "cannot flush compiled-model output: " + output_path;
        std::filesystem::remove(staging, ignored);
        return false;
    }
    std::error_code publish_error;
    std::filesystem::rename(staging, output, publish_error);
    if (publish_error) {
        error = "cannot publish compiled-model output: " + output_path;
        std::filesystem::remove(staging, ignored);
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_compiled_model_read(
    const std::string & input_path,
    astc_vulkan_compiled_model & model,
    std::string & error) {
    model = {};
    std::ifstream file(input_path, std::ios::binary);
    if (!file) {
        error = "cannot open compiled-model input: " + input_path;
        return false;
    }
    std::array<char, 8> magic{};
    uint32_t version = 0, count = 0, fingerprint_size = 0, reserved = 0;
    if (!file.read(magic.data(), magic.size()) || magic != kMagic ||
        !read_u32(file, version) || !read_u32(file, count) ||
        !read_u32(file, fingerprint_size) || !read_u32(file, reserved) ||
        (version != 1 && version != astc_vulkan_compiled_model::kCurrentVersion) ||
        count != (version >= 2 ? kV2SectionCount : kV1SectionCount) ||
        fingerprint_size > (1u << 20)) {
        error = "invalid compiled-model header";
        return false;
    }
    model.version = version;
    // Early v2 files used the reserved field as zero and were strict by
    // construction. Preserve that behavior instead of treating them as the
    // new safe hybrid default.
    if (version >= 2) {
        if (reserved == 0) model.storage_mode = astc_vulkan_compiled_storage_mode::strict;
        else model.storage_mode = static_cast<astc_vulkan_compiled_storage_mode>(reserved);
        if (!valid_storage_mode(model.storage_mode)) {
            error = "invalid bootstrap compiled-model storage mode";
            return false;
        }
    }
    model.source_model_fingerprint.resize(fingerprint_size);
    if (fingerprint_size &&
        !file.read(model.source_model_fingerprint.data(), fingerprint_size)) {
        error = "truncated compiled-model fingerprint";
        return false;
    }
    std::array<section_header, kV2SectionCount> headers{};
    uint64_t total = 0;
    for (uint32_t index = 0; index < count; ++index) {
        auto & header = headers[index];
        if (!read_u64(file, header.size) || !read_u64(file, header.hash) ||
            header.size > kMaxSectionBytes ||
            header.size > std::numeric_limits<uint64_t>::max() - total) {
            error = "invalid compiled-model section table";
            return false;
        }
        total += header.size;
    }
    auto target = mutable_sections(model);
    for (size_t index = 0; index < count; ++index) {
        if (!read_section(file, headers[index], *target[index], error)) return false;
    }
    if (file.peek() != std::ifstream::traits_type::eof()) {
        error = "compiled-model has trailing bytes";
        return false;
    }
    return astc_vulkan_compiled_model_validate(model, error);
}

bool astc_vulkan_compiled_model_extract_gguf(
    const astc_vulkan_compiled_model & model,
    const std::string & output_path,
    std::string & error) {
    if (!astc_vulkan_compiled_model_validate(model, error)) return false;
    std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
    if (!file || !write_section(file, model.gguf)) {
        error = "cannot write extracted GGUF: " + output_path;
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_compiled_model_pack(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const std::string & output_path,
    std::string & error) {
    astc_vulkan_cache_validation validation;
    if (!astc_vulkan_cache_validate(model_path, requested_cache_path, validation, error)) {
        return false;
    }
    astc_vulkan_compiled_model model;
    if (!read_file(model_path, model.gguf, error)) return false;
    if (!read_file(validation.paths.manifest, model.manifest, error) ||
        !read_file(validation.paths.payload, model.payload, error) ||
        !read_file(validation.paths.layout, model.layout, error, false) ||
        !read_file(validation.paths.row_scales, model.row_scales, error, false) ||
        !read_file(validation.paths.pair_map, model.pair_map, error, false) ||
        !read_file(validation.paths.provenance, model.provenance, error, false) ||
        !read_file(validation.paths.catalog, model.catalog, error, false)) {
        return false;
    }
    // Keep the source fingerprint as text; it is provenance rather than a
    // separate payload section.
    std::vector<uint8_t> fingerprint_bytes;
    if (!read_file(validation.paths.source_sha256, fingerprint_bytes, error)) return false;
    model.source_model_fingerprint.assign(
        reinterpret_cast<const char *>(fingerprint_bytes.data()), fingerprint_bytes.size());
    while (!model.source_model_fingerprint.empty() &&
           (model.source_model_fingerprint.back() == '\n' ||
            model.source_model_fingerprint.back() == '\r')) {
        model.source_model_fingerprint.pop_back();
    }
    return astc_vulkan_compiled_model_write(output_path, model, error);
}

bool astc_vulkan_compiled_model_pack_bootstrap(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const std::string & output_path,
    astc_vulkan_compiled_storage_mode storage_mode,
    std::string & error) {
    astc_vulkan_cache_validation validation;
    if (!astc_vulkan_cache_validate(model_path, requested_cache_path, validation, error)) return false;

    astc_vulkan_compiled_model model;
    model.version = astc_vulkan_compiled_model::kCurrentVersion;
    model.storage_mode = storage_mode;
    if (!valid_storage_mode(storage_mode)) { error = "invalid bootstrap storage mode"; return false; }
    if (!read_file(validation.paths.manifest, model.manifest, error) ||
        !read_file(validation.paths.payload, model.payload, error) ||
        !read_file(validation.paths.layout, model.layout, error, false) ||
        !read_file(validation.paths.row_scales, model.row_scales, error, false) ||
        !read_file(validation.paths.pair_map, model.pair_map, error, false) ||
        !read_file(validation.paths.provenance, model.provenance, error, false) ||
        !read_file(validation.paths.catalog, model.catalog, error, false)) return false;
    std::vector<uint8_t> fingerprint_bytes;
    if (!read_file(validation.paths.source_sha256, fingerprint_bytes, error)) return false;
    model.source_model_fingerprint.assign(reinterpret_cast<const char *>(fingerprint_bytes.data()), fingerprint_bytes.size());
    while (!model.source_model_fingerprint.empty() &&
           (model.source_model_fingerprint.back() == '\n' || model.source_model_fingerprint.back() == '\r')) {
        model.source_model_fingerprint.pop_back();
    }

    astc_vulkan_manifest manifest;
    if (!read_embedded_manifest(model.manifest, manifest, error)) return false;
    // E1 is deliberately an annex rather than a matrix-manifest artifact.
    // Carry its immutable resources in ASTCCM v2, but retain the native
    // embedding until the E1 provider has a strict resource gate of its own.
    const std::filesystem::path cache_root = validation.paths.root;
    const std::filesystem::path embedding_meta = cache_root / "embedding-10x5.astce";
    const std::filesystem::path embedding_payload = cache_root / "embedding-10x5.astcpack";
    const std::filesystem::path embedding_affine = cache_root / "embedding-10x5-affine.bin";
    std::error_code embedding_ec;
    const bool any_embedding = std::filesystem::exists(embedding_meta, embedding_ec) ||
        std::filesystem::exists(embedding_payload, embedding_ec) ||
        std::filesystem::exists(embedding_affine, embedding_ec);
    if (any_embedding && (!read_file(embedding_meta.string(), model.embedding_metadata, error) ||
                          !read_file(embedding_payload.string(), model.embedding_payload, error) ||
                          !read_file(embedding_affine.string(), model.embedding_affine, error))) {
        return false;
    }
    // The E1 annex is always optional today, so token_embd remains native in
    // both modes. In strict mode only evidence-approved matrix artifacts are
    // allowed to displace native bytes; rejected/experimental entries remain
    // native-only and are removed from the embedded ASTC manifest.
    if (storage_mode == astc_vulkan_compiled_storage_mode::strict) {
        if (!strict_manifest_filter(manifest, error)) return false;
        if (!write_embedded_manifest(manifest, model.manifest, error)) return false;
        astc_vulkan_compiled_catalog filtered_catalog;
        if (!astc_vulkan_compiled_catalog_from_manifest(manifest, filtered_catalog, error)) {
            return false;
        }
        const auto nonce = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&filtered_catalog));
        const std::filesystem::path catalog_path = std::filesystem::temp_directory_path() /
            ("astc-vulkan-compiled-model-catalog-write-" + std::to_string(nonce) + ".astcc");
        if (!astc_vulkan_write_compiled_catalog(catalog_path.string(), filtered_catalog, error) ||
            !read_file(catalog_path.string(), model.catalog, error)) {
            std::error_code ignored;
            std::filesystem::remove(catalog_path, ignored);
            return false;
        }
        std::error_code ignored;
        std::filesystem::remove(catalog_path, ignored);
    }
    const auto astc_names = manifest_astc_tensor_names(manifest);

    gguf_init_params params{true, nullptr};
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) { error = "cannot parse source GGUF for bootstrap compilation"; return false; }
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t count = gguf_get_n_tensors(ctx);
    if (count <= 0) { gguf_free(ctx); error="source GGUF has no tensors"; return false; }
    if (!read_file_range(model_path, 0, data_offset, model.gguf, error)) { gguf_free(ctx); return false; }
    astc_vulkan_native_tensor_table table;
    table.tensors.reserve(static_cast<size_t>(count));
    for (int64_t i=0; i<count; ++i) {
        astc_vulkan_native_tensor_record record;
        record.name = gguf_get_tensor_name(ctx, i);
        record.ggml_type = static_cast<uint32_t>(gguf_get_tensor_type(ctx, i));
        const int64_t * ne = gguf_get_tensor_ne(ctx, i);
        record.n_dims = 1;
        for (uint32_t d=0; d<4; ++d) { record.ne[d] = static_cast<uint64_t>(ne[d]); if (ne[d] > 1) record.n_dims=d+1; }
        if (storage_mode == astc_vulkan_compiled_storage_mode::strict && astc_names.count(record.name)) {
            record.storage = astc_vulkan_native_tensor_storage::astc;
        } else {
            record.storage = astc_vulkan_native_tensor_storage::native;
            record.native_offset = model.native_payload.size();
            record.native_size = gguf_get_tensor_size(ctx, i);
            std::vector<uint8_t> tensor_bytes;
            if (!read_file_range(model_path, data_offset + gguf_get_tensor_offset(ctx, i),
                                 record.native_size, tensor_bytes, error)) { gguf_free(ctx); return false; }
            record.native_hash64 = astc_vulkan_payload_hash64(tensor_bytes.data(), tensor_bytes.size());
            model.native_payload.insert(model.native_payload.end(), tensor_bytes.begin(), tensor_bytes.end());
        }
        table.tensors.push_back(std::move(record));
    }
    gguf_free(ctx);
    if (!astc_vulkan_encode_native_tensor_table(table, model.native_table, error)) return false;
    return astc_vulkan_compiled_model_write(output_path, model, error);
}

const char * astc_vulkan_compiled_storage_mode_name(astc_vulkan_compiled_storage_mode mode) {
    switch (mode) {
        case astc_vulkan_compiled_storage_mode::hybrid: return "hybrid";
        case astc_vulkan_compiled_storage_mode::strict: return "strict";
    }
    return "invalid";
}

bool astc_vulkan_compiled_model_convert_to_strict(
    const std::string & input_path,
    const std::string & output_path,
    std::string & error) {
    astc_vulkan_compiled_model model;
    if (!astc_vulkan_compiled_model_read(input_path, model, error)) return false;
    if (!model.is_bootstrap_v2()) {
        error = "only ASTCCM v2 bootstrap containers can be converted to strict";
        return false;
    }
    if (model.is_strict()) {
        error = "compiled model is already strict";
        return false;
    }
    astc_vulkan_manifest manifest;
    if (!read_embedded_manifest(model.manifest, manifest, error)) return false;
    if (!strict_manifest_filter(manifest, error)) return false;
    if (!write_embedded_manifest(manifest, model.manifest, error)) return false;
    astc_vulkan_compiled_catalog filtered_catalog;
    if (!astc_vulkan_compiled_catalog_from_manifest(manifest, filtered_catalog, error)) {
        return false;
    }
    const auto nonce = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&filtered_catalog));
    const std::filesystem::path catalog_path = std::filesystem::temp_directory_path() /
        ("astc-vulkan-compiled-model-catalog-write-" + std::to_string(nonce) + ".astcc");
    if (!astc_vulkan_write_compiled_catalog(catalog_path.string(), filtered_catalog, error) ||
        !read_file(catalog_path.string(), model.catalog, error)) {
        std::error_code ignored;
        std::filesystem::remove(catalog_path, ignored);
        return false;
    }
    std::error_code ignored;
    std::filesystem::remove(catalog_path, ignored);
    const auto astc_names = manifest_astc_tensor_names(manifest);
    astc_vulkan_native_tensor_table table;
    if (!astc_vulkan_decode_native_tensor_table(model.native_table, table, error)) return false;

    std::vector<uint8_t> strict_native_payload;
    for (auto & record : table.tensors) {
        if (astc_names.count(record.name)) {
            record.storage = astc_vulkan_native_tensor_storage::astc;
            record.native_offset = 0;
            record.native_size = 0;
            record.native_hash64 = 0;
            continue;
        }
        if (record.storage != astc_vulkan_native_tensor_storage::native ||
            record.native_offset > model.native_payload.size() ||
            record.native_size > model.native_payload.size() - record.native_offset) {
            error = "hybrid compiled model lacks native bytes for: " + record.name;
            return false;
        }
        const uint8_t * bytes = model.native_payload.data() + record.native_offset;
        record.native_offset = strict_native_payload.size();
        strict_native_payload.insert(strict_native_payload.end(), bytes, bytes + record.native_size);
        record.native_hash64 = astc_vulkan_payload_hash64(
            strict_native_payload.data() + record.native_offset, record.native_size);
    }
    model.native_payload = std::move(strict_native_payload);
    model.storage_mode = astc_vulkan_compiled_storage_mode::strict;
    if (!astc_vulkan_encode_native_tensor_table(table, model.native_table, error)) return false;
    return astc_vulkan_compiled_model_write(output_path, model, error);
}
