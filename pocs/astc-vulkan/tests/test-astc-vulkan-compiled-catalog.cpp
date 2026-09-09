#include "astc-vulkan-compiled-catalog.h"
#include "astc-vulkan-paired-layout.h"

#include <filesystem>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#define REQUIRE(condition) do { if (!(condition)) { std::fprintf(stderr, "failed: %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } } while (false)

namespace {

bool write_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    if (!bytes.empty()) file.write(reinterpret_cast<const char *>(bytes.data()),
                                   static_cast<std::streamsize>(bytes.size()));
    return file.good();
}

std::vector<uint8_t> read_bytes(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size < 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty()) file.read(reinterpret_cast<char *>(bytes.data()), size);
    return file.good() ? bytes : std::vector<uint8_t>{};
}

} // namespace

int main() {
    astc_vulkan_manifest manifest;
    manifest.version = 8;
    manifest.model_fingerprint = "compiled-catalog-fixture";

    astc_vulkan_artifact_record artifact;
    artifact.id = "blk.12.ffn_down.weight/d2-la/pairing-selected";
    artifact.storage.name = "blk.12.ffn_down.weight";
    artifact.storage.width = 8;
    artifact.storage.height = 10;
    artifact.storage.footprint = astc_vulkan_footprint::k8x5;
    artifact.storage.representation = astc_vulkan_representation::kPairedD2;
    artifact.storage.byte_size = astc_vulkan_image_bytes(
        artifact.storage.footprint, artifact.storage.width,
        astc_vulkan_paired_storage_height(artifact.storage.height));
    artifact.storage.layout_byte_size = astc_vulkan_paired_layout_bytes(
        artifact.storage.footprint, artifact.storage.width, artifact.storage.height);
    artifact.storage.semantic_role = "ffn.down";
    artifact.storage.canonical_path = "layers/12/ffn/down/weight";
    artifact.paired_semantic = astc_vulkan_paired_semantic::luminance_alpha;
    artifact.normalization = astc_vulkan_normalization::per_row_absmax;
    artifact.row_scale_byte_size = artifact.storage.height * sizeof(float);
    artifact.pair_map_byte_size = 10;
    manifest.artifacts.push_back(artifact);

    astc_vulkan_artifact_record d1;
    d1.id = "blk.0.ffn_down.weight/d1-gauge";
    d1.storage.name = "blk.0.ffn_down.weight";
    d1.storage.width = 6;
    d1.storage.height = 6;
    d1.storage.footprint = astc_vulkan_footprint::k6x6;
    d1.storage.representation = astc_vulkan_representation::kGaugeLumaAlpha;
    d1.storage.byte_size = astc_vulkan_image_bytes(
        d1.storage.footprint, d1.storage.width, d1.storage.height);
    d1.storage.byte_offset = artifact.storage.byte_size;
    manifest.artifacts.push_back(d1);

    std::string error;
    astc_vulkan_compiled_catalog catalog;
    if (!astc_vulkan_compiled_catalog_from_manifest(manifest, catalog, error)) {
        std::fprintf(stderr, "catalog fixture error: %s\n", error.c_str());
        return 1;
    }
    REQUIRE(catalog.version == 1);
    REQUIRE(catalog.logical_model_id == "compiled-catalog-fixture");
    REQUIRE(catalog.tensors.size() == 2);
    const auto & record = catalog.tensors.front();
    REQUIRE(record.logical_name == "blk.12.ffn_down.weight");
    REQUIRE(record.semantic_role == "ffn.down");
    REQUIRE(record.canonical_path == "layers/12/ffn/down/weight");
    REQUIRE(record.storage_class == "astc.d2.la.8x5");
    REQUIRE(record.storage_kind == astc_vulkan_compiled_storage_kind::astc_d2);
    REQUIRE(record.payload_size == 16);
    REQUIRE(record.layout_size == 4);
    REQUIRE(record.row_scale_size == 40);
    REQUIRE(record.pair_map_size == 10);
    REQUIRE(catalog.tensors[1].storage_class == "astc.d1.la.6x6");
    REQUIRE(catalog.tensors[1].storage_kind == astc_vulkan_compiled_storage_kind::astc_d1);
    REQUIRE(astc_vulkan_compiled_catalog_matches_manifest(catalog, manifest, error));
    astc_vulkan_manifest changed = manifest;
    changed.artifacts[0].id += ".stale";
    REQUIRE(!astc_vulkan_compiled_catalog_matches_manifest(catalog, changed, error));

    astc_vulkan_compiled_catalog duplicate = catalog;
    duplicate.tensors[1].artifact_id = duplicate.tensors[0].artifact_id;
    REQUIRE(!astc_vulkan_validate_compiled_catalog(duplicate, error));
    astc_vulkan_compiled_catalog empty_id = catalog;
    empty_id.tensors[0].artifact_id.clear();
    REQUIRE(!astc_vulkan_validate_compiled_catalog(empty_id, error));

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "astc-vulkan-compiled-catalog-test.astcc";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    REQUIRE(astc_vulkan_write_compiled_catalog(path.string(), catalog, error));
    astc_vulkan_compiled_catalog read_back;
    REQUIRE(astc_vulkan_read_compiled_catalog(path.string(), read_back, error));
    REQUIRE(read_back.version == catalog.version);
    REQUIRE(read_back.logical_model_id == catalog.logical_model_id);
    REQUIRE(read_back.tensors.size() == catalog.tensors.size());
    REQUIRE(read_back.tensors.front().logical_name == record.logical_name);
    REQUIRE(read_back.tensors.front().storage_class == record.storage_class);
    REQUIRE(read_back.tensors.front().pair_map_size == record.pair_map_size);

    const std::vector<uint8_t> valid = read_bytes(path);
    REQUIRE(valid.size() > 12);
    std::vector<uint8_t> bad_magic = valid;
    bad_magic[0] ^= 0xff;
    REQUIRE(write_bytes(path, bad_magic));
    REQUIRE(!astc_vulkan_read_compiled_catalog(path.string(), read_back, error));
    std::vector<uint8_t> bad_version = valid;
    // version is the first little-endian u32 after the eight-byte magic.
    bad_version[8] = 0xff;
    REQUIRE(write_bytes(path, bad_version));
    REQUIRE(!astc_vulkan_read_compiled_catalog(path.string(), read_back, error));
    std::vector<uint8_t> truncated(valid.begin(), valid.end() - 1);
    REQUIRE(write_bytes(path, truncated));
    REQUIRE(!astc_vulkan_read_compiled_catalog(path.string(), read_back, error));
    std::vector<uint8_t> trailing = valid;
    trailing.push_back(0);
    REQUIRE(write_bytes(path, trailing));
    REQUIRE(!astc_vulkan_read_compiled_catalog(path.string(), read_back, error));
    std::filesystem::remove(path, ignored);
    return 0;
}

#undef REQUIRE
