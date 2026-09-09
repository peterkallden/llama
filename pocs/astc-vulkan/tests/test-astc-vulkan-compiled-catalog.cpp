#include "astc-vulkan-compiled-catalog.h"
#include "astc-vulkan-paired-layout.h"

#include <cassert>
#include <filesystem>
#include <string>

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
    manifest.artifacts.push_back(d1);

    std::string error;
    astc_vulkan_compiled_catalog catalog;
    assert(astc_vulkan_compiled_catalog_from_manifest(manifest, catalog, error));
    assert(catalog.version == 1);
    assert(catalog.logical_model_id == "compiled-catalog-fixture");
    assert(catalog.tensors.size() == 2);
    const auto & record = catalog.tensors.front();
    assert(record.logical_name == "blk.12.ffn_down.weight");
    assert(record.semantic_role == "ffn.down");
    assert(record.canonical_path == "layers/12/ffn/down/weight");
    assert(record.storage_class == "astc.d2.la.8x5");
    assert(record.storage_kind == astc_vulkan_compiled_storage_kind::astc_d2);
    assert(record.payload_size == 16);
    assert(record.layout_size == 4);
    assert(record.row_scale_size == 40);
    assert(record.pair_map_size == 10);
    assert(catalog.tensors[1].storage_class == "astc.d1.la.6x6");
    assert(catalog.tensors[1].storage_kind == astc_vulkan_compiled_storage_kind::astc_d1);
    assert(astc_vulkan_compiled_catalog_matches_manifest(catalog, manifest, error));
    astc_vulkan_manifest changed = manifest;
    changed.artifacts[0].id += ".stale";
    assert(!astc_vulkan_compiled_catalog_matches_manifest(catalog, changed, error));

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "astc-vulkan-compiled-catalog-test.astcc";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    assert(astc_vulkan_write_compiled_catalog(path.string(), catalog, error));
    astc_vulkan_compiled_catalog read_back;
    assert(astc_vulkan_read_compiled_catalog(path.string(), read_back, error));
    assert(read_back.version == catalog.version);
    assert(read_back.logical_model_id == catalog.logical_model_id);
    assert(read_back.tensors.size() == catalog.tensors.size());
    assert(read_back.tensors.front().logical_name == record.logical_name);
    assert(read_back.tensors.front().storage_class == record.storage_class);
    assert(read_back.tensors.front().pair_map_size == record.pair_map_size);
    std::filesystem::remove(path, ignored);
    return 0;
}
