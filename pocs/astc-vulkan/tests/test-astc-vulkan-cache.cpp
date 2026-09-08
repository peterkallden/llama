#include "astc-vulkan-cache.h"
#include "astc-vulkan-paired-layout.h"

#include "gguf.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>

namespace {

void write_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    assert(file.good());
}

void write_schema_fixture_gguf(const std::string & path, const char * name) {
    gguf_context * context = gguf_init_empty();
    assert(context != nullptr);
    gguf_set_val_str(context, "general.architecture", "astc-cache-fixture");
    // Deliberately differs between the source and runtime fixture. It must not
    // affect structural compatibility, but it does give each GGUF a distinct
    // file hash.
    gguf_set_val_str(context, "general.name", name);
    assert(gguf_write_to_file(context, path.c_str(), false));
    gguf_free(context);
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root("astc-vulkan-cache-test-dir");
    const fs::path model("astc-vulkan-cache-test.gguf");
    const fs::path manifest_path("astc-vulkan-cache-test.manifest");
    const fs::path payload_path("astc-vulkan-cache-test.payload");
    const fs::path layout_path("astc-vulkan-cache-test.layout");
    std::error_code ignored;
    fs::remove_all(root, ignored);
    fs::remove(model, ignored);
    fs::remove(manifest_path, ignored);
    fs::remove(payload_path, ignored);
    fs::remove(layout_path, ignored);

    write_bytes(model.string(), {0x47, 0x47, 0x55, 0x46, 1, 2, 3, 4});
    const std::vector<uint8_t> payload(16, 0x5a);
    const std::vector<uint8_t> layout(4, 0);
    write_bytes(payload_path.string(), payload);
    write_bytes(layout_path.string(), layout);

    astc_vulkan_manifest manifest;
    manifest.model_fingerprint = "fixture";
    astc_vulkan_tensor_record record;
    record.name = "blk.0.ffn_down.weight";
    record.width = 8;
    record.height = 10;
    record.footprint = astc_vulkan_footprint::k8x5;
    record.byte_size = payload.size();
    record.representation = astc_vulkan_representation::kPairedD2;
    record.payload_hash64 = astc_vulkan_payload_hash64(payload.data(), payload.size());
    record.layout_byte_size = astc_vulkan_paired_layout_bytes(record.footprint, record.width, record.height);
    record.layout_hash64 = astc_vulkan_payload_hash64(layout.data(), layout.size());
    assert(record.layout_byte_size == layout.size());
    manifest.tensors.push_back(record);
    std::string error;
    assert(astc_vulkan_write_manifest(manifest_path.string(), manifest, error));

    astc_vulkan_cache_paths paths;
    assert(astc_vulkan_cache_create(model.string(), manifest_path.string(), payload_path.string(),
                                    layout_path.string(), {}, root.string(), paths, error));
    astc_vulkan_cache_validation validation;
    assert(astc_vulkan_cache_validate(model.string(), root.string(), validation, error));
    assert(validation.has_paired_d2 && validation.manifest.tensors.size() == 1);
    assert(astc_vulkan_cache_validate(model.string(), paths.manifest, validation, error));

    write_bytes(model.string(), {0x47, 0x47, 0x55, 0x46, 9, 9, 9, 9});
    assert(!astc_vulkan_cache_validate(model.string(), root.string(), validation, error));

    // v4 keeps neutral and selected artifacts side by side and validates an
    // optional row-scale blob independently from payload/layout bytes.
    const fs::path v4_root("astc-vulkan-cache-v4-test-dir");
    const fs::path v4_manifest_path("astc-vulkan-cache-v4-test.manifest");
    const fs::path v4_payload_path("astc-vulkan-cache-v4-test.payload");
    const fs::path v4_layout_path("astc-vulkan-cache-v4-test.layout");
    const fs::path v4_scales_path("astc-vulkan-cache-v4-test.scales");
    fs::remove_all(v4_root, ignored);
    fs::remove(v4_manifest_path, ignored);
    fs::remove(v4_payload_path, ignored);
    fs::remove(v4_layout_path, ignored);
    fs::remove(v4_scales_path, ignored);
    write_bytes(model.string(), {0x47, 0x47, 0x55, 0x46, 1, 2, 3, 4});
    std::vector<uint8_t> v4_payload(32, 0x31);
    std::fill(v4_payload.begin() + 16, v4_payload.end(), 0x42);
    write_bytes(v4_payload_path.string(), v4_payload);
    write_bytes(v4_layout_path.string(), layout);
    const std::array<float, 10> scales = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f};
    std::vector<uint8_t> scale_bytes(sizeof(scales));
    std::memcpy(scale_bytes.data(), scales.data(), scale_bytes.size());
    write_bytes(v4_scales_path.string(), scale_bytes);
    astc_vulkan_manifest v4;
    v4.version = 4;
    v4.model_fingerprint = "fixture-v4";
    astc_vulkan_artifact_record neutral;
    neutral.id = "blk.0.ffn_down.weight/d2-la/neutral";
    neutral.storage = record;
    neutral.storage.byte_offset = 0;
    neutral.storage.payload_hash64 = astc_vulkan_payload_hash64(v4_payload.data(), 16);
    neutral.paired_semantic = astc_vulkan_paired_semantic::luminance_alpha;
    neutral.encoder_profile = "d2-la";
    neutral.evidence = {true, true, .1f, .2f, .3f, 90.f, "cal", "replay"};
    astc_vulkan_artifact_record selected = neutral;
    selected.id = "blk.0.ffn_down.weight/d2-la/absmax-selected";
    selected.storage.byte_offset = 16;
    selected.storage.payload_hash64 = astc_vulkan_payload_hash64(v4_payload.data() + 16, 16);
    selected.variant = astc_vulkan_artifact_variant::validation_selected;
    selected.normalization = astc_vulkan_normalization::per_row_absmax;
    selected.row_scale_byte_size = scale_bytes.size();
    selected.row_scale_hash64 = astc_vulkan_payload_hash64(scale_bytes.data(), scale_bytes.size());
    selected.evidence.loss_delta = .1f;
    v4.artifacts = {neutral, selected};
    assert(astc_vulkan_write_manifest(v4_manifest_path.string(), v4, error));
    assert(astc_vulkan_cache_create_with_row_scales(
        model.string(), v4_manifest_path.string(), v4_payload_path.string(), v4_layout_path.string(),
        v4_scales_path.string(), {}, v4_root.string(), paths, error));
    assert(astc_vulkan_cache_validate(model.string(), v4_root.string(), validation, error));
    assert(validation.manifest.version == 4 && validation.manifest.artifacts.size() == 2);
    assert(validation.has_paired_d2 && validation.has_row_scales);

    // v5 makes an optimized ten-row D2 pairing an independently verified
    // artifact resource. The direct slot->logical-row encoding deliberately
    // avoids coupling this fixture to any matching-enumerator order.
    const fs::path v5_root("astc-vulkan-cache-v5-test-dir");
    const fs::path v5_manifest_path("astc-vulkan-cache-v5-manifest.astcv");
    const fs::path v5_pair_map_path("astc-vulkan-cache-v5-pair-map.bin");
    fs::remove_all(v5_root, ignored);
    fs::remove(v5_manifest_path, ignored);
    fs::remove(v5_pair_map_path, ignored);
    const std::array<uint8_t, 10> pair_map = {0, 2, 1, 3, 4, 5, 6, 7, 8, 9};
    write_bytes(v5_pair_map_path.string(),
                std::vector<uint8_t>(pair_map.begin(), pair_map.end()));
    astc_vulkan_manifest v5 = v4;
    v5.version = 5;
    v5.model_fingerprint = "fixture-v5-pair-map";
    v5.artifacts[0].pair_map_byte_size = pair_map.size();
    v5.artifacts[0].pair_map_hash64 = astc_vulkan_payload_hash64(pair_map.data(), pair_map.size());
    assert(astc_vulkan_write_manifest(v5_manifest_path.string(), v5, error));
    assert(astc_vulkan_cache_create_with_metadata(
        model.string(), v5_manifest_path.string(), v4_payload_path.string(), v4_layout_path.string(),
        v4_scales_path.string(), v5_pair_map_path.string(), {}, v5_root.string(), paths, error));
    assert(astc_vulkan_cache_validate(model.string(), v5_root.string(), validation, error));
    assert(validation.manifest.version == 5 && validation.has_paired_d2 && validation.has_pair_map);
    assert(validation.manifest.artifacts[0].pair_map_byte_size == pair_map.size());

    // A cache may record a structurally compatible quantized runtime base
    // without weakening the strict source-model validation contract. The
    // binding starts with no model/Vulkan quality evidence.
    const fs::path admit_source("astc-vulkan-cache-admit-source.gguf");
    const fs::path admit_runtime("astc-vulkan-cache-admit-runtime-q3.gguf");
    const fs::path admit_runtime_tq("astc-vulkan-cache-admit-runtime-tq2.gguf");
    const fs::path admit_root("astc-vulkan-cache-admit-test-dir");
    fs::remove_all(admit_root, ignored);
    fs::remove(admit_source, ignored);
    fs::remove(admit_runtime, ignored);
    fs::remove(admit_runtime_tq, ignored);
    write_schema_fixture_gguf(admit_source.string(), "source-f16");
    write_schema_fixture_gguf(admit_runtime.string(), "runtime-q3");
    write_schema_fixture_gguf(admit_runtime_tq.string(), "runtime-tq2");
    assert(astc_vulkan_cache_create(
        admit_source.string(), manifest_path.string(), payload_path.string(), layout_path.string(), {},
        admit_root.string(), paths, error));
    astc_vulkan_cache_runtime_base binding;
    assert(astc_vulkan_cache_admit_runtime_base(
        admit_source.string(), admit_runtime.string(), admit_root.string(), "q3_k_m", binding, error));
    assert(binding.admitted && binding.family == "q3_k_m" &&
           !binding.model_gate_passed && !binding.vulkan_gate_passed);
    assert(fs::is_regular_file(fs::path(paths.compatible_bases) / (binding.runtime_sha256 + ".astcbase")));
    astc_vulkan_cache_runtime_base tq_binding;
    assert(astc_vulkan_cache_admit_runtime_base(
        admit_source.string(), admit_runtime_tq.string(), admit_root.string(), "tq2_0", tq_binding, error));
    assert(tq_binding.admitted && tq_binding.family == "tq2_0");
    // Admission is provenance only: regular runtime validation remains tied to
    // the exact source GGUF until a future evidence-gated runtime path exists.
    assert(!astc_vulkan_cache_validate(admit_runtime.string(), admit_root.string(), validation, error));

    fs::remove_all(root, ignored);
    fs::remove(model, ignored);
    fs::remove(manifest_path, ignored);
    fs::remove(payload_path, ignored);
    fs::remove(layout_path, ignored);
    fs::remove_all(v4_root, ignored);
    fs::remove(v4_manifest_path, ignored);
    fs::remove(v4_payload_path, ignored);
    fs::remove(v4_layout_path, ignored);
    fs::remove(v4_scales_path, ignored);
    fs::remove_all(admit_root, ignored);
    fs::remove(admit_source, ignored);
    fs::remove(admit_runtime, ignored);
    fs::remove(admit_runtime_tq, ignored);
    std::puts("ASTC Vulkan cache D1/D2 contract passed");
    return 0;
}
