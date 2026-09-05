#include "astc-vulkan-scheduler-adapter.h"
#include "astc-vulkan-cache.h"
#include "astc-vulkan-paired-layout.h"

#include <cassert>
#include <array>
#include <filesystem>
#include <fstream>
#include <cstring>

namespace {

void write_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    assert(file.good());
}

} // namespace

int main() {
    astc_vulkan_scheduler_adapter adapter;
    std::string error;
    assert(!adapter.ready());
    std::vector<float> output;
    assert(!adapter.run({}, {}, output, error));
    assert(error == "ASTC scheduler adapter is not ready; use normal fallback");
    error.clear();
    assert(!adapter.prepare("missing.manifest", "missing.payload", "tensor",
                            astc_vulkan_footprint::k6x6, error));
    assert(!adapter.ready());
    error.clear();
    assert(!adapter.prepare_from_cache("missing.gguf", "auto", "tensor",
                                       astc_vulkan_footprint::k6x6, error));
    assert(!adapter.ready());
    assert(adapter.binding().status == astc_vulkan_binding_status::kFallback);
    adapter.reset();
    assert(!adapter.ready());

    // D2 cache discovery is execution-independent. The paired dispatcher sees
    // exactly these bytes and layout words; a device that cannot admit the
    // artifact still returns a complete normal llama fallback.
    namespace fs = std::filesystem;
    const fs::path root("astc-vulkan-scheduler-adapter-cache");
    const fs::path model("astc-vulkan-scheduler-adapter.gguf");
    const fs::path manifest_path("astc-vulkan-scheduler-adapter.manifest");
    const fs::path payload_path("astc-vulkan-scheduler-adapter.payload");
    const fs::path layout_path("astc-vulkan-scheduler-adapter.layout");
    std::error_code ignored;
    fs::remove_all(root, ignored);
    fs::remove(model, ignored);
    fs::remove(manifest_path, ignored);
    fs::remove(payload_path, ignored);
    fs::remove(layout_path, ignored);
    write_bytes(model.string(), {0x47, 0x47, 0x55, 0x46, 1, 2, 3, 4});
    const std::vector<uint8_t> d1_payload(16, 0x3c);
    const std::vector<uint8_t> d2_payload(16, 0x5a);
    std::vector<uint8_t> payload = d1_payload;
    payload.reserve(d1_payload.size() + d2_payload.size());
    payload.insert(payload.end(), d2_payload.begin(), d2_payload.end());
    const std::vector<uint8_t> layout(4, 0);
    write_bytes(payload_path.string(), payload);
    write_bytes(layout_path.string(), layout);
    astc_vulkan_manifest manifest;
    astc_vulkan_tensor_record d1_record;
    d1_record.name = "d1";
    d1_record.width = 4;
    d1_record.height = 4;
    d1_record.footprint = astc_vulkan_footprint::k4x4;
    d1_record.byte_size = d1_payload.size();
    d1_record.representation = astc_vulkan_representation::kScalar;
    d1_record.payload_hash64 = astc_vulkan_payload_hash64(d1_payload.data(), d1_payload.size());
    manifest.tensors.push_back(d1_record);
    astc_vulkan_tensor_record d2_record;
    d2_record.name = "tensor";
    d2_record.width = 6;
    d2_record.height = 10;
    d2_record.footprint = astc_vulkan_footprint::k6x5;
    d2_record.byte_offset = d1_payload.size();
    d2_record.byte_size = d2_payload.size();
    d2_record.representation = astc_vulkan_representation::kPairedD2;
    d2_record.payload_hash64 = astc_vulkan_payload_hash64(d2_payload.data(), d2_payload.size());
    d2_record.layout_byte_size = astc_vulkan_paired_layout_bytes(d2_record.footprint, d2_record.width, d2_record.height);
    d2_record.layout_hash64 = astc_vulkan_payload_hash64(layout.data(), layout.size());
    manifest.tensors.push_back(d2_record);
    assert(astc_vulkan_write_manifest(manifest_path.string(), manifest, error));
    astc_vulkan_cache_paths paths;
    assert(astc_vulkan_cache_create(model.string(), manifest_path.string(), payload_path.string(),
                                    layout_path.string(), {}, root.string(), paths, error));
    astc_vulkan_scheduler_artifact artifact;
    assert(adapter.resolve_from_cache(model.string(), root.string(), "d1",
                                      astc_vulkan_footprint::k4x4, artifact, error));
    assert(artifact.kind == astc_vulkan_scheduler_artifact_kind::kD1);
    assert(artifact.payload == d1_payload && artifact.layout.empty());
    assert(adapter.resolve_from_cache(model.string(), root.string(), "tensor",
                                      astc_vulkan_footprint::k6x5, artifact, error));
    assert(artifact.kind == astc_vulkan_scheduler_artifact_kind::kD2);
    assert(artifact.payload == d2_payload && artifact.layout == layout);
    const bool d2_prepared = adapter.prepare_from_cache(model.string(), root.string(), "tensor",
                                                        astc_vulkan_footprint::k6x5, error, true);
    // Device availability is intentionally not a unit-test prerequisite. If
    // supported, the same verified bytes reach the paired runtime; otherwise
    // the adapter returns a complete normal-quant fallback.
    assert(d2_prepared || adapter.binding().status == astc_vulkan_binding_status::kFallback);
    assert(adapter.dispatch_kind() == astc_vulkan_scheduler_dispatch_kind::kD2Paired);
    assert(!astc_vulkan_scheduler_adapter::jit_cache_build_enabled());

    // v4 selection is evidence driven. Both records belong to the same tensor
    // and footprint, but only the selected absmax artifact wins policy ranking
    // and carries a streamable row-scale sidecar.
    const fs::path v4_root("astc-vulkan-scheduler-adapter-v4-cache");
    const fs::path v4_manifest_path("astc-vulkan-scheduler-adapter-v4.manifest");
    const fs::path v4_payload_path("astc-vulkan-scheduler-adapter-v4.payload");
    const fs::path v4_layout_path("astc-vulkan-scheduler-adapter-v4.layout");
    const fs::path v4_scales_path("astc-vulkan-scheduler-adapter-v4.scales");
    fs::remove_all(v4_root, ignored);
    fs::remove(v4_manifest_path, ignored);
    fs::remove(v4_payload_path, ignored);
    fs::remove(v4_layout_path, ignored);
    fs::remove(v4_scales_path, ignored);
    std::vector<uint8_t> v4_payload(32, 0x11);
    std::fill(v4_payload.begin() + 16, v4_payload.end(), 0x22);
    write_bytes(v4_payload_path.string(), v4_payload);
    write_bytes(v4_layout_path.string(), layout);
    const std::array<float, 10> scale_values = {1.f, 1.1f, 1.2f, 1.3f, 1.4f,
                                                 1.5f, 1.6f, 1.7f, 1.8f, 1.9f};
    std::vector<uint8_t> scale_bytes(sizeof(scale_values));
    std::memcpy(scale_bytes.data(), scale_values.data(), scale_bytes.size());
    write_bytes(v4_scales_path.string(), scale_bytes);
    astc_vulkan_manifest v4;
    v4.version = 4;
    astc_vulkan_artifact_record neutral;
    neutral.id = "tensor/d2-la/neutral";
    neutral.storage = d2_record;
    neutral.storage.width = 8;
    neutral.storage.footprint = astc_vulkan_footprint::k8x5;
    neutral.storage.layout_byte_size = astc_vulkan_paired_layout_bytes(
        neutral.storage.footprint, neutral.storage.width, neutral.storage.height);
    neutral.storage.byte_offset = 0;
    neutral.storage.payload_hash64 = astc_vulkan_payload_hash64(v4_payload.data(), 16);
    neutral.paired_semantic = astc_vulkan_paired_semantic::luminance_alpha;
    neutral.encoder_profile = "d2-la";
    neutral.evidence = {true, true, .2f, .2f, .4f, 80.f, "cal", "replay"};
    astc_vulkan_artifact_record selected = neutral;
    selected.id = "tensor/d2-la/absmax-selected";
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
    assert(adapter.resolve_best_from_cache(model.string(), v4_root.string(), "tensor",
                                           astc_vulkan_footprint::k8x5,
                                           astc_vulkan_quality_policy::quality, artifact, error));
    assert(artifact.artifact_id == selected.id);
    assert(artifact.paired_semantic == astc_vulkan_paired_semantic::luminance_alpha);
    assert(artifact.normalization == astc_vulkan_normalization::per_row_absmax);
    assert(artifact.payload == std::vector<uint8_t>(v4_payload.begin() + 16, v4_payload.end()));
    assert(artifact.row_scales.size() == scale_values.size());
    assert(artifact.row_scales[3] == scale_values[3]);
    // Experimental D2 is never implicit. The same evidence-approved v4
    // artifact becomes runnable only after the caller explicitly opts in.
    assert(adapter.prepare_from_cache(model.string(), v4_root.string(), "tensor",
                                      astc_vulkan_footprint::k8x5, error, false));
    assert(adapter.binding().status == astc_vulkan_binding_status::kFallback);
    assert(error == "ASTC paired-D2 requires an explicit, evidence-approved D2_6x5 or D2_8x5 L+A artifact");
    const bool evidence_d2_prepared = adapter.prepare_from_cache(
        model.string(), v4_root.string(), "tensor", astc_vulkan_footprint::k8x5, error, true);
    assert(evidence_d2_prepared || adapter.binding().status == astc_vulkan_binding_status::kFallback);
    assert(adapter.dispatch_kind() == astc_vulkan_scheduler_dispatch_kind::kD2Paired);
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
    return 0;
}
