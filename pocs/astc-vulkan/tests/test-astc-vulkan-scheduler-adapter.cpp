#include "astc-vulkan-scheduler-adapter.h"
#include "astc-vulkan-cache.h"
#include "astc-vulkan-paired-layout.h"

#include <cassert>
#include <filesystem>
#include <fstream>

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

    // D2 cache discovery is execution-independent. A future paired dispatcher
    // receives exactly these bytes and layout words; today's scalar scheduler
    // remains on the normal llama fallback instead of attempting a fake run.
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
    const std::vector<uint8_t> payload(16, 0x5a);
    const std::vector<uint8_t> layout(4, 0);
    write_bytes(payload_path.string(), payload);
    write_bytes(layout_path.string(), layout);
    astc_vulkan_manifest manifest;
    astc_vulkan_tensor_record record;
    record.name = "tensor";
    record.width = 8;
    record.height = 10;
    record.footprint = astc_vulkan_footprint::k8x5;
    record.byte_size = payload.size();
    record.representation = astc_vulkan_representation::kPairedD2;
    record.payload_hash64 = astc_vulkan_payload_hash64(payload.data(), payload.size());
    record.layout_byte_size = astc_vulkan_paired_layout_bytes(record.footprint, record.width, record.height);
    record.layout_hash64 = astc_vulkan_payload_hash64(layout.data(), layout.size());
    manifest.tensors.push_back(record);
    assert(astc_vulkan_write_manifest(manifest_path.string(), manifest, error));
    astc_vulkan_cache_paths paths;
    assert(astc_vulkan_cache_create(model.string(), manifest_path.string(), payload_path.string(),
                                    layout_path.string(), {}, root.string(), paths, error));
    astc_vulkan_scheduler_artifact artifact;
    assert(adapter.resolve_from_cache(model.string(), root.string(), "tensor",
                                      astc_vulkan_footprint::k8x5, artifact, error));
    assert(artifact.kind == astc_vulkan_scheduler_artifact_kind::kD2);
    assert(artifact.payload == payload && artifact.layout == layout);
    assert(adapter.prepare_from_cache(model.string(), root.string(), "tensor",
                                      astc_vulkan_footprint::k8x5, error, true));
    assert(!adapter.ready());
    assert(adapter.binding().fallback_reason.find("paired-D2 artifact is verified") != std::string::npos);
    assert(!astc_vulkan_scheduler_adapter::jit_cache_build_enabled());
    fs::remove_all(root, ignored);
    fs::remove(model, ignored);
    fs::remove(manifest_path, ignored);
    fs::remove(payload_path, ignored);
    fs::remove(layout_path, ignored);
    return 0;
}
