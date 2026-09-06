#include "astc-vulkan-page-owner.h"
#include "astc-vulkan-sidecar.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    const fs::path payload_path = fs::temp_directory_path() /
        "astc-vulkan-page-owner-device-payload.bin";
    {
        std::ofstream payload(payload_path, std::ios::binary | std::ios::trunc);
        const std::vector<uint8_t> bytes(16, 0);
        payload.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }
    const std::vector<uint8_t> payload_bytes(16, 0);

    astc_vulkan_manifest manifest;
    astc_vulkan_tensor_record record;
    record.name = "blk.0.ffn_down.weight";
    record.width = 6;
    record.height = 6;
    record.footprint = astc_vulkan_footprint::k6x6;
    record.byte_size = astc_vulkan_image_bytes(record.footprint, record.width, record.height);
    record.payload_hash64 = astc_vulkan_payload_hash64(
        payload_bytes.data(), payload_bytes.size());
    manifest.tensors.push_back(record);

    astc_vulkan_model_cache_plan plan;
    std::string error;
    astc_vulkan_model_cache_plan_options options;
    assert(astc_vulkan_model_cache_make_plan(manifest, options, plan, error));
    astc_vulkan_model_cache_storage_page page;
    page.key.footprint = record.footprint;
    page.key.representation = record.representation;
    page.entry_indices = {0};
    page.payload_bytes = record.byte_size;
    page.host_bytes = record.byte_size;
    astc_vulkan_page_owner owner;
    assert(owner.prepare(plan, {page}, 1, error));
    assert(owner.set_resident_prefix(1, error));

    astc_vulkan_model_cache_catalog catalog;
    catalog.plan = plan;
    catalog.validation.paths.payload = payload_path.string();
    astc_vulkan_page_material material;
    assert(owner.load_entry_material(catalog, 0, material, error));
    assert(material.payload.size() == record.byte_size);

    astc_vulkan_sidecar sidecar;
    if (!sidecar.init(record.footprint, error)) {
        // A build without a sampled-ASTC/compute-capable device is a valid
        // environment for the contract test; the CPU owner test remains the
        // required gate in that case.
        std::puts("ASTC Vulkan page-owner device smoke skipped: no compatible device");
        fs::remove(payload_path);
        return 0;
    }
    assert(sidecar.set_manifest(manifest, error));
    astc_vulkan_ffn_binding binding;
    assert(sidecar.bind_tensor(record.name, record.width, record.height,
                               material.payload, binding, error));
    assert(binding.status == astc_vulkan_binding_status::kReady);
    std::puts("ASTC Vulkan page-owner device upload passed");
    sidecar.reset();
    fs::remove(payload_path);
    return 0;
}
