#include "astc-vulkan-page-owner.h"

#include <cassert>
#include <string>
#include <vector>

namespace {

astc_vulkan_model_cache_entry make_entry(const char * name, uint64_t bytes) {
    astc_vulkan_model_cache_entry entry;
    entry.tensor_name = name;
    entry.artifact_id = std::string(name) + "/artifact";
    entry.storage.name = name;
    entry.storage.width = 6;
    entry.storage.height = 6;
    entry.storage.footprint = astc_vulkan_footprint::k6x6;
    entry.storage.representation = astc_vulkan_representation::kScalar;
    entry.storage.byte_size = bytes;
    entry.device_bytes = bytes;
    entry.host_bytes = bytes;
    entry.evidence.model_gate_passed = true;
    entry.evidence.vulkan_gate_passed = true;
    return entry;
}

} // namespace

int main() {
    astc_vulkan_model_cache_plan plan;
    plan.entries.push_back(make_entry("blk.0.ffn_down.weight", 16));
    plan.entries.push_back(make_entry("blk.1.ffn_down.weight", 16));
    auto fallback = make_entry("blk.2.ffn_down.weight", 16);
    fallback.use_native_fallback = true;
    plan.entries.push_back(fallback);

    astc_vulkan_model_cache_storage_page page0;
    page0.key.footprint = astc_vulkan_footprint::k6x6;
    page0.key.representation = astc_vulkan_representation::kScalar;
    page0.entry_indices = {0};
    page0.payload_bytes = 16;
    page0.host_bytes = 16;
    astc_vulkan_model_cache_storage_page page1 = page0;
    page1.entry_indices = {1};
    std::vector<astc_vulkan_model_cache_storage_page> pages = {page0, page1};

    astc_vulkan_page_owner owner;
    std::string error;
    assert(owner.prepare(plan, pages, 2, error));
    assert(owner.pages().size() == 2);
    assert(owner.pages()[0].slot_x == 0 && owner.pages()[0].slot_y == 0);
    assert(owner.pages()[1].slot_x == 1 && owner.pages()[1].slot_y == 0);

    astc_vulkan_page_resolve resolve;
    assert(owner.resolve_tensor("blk.0.ffn_down.weight", resolve, error));
    assert(!resolve.resident && resolve.use_native_fallback && resolve.page_index == 0);
    assert(owner.set_resident_prefix(1, error));
    assert(owner.resolve_tensor("blk.0.ffn_down.weight", resolve, error));
    assert(resolve.resident && !resolve.use_native_fallback && resolve.slot_index == 0);
    assert(owner.resolve_tensor("blk.1.ffn_down.weight", resolve, error));
    assert(!resolve.resident && resolve.use_native_fallback && resolve.page_index == 1);

    assert(owner.begin_upload(1, error));
    assert(owner.finish_upload(1, error));
    assert(owner.resolve_tensor("blk.1.ffn_down.weight", resolve, error));
    assert(resolve.resident && resolve.slot_index == 1);
    assert(owner.begin_evict(0, error));
    assert(owner.finish_evict(0, error));
    assert(owner.resolve_tensor("blk.0.ffn_down.weight", resolve, error));
    assert(!resolve.resident && resolve.use_native_fallback);
    assert(owner.resolve_tensor("blk.2.ffn_down.weight", resolve, error));
    assert(resolve.use_native_fallback && resolve.page_index == static_cast<size_t>(-1));

    std::puts("ASTC Vulkan page owner contract passed");
    return 0;
}
