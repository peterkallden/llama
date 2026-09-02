#include "astc-vulkan-ffn-adapter.h"

#include <cassert>
#include <cstdio>

int main() {
    astc_vulkan_manifest manifest;
    manifest.tensors.push_back({
        "blk.0.ffn_down.weight", 1536, 32, astc_vulkan_footprint::k6x6, 0,
        astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 1536, 32),
        astc_vulkan_representation::kGaugeLumaAlpha, 2.0f, 0.25f, -1.0f});
    astc_vulkan_ffn_adapter adapter;
    astc_vulkan_ffn_binding binding;
    std::string error;
    assert(adapter.prepare(manifest, "blk.0.ffn_down.weight", 1536, true, binding, error));
    assert(binding.status == astc_vulkan_binding_status::kReady);
    assert(binding.reconstruction.scale_l == 2.0f);
    assert(binding.reconstruction.scale_a == 0.25f);
    assert(binding.reconstruction.offset == -1.0f);
    assert(adapter.prepare(manifest, "blk.0.ffn_down.weight", 576, true, binding, error));
    assert(binding.status == astc_vulkan_binding_status::kFallback);
    assert(!binding.fallback_reason.empty());
    assert(adapter.prepare(manifest, "blk.0.ffn_down.weight", 1536, false, binding, error));
    assert(binding.status == astc_vulkan_binding_status::kFallback);
    assert(adapter.prepare(manifest, "missing", 1536, true, binding, error));
    assert(binding.status == astc_vulkan_binding_status::kFallback);
    std::puts("ASTC Vulkan FFN adapter contract passed");
    return 0;
}
