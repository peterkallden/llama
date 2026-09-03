#include "astc-vulkan-sidecar.h"

#include <cassert>
#include <cstdio>

int main() {
    astc_vulkan_sidecar sidecar;
    std::string error;
    std::vector<float> output;
    assert(!sidecar.ready());
    assert(!sidecar.run({}, {}, output, error));
    assert(error == "ASTC Vulkan sidecar has no ready tensor");

    astc_vulkan_manifest manifest;
    manifest.tensors.push_back({
        "blk.0.ffn_down.weight", 1536, 32, astc_vulkan_footprint::k6x6, 0,
        astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 1536, 32)});
    assert(sidecar.set_manifest(manifest, error));

    astc_vulkan_ffn_binding binding;
    assert(!sidecar.bind_tensor("missing", 1, 1, {}, binding, error));
    assert(error == "ASTC Vulkan sidecar is not initialized");

    astc_vulkan_manifest invalid = manifest;
    invalid.tensors[0].width = 0;
    assert(!sidecar.set_manifest(invalid, error));
    assert(!sidecar.ready());

    const bool initialized = sidecar.init(astc_vulkan_footprint::k6x6, error);
    if (initialized) {
        assert(sidecar.ready());
        const astc_vulkan_memory_budget & budget = sidecar.memory_budget();
        assert(budget.fraction == ASTC_VULKAN_DEFAULT_MEMORY_FRACTION);
        assert(budget.device_available_bytes != 0);
        assert(budget.effective_device_limit_bytes != 0);
        assert(budget.effective_device_limit_bytes <= budget.device_available_bytes);
        sidecar.reset();
        assert(!sidecar.ready());
    } else {
        assert(!error.empty());
    }
    std::puts("ASTC Vulkan sidecar lifecycle contract passed");
    return 0;
}
