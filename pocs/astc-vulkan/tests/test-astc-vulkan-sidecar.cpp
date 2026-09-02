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

    astc_vulkan_ffn_binding binding;
    assert(!sidecar.bind_tensor("missing", 1, 1, {}, binding, error));
    assert(error == "ASTC Vulkan sidecar is not initialized");

    const bool initialized = sidecar.init(astc_vulkan_footprint::k6x6, error);
    if (initialized) {
        assert(sidecar.ready());
        sidecar.reset();
        assert(!sidecar.ready());
    } else {
        assert(!error.empty());
    }
    std::puts("ASTC Vulkan sidecar lifecycle contract passed");
    return 0;
}
