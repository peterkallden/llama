#include "astc-vulkan-scheduler-adapter.h"

#include <cassert>

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
    adapter.reset();
    assert(!adapter.ready());
    return 0;
}
