#include "astc-vulkan-scheduler-adapter.h"

#include <cassert>

int main() {
    astc_vulkan_scheduler_adapter adapter;
    std::string error;
    assert(!adapter.ready());
    std::vector<float> output;
    assert(!adapter.run({}, {}, output, error));
    assert(error == "ASTC scheduler adapter is not ready; use normal fallback");
    adapter.reset();
    assert(!adapter.ready());
    return 0;
}
