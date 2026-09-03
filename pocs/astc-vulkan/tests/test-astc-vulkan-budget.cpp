#include "astc-vulkan-budget.h"

#include <iostream>
#include <string>

int main() {
    astc_vulkan_memory_budget host;
    std::string error;
    if (!astc_vulkan_query_host_memory_budget(ASTC_VULKAN_DEFAULT_MEMORY_FRACTION, host, error) ||
        host.host_available_bytes == 0 || host.host_limit_bytes == 0 ||
        host.host_limit_bytes > host.host_available_bytes) {
        std::cerr << "host budget auto-detection failed: " << error << '\n';
        return 1;
    }

    astc_vulkan_memory_budget synthetic;
    synthetic.host_limit_bytes = 100;
    synthetic.effective_device_limit_bytes = 80;
    if (!astc_vulkan_budget_can_reserve(synthetic, 16, 64, 80, error)) {
        std::cerr << "expected budget reservation to fit: " << error << '\n';
        return 1;
    }
    if (astc_vulkan_budget_can_reserve(synthetic, 16, 65, 80, error) || error.empty()) {
        std::cerr << "oversized device reservation was accepted\n";
        return 1;
    }
    if (astc_vulkan_budget_can_reserve(synthetic, 0, 1, 101, error) || error.empty()) {
        std::cerr << "oversized staging reservation was accepted\n";
        return 1;
    }
    std::cout << "ASTC Vulkan memory budget contract passed\n";
    return 0;
}
