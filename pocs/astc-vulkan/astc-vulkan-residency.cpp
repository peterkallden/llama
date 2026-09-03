#include "astc-vulkan-residency.h"

#include <limits>

bool astc_vulkan_plan_residency(const std::vector<astc_vulkan_residency_item> & items,
                                const astc_vulkan_memory_budget & budget,
                                size_t max_items,
                                astc_vulkan_residency_plan & result,
                                std::string & error) {
    result = {};
    if (budget.effective_device_limit_bytes == 0) {
        error = "ASTC residency planner has no effective device limit";
        return false;
    }
    for (size_t index = 0; index < items.size(); ++index) {
        if (max_items != 0 && result.resident_items.size() >= max_items) break;
        const auto & item = items[index];
        if (item.device_bytes > budget.effective_device_limit_bytes -
                                std::min(result.device_bytes, budget.effective_device_limit_bytes) ||
            (budget.host_limit_bytes != 0 &&
             (item.host_bytes > budget.host_limit_bytes -
                               std::min(result.host_bytes, budget.host_limit_bytes)))) {
            break;
        }
        result.resident_items.push_back(index);
        result.device_bytes += item.device_bytes;
        result.host_bytes += item.host_bytes;
    }
    result.preload_all = result.resident_items.size() == items.size();
    result.requires_streaming = !result.preload_all;
    if (!items.empty() && result.resident_items.empty()) {
        error = "no ASTC cache item fits the configured residency budget";
        return false;
    }
    error.clear();
    return true;
}
