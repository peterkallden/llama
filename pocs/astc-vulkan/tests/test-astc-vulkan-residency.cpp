#include "astc-vulkan-residency.h"

#include <cassert>
#include <iostream>

int main() {
    astc_vulkan_memory_budget budget;
    budget.effective_device_limit_bytes = 100;
    budget.host_limit_bytes = 100;
    const std::vector<astc_vulkan_residency_item> items = {
        {"layer0", 40, 20}, {"layer1", 40, 20}, {"layer2", 40, 20},
    };
    astc_vulkan_residency_plan plan;
    std::string error;
    assert(astc_vulkan_plan_residency(items, budget, 0, plan, error));
    assert(plan.requires_streaming && !plan.preload_all);
    assert(plan.resident_items.size() == 2 && plan.device_bytes == 80);
    assert(astc_vulkan_plan_residency(items, budget, 1, plan, error));
    assert(plan.resident_items.size() == 1 && plan.device_bytes == 40);

    budget.effective_device_limit_bytes = 200;
    assert(astc_vulkan_plan_residency(items, budget, 0, plan, error));
    assert(plan.preload_all && !plan.requires_streaming);
    std::cout << "ASTC Vulkan residency planner contract passed\n";
    return 0;
}
