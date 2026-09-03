#pragma once

#include "astc-vulkan-budget.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// One approved cache record with allocations measured (or conservatively
// estimated) by the Vulkan owner. Keeping this separate from the manifest
// lets the planner account for image alignment and staging policy.
struct astc_vulkan_residency_item {
    std::string tensor_name;
    uint64_t device_bytes = 0;
    uint64_t host_bytes = 0;
};

struct astc_vulkan_residency_plan {
    std::vector<size_t> resident_items;
    uint64_t device_bytes = 0;
    uint64_t host_bytes = 0;
    bool preload_all = false;
    bool requires_streaming = false;
};

// Selects an ordered resident prefix. If all records fit, the result is a
// preload plan. Otherwise it returns the largest prefix that fits both memory
// limits, bounded by max_items when nonzero. The ordering is intentional: a
// scheduler can arrange items by layer/tensor locality before calling this
// function, without the planner inventing eviction policy.
bool astc_vulkan_plan_residency(const std::vector<astc_vulkan_residency_item> & items,
                                const astc_vulkan_memory_budget & budget,
                                size_t max_items,
                                astc_vulkan_residency_plan & result,
                                std::string & error);
