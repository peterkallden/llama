#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

// Conservative resource admission policy for the ASTC sidecar.  The cache is
// streamed and a single tensor payload is resident on the host, but the image
// allocation itself can still be larger than its compressed payload due to
// Vulkan alignment.  We therefore budget actual Vulkan allocation bytes.
//
// The default deliberately keeps 20% of currently available memory free. On
// UMA devices the effective device cap is also constrained by host memory.
constexpr float ASTC_VULKAN_DEFAULT_MEMORY_FRACTION = 0.80f;

struct astc_vulkan_memory_budget {
    float fraction = ASTC_VULKAN_DEFAULT_MEMORY_FRACTION;
    uint64_t host_available_bytes = 0;
    uint64_t host_limit_bytes = 0;
    uint64_t device_available_bytes = 0;
    uint64_t device_limit_bytes = 0;
    uint64_t effective_device_limit_bytes = 0;
    bool integrated_gpu = false;
    bool uses_vk_ext_memory_budget = false;
};

// Detects currently available host RAM on supported platforms. It is a
// capacity limit, not a reservation or an OS-level memory lock.
bool astc_vulkan_query_host_memory_budget(float fraction,
                                          astc_vulkan_memory_budget & result,
                                          std::string & error);

// Detects the largest device-local Vulkan heap. VK_EXT_memory_budget is used
// when exposed by the driver; otherwise the physical heap size is a safe,
// conservative upper bound. For an integrated GPU, the host limit also caps
// the effective device limit.
bool astc_vulkan_query_memory_budget(VkPhysicalDevice physical_device,
                                     float fraction,
                                     astc_vulkan_memory_budget & result,
                                     std::string & error);

// Tests a proposed persistent image allocation plus its temporary upload
// staging requirement against the detected budget. `resident_device_bytes`
// allows an atlas/multi-tensor owner to apply the same rule cumulatively.
bool astc_vulkan_budget_can_reserve(const astc_vulkan_memory_budget & budget,
                                    uint64_t resident_device_bytes,
                                    uint64_t image_bytes,
                                    uint64_t staging_bytes,
                                    std::string & error);
