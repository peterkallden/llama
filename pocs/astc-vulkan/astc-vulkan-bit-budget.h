#pragma once

#include "astc-vulkan-ise.h"

#include <cstdint>
#include <vector>

struct astc_vulkan_bit_budget_request {
    uint32_t fixed_block_bits = 0;
    uint32_t endpoint_value_count = 0;
    uint32_t weight_value_count = 0;
    uint32_t endpoint_max_value = 0;
    uint32_t weight_max_value = 0;
    uint32_t payload_bits = 128;
};

struct astc_vulkan_bit_budget_result {
    astc_vulkan_ise_range endpoint_range{};
    astc_vulkan_ise_range weight_range{};
    uint32_t endpoint_bits = 0;
    uint32_t weight_bits = 0;
    uint32_t fixed_block_bits = 0;
    uint32_t used_bits = 0;
    uint32_t remaining_bits = 0;
    bool legal = false;
};

// Chooses the lowest-cost legal endpoint/weight pair under the supplied
// payload budget. It does not assume an ASTC mode; callers provide all fixed
// header/partition/dual-plane bits from the audited mode descriptor.
astc_vulkan_bit_budget_result astc_vulkan_evaluate_bit_budget(
    const astc_vulkan_bit_budget_request & request);

// Enumerates every legal pair of ISE ranges under the same mode budget. The
// caller can rank the returned candidates for quality; this layer does not
// decide whether endpoint or weight precision is more valuable.
std::vector<astc_vulkan_bit_budget_result> astc_vulkan_enumerate_bit_budget(
    const astc_vulkan_bit_budget_request & request);
