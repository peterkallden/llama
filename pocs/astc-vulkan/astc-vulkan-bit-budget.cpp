#include "astc-vulkan-bit-budget.h"

namespace {

std::vector<astc_vulkan_ise_range> ranges_for(uint32_t max_value) {
    std::vector<astc_vulkan_ise_range> ranges;
    for (const auto family : {astc_vulkan_ise_family::binary,
                              astc_vulkan_ise_family::trit,
                              astc_vulkan_ise_family::quint}) {
        for (uint32_t bits = 0; bits <= 8; ++bits) {
            const uint32_t capacity = family == astc_vulkan_ise_family::binary
                ? (1u << bits) : (family == astc_vulkan_ise_family::trit
                    ? (3u << bits) : (5u << bits));
            if (capacity > max_value) {
                ranges.push_back({family, static_cast<uint8_t>(bits),
                                  static_cast<uint16_t>(capacity)});
            }
        }
    }
    return ranges;
}

} // namespace

std::vector<astc_vulkan_bit_budget_result> astc_vulkan_enumerate_bit_budget(
    const astc_vulkan_bit_budget_request & request) {
    std::vector<astc_vulkan_bit_budget_result> results;
    if (request.payload_bits == 0 || request.fixed_block_bits > request.payload_bits) return results;
    const auto endpoints = ranges_for(request.endpoint_max_value);
    const auto weights = ranges_for(request.weight_max_value);
    for (const auto endpoint_range : endpoints) {
        const auto endpoint = astc_vulkan_ise_evaluate(endpoint_range, request.endpoint_value_count);
        for (const auto weight_range : weights) {
            const auto weight = astc_vulkan_ise_evaluate(weight_range, request.weight_value_count);
            if (!endpoint.valid || !weight.valid) continue;
            astc_vulkan_bit_budget_result result;
            result.endpoint_range = endpoint_range;
            result.weight_range = weight_range;
            result.endpoint_bits = endpoint.total_bits;
            result.weight_bits = weight.total_bits;
            result.fixed_block_bits = request.fixed_block_bits;
            result.used_bits = request.fixed_block_bits + result.endpoint_bits + result.weight_bits;
            if (result.used_bits > request.payload_bits) continue;
            result.remaining_bits = request.payload_bits - result.used_bits;
            result.legal = true;
            results.push_back(result);
        }
    }
    return results;
}

astc_vulkan_bit_budget_result astc_vulkan_evaluate_bit_budget(
    const astc_vulkan_bit_budget_request & request) {
    astc_vulkan_bit_budget_result best;
    best.fixed_block_bits = request.fixed_block_bits;
    if (request.payload_bits == 0 || request.fixed_block_bits > request.payload_bits) return best;
    best.endpoint_range = astc_vulkan_ise_choose_range(request.endpoint_max_value,
                                                       request.endpoint_value_count);
    best.weight_range = astc_vulkan_ise_choose_range(request.weight_max_value,
                                                     request.weight_value_count);
    const auto endpoint = astc_vulkan_ise_evaluate(best.endpoint_range, request.endpoint_value_count);
    const auto weight = astc_vulkan_ise_evaluate(best.weight_range, request.weight_value_count);
    if (!endpoint.valid || !weight.valid) return best;
    best.endpoint_bits = endpoint.total_bits;
    best.weight_bits = weight.total_bits;
    best.used_bits = request.fixed_block_bits + best.endpoint_bits + best.weight_bits;
    if (best.used_bits <= request.payload_bits) {
        best.remaining_bits = request.payload_bits - best.used_bits;
        best.legal = true;
    }
    return best;
}
