#include "astc-vulkan-d2-pair-transform.h"

#include <cmath>

namespace {

void enumerate_from_remaining(std::array<uint8_t, astc_vulkan_d2_pair_group_rows> & order,
                              uint32_t write_index, uint16_t remaining,
                              std::vector<astc_vulkan_d2_pairing> & result) {
    if (remaining == 0) {
        result.push_back({order});
        return;
    }
    uint8_t first = 0;
    while ((remaining & (uint16_t{1} << first)) == 0) ++first;
    const uint16_t without_first = static_cast<uint16_t>(remaining & ~(uint16_t{1} << first));
    for (uint8_t second = static_cast<uint8_t>(first + 1);
         second < astc_vulkan_d2_pair_group_rows; ++second) {
        if ((without_first & (uint16_t{1} << second)) == 0) continue;
        order[write_index] = first;
        order[write_index + 1] = second;
        enumerate_from_remaining(order, write_index + 2,
                                 static_cast<uint16_t>(without_first & ~(uint16_t{1} << second)),
                                 result);
    }
}

} // namespace

astc_vulkan_d2_pairing astc_vulkan_d2_identity_pairing() {
    astc_vulkan_d2_pairing result{};
    for (uint8_t index = 0; index < astc_vulkan_d2_pair_group_rows; ++index) {
        result.row_order[index] = index;
    }
    return result;
}

bool astc_vulkan_d2_pairing_is_valid(const astc_vulkan_d2_pairing & pairing) {
    uint16_t seen = 0;
    for (const uint8_t row : pairing.row_order) {
        if (row >= astc_vulkan_d2_pair_group_rows) return false;
        const uint16_t bit = static_cast<uint16_t>(uint16_t{1} << row);
        if ((seen & bit) != 0) return false;
        seen = static_cast<uint16_t>(seen | bit);
    }
    return seen == ((uint16_t{1} << astc_vulkan_d2_pair_group_rows) - 1);
}

std::vector<astc_vulkan_d2_pairing> astc_vulkan_d2_enumerate_pairings() {
    std::vector<astc_vulkan_d2_pairing> result;
    result.reserve(945);
    std::array<uint8_t, astc_vulkan_d2_pair_group_rows> order{};
    enumerate_from_remaining(order, 0,
                             static_cast<uint16_t>((uint16_t{1} << astc_vulkan_d2_pair_group_rows) - 1),
                             result);
    return result;
}

void astc_vulkan_d2_pair_forward(float w0, float w1,
                                 const astc_vulkan_d2_givens_transform & transform,
                                 float & u, float & v) {
    const float c = std::cos(transform.radians);
    const float s = std::sin(transform.radians);
    u = c * w0 + s * w1;
    v = -s * w0 + c * w1;
}

void astc_vulkan_d2_pair_inverse(float u, float v,
                                 const astc_vulkan_d2_givens_transform & transform,
                                 float & w0, float & w1) {
    const float c = std::cos(transform.radians);
    const float s = std::sin(transform.radians);
    w0 = c * u - s * v;
    w1 = s * u + c * v;
}

