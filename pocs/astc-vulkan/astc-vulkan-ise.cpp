#include "astc-vulkan-ise.h"

#include <algorithm>
#include <limits>

namespace {

uint32_t ceil_div(uint32_t a, uint32_t b) { return b == 0 ? 0 : (a + b - 1u) / b; }

uint32_t capacity(astc_vulkan_ise_family family, uint32_t bits) {
    switch (family) {
    case astc_vulkan_ise_family::binary: return 1u << bits;
    case astc_vulkan_ise_family::trit: return 3u << bits;
    case astc_vulkan_ise_family::quint: return 5u << bits;
    }
    return 0;
}

uint32_t auxiliary_bits(astc_vulkan_ise_family family, uint32_t count) {
    switch (family) {
    case astc_vulkan_ise_family::binary: return 0;
    case astc_vulkan_ise_family::trit: return ceil_div(8u * count, 5u);
    case astc_vulkan_ise_family::quint: return ceil_div(7u * count, 3u);
    }
    return 0;
}

void put_bits(std::vector<uint8_t> & out, uint32_t & offset, uint32_t value, uint32_t count) {
    for (uint32_t bit = 0; bit < count; ++bit) {
        if ((offset >> 3u) >= out.size()) out.push_back(0);
        if ((value >> bit) & 1u) out[offset >> 3u] |= uint8_t(1u << (offset & 7u));
        ++offset;
    }
}

uint32_t get_bits(const std::vector<uint8_t> & in, uint32_t & offset, uint32_t count) {
    uint32_t value = 0;
    for (uint32_t bit = 0; bit < count; ++bit) {
        if ((offset >> 3u) >= in.size()) return std::numeric_limits<uint32_t>::max();
        value |= uint32_t((in[offset >> 3u] >> (offset & 7u)) & 1u) << bit;
        ++offset;
    }
    return value;
}

uint32_t pack_group(const std::vector<uint16_t> & values, uint32_t first,
                    uint32_t count, uint32_t radix) {
    uint32_t packed = 0, multiplier = 1;
    for (uint32_t index = 0; index < count; ++index) {
        packed += uint32_t(values[first + index]) * multiplier;
        multiplier *= radix;
    }
    return packed;
}

} // namespace

astc_vulkan_ise_range astc_vulkan_ise_choose_range(uint32_t max_value,
                                                   uint32_t value_count) {
    astc_vulkan_ise_range best{};
    uint32_t best_bits = std::numeric_limits<uint32_t>::max();
    for (const auto family : {astc_vulkan_ise_family::binary,
                              astc_vulkan_ise_family::trit,
                              astc_vulkan_ise_family::quint}) {
        for (uint32_t bits = 0; bits <= 8; ++bits) {
            if (capacity(family, bits) <= max_value) continue;
            const uint32_t cost = value_count * bits + auxiliary_bits(family, value_count);
            if (cost < best_bits) {
                best = {family, static_cast<uint8_t>(bits),
                        static_cast<uint16_t>(capacity(family, bits))};
                best_bits = cost;
            }
        }
    }
    return best;
}

astc_vulkan_ise_layout astc_vulkan_ise_evaluate(
    astc_vulkan_ise_range range, uint32_t value_count) {
    astc_vulkan_ise_layout result;
    result.range = range;
    result.value_count = value_count;
    if (range.symbol_count == 0 || value_count == 0 ||
        range.symbol_count != capacity(range.family, range.binary_bits)) return result;
    result.binary_bits = value_count * range.binary_bits;
    result.auxiliary_bits = auxiliary_bits(range.family, value_count);
    result.total_bits = result.binary_bits + result.auxiliary_bits;
    result.valid = true;
    return result;
}

bool astc_vulkan_ise_pack(astc_vulkan_ise_range range,
                          const std::vector<uint16_t> & values,
                          std::vector<uint8_t> & bits) {
    const auto layout = astc_vulkan_ise_evaluate(range, values.size());
    if (!layout.valid) return false;
    for (const uint16_t value : values) if (value >= range.symbol_count) return false;
    bits.assign((layout.total_bits + 7u) / 8u, 0);
    uint32_t offset = 0;
    if (range.family == astc_vulkan_ise_family::binary) {
        for (const uint16_t value : values) put_bits(bits, offset, value, range.binary_bits);
        return true;
    }
    const uint32_t radix = range.family == astc_vulkan_ise_family::trit ? 3u : 5u;
    const uint32_t group_size = range.family == astc_vulkan_ise_family::trit ? 5u : 3u;
    const uint32_t group_bits = range.family == astc_vulkan_ise_family::trit ? 8u : 7u;
    for (uint32_t first = 0; first < values.size(); first += group_size) {
        const uint32_t count = std::min(group_size, uint32_t(values.size() - first));
        uint32_t packed = pack_group(values, first, count, radix);
        put_bits(bits, offset, packed, count == group_size ? group_bits :
                 (range.family == astc_vulkan_ise_family::trit ? ceil_div(8u * count, 5u)
                                                                : ceil_div(7u * count, 3u)));
    }
    return offset == layout.total_bits;
}

bool astc_vulkan_ise_unpack(astc_vulkan_ise_range range, uint32_t value_count,
                            const std::vector<uint8_t> & bits,
                            std::vector<uint16_t> & values) {
    const auto layout = astc_vulkan_ise_evaluate(range, value_count);
    if (!layout.valid || bits.size() * 8u < layout.total_bits) return false;
    values.assign(value_count, 0);
    uint32_t offset = 0;
    if (range.family == astc_vulkan_ise_family::binary) {
        for (uint16_t & value : values) {
            const uint32_t decoded = get_bits(bits, offset, range.binary_bits);
            if (decoded == std::numeric_limits<uint32_t>::max()) return false;
            value = static_cast<uint16_t>(decoded);
        }
        return true;
    }
    const uint32_t radix = range.family == astc_vulkan_ise_family::trit ? 3u : 5u;
    const uint32_t group_size = range.family == astc_vulkan_ise_family::trit ? 5u : 3u;
    const uint32_t group_bits = range.family == astc_vulkan_ise_family::trit ? 8u : 7u;
    for (uint32_t first = 0; first < value_count; first += group_size) {
        const uint32_t count = std::min(group_size, value_count - first);
        const uint32_t encoded_bits = count == group_size ? group_bits :
            (range.family == astc_vulkan_ise_family::trit ? ceil_div(8u * count, 5u)
                                                           : ceil_div(7u * count, 3u));
        const uint32_t packed = get_bits(bits, offset, encoded_bits);
        if (packed == std::numeric_limits<uint32_t>::max()) return false;
        uint32_t value = packed;
        for (uint32_t index = 0; index < count; ++index) {
            values[first + index] = static_cast<uint16_t>(value % radix);
            value /= radix;
        }
    }
    return offset == layout.total_bits;
}
