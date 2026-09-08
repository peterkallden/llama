#pragma once

// CPU-only BISE/ISE arithmetic used by the offline ASTC quality planner.
//
// This layer deliberately owns symbol alphabets and exact bit costs, but not
// ASTC block headers or endpoint/weight field placement. The latter belongs to
// astc-vulkan-bit-budget and is validated against astcenc block information.

#include <cstdint>
#include <vector>

enum class astc_vulkan_ise_family : uint8_t {
    binary,
    trit,
    quint,
};

struct astc_vulkan_ise_range {
    astc_vulkan_ise_family family = astc_vulkan_ise_family::binary;
    uint8_t binary_bits = 0;
    uint16_t symbol_count = 0;
};

struct astc_vulkan_ise_layout {
    astc_vulkan_ise_range range{};
    uint32_t value_count = 0;
    uint32_t binary_bits = 0;
    uint32_t auxiliary_bits = 0;
    uint32_t total_bits = 0;
    bool valid = false;
};

// Chooses the least-cost BISE family whose alphabet contains [0, max_value].
// Ties prefer binary, then trit, then quint for deterministic diagnostics.
astc_vulkan_ise_range astc_vulkan_ise_choose_range(uint32_t max_value,
                                                   uint32_t value_count = 1);

astc_vulkan_ise_layout astc_vulkan_ise_evaluate(
    astc_vulkan_ise_range range, uint32_t value_count);

// Canonical CPU round-trip stream for testing and cost validation. It is not
// yet the final ASTC physical bit interleave; GPU block packing remains gated.
bool astc_vulkan_ise_pack(
    astc_vulkan_ise_range range,
    const std::vector<uint16_t> & values,
    std::vector<uint8_t> & bits);

bool astc_vulkan_ise_unpack(
    astc_vulkan_ise_range range,
    uint32_t value_count,
    const std::vector<uint8_t> & bits,
    std::vector<uint16_t> & values);
