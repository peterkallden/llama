#pragma once

#include <cstddef>
#include <vector>

// D2 per-output-row symmetric normalization.
//
// D2 packs two output rows into one ASTC texel. A single global affine range
// can make that pairing unnecessarily difficult when the two rows have very
// different dynamic ranges. This offline-only primitive derives one positive
// absmax scale per output row, so the paired source is normalized as
// z[r, c] = w[r, c] / scale[r]. Runtime reconstructs the output with one
// multiply per output row after the reduction.
//
// Suitable for: a separate low-rate D2-LA representation experiment. The
// scales are side metadata (normally FP16 in a later artifact ABI), not ASTC
// payload bytes. This module intentionally does not choose an ASTC candidate
// or change a selector objective.
//
// References: ParoQuant (arXiv:2501.01558) motivates inexpensive pair/channel
// normalization at low bit rates; the project ASTC-Vulkan plan defines the
// artifact and runtime gates required before this becomes deployable.

struct astc_vulkan_d2_row_scale {
    float value = 1.0f;
};

// Returns one finite, positive absmax scale per logical output row. Zero or
// non-finite rows use 1.0, preserving finite normalization and deterministic
// padding behavior.
std::vector<astc_vulkan_d2_row_scale> astc_vulkan_d2_make_absmax_row_scales(
    const std::vector<float> & weights, unsigned int rows, unsigned int columns);

// Applies/removes symmetric row scaling for a dense row-major matrix. Invalid
// dimensions return an empty vector; callers must preserve the original shape.
std::vector<float> astc_vulkan_d2_normalize_rows(
    const std::vector<float> & weights, unsigned int rows, unsigned int columns,
    const std::vector<astc_vulkan_d2_row_scale> & scales);
std::vector<float> astc_vulkan_d2_restore_rows(
    const std::vector<float> & normalized, unsigned int rows, unsigned int columns,
    const std::vector<astc_vulkan_d2_row_scale> & scales);

// The planned artifact format writes one FP16 scale per logical output row.
constexpr size_t astc_vulkan_d2_row_scale_metadata_bytes(unsigned int rows) {
    return static_cast<size_t>(rows) * sizeof(unsigned short);
}
