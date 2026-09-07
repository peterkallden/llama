#pragma once

#include <array>
#include <cstdint>
#include <vector>

// D2 pair preconditioner primitives.
//
// A physical D2 8x5 block represents five pairs of logical output rows. This
// module keeps pair choice and a 2x2 orthogonal transform explicit, rather
// than treating row scale, common/difference and future predictive variants as
// unrelated codecs. It is deliberately offline-only: cache serialization and
// Vulkan shader support are introduced only after an exact ASTC/model gate.
//
// Suitable for: bounded low-rate D2-LA experiments where row-pair choice is
// expected to affect ASTC's two-channel representation. Not suitable for
// runtime use until a versioned pair-map/transform metadata contract exists.
//
// Reference: ParoQuant, "Pairwise Rotations for Quantization" (ICLR 2026);
// the current implementation uses only the inexpensive orthogonal 2x2 part.

constexpr uint32_t astc_vulkan_d2_pair_group_rows = 10;
constexpr uint32_t astc_vulkan_d2_pair_group_pairs = 5;

// row_order is [first0, second0, first1, second1, ...]. It maps physical D2
// texel rows to original logical output rows and must be a permutation of 0..9.
struct astc_vulkan_d2_pairing {
    std::array<uint8_t, astc_vulkan_d2_pair_group_rows> row_order{};
};

struct astc_vulkan_d2_givens_transform {
    float radians = 0.0f;
};

astc_vulkan_d2_pairing astc_vulkan_d2_identity_pairing();
bool astc_vulkan_d2_pairing_is_valid(const astc_vulkan_d2_pairing & pairing);

// There are (10 - 1)!! = 945 perfect matchings. Pair orientation is canonical
// here; runtime orientation remains a separate D2 layout decision.
std::vector<astc_vulkan_d2_pairing> astc_vulkan_d2_enumerate_pairings();

// Applies [u v]^T = [[cos, sin], [-sin, cos]] [w0 w1]^T and its exact inverse.
void astc_vulkan_d2_pair_forward(float w0, float w1,
                                 const astc_vulkan_d2_givens_transform & transform,
                                 float & u, float & v);
void astc_vulkan_d2_pair_inverse(float u, float v,
                                 const astc_vulkan_d2_givens_transform & transform,
                                 float & w0, float & w1);

// A perfect matching of ten rows needs ceil(log2(945)) = 10 bits. Angle and
// layout metadata are intentionally outside this helper until their artifact
// encoding is versioned.
constexpr uint32_t astc_vulkan_d2_pairing_metadata_bits = 10;

