#include "astc-vulkan-d2-pair-transform.h"

#include <cassert>
#include <cmath>

int main() {
    const auto identity = astc_vulkan_d2_identity_pairing();
    assert(astc_vulkan_d2_pairing_is_valid(identity));
    const auto pairings = astc_vulkan_d2_enumerate_pairings();
    assert(pairings.size() == 945);
    for (const auto & pairing : pairings) assert(astc_vulkan_d2_pairing_is_valid(pairing));

    const astc_vulkan_d2_givens_transform transform{0.5235987756f};
    float u = 0.0f, v = 0.0f, w0 = 0.0f, w1 = 0.0f;
    astc_vulkan_d2_pair_forward(0.25f, -0.75f, transform, u, v);
    astc_vulkan_d2_pair_inverse(u, v, transform, w0, w1);
    assert(std::fabs(w0 - 0.25f) < 1e-6f);
    assert(std::fabs(w1 + 0.75f) < 1e-6f);
    static_assert(astc_vulkan_d2_pairing_metadata_bits == 10,
                  "D2 perfect matching metadata changed");
    return 0;
}
