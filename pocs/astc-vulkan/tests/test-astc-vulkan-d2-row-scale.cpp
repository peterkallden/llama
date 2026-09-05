#include "astc-vulkan-d2-row-scale.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    const std::vector<float> weights{
        -2.0f, 1.0f, 0.5f,
         8.0f, -4.0f, 2.0f,
         0.0f, 0.0f, 0.0f,
    };
    const auto scales = astc_vulkan_d2_make_absmax_row_scales(weights, 3, 3);
    assert(scales.size() == 3);
    assert(scales[0].value == 2.0f);
    assert(scales[1].value == 8.0f);
    assert(scales[2].value == 1.0f);
    const auto normalized = astc_vulkan_d2_normalize_rows(weights, 3, 3, scales);
    assert(normalized.size() == weights.size());
    assert(std::fabs(normalized[0] + 1.0f) < 1e-6f);
    assert(std::fabs(normalized[3] - 1.0f) < 1e-6f);
    const auto restored = astc_vulkan_d2_restore_rows(normalized, 3, 3, scales);
    for (size_t index = 0; index < weights.size(); ++index) {
        assert(std::fabs(restored[index] - weights[index]) < 1e-6f);
    }
    assert(astc_vulkan_d2_row_scale_metadata_bytes(2048) == 4096);
    assert(astc_vulkan_d2_normalize_rows(weights, 2, 3, scales).empty());
    return 0;
}
