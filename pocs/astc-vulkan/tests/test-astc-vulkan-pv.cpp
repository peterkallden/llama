#include "astc-vulkan-pv.h"

#include <cassert>
#include <cmath>

int main() {
    const std::vector<float> initial{ 0.0f, 0.0f };
    const std::vector<float> steps{ 0.25f, 0.25f };
    astc_vulkan_pv_result result;
    const bool ok = astc_vulkan_pv_alternate(
        initial, steps, 8,
        [](const std::vector<float> & continuous, std::vector<float> & deployed) {
            deployed = continuous;
            return true;
        },
        [](const std::vector<float> & deployed) {
            const double dx = deployed[0] - 0.5;
            const double dy = deployed[1] + 0.25;
            return dx * dx + dy * dy;
        }, result);
    assert(ok);
    assert(result.accepted_steps > 0);
    assert(std::abs(result.continuous[0] - 0.5f) < 1e-6f);
    assert(std::abs(result.continuous[1] + 0.25f) < 1e-6f);
    assert(result.objective < 1e-12);
    return 0;
}
