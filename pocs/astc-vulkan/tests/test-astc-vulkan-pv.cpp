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
    astc_vulkan_pv_result batched;
    const bool batch_ok = astc_vulkan_pv_alternate_batched(
        initial, steps, 8,
        [](const std::vector<float> & continuous, std::vector<float> & deployed) {
            deployed = continuous;
            return true;
        },
        [](const std::vector<std::vector<float>> & deployed, std::vector<double> & scores) {
            scores.clear();
            for (const auto & value : deployed) {
                const double dx = value[0] - 0.5;
                const double dy = value[1] + 0.25;
                scores.push_back(dx * dx + dy * dy);
            }
            return true;
        }, batched);
    assert(batch_ok && std::abs(batched.continuous[0] - 0.5f) < 1e-6f &&
           std::abs(batched.continuous[1] + 0.25f) < 1e-6f);
    return 0;
}
