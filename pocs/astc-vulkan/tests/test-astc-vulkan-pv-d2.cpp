#include "astc-vulkan-pv.h"

#include <cassert>
#include <cmath>

namespace {
double paired_loss(const std::vector<float> & deployed, bool rg_b) {
    double loss = 0.0;
    for (size_t texel = 0; texel < 4; ++texel) {
        const float first = deployed[texel * 2 + (rg_b ? 0 : 1)];
        const float second = deployed[texel * 2 + (rg_b ? 1 : 0)];
        loss += (first - 0.25f) * (first - 0.25f);
        loss += (second + 0.50f) * (second + 0.50f);
    }
    return loss;
}
bool run_layout(bool rg_b, astc_vulkan_pv_result & result) {
    return astc_vulkan_pv_alternate_batched(
        {0.0f, 0.0f}, {0.25f, 0.25f}, 4,
        [rg_b](const std::vector<float> & coefficients, std::vector<float> & deployed) {
            deployed.resize(8);
            for (size_t texel = 0; texel < 4; ++texel) {
                deployed[texel * 2 + (rg_b ? 0 : 1)] = coefficients[0];
                deployed[texel * 2 + (rg_b ? 1 : 0)] = coefficients[1];
            }
            return true;
        },
        [rg_b](const std::vector<std::vector<float>> & deployed, std::vector<double> & scores) {
            scores.clear();
            for (const auto & candidate : deployed) scores.push_back(paired_loss(candidate, rg_b));
            return true;
        }, result);
}
}

int main() {
    astc_vulkan_pv_result rg_b, r_gb;
    assert(run_layout(true, rg_b));
    assert(run_layout(false, r_gb));
    assert(rg_b.accepted_steps > 0 && r_gb.accepted_steps > 0);
    assert(std::abs(rg_b.objective - r_gb.objective) < 1e-12);
    assert(rg_b.deployed.size() == 8 && r_gb.deployed.size() == 8);
    return 0;
}
