#include "astc-vulkan-objective.h"

#include <cassert>
#include <cmath>
#include <cstring>

int main() {
    const std::vector<float> error{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> trace{1.0f, 0.0f, 0.0f, 1.0f};
    const double activation = astc_vulkan_activation_score(error, 2, 2, trace, 2);
    assert(std::fabs(activation - 30.0) < 1e-12);
    const std::vector<float> sensitivity{2.0f, 0.5f};
    const double weighted = astc_vulkan_weighted_activation_score(
        error, 2, 2, trace, 2, sensitivity);
    assert(std::fabs(weighted - 22.5) < 1e-12);
    assert(std::strcmp(astc_vulkan_objective_name(
        astc_vulkan_objective::two_sided_trace), "two-sided-trace") == 0);
    assert(std::isnan(astc_vulkan_weighted_activation_score(
        error, 2, 2, trace, 2, {1.0f})));
    return 0;
}
