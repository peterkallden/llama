#include "astc-vulkan-pv.h"

#include <cmath>

bool astc_vulkan_pv_alternate(
    const std::vector<float> & initial,
    const std::vector<float> & coordinate_steps,
    uint32_t max_iterations,
    const astc_vulkan_pv_projector & project,
    const astc_vulkan_pv_objective & objective,
    astc_vulkan_pv_result & result) {
    if (initial.empty() || initial.size() != coordinate_steps.size() ||
        !project || !objective || max_iterations == 0) return false;

    result = {};
    result.continuous = initial;
    if (!project(result.continuous, result.deployed) || result.deployed.empty()) return false;
    result.objective = objective(result.deployed);
    if (!std::isfinite(result.objective)) return false;

    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        ++result.iterations;
        bool improved = false;
        for (size_t coordinate = 0; coordinate < result.continuous.size(); ++coordinate) {
            const float step = coordinate_steps[coordinate];
            if (!std::isfinite(step) || step == 0.0f) continue;
            std::vector<float> best_continuous = result.continuous;
            std::vector<float> best_deployed = result.deployed;
            double best_objective = result.objective;
            for (const float direction : { -1.0f, 1.0f }) {
                std::vector<float> trial = result.continuous;
                trial[coordinate] += direction * step;
                std::vector<float> deployed;
                if (!project(trial, deployed) || deployed.empty()) continue;
                const double value = objective(deployed);
                if (std::isfinite(value) && value + 1e-15 < best_objective) {
                    best_continuous = std::move(trial);
                    best_deployed = std::move(deployed);
                    best_objective = value;
                }
            }
            if (best_objective + 1e-15 < result.objective) {
                result.continuous = std::move(best_continuous);
                result.deployed = std::move(best_deployed);
                result.objective = best_objective;
                ++result.accepted_steps;
                improved = true;
            }
        }
        if (!improved) break;
    }
    return true;
}
