#include "astc-vulkan-composition-gate.h"

#include <unordered_set>
#include <utility>

bool astc_vulkan_select_composed_artifacts(
        const std::vector<astc_vulkan_composition_stage> & stages,
        const astc_vulkan_composition_options & options,
        astc_vulkan_composition_result & result,
        std::string & error) {
    result = {};
    std::unordered_set<std::string> names;
    for (const auto & stage : stages) {
        if (stage.tensor_name.empty() || !names.insert(stage.tensor_name).second) {
            error = "composition stages require unique non-empty tensor names";
            return false;
        }
        astc_vulkan_composition_decision decision;
        decision.tensor_name = stage.tensor_name;
        for (const auto & trial : stage.trials) {
            if (trial.artifact_id.empty()) {
                error = "composition trial requires a non-empty artifact id";
                return false;
            }
            if (!astc_vulkan_artifact_is_eligible(
                    trial.candidate, true, true, options.quality_rules)) {
                continue;
            }
            decision.artifact_id = trial.artifact_id;
            decision.reason = "global-replay-gate-passed";
            break;
        }
        if (decision.artifact_id.empty()) {
            decision.use_native_fallback = true;
            decision.reason = "no-artifact-passed-global-replay-gate";
        }
        result.decisions.push_back(std::move(decision));
    }
    error.clear();
    return true;
}
