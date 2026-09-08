#include "astc-vulkan-composition-gate.h"

#include <cassert>

namespace {

astc_vulkan_composition_trial make_trial(const char * id, astc_vulkan_tensor_record & tensor,
                                         float median, float worst, float top1) {
    astc_vulkan_composition_trial value;
    value.artifact_id = id;
    value.candidate.tensor = &tensor;
    value.candidate.rate_bpw = 1.6;
    value.candidate.evidence = {true, true, 0.0f, median, median, top1, "cal", "replay"};
    value.candidate.evidence.replay_case_count = 3;
    value.candidate.evidence.median_loss_delta = median;
    value.candidate.evidence.worst_loss_delta = worst;
    value.candidate.evidence.worst_top1_agreement = top1;
    return value;
}

} // namespace

int main() {
    astc_vulkan_tensor_record first; first.name = "first";
    astc_vulkan_tensor_record second; second.name = "second";
    astc_vulkan_composition_options options;
    options.quality_rules.max_worst_loss_delta = .05f;
    options.quality_rules.min_worst_top1_agreement = .9f;
    const std::vector<astc_vulkan_composition_stage> stages = {
        {"first", {make_trial("first-selected", first, .01f, .02f, 1.0f)}},
        {"second", {make_trial("second-bad", second, .005f, .07f, 1.0f),
                    make_trial("second-neutral", second, .02f, .03f, 1.0f)}},
    };
    astc_vulkan_composition_result result;
    std::string error;
    assert(astc_vulkan_select_composed_artifacts(stages, options, result, error));
    assert(result.decisions.size() == 2);
    assert(result.decisions[0].artifact_id == "first-selected");
    assert(result.decisions[1].artifact_id == "second-neutral");
    assert(!result.decisions[1].use_native_fallback);

    const std::vector<astc_vulkan_composition_stage> fallback = {
        {"second", {make_trial("second-bad", second, .005f, .07f, .8f)}},
    };
    assert(astc_vulkan_select_composed_artifacts(fallback, options, result, error));
    assert(result.decisions[0].use_native_fallback);
    return 0;
}
