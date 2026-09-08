#include "astc-vulkan-composition-gate.h"

#include <cstdio>

namespace {

astc_vulkan_composition_trial trial(const char * id, astc_vulkan_tensor_record & tensor,
                                    float median_loss, float worst_loss, float worst_top1) {
    astc_vulkan_composition_trial result;
    result.artifact_id = id;
    result.candidate.tensor = &tensor;
    result.candidate.rate_bpw = 1.6;
    result.candidate.evidence.model_gate_passed = true;
    result.candidate.evidence.vulkan_gate_passed = true;
    result.candidate.evidence.loss_delta = median_loss;
    result.candidate.evidence.logits_relative_mse = median_loss;
    result.candidate.evidence.top1_agreement = worst_top1;
    result.candidate.evidence.replay_case_count = 3;
    result.candidate.evidence.median_loss_delta = median_loss;
    result.candidate.evidence.worst_loss_delta = worst_loss;
    result.candidate.evidence.worst_top1_agreement = worst_top1;
    return result;
}

} // namespace

int main() {
    astc_vulkan_tensor_record tensor_a; tensor_a.name = "blk.0.ffn_down.weight";
    astc_vulkan_tensor_record tensor_b; tensor_b.name = "blk.1.ffn_down.weight";
    astc_vulkan_tensor_record tensor_c; tensor_c.name = "blk.2.ffn_down.weight";
    astc_vulkan_composition_options options;
    options.quality_rules.max_worst_loss_delta = 0.05f;
    options.quality_rules.min_worst_top1_agreement = 0.90f;
    const std::vector<astc_vulkan_composition_stage> stages = {
        {tensor_a.name, {trial("a/la-selected", tensor_a, .010f, .020f, 1.0f)}},
        // First candidate has a bad prompt tail; the neutral alternative is
        // accepted for the same prefix instead.
        {tensor_b.name, {trial("b/la-selected", tensor_b, .008f, .080f, .95f),
                         trial("b/la-neutral", tensor_b, .018f, .030f, 1.0f)}},
        // No acceptable global replay result: retain the native GGUF tensor.
        {tensor_c.name, {trial("c/la-selected", tensor_c, .009f, .060f, .88f)}},
    };
    astc_vulkan_composition_result result;
    std::string error;
    if (!astc_vulkan_select_composed_artifacts(stages, options, result, error)) {
        std::fprintf(stderr, "composition gate failed: %s\n", error.c_str());
        return 1;
    }
    for (const auto & decision : result.decisions) {
        std::printf("composition tensor=%s artifact=%s fallback=%s reason=%s\n",
                    decision.tensor_name.c_str(), decision.artifact_id.c_str(),
                    decision.use_native_fallback ? "true" : "false", decision.reason.c_str());
    }
    return result.decisions.size() == 3 &&
           result.decisions[0].artifact_id == "a/la-selected" &&
           result.decisions[1].artifact_id == "b/la-neutral" &&
           result.decisions[2].use_native_fallback ? 0 : 2;
}
