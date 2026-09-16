#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_intervention_region_trial arm(uint32_t layer, float score, bool helped = false) {
    common_flydelta_intervention_region_trial value;
    value.candidate.layer_indices = {layer};
    value.candidate.anchor_layer_index = layer;
    value.candidate.total_scale = 0.1f;
    value.candidate.per_layer_scale = 0.1f;
    value.executed = true;
    value.verifier_known = true;
    value.outcome = helped ? common_flydelta_counterfactual_outcome::helped :
        common_flydelta_counterfactual_outcome::unknown;
    value.search_score = score;
    value.promising = true;
    value.safe_to_continue = true;
    value.evidence_ref = "evidence:orchestration";
    return value;
}

int main() {
    std::string error;
    common_flydelta_search_pipeline_result pipeline;
    common_flydelta_search_pipeline_direction_result direction;
    direction.region_trials = {arm(22, 0.3f), arm(24, 0.6f), arm(25, 0.5f)};
    pipeline.directions.push_back(direction);
    common_flydelta_search_continuation continuation;
    CHECK(common_flydelta_select_search_continuation(pipeline, continuation, error));
    CHECK(continuation.region.anchor_layer_index == 24 && !continuation.host_helped);
    pipeline.directions.front().region_trials.push_back(arm(25, 0.4f, true));
    CHECK(common_flydelta_select_search_continuation(pipeline, continuation, error));
    CHECK(continuation.region.anchor_layer_index == 25 && continuation.host_helped);

    common_flydelta_evidence_depth_result depth;
    depth.compatible_samples = 1;
    depth.effective_rank = 1;
    common_flydelta_experiment_plan plan;
    CHECK(common_flydelta_plan_search_continuation(continuation, depth, plan, error));
    CHECK(plan.phase == common_flydelta_experiment_phase::bootstrap);
    depth.compatible_samples = 2;
    depth.effective_rank = 2;
    depth.depth = common_flydelta_search_depth::shallow;
    CHECK(common_flydelta_plan_search_continuation(continuation, depth, plan, error));
    CHECK(plan.run_rank_two_controls_first && !plan.allow_tfo_lite_after_controls);
    depth.compatible_samples = 6;
    depth.depth = common_flydelta_search_depth::deep;
    CHECK(common_flydelta_plan_search_continuation(continuation, depth, plan, error));
    CHECK(plan.run_rank_two_controls_first && plan.allow_tfo_lite_after_controls);
    return 0;
}
