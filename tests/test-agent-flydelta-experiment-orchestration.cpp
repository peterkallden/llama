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
    CHECK(plan.phase == common_flydelta_experiment_phase::bootstrap &&
        !plan.run_rank_two_controls_first && !plan.tfo_lite_permitted_by_evidence);
    depth.compatible_samples = 6;
    depth.depth = common_flydelta_search_depth::deep;
    CHECK(common_flydelta_plan_search_continuation(continuation, depth, plan, error));
    CHECK(plan.phase == common_flydelta_experiment_phase::bootstrap &&
        !plan.run_rank_two_controls_first && plan.tfo_lite_permitted_by_evidence &&
        plan.tfo_lite_requires_utility_gate && !plan.run_tfo_lite);

    common_flydelta_utility_gate_config utility_config;
    common_flydelta_subspace_utility_observation utility;
    utility.safe_to_continue = true;
    utility.decision_margin_available = true;
    utility.decision_margin_delta = 0.2f;
    utility.geometry_available = true;
    utility.geometry.cosine = 0.8f;
    utility.geometry.progress = 0.2f;
    utility.geometry.leakage = 0.1f;
    utility.geometry.shift_norm = 0.2f;
    common_flydelta_utility_gate_decision utility_decision;
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, common_flydelta_search_depth::bootstrap,
        common_flydelta_experiment_phase::bootstrap, {utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::refine_bootstrap);
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, common_flydelta_search_depth::shallow,
        common_flydelta_experiment_phase::bootstrap, {utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::escalate_shallow);
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, common_flydelta_search_depth::deep,
        common_flydelta_experiment_phase::shallow_controls, {utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::escalate_deep);
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, common_flydelta_search_depth::deep,
        common_flydelta_experiment_phase::deep_controls, {utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::allow_tfo_lite);

    // Capacity may be Deep, but the execution path remains ordered:
    // Bootstrap/Whirlpool -> Shallow controls -> Deep controls -> TFO-lite.
    CHECK(common_flydelta_plan_search_continuation(continuation, depth, plan, error));
    bool advanced = false;
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, plan.depth, plan.phase, {utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::escalate_shallow);
    common_flydelta_experiment_plan shallow_plan;
    CHECK(common_flydelta_advance_experiment_plan(
        plan, utility_decision, shallow_plan, advanced, error));
    CHECK(advanced && shallow_plan.phase == common_flydelta_experiment_phase::shallow_controls &&
        shallow_plan.run_rank_two_controls_first && shallow_plan.required_compatible_directions == 2 &&
        !shallow_plan.run_tfo_lite);
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, shallow_plan.depth, shallow_plan.phase,
        {utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::escalate_deep);
    common_flydelta_experiment_plan deep_plan;
    CHECK(common_flydelta_advance_experiment_plan(
        shallow_plan, utility_decision, deep_plan, advanced, error));
    CHECK(advanced && deep_plan.phase == common_flydelta_experiment_phase::deep_controls &&
        deep_plan.run_rank_two_controls_first && !deep_plan.run_tfo_lite);
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, deep_plan.depth, deep_plan.phase,
        {utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::allow_tfo_lite);
    common_flydelta_experiment_plan tfo_plan;
    CHECK(common_flydelta_advance_experiment_plan(
        deep_plan, utility_decision, tfo_plan, advanced, error));
    CHECK(advanced && tfo_plan.phase == common_flydelta_experiment_phase::deep_controls &&
        tfo_plan.run_tfo_lite);
    utility.safe_to_continue = false;
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, common_flydelta_search_depth::deep,
        common_flydelta_experiment_phase::deep_controls, {utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::stop);

    common_flydelta_bootstrap_zoom_config zoom_config;
    std::vector<common_flydelta_bootstrap_zoom_candidate> zoom;
    CHECK(common_flydelta_propose_bootstrap_alpha_zoom(24, 0.1f, zoom_config, zoom, error));
    CHECK(zoom.size() == 3 && zoom[0].total_scale == 0.05f &&
        zoom[1].total_scale == 0.1f && zoom[2].total_scale == 0.15f);
    CHECK(common_flydelta_propose_bootstrap_profile_zoom(
        {23, 24, 25}, 24, 0.1f, zoom_config, zoom, error));
    CHECK(zoom.size() == 5);
    CHECK(zoom[0].layer_indices == std::vector<uint32_t>{24});
    CHECK(zoom[1].layer_indices == std::vector<uint32_t>({23, 24}));
    CHECK(zoom[2].layer_indices == std::vector<uint32_t>({24, 25}));
    CHECK(zoom[3].layer_indices == std::vector<uint32_t>({23, 24, 25}));
    CHECK(zoom[4].opposite_sign_control);

    common_flydelta_bootstrap_zoom_trial first_trial;
    first_trial.candidate = zoom[0];
    first_trial.host_verified = true;
    first_trial.margin_available = true;
    first_trial.margin_delta = 0.1f;
    first_trial.diagnostics_available = true;
    first_trial.diagnostics = {1, 24, 0.7f, 0.2f, 0.1f, 0.2f};
    common_flydelta_bootstrap_zoom_trial best_trial = first_trial;
    best_trial.candidate = zoom[2];
    best_trial.margin_delta = 0.2f;
    std::vector<common_flydelta_bootstrap_zoom_trial> zoom_trials = {
        first_trial, best_trial,
    };
    common_flydelta_bootstrap_zoom_selection zoom_selection;
    CHECK(common_flydelta_select_bootstrap_zoom_trial(
        zoom_trials, zoom_selection, error));
    CHECK(zoom_selection.selected && zoom_selection.trial_index == 1 &&
        zoom_selection.search_score == 0.2f);
    return 0;
}
