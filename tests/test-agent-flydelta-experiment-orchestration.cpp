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

    // A stale diagnostic safety flag must never allow a fully verified
    // HARMED arm to become a continuation.
    auto harmed = arm(27, 9.0f);
    harmed.outcome = common_flydelta_counterfactual_outcome::harmed;
    pipeline.directions.front().region_trials.push_back(harmed);
    CHECK(common_flydelta_select_search_continuation(pipeline, continuation, error));
    CHECK(continuation.region.anchor_layer_index == 25);

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
    common_flydelta_subspace_utility_observation alpha_utility = utility;
    alpha_utility.alpha_response_available = true;
    alpha_utility.alpha_response_status = common_flydelta_alpha_response_status::budget_limited;
    alpha_utility.alpha_range_not_exhausted = true;
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, common_flydelta_search_depth::bootstrap,
        common_flydelta_experiment_phase::bootstrap, {alpha_utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::refine_bootstrap);
    common_flydelta_experiment_plan zoom_plan;
    common_flydelta_evidence_depth_result zoom_depth = depth;
    zoom_depth.compatible_samples = 1;
    zoom_depth.effective_rank = 1.0f;
    zoom_depth.depth = common_flydelta_search_depth::bootstrap;
    CHECK(common_flydelta_plan_search_continuation(
        continuation, zoom_depth, zoom_plan, error));
    CHECK(zoom_plan.bootstrap_refinement ==
        common_flydelta_bootstrap_refinement_kind::bootstrap_zoom);
    common_flydelta_experiment_plan alpha_plan;
    bool alpha_advanced = false;
    CHECK(common_flydelta_advance_experiment_plan(
        zoom_plan, utility_decision, alpha_plan, alpha_advanced, error));
    CHECK(alpha_advanced && alpha_plan.bootstrap_refinement ==
        common_flydelta_bootstrap_refinement_kind::adaptive_alpha);
    alpha_utility.alpha_range_not_exhausted = false;
    alpha_utility.alpha_response_status = common_flydelta_alpha_response_status::saturated;
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, common_flydelta_search_depth::bootstrap,
        common_flydelta_experiment_phase::bootstrap, {alpha_utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::retain);
    alpha_utility.alpha_response_status = common_flydelta_alpha_response_status::helped;
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, common_flydelta_search_depth::shallow,
        common_flydelta_experiment_phase::bootstrap, {alpha_utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::escalate_shallow);
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

    // The worker-facing orchestration seam combines the existing utility and
    // phase policies. It returns a bounded next action; it does not recurse
    // into the next model slice.
    common_flydelta_slice_orchestration_result slice;
    CHECK(common_flydelta_orchestrate_search_slice(
        plan, utility_config, {utility}, {}, slice, error));
    CHECK(slice.plan_advanced &&
        slice.next_action == common_flydelta_next_action::run_shallow_controls &&
        slice.plan.phase == common_flydelta_experiment_phase::shallow_controls &&
        slice.plan.depth == common_flydelta_search_depth::deep);

    utility.safe_to_continue = false;
    CHECK(common_flydelta_orchestrate_search_slice(
        plan, utility_config, {utility}, {}, slice, error));
    CHECK(!slice.plan_advanced && slice.next_action == common_flydelta_next_action::stop);
    utility.safe_to_continue = true;

    common_flydelta_rank1_plateau_config plateau_config;
    std::vector<common_flydelta_rank1_plateau_round> plateau_rounds = {
        {true, 2, 24, 0.100f},
        {true, 2, 24, 0.105f},
        {true, 2, 25, 0.109f},
    };
    common_flydelta_rank1_plateau_result plateau;
    CHECK(common_flydelta_evaluate_rank1_plateau(
        plateau_config, 1.08f, plateau_rounds, plateau, error));
    CHECK(plateau.eligible && plateau.plateau && plateau.plateau_streak == 2 &&
        plateau.action == common_flydelta_rank1_plateau_action::orthogonal_search &&
        plateau.safe_arm_count == 6 && plateau.minimum_anchor_layer == 24 &&
        plateau.maximum_anchor_layer == 25);
    common_flydelta_utility_gate_decision plateau_decision;
    CHECK(common_flydelta_decide_rank1_plateau_utility(
        plateau, plateau_decision, error));
    CHECK(plateau_decision.utility_qualified &&
        plateau_decision.action == common_flydelta_utility_gate_action::orthogonal_search);

    common_flydelta_orthogonal_search_config orthogonal_config;
    std::vector<common_flydelta_orthogonal_search_arm> orthogonal_arms = {
        {{1.0f, 0.0f, 0.0f}, true, 0.0f, false, 0.0f, true,
            common_flydelta_counterfactual_outcome::unknown},
        {{1.0f, 1.0f, 0.0f}, true, 0.5f, false, 0.0f, true,
            common_flydelta_counterfactual_outcome::neutral},
        {{1.0f, -1.0f, 0.0f}, true, -0.5f, false, 0.0f, true,
            common_flydelta_counterfactual_outcome::unknown},
        {{1.0f, 2.0f, 0.0f}, true, 1.0f, false, 0.0f, true,
            common_flydelta_counterfactual_outcome::unknown},
        {{1.0f, -2.0f, 0.0f}, true, -1.0f, false, 0.0f, true,
            common_flydelta_counterfactual_outcome::unknown},
    };
    common_flydelta_orthogonal_search_result orthogonal;
    CHECK(common_flydelta_build_orthogonal_search_direction(
        orthogonal_config, {1.0f, 0.0f, 0.0f}, orthogonal_arms, orthogonal, error));
    CHECK(orthogonal.available && orthogonal.experimental_only &&
        orthogonal.source_arm_count == orthogonal_arms.size() && orthogonal.fit_quality > 0.99f &&
        orthogonal.response_signal == common_flydelta_orthogonal_response_signal::decision_margin &&
        std::fabs(orthogonal.direction[0]) < 0.001f && orthogonal.direction[1] > 0.99f &&
        std::fabs(orthogonal.direction[2]) < 0.001f);

    std::vector<common_flydelta_orthogonal_search_arm> geometric_arms = {
        {{1.0f, 0.0f, 0.0f}, false, 0.0f, true, 0.0f, true,
            common_flydelta_counterfactual_outcome::neutral},
        {{1.0f, 1.0f, 0.0f}, false, 0.0f, true, 0.5f, true,
            common_flydelta_counterfactual_outcome::neutral},
        {{1.0f, -1.0f, 0.0f}, false, 0.0f, true, -0.5f, true,
            common_flydelta_counterfactual_outcome::unknown},
        {{1.0f, 2.0f, 0.0f}, false, 0.0f, true, 1.0f, true,
            common_flydelta_counterfactual_outcome::neutral},
        {{1.0f, -2.0f, 0.0f}, false, 0.0f, true, -1.0f, true,
            common_flydelta_counterfactual_outcome::unknown},
    };
    common_flydelta_orthogonal_search_result geometric_orthogonal;
    CHECK(common_flydelta_build_orthogonal_search_direction(
        orthogonal_config, {1.0f, 0.0f, 0.0f}, geometric_arms, geometric_orthogonal, error));
    CHECK(geometric_orthogonal.available && geometric_orthogonal.experimental_only &&
        geometric_orthogonal.response_signal ==
            common_flydelta_orthogonal_response_signal::geometric_response &&
        geometric_orthogonal.source_arm_count == geometric_arms.size() &&
        geometric_orthogonal.fit_quality > 0.99f &&
        std::fabs(geometric_orthogonal.direction[0]) < 0.001f &&
        geometric_orthogonal.direction[1] > 0.99f &&
        std::fabs(geometric_orthogonal.direction[2]) < 0.001f);

    common_flydelta_experiment_plan bootstrap_plan;
    bool advanced = false;
    common_flydelta_evidence_depth_result bootstrap_depth = depth;
    bootstrap_depth.depth = common_flydelta_search_depth::bootstrap;
    bootstrap_depth.compatible_samples = 1;
    bootstrap_depth.effective_rank = 1.0f;
    CHECK(common_flydelta_plan_search_continuation(
        continuation, bootstrap_depth, bootstrap_plan, error));
    common_flydelta_utility_gate_decision orthogonal_decision;
    orthogonal_decision = plateau_decision;
    common_flydelta_experiment_plan orthogonal_plan;
    CHECK(common_flydelta_advance_experiment_plan(
        bootstrap_plan, orthogonal_decision, orthogonal_plan, advanced, error));
    CHECK(advanced && orthogonal_plan.run_orthogonal_search &&
        orthogonal_plan.phase == common_flydelta_experiment_phase::bootstrap &&
        orthogonal_plan.run_rank_two_controls_first && !orthogonal_plan.run_tfo_lite);

    // Capacity may be Deep, but the execution path remains ordered:
    // Bootstrap/Whirlpool -> Shallow controls -> Deep controls -> TFO-lite.
    CHECK(common_flydelta_plan_search_continuation(continuation, depth, plan, error));
    CHECK(common_flydelta_decide_subspace_utility(
        utility_config, plan.depth, plan.phase, {utility}, {}, utility_decision, error));
    CHECK(utility_decision.action == common_flydelta_utility_gate_action::escalate_shallow);
    common_flydelta_experiment_plan shallow_plan;
    CHECK(common_flydelta_advance_experiment_plan(
        plan, utility_decision, shallow_plan, advanced, error));
    CHECK(advanced && shallow_plan.phase == common_flydelta_experiment_phase::shallow_controls &&
        shallow_plan.run_rank_two_controls_first && shallow_plan.required_compatible_directions == 2 &&
        !shallow_plan.run_tfo_lite && !shallow_plan.run_orthogonal_search);
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
    first_trial.host_evaluated = true;
    first_trial.verifier_known = true;
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

    // The model host receives this reference-free preparation from persisted
    // BootstrapZoom state; it alone owns the following fresh model probes.
    std::vector<common_flydelta_bootstrap_zoom_trial> geometric_trials;
    for (size_t index = 0; index < zoom.size(); ++index) {
        common_flydelta_bootstrap_zoom_trial trial = first_trial;
        trial.candidate = zoom[index];
        trial.margin_available = false;
        trial.margin_delta = 0.0f;
        trial.outcome = common_flydelta_counterfactual_outcome::neutral;
        trial.diagnostics.progress = 0.10f * static_cast<float>(index + 1);
        trial.diagnostics.leakage = 0.05f * static_cast<float>(index + 1);
        geometric_trials.push_back(std::move(trial));
    }
    // A host-evaluated arm may remain in lineage while falling outside the
    // orthogonal safety envelope.  Keep it valid persisted search state and
    // make the eligibility decision come from its diagnostics instead of
    // conflating "not eligible" with "not evaluated".
    geometric_trials.back().diagnostics.shift_norm = 1.5f;
    common_flydelta_bootstrap_zoom_state orthogonal_input_state;
    orthogonal_input_state.behavior_key = "tool_choice/data_query";
    orthogonal_input_state.model_profile_fingerprint = "model:test";
    orthogonal_input_state.capture_layout_revision = "layer-input:v1";
    orthogonal_input_state.phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
    orthogonal_input_state.anchor_layer = 24;
    orthogonal_input_state.selected_scale = 0.1f;
    orthogonal_input_state.extra_model_trials = geometric_trials.size();
    orthogonal_input_state.next_candidate_index = geometric_trials.size();
    orthogonal_input_state.local_layers = {23, 24, 25};
    orthogonal_input_state.completed_trials = geometric_trials;
    orthogonal_input_state.selection = {true, 0, 0.0f};
    common_flydelta_orthogonal_search_input orthogonal_input;
    CHECK(common_flydelta_prepare_orthogonal_search_input(
        orthogonal_config, orthogonal_input_state, orthogonal_input, error));
    CHECK(orthogonal_input.local_layers == std::vector<uint32_t>({23, 24, 25}) &&
        orthogonal_input.rank1_intervention == std::vector<float>({0.0f, 1.0f, 0.0f}) &&
        orthogonal_input.arms.size() == geometric_trials.size());
    for (size_t index = 0; index < orthogonal_input.arms.size(); ++index) {
        const auto & prepared_arm = orthogonal_input.arms[index];
        CHECK(prepared_arm.geometric_response_available &&
            prepared_arm.safe_to_continue == (index + 1 < orthogonal_input.arms.size()) &&
            !prepared_arm.decision_margin_available);
    }
    CHECK(std::fabs(orthogonal_input.arms.front().geometric_response - 0.065f) < 1.0e-6f);

    common_flydelta_bootstrap_zoom_state surface_state;
    surface_state.behavior_key = "tool_choice/data_query";
    surface_state.model_profile_fingerprint = "model:test";
    surface_state.capture_layout_revision = "layer-input:v1";
    surface_state.phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
    surface_state.anchor_layer = 24;
    surface_state.selected_scale = 0.1f;
    surface_state.best_margin_delta = 0.2f;
    surface_state.best_search_score = 0.2f;
    surface_state.extra_model_trials = 1;
    surface_state.next_candidate_index = 1;
    surface_state.surface_revision = 2;
    surface_state.parent_surface_revision = 1;
    surface_state.search_rank = 2;
    surface_state.evidence_rank = 1.0f;
    surface_state.surface_origin = "orthogonal_search";
    surface_state.local_layers = {23, 24, 25};
    surface_state.completed_trials = {first_trial};
    surface_state.selection = {true, 0, 0.1f};
    surface_state.surface_trials = {best_trial};
    CHECK(common_flydelta_bootstrap_zoom_state_validate(surface_state, error));
    surface_state.phase = common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
    surface_state.refinement_kind = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
    surface_state.alpha_response_available = true;
    surface_state.alpha_response.selected = true;
    surface_state.alpha_response.scale = 0.2f;
    surface_state.alpha_response.utility = 0.4f;
    surface_state.alpha_response.best_margin_available = true;
    surface_state.alpha_response.best_margin_delta_normalized = 0.2f;
    surface_state.alpha_response.response_status =
        common_flydelta_alpha_response_status::budget_limited;
    surface_state.alpha_response.range_not_exhausted = true;
    CHECK(common_flydelta_bootstrap_zoom_state_validate(surface_state, error));
    surface_state.parent_surface_revision = surface_state.surface_revision;
    CHECK(!common_flydelta_bootstrap_zoom_state_validate(surface_state, error));
    return 0;
}
