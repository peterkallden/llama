#include "agent/adaptation/flydelta/flydelta-trace-internal.h"
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

namespace {
using json = nlohmann::ordered_json;

bool finite(const float value) {
    return std::isfinite(value);
}

const char * phase_name(const common_flydelta_experiment_job_kind kind) {
    switch (kind) {
        case common_flydelta_experiment_job_kind::counterfactual: return "counterfactual";
        case common_flydelta_experiment_job_kind::direction: return "direction";
        case common_flydelta_experiment_job_kind::basis: return "basis";
        case common_flydelta_experiment_job_kind::delta_memory: return "delta_memory";
        case common_flydelta_experiment_job_kind::search_pipeline: return "search_pipeline";
        case common_flydelta_experiment_job_kind::donor_capture: return "donor_capture";
        case common_flydelta_experiment_job_kind::concept_capture: return "concept_capture";
        case common_flydelta_experiment_job_kind::concept_synthesis: return "concept_synthesis";
        case common_flydelta_experiment_job_kind::evaluation: return "evaluation";
        case common_flydelta_experiment_job_kind::causal_diagnostic: return "causal_diagnostic";
    }
    return "unknown";
}

void copy_geometry(const common_flydelta_representation_diagnostics & source,
        common_flydelta_trace_arm & target) {
    target.geometry_available = true;
    target.cosine = source.cosine;
    target.progress = source.progress;
    target.leakage = source.leakage;
    target.shift_norm = source.shift_norm;
}

void copy_margin(const common_flydelta_margin_comparison & source,
        common_flydelta_trace_arm & target) {
    if (!source.available) return;
    target.margin_available = true;
    target.margin_total = source.candidate.total_delta();
    target.margin_normalized = source.candidate.normalized_delta();
    target.margin_delta_total = source.total_delta();
    target.margin_delta_normalized = source.normalized_delta();
}

void append_region_arm(const common_flydelta_intervention_region_trial & trial,
        const size_t direction_index, const size_t trial_index,
        common_flydelta_trace & trace) {
    common_flydelta_trace_arm arm;
    arm.phase = "region";
    arm.arm_id = "region:" + std::to_string(direction_index) + ":" +
        std::to_string(trial_index);
    if (!trial.arm_id.empty()) arm.arm_id = trial.arm_id;
    arm.coefficients = trial.coefficients;
    arm.requested_coefficients = trial.requested_coefficients;
    arm.intervention_norms_available = trial.intervention_norms_available;
    arm.requested_intervention_norm = trial.requested_intervention_norm;
    arm.applied_intervention_norm = trial.applied_intervention_norm;
    arm.requested_residual_fraction = trial.requested_residual_fraction;
    arm.applied_residual_fraction = trial.applied_residual_fraction;
    arm.residual_calibration_ref = trial.residual_calibration_ref;

    arm.layer_indices = trial.candidate.layer_indices;
    arm.scale = trial.candidate.total_scale;
    arm.requested_scale = trial.dose_evaluated || trial.requested_total_scale > 0.0f
        ? trial.requested_total_scale : trial.candidate.total_scale;
    arm.executed_scale = trial.dose_evaluated || trial.executed_total_scale > 0.0f
        ? trial.executed_total_scale : trial.candidate.total_scale;
    arm.dose_evaluated = trial.dose_evaluated;
    arm.relative_dose = trial.relative_dose;
    arm.dose_comparable = trial.dose_comparable;
    arm.dose_safety_limited = trial.dose_safety_limited;
    arm.dose_action = trial.dose_evaluated
        ? common_flydelta_dose_action_name(trial.dose_action) : "not_evaluated";
    arm.dose_reason = trial.dose_reason;
    copy_margin(trial.margin_comparison, arm);
    if (!arm.margin_available && trial.margin.available) {
        arm.margin_available = true;
        arm.margin_total = trial.margin.total_delta();
        arm.margin_normalized = trial.margin.normalized_delta();
    }
    if (trial.geometry_available) copy_geometry(trial.geometry, arm);
    arm.search_score = trial.search_score;
    arm.promising = trial.promising;
    arm.safe_to_continue = trial.safe_to_continue;
    arm.execution_class = trial.host_evaluated ? "full" :
        trial.executed ? "diagnostic" : "not_executed";
    arm.host_evaluated = trial.host_evaluated;
    arm.verifier_known = trial.verifier_known;
    arm.host_outcome = trial.outcome;
    arm.observed_decision_summary = trial.observed_decision_summary;
    arm.expected_decision_summary = trial.expected_decision_summary;
    arm.verifier_reason = trial.verifier_reason;
    arm.semantic_progress_observation = trial.semantic_progress_observation;
    arm.semantic_progress = trial.semantic_progress;
    if (trial.candidate_observation) {
        const auto & candidate = *trial.candidate_observation;
        arm.candidate_evaluation_available = candidate.host_evaluated;
        arm.host_evaluated = candidate.host_evaluated;
        arm.execution_class = candidate.host_evaluated ? "full" :
            candidate.executed ? "diagnostic" : "not_executed";
        arm.verifier_known = candidate.verifier_known;
        arm.verifier_passed = candidate.verifier_known && candidate.passed;
        arm.candidate_passed = candidate.passed;
        arm.verifier_violation_code = candidate.verifier_violation_code;
        arm.verifier_violation_kind = candidate.verifier_violation_kind;
        arm.verifier_violation_dimensions = candidate.verifier_violation_dimensions;
    }
    if (trial.baseline_observation) {
        const auto & baseline = *trial.baseline_observation;
        arm.has_baseline = true;
        arm.baseline_executed = baseline.executed;
        arm.baseline_verifier_known = baseline.verifier_known;
        arm.baseline_passed = baseline.passed;
    }
    arm.evidence_ref = trial.evidence_ref;
    trace.arms.push_back(std::move(arm));
}

void append_counterfactual_arm(const common_flydelta_counterfactual_report & report,
        const common_flydelta_experiment_job & job, common_flydelta_trace & trace) {
    common_flydelta_trace_arm arm;
    arm.phase = phase_name(job.kind);
    arm.arm_id = report.candidate_id;
    arm.scale = job.alpha_search.candidates.empty() ? 0.0f :
        job.alpha_search.candidates.front();
    arm.host_evaluated = report.candidate.host_evaluated;
    arm.verifier_known = report.candidate.verifier_known;
    arm.verifier_passed = report.candidate.verifier_known && report.candidate.passed;
    arm.candidate_evaluation_available = report.candidate.host_evaluated;
    arm.candidate_passed = report.candidate.passed;
    arm.host_outcome = report.outcome;
    arm.observed_decision_summary = report.candidate.observed_decision_summary;
    arm.expected_decision_summary = report.candidate.expected_decision_summary;
    arm.verifier_reason = report.candidate.verifier_reason;
    arm.verifier_violation_code = report.candidate.verifier_violation_code;
    arm.verifier_violation_kind = report.candidate.verifier_violation_kind;
    arm.verifier_violation_dimensions = report.candidate.verifier_violation_dimensions;
    arm.semantic_progress_observation = report.candidate.semantic_progress_observation;
    arm.semantic_progress = report.semantic_progress;
    arm.evidence_ref = report.candidate.evidence_ref;
    arm.has_baseline = true;
    arm.baseline_executed = report.baseline.executed;
    arm.baseline_verifier_known = report.baseline.verifier_known;
    arm.baseline_passed = report.baseline.passed;
    trace.arms.push_back(std::move(arm));
}

void append_derived_trace(const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_worker_result & result,
        common_flydelta_trace & trace) {
    trace.job_id = job.id;
    trace.phase = phase_name(job.kind);
    trace.behavior_key = job.seed.behavior_key;
    trace.fixture_baseline_ref = job.seed.baseline_ref;
    trace.surface_parent_best_ref = job.seed.candidate_ref;
    trace.evidence_rank = result.evidence_depth.stable_rank;
    trace.search_rank = result.basis_directions.size();
    trace.region_budget = result.search_budget.max_region_trials;
    trace.coefficient_budget = result.search_budget.max_coefficient_trials;
    trace.tfo_lite_allowed = result.search_budget.allow_tfo_lite;
    if (!result.search_pipeline_results.empty()) {
        trace.search_status = common_flydelta_search_status_name(
            result.search_pipeline_results.back().search_status);
        trace.search_model_call_limit = result.search_pipeline_results.back().search_model_call_limit;
        trace.search_model_calls_used = result.search_pipeline_results.back().search_model_calls_used;
        trace.physical_rows_started = result.search_pipeline_results.back().physical_rows_started;
        trace.physical_rows_completed = result.search_pipeline_results.back().physical_rows_completed;
        trace.physical_rows_unresolved = result.search_pipeline_results.back().physical_rows_unresolved;
    }
    if (result.has_experiment_plan) {
        trace.search_anchor_ref = result.experiment_plan.search_anchor_ref;
        trace.search_anchor_round = result.experiment_plan.search_anchor_round;
        trace.bootstrap_refinement = common_flydelta_bootstrap_refinement_kind_name(
            result.experiment_plan.bootstrap_refinement);
    }
    if (result.has_bootstrap_zoom_state &&
            result.bootstrap_zoom_state.alpha_response_available) {
        trace.alpha_response_available = true;
        trace.alpha_response_status =
            result.bootstrap_zoom_state.alpha_response.response_status;
        trace.alpha_range_not_exhausted =
            result.bootstrap_zoom_state.alpha_response.range_not_exhausted;
        trace.alpha_last_scale = result.bootstrap_zoom_state.alpha_response.last_scale;
        trace.alpha_last_requested_scale = result.bootstrap_zoom_state.alpha_response.last_requested_scale;
        trace.alpha_last_executed_scale = result.bootstrap_zoom_state.alpha_response.last_executed_scale;
        trace.alpha_last_relative_dose = result.bootstrap_zoom_state.alpha_response.last_relative_dose;
        trace.alpha_last_dose_action = common_flydelta_dose_action_name(
            result.bootstrap_zoom_state.alpha_response.last_dose_action);
        trace.alpha_last_dose_evaluated = result.bootstrap_zoom_state.alpha_response.last_dose_evaluated;
        trace.alpha_last_dose_safety_limited = result.bootstrap_zoom_state.alpha_response.last_dose_safety_limited;
        trace.alpha_utility_slope = result.bootstrap_zoom_state.alpha_response.utility_slope;
        trace.alpha_best_margin_delta_normalized =
            result.bootstrap_zoom_state.alpha_response.best_margin_delta_normalized;
    }
    trace.has_next_action = result.has_next_action;
    trace.next_action = result.next_action;
    trace.next_action_reason = result.next_action_reason;
    trace.graft_direction_ref = result.graft_direction_ref;
    trace.graft_direction_refs = result.graft_direction_refs;
    trace.has_selected_candidate = result.has_selected_candidate;
    trace.selected_candidate = result.selected_candidate;
    trace.has_paired_intervention = result.has_paired_intervention;
    trace.paired_intervention = result.paired_intervention;

    if (result.counterfactual_reports.size() <= 256) {
        for (const auto & report : result.counterfactual_reports) {
            append_counterfactual_arm(report, job, trace);
        }
    }
    if (result.has_causal_diagnostic_report) {
        for (const auto & diagnostic : result.causal_diagnostic_report.arms) {
            if (trace.arms.size() >= 256) break;
            common_flydelta_trace_arm arm;
            arm.phase = "causal_diagnostic";
            arm.arm_id = std::string(common_flydelta_causal_patch_kind_name(
                diagnostic.patch_kind)) + ":" + diagnostic.arm_id;
            arm.layer_indices = {diagnostic.layer};
            arm.margin_available = diagnostic.margin_available;
            arm.margin_total = diagnostic.candidate_margin;
            arm.margin_delta_total = diagnostic.margin_delta;
            arm.host_evaluated = diagnostic.host_evaluated;
            arm.verifier_known = diagnostic.verifier_known;
            arm.host_outcome = diagnostic.host_outcome;
            arm.execution_class = diagnostic.host_evaluated ? "full" :
                diagnostic.executed ? "diagnostic" : "not_executed";
            arm.evidence_ref = diagnostic.evidence_ref;
            trace.arms.push_back(std::move(arm));
        }
    }
    size_t direction_index = 0;
    for (const auto & pipeline : result.search_pipeline_results) {
        if (pipeline.alpha_response_available) {
            trace.alpha_response_available = true;
            trace.alpha_response_status = pipeline.alpha_response.response_status;
            trace.alpha_range_not_exhausted = pipeline.alpha_response.range_not_exhausted;
            trace.alpha_last_scale = pipeline.alpha_response.last_scale;
            trace.alpha_last_requested_scale = pipeline.alpha_response.last_requested_scale;
            trace.alpha_last_executed_scale = pipeline.alpha_response.last_executed_scale;
            trace.alpha_last_relative_dose = pipeline.alpha_response.last_relative_dose;
            trace.alpha_last_dose_action = common_flydelta_dose_action_name(
                pipeline.alpha_response.last_dose_action);
            trace.alpha_last_dose_evaluated = pipeline.alpha_response.last_dose_evaluated;
            trace.alpha_last_dose_safety_limited = pipeline.alpha_response.last_dose_safety_limited;
            trace.alpha_utility_slope = pipeline.alpha_response.utility_slope;
            trace.alpha_best_margin_delta_normalized =
                pipeline.alpha_response.best_margin_delta_normalized;
        }
        for (const auto & direction : pipeline.directions) {
            trace.whirlpool.push_back(direction.whirlpool_trace);
            trace.model_evaluations += direction.whirlpool_trace.model_evaluations;
            for (size_t trial_index = 0; trial_index < direction.region_trials.size();
                    ++trial_index) {
                if (trace.arms.size() >= 256) break;
                append_region_arm(direction.region_trials[trial_index], direction_index,
                    trial_index, trace);
            }
            ++direction_index;
        }
    }
    if (trace.model_evaluations == 0) trace.model_evaluations = trace.arms.size();
}

json trace_arm_json(const common_flydelta_trace_arm & arm) {
    return {
        {"phase", arm.phase}, {"arm_id", arm.arm_id},
        {"layer_indices", arm.layer_indices}, {"scale", arm.scale},
        {"requested_scale", arm.requested_scale},
        {"executed_scale", arm.executed_scale},
        {"dose_evaluated", arm.dose_evaluated},
        {"relative_dose", arm.relative_dose},
        {"dose_comparable", arm.dose_comparable},
        {"dose_safety_limited", arm.dose_safety_limited},
        {"dose_action", arm.dose_action},
        {"dose_reason", arm.dose_reason},
        {"coefficients", arm.coefficients},
        {"requested_coefficients", arm.requested_coefficients},
        {"intervention_norms_available", arm.intervention_norms_available},
        {"requested_intervention_norm", arm.requested_intervention_norm},
        {"applied_intervention_norm", arm.applied_intervention_norm},
        {"requested_residual_fraction", arm.requested_residual_fraction},
        {"applied_residual_fraction", arm.applied_residual_fraction},
        {"residual_calibration_ref", arm.residual_calibration_ref},

        {"margin_available", arm.margin_available},
        {"margin_total", arm.margin_total},
        {"margin_normalized", arm.margin_normalized},
        {"margin_delta_total", arm.margin_delta_total},
        {"margin_delta_normalized", arm.margin_delta_normalized},
        {"geometry_available", arm.geometry_available},
        {"cosine", arm.cosine}, {"progress", arm.progress},
        {"leakage", arm.leakage}, {"shift_norm", arm.shift_norm},
        {"search_score", arm.search_score}, {"promising", arm.promising},
        {"safe_to_continue", arm.safe_to_continue},
        {"execution_class", arm.execution_class},
        {"host_evaluated", arm.host_evaluated},
        {"verifier_known", arm.verifier_known},
        {"candidate_evaluation_available", arm.candidate_evaluation_available},
        {"verifier_passed", arm.verifier_passed},
        {"candidate_passed", arm.candidate_passed},
        {"has_baseline", arm.has_baseline},
        {"baseline_executed", arm.baseline_executed},
        {"baseline_verifier_known", arm.baseline_verifier_known},
        {"baseline_passed", arm.baseline_passed},
        {"host_outcome", common_flydelta_counterfactual_outcome_name(arm.host_outcome)},
        {"outcome_source", "flydelta_intervention_counterfactual"},
        {"comparison_available", arm.has_baseline && arm.candidate_evaluation_available && arm.baseline_executed &&
            arm.baseline_verifier_known && arm.host_evaluated && arm.verifier_known},
        {"observed_decision", arm.observed_decision_summary},
        {"expected_decision", arm.expected_decision_summary},
        {"verifier_reason", arm.verifier_reason},
        {"verifier_violation_code", arm.verifier_violation_code},
        {"verifier_violation_kind", arm.verifier_violation_kind},
        {"verifier_violation_dimensions", arm.verifier_violation_dimensions},
        {"objective_assessment", json::parse(common_flydelta_progress_assessment_to_json(
            arm.semantic_progress_observation.objective))},
        {"semantic_progress", {
            {"outcome", common_flydelta_semantic_progress_outcome_name(arm.semantic_progress.outcome)},
            {"comparable", arm.semantic_progress.comparable},
            {"regression_known", arm.semantic_progress.regression_known},
            {"baseline_score", arm.semantic_progress.baseline_score},
            {"candidate_score", arm.semantic_progress.candidate_score},
            {"improved_dimensions", arm.semantic_progress.improved_dimensions},
            {"regressed_dimensions", arm.semantic_progress.regressed_dimensions},
            {"residual_dimensions", arm.semantic_progress.residual_dimensions},
        }},
        {"evidence_ref", arm.evidence_ref}
    };
}


} // namespace

bool common_flydelta_trace_validate(
        const common_flydelta_trace & trace, std::string & error) {
    error.clear();
    if (trace.schema_version != 1 || trace.job_id.size() > 512 ||
            trace.phase.size() > 64 || trace.behavior_key.size() > 256 ||
            trace.fixture_baseline_ref.size() > 512 || trace.search_anchor_ref.size() > 512 ||
            trace.search_model_calls_used > trace.search_model_call_limit ||
            trace.physical_rows_completed > trace.physical_rows_started ||
            trace.physical_rows_unresolved != trace.physical_rows_started - trace.physical_rows_completed ||
            trace.surface_parent_best_ref.size() > 512 ||
            trace.next_action_reason.size() > 512 || trace.graft_direction_ref.size() > 512 ||
            trace.graft_direction_refs.size() > 8 ||
            trace.arms.size() > 256 ||
            trace.whirlpool.size() > 32 || trace.bootstrap_refinement.size() > 64 ||
            !finite(trace.alpha_last_scale) || !finite(trace.alpha_utility_slope) ||
            !finite(trace.alpha_best_margin_delta_normalized)) {
        error = "FlyDelta trace exceeds its bounds";
        return false;
    }
    if (trace.has_paired_intervention &&
            !common_flydelta_paired_intervention_proposal_validate(
                trace.paired_intervention, error)) return false;
    if (trace.has_selected_candidate &&
            !common_flydelta_selected_candidate_descriptor_validate(
                trace.selected_candidate, error)) return false;
    for (const auto & direction_ref : trace.graft_direction_refs) {
        if (direction_ref.empty() || direction_ref.size() > 512) {
            error = "FlyDelta trace contains an invalid graft direction portfolio reference";
            return false;
        }
    }
    if (!trace.graft_direction_refs.empty() &&
            trace.graft_direction_refs.front() != trace.graft_direction_ref) {
        error = "FlyDelta trace primary graft reference does not match its portfolio";
        return false;
    }
    for (const auto & arm : trace.arms) {
        if (!common_flydelta_progress_assessment_validate(arm.semantic_progress_observation.objective, error)) return false;
        if (arm.phase.size() > 64 || arm.arm_id.size() > 512 ||
                arm.evidence_ref.size() > 512 || arm.layer_indices.size() > 64 ||
                arm.coefficients.size() > 64 || !finite(arm.scale) ||
                !finite(arm.margin_total) || !finite(arm.margin_normalized) ||
                !finite(arm.margin_delta_total) || !finite(arm.margin_delta_normalized) ||
                !finite(arm.cosine) || !finite(arm.progress) || !finite(arm.leakage) ||
                !finite(arm.shift_norm) || !finite(arm.search_score)) {
            error = "FlyDelta trace arm is invalid";
            return false;
        }
    }
    return true;
}

std::string common_flydelta_trace_to_json(const common_flydelta_trace & trace) {
    json arms = json::array();
    for (const auto & arm : trace.arms) arms.push_back(trace_arm_json(arm));
    json whirlpool = json::array();
    for (const auto & value : trace.whirlpool) {
        json rounds = json::array();
        for (const auto & round : value.rounds) {
            rounds.push_back({
                {"round", round.round}, {"centre_before", round.centre_before},
                {"radius_before", round.radius_before},
                {"probed_layers", round.probed_layers},
                {"global_probed_layers", round.global_probed_layers},
                {"guidance_decision_id", round.guidance_decision_id},
                {"guidance_original_layers", round.guidance_original_layers},
                {"guidance_posterior_layers", round.guidance_posterior_layers},
                {"guidance_hypothetical_layers", round.guidance_hypothetical_layers},
                {"guidance_selected_layers", round.guidance_selected_layers},
                {"guidance_history_fingerprint", round.guidance_history_fingerprint},
                {"guidance_fallback_reason", round.guidance_fallback_reason},
                {"best_probe_layer", round.best_probe_layer},
                {"best_probe_score", round.best_probe_score},
                {"centre_after", round.centre_after},
                {"radius_after", round.radius_after}
            });
        }
        whirlpool.push_back({
            {"schema_version", value.schema_version},
            {"model_evaluations", value.model_evaluations},
            {"proposal_policy", value.proposal_policy},
            {"best_trial_index", value.best_trial_index},
            {"best_search_score", value.best_search_score},
            {"final_centre", value.final_centre},
            {"final_radius", value.final_radius},
            {"initial_centre_source", value.initial_centre_source},
            {"initial_centre_guidance_decision_id", value.initial_centre_guidance_decision_id},
            {"initial_centre_guidance_original_layers", value.initial_centre_guidance_original_layers},
            {"initial_centre_guidance_posterior_layers", value.initial_centre_guidance_posterior_layers},
            {"initial_centre_guidance_hypothetical_layers", value.initial_centre_guidance_hypothetical_layers},
            {"initial_centre_guidance_selected_layers", value.initial_centre_guidance_selected_layers},
            {"initial_centre_guidance_history_fingerprint", value.initial_centre_guidance_history_fingerprint},
            {"initial_centre_guidance_fallback_reason", value.initial_centre_guidance_fallback_reason},
            {"rounds", std::move(rounds)}
        });
    }
    const json payload = {
        {"kind", "flydelta_trace"}, {"schema_version", trace.schema_version},
        {"job_id", trace.job_id}, {"phase", trace.phase},
        {"behavior_key", trace.behavior_key},
        {"fixture_baseline_ref", trace.fixture_baseline_ref},
        {"surface_parent_best_ref", trace.surface_parent_best_ref},
        {"evidence_rank", trace.evidence_rank}, {"search_rank", trace.search_rank},
        {"model_evaluations", trace.model_evaluations},
        {"search_model_call_limit", trace.search_model_call_limit},
        {"search_model_calls_used", trace.search_model_calls_used},
        {"physical_rows_started", trace.physical_rows_started},
        {"physical_rows_completed", trace.physical_rows_completed},
        {"physical_rows_unresolved", trace.physical_rows_unresolved},
        {"search_anchor_ref", trace.search_anchor_ref},
        {"search_anchor_round", trace.search_anchor_round},
        {"region_budget", trace.region_budget},
        {"coefficient_budget", trace.coefficient_budget},
        {"tfo_lite_allowed", trace.tfo_lite_allowed},
        {"search_status", trace.search_status},
        {"bootstrap_refinement", trace.bootstrap_refinement},
        {"alpha_response_available", trace.alpha_response_available},
        {"alpha_response_status", common_flydelta_alpha_response_status_name(
            trace.alpha_response_status)},
        {"alpha_range_not_exhausted", trace.alpha_range_not_exhausted},
        {"alpha_last_scale", trace.alpha_last_scale},
        {"alpha_last_requested_scale", trace.alpha_last_requested_scale},
        {"alpha_last_executed_scale", trace.alpha_last_executed_scale},
        {"alpha_last_relative_dose", trace.alpha_last_relative_dose},
        {"alpha_last_dose_action", trace.alpha_last_dose_action},
        {"alpha_last_dose_evaluated", trace.alpha_last_dose_evaluated},
        {"alpha_last_dose_safety_limited", trace.alpha_last_dose_safety_limited},
        {"alpha_utility_slope", trace.alpha_utility_slope},
        {"alpha_best_margin_delta_normalized",
            trace.alpha_best_margin_delta_normalized},
        {"has_next_action", trace.has_next_action},
        {"next_action", common_flydelta_next_action_name(trace.next_action)},
        {"next_action_reason", trace.next_action_reason},
        {"graft_direction_ref", trace.graft_direction_ref},
        {"graft_direction_refs", trace.graft_direction_refs},
        {"selected_candidate", trace.has_selected_candidate ? json({
            {"pipeline_index", trace.selected_candidate.pipeline_index},
            {"direction_index", trace.selected_candidate.direction_index},
            {"intervention_region", trace.selected_candidate.intervention_region},
            {"layer_result_index", trace.selected_candidate.layer_result_index},
            {"region_trial_index", trace.selected_candidate.region_trial_index},
            {"direction_kind", common_flydelta_direction_kind_name(
                trace.selected_candidate.direction_kind)},
            {"strategy", common_flydelta_synthesis_strategy_name(
                trace.selected_candidate.strategy)},
            {"estimator", common_flydelta_synthesis_estimator_name(
                trace.selected_candidate.estimator)},
            {"strategy_revision", trace.selected_candidate.strategy_revision},
            {"layer_index", trace.selected_candidate.layer_index},
            {"direction_ref", trace.selected_candidate.direction_ref},
            {"source_material_ref", trace.selected_candidate.source_material_ref},
            {"selection_score", trace.selected_candidate.selection_score},
            {"outcome", common_flydelta_counterfactual_outcome_name(
                trace.selected_candidate.outcome)},
            {"host_evaluated", trace.selected_candidate.host_evaluated},
            {"verifier_known", trace.selected_candidate.verifier_known},
            {"selection_evidence", trace.selected_candidate.selection_evidence}
        }) : json(nullptr)},
        {"paired_intervention", trace.has_paired_intervention ? json({
            {"proposal_id", trace.paired_intervention.proposal_id},
            {"prefer_role", common_flydelta_intervention_component_role_name(
                trace.paired_intervention.prefer_role)},
            {"avoid_role", common_flydelta_intervention_component_role_name(
                trace.paired_intervention.avoid_role)},
            {"prefer_candidate_ref", trace.paired_intervention.prefer_candidate_ref},
            {"avoid_candidate_ref", trace.paired_intervention.avoid_candidate_ref},
            {"prefer_direction_kind", common_flydelta_direction_kind_name(
                trace.paired_intervention.prefer_direction_kind)},
            {"avoid_direction_kind", common_flydelta_direction_kind_name(
                trace.paired_intervention.avoid_direction_kind)},
            {"behavior_key", trace.paired_intervention.behavior_key},
            {"fixture_ref", trace.paired_intervention.fixture_ref},
            {"oracle_ref", trace.paired_intervention.oracle_ref},
            {"layer_index", trace.paired_intervention.layer_index}
        }) : json(nullptr)},
        {"arms", std::move(arms)}, {"whirlpool", std::move(whirlpool)}
    };
    return payload.dump();
}


namespace common_flydelta_trace_detail {
void append_worker_result(const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_worker_result & result, common_flydelta_trace & trace) {
    append_derived_trace(job, result, trace);
}
} // namespace common_flydelta_trace_detail
