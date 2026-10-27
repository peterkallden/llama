#include "../agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {

void daemon_flydelta_record_arm_observations(
        common_flydelta_intervention_region_trial & trial,
        const common_flydelta_arm_result & candidate,
        const common_flydelta_arm_result * baseline) {
    const auto observation = [](const common_flydelta_arm_result & arm) {
        common_flydelta_counterfactual_trial value;
        value.arm_id = arm.arm_id;
        value.intervention_ref = arm.intervention_ref;
        value.executed = arm.executed;
        value.overlay_applied = arm.overlay_applied;
        value.host_evaluated = arm.host_evaluated;
        value.verifier_known = arm.verifier_known;
        value.passed = daemon_flydelta_arm_semantic_passed(arm);
        value.quality = arm.quality;
        value.observed_decision_summary = arm.observed_decision_summary;
        value.expected_decision_summary = arm.expected_decision_summary;
        value.verifier_reason = arm.verifier_reason;
        value.verifier_violation_code = arm.verifier_violation_code;
        value.verifier_violation_kind = arm.verifier_violation_kind;
        value.verifier_violation_dimensions = arm.verifier_violation_dimensions;
        value.semantic_progress_observation = arm.semantic_progress_observation;
        value.evidence_ref = arm.generation_ref;
        return value;
    };
    trial.candidate_observation = observation(candidate);
    if (baseline) trial.baseline_observation = observation(*baseline);
}

void daemon_flydelta_bind_execution_callbacks(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::shared_ptr<common_flydelta_teaching_material_runtime> & teaching_material_runtime,
        common_agent_server_flydelta_binding & binding) {
    binding.teaching_material_runtime = teaching_material_runtime;
    binding.primitives.capture = true;
    binding.primitives.overlay = true;
    binding.primitives.generation = true;
    binding.primitives.teacher_forced_scoring = true;
    binding.primitives.host_verification = true;
    binding.run_concept_capture = [provider](
            const common_flydelta_experiment_job & job,
            std::vector<std::string> & trajectory_refs,
            std::string & callback_error) {
        return daemon_flydelta_run_concept_capture(
            provider, job, trajectory_refs, callback_error);
    };
    binding.prepare_arm = [provider](const auto & arm, auto & request, std::string & callback_error) {
        return daemon_flydelta_prepare_arm(provider, arm, request, callback_error);
    };
    binding.finalize_arm = [provider](const auto & arm, const auto & generation,
            auto & result, std::string & callback_error) {
        return daemon_flydelta_finalize_arm(provider, arm, generation, result, callback_error);
    };
    binding.score_teacher_forced_margin_batch = [provider](
            const auto & arms, const auto & contexts, auto & scorer, auto & margins,
            std::string & callback_error) {
        return daemon_flydelta_score_teacher_forced_margin_batch(
            provider, arms, contexts, scorer, margins, callback_error);
    };
    binding.run_decision_margin_challenger = [provider](
            const auto & job,
            auto & candidates,
            std::string & callback_error) {
        return daemon_flydelta_run_decision_margin_challenger(
            provider, job, candidates, callback_error);
    };
    // Reuse the existing counterfactual contract and batch host. This is
    // the host-owned comparison seam: individual arm finalization remains
    // NEUTRAL/HARMED, while the paired callback derives HELPED from the
    // baseline/candidate relation.
    binding.run_counterfactual = [provider](
            const common_flydelta_experiment_job & job,
            std::vector<common_flydelta_counterfactual_report> & reports,
            std::string & callback_error) {
        return daemon_flydelta_run_counterfactual(
            provider, job, reports, callback_error);
    };
}

} // namespace agent_daemon_flydelta_internal
