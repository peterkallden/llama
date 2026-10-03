#include "agent-daemon-flydelta-internal.h"

#include "agent/agent-residual-patch.h"

#include <iterator>

namespace agent_daemon_flydelta_internal {

namespace {

bool daemon_flydelta_negative_violation_allowed(const std::string & kind) {
    return kind == "wrong_tool" ||
        kind == "missing_required_grouping" ||
        kind == "wrong_grouping_field" ||
        kind == "missing_measure" ||
        kind == "wrong_measure_column" ||
        kind == "wrong_measure_function";
}

} // namespace

bool daemon_flydelta_run_concept_capture(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        std::vector<std::string> & trajectory_refs,
        std::string & error) {
    error.clear();
    trajectory_refs.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (!provider || !provider->teaching_material_runtime ||
            job.teaching_material_group_ref.empty()) {
        error = "MODEL_ADAPTER_CAPABILITY_UNAVAILABLE: FlyDelta teaching material runtime is not bound";
        return false;
    }
    common_flydelta_teaching_material_group group;
    if (!provider->teaching_material_runtime->store().resolve_relation_set(
            job.teaching_material_group_ref, group, error)) return false;
    if (!group.relation_set_ready || group.relation_refs.size() < group.minimum_relations) {
        error = "FlyDelta concept capture requires a relation-ready teaching material group";
        return false;
    }

    daemon_flydelta_concept_target target;
    if (!daemon_flydelta_resolve_concept_target(provider, job, target, error)) return false;

    std::vector<daemon_flydelta_concept_relation_material> materials;
    materials.reserve(group.relation_refs.size());
    for (const auto & relation_ref : group.relation_refs) {
        daemon_flydelta_concept_relation_material material;
        if (!daemon_flydelta_parse_teaching_relation(*provider, relation_ref, material, error)) {
            return false;
        }
        if (material.relation.behavior_key != group.behavior_key ||
                material.relation.teaching_key != group.teaching_key ||
                material.relation.control_ref.empty()) {
            error = "FlyDelta concept relation is incompatible with its material group";
            return false;
        }
        common_flydelta_concept_capture_plan plan;
        const std::string plan_identity = job.id + "\n" + relation_ref;
        plan.id = "flydelta://concept-plan/" +
            hash_sha256_hex(plan_identity.data(), plan_identity.size()).substr(0, 32);
        plan.group_ref = job.teaching_material_group_ref;
        plan.relation_ref = relation_ref;
        plan.baseline_ref = material.relation.baseline_ref;
        plan.conditioned_ref = material.relation.conditioned_ref;
        plan.control_ref = material.relation.control_ref;
        plan.semantic_anchor = material.semantic_anchor;
        plan.layer_index = target.layer_index;
        plan.identity = group.identity;
        if (!common_flydelta_concept_capture_plan_validate(plan, error)) return false;
        material.layer_index = plan.layer_index;
        material.parent_surface_ref = target.parent_surface_ref;
        material.parent_surface_revision = target.parent_surface_revision;
        material.parent_evidence_rank = target.parent_evidence_rank;
        materials.push_back(std::move(material));
    }

    common_flydelta_arm_batch_request batch;
    batch.batch_id = job.id + ":concept-capture";
    batch.wave_id = "concept-capture";
    struct capture_binding {
        size_t material_index = 0;
        size_t baseline_arm = 0;
        size_t conditioned_arm = 0;
        size_t control_arm = 0;
    };
    std::vector<capture_binding> bindings;
    bindings.reserve(materials.size());
    const std::string intervention_ref = job.seed.candidate_ref.empty()
        ? job.seed.verifier_ref : job.seed.candidate_ref;
    for (size_t index = 0; index < materials.size(); ++index) {
        const auto & material = materials[index];
        capture_binding binding;
        binding.material_index = index;
        const std::array<std::string, 3> contexts = {
            material.relation.baseline_ref,
            material.relation.conditioned_ref,
            material.relation.control_ref,
        };
        for (size_t side = 0; side < contexts.size(); ++side) {
            common_flydelta_arm_request arm;
            arm.job_id = job.id;
            arm.wave_id = batch.wave_id;
            arm.proposal_index = batch.arms.size();
            arm.context_ref = contexts[side];
            arm.fixture_ref = material.relation.verifier_ref;
            arm.intervention_ref = intervention_ref;
            arm.layer_indices = {static_cast<uint32_t>(material.layer_index)};
            arm.coefficients = {1.0f};
            arm.alpha = 0.0f;
            arm.apply_overlay = false;
            arm.fresh_context = true;
            arm.request_capture = true;
            arm.request_generation = true;
            arm.request_host_verification = true;
            arm.max_capture_bytes = 4U * 1024U * 1024U;
            arm.max_generated_tokens = std::max<size_t>(
                64U, provider->n_predict > 0 ? provider->n_predict : 64U);
            const std::string identity = job.id + "\n" + material.relation.id + "\n" +
                std::to_string(side) + "\n" + contexts[side];
            arm.arm_id = "flydelta://concept-arm/" +
                hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
            if (!common_flydelta_arm_request_validate(arm, error)) return false;
            const size_t arm_index = batch.arms.size();
            batch.arms.push_back(std::move(arm));
            if (side == 0) binding.baseline_arm = arm_index;
            else if (side == 1) binding.conditioned_arm = arm_index;
            else binding.control_arm = arm_index;
        }
        bindings.push_back(binding);
    }

    common_flydelta_arm_batch_result batch_result;
    if (!daemon_flydelta_execute_batch(provider, batch, batch_result, error) ||
            batch_result.arms.size() != batch.arms.size()) {
        if (error.empty()) error = "FlyDelta concept capture batch returned incomplete arms";
        return false;
    }
    for (const auto & binding : bindings) {
        const auto & material = materials[binding.material_index];
        const auto & baseline_result = batch_result.arms[binding.baseline_arm];
        const auto & conditioned_result = batch_result.arms[binding.conditioned_arm];
        const auto & control_result = batch_result.arms[binding.control_arm];
        if (!baseline_result.executed || !conditioned_result.executed || !control_result.executed) {
            error = "FlyDelta concept capture contains an unexecuted arm";
            return false;
        }
        if (!conditioned_result.host_evaluated ||
                !conditioned_result.verifier_known ||
                !conditioned_result.verifier_passed) {
            error = "FlyDelta concept capture conditioned arm is not host verified";
            return false;
        }
        std::shared_ptr<const common_flydelta_hidden_state_capture> baseline;
        std::shared_ptr<const common_flydelta_hidden_state_capture> conditioned;
        std::shared_ptr<const common_flydelta_hidden_state_capture> control;
        {
            std::lock_guard<std::mutex> lock(provider->capture_mutex);
            const auto baseline_it = provider->captures.find(batch.arms[binding.baseline_arm].arm_id);
            const auto conditioned_it = provider->captures.find(batch.arms[binding.conditioned_arm].arm_id);
            const auto control_it = provider->captures.find(batch.arms[binding.control_arm].arm_id);
            if (baseline_it != provider->captures.end()) baseline = baseline_it->second;
            if (conditioned_it != provider->captures.end()) conditioned = conditioned_it->second;
            if (control_it != provider->captures.end()) control = control_it->second;
        }
        if (!baseline || !conditioned || !control) {
            error = "FlyDelta concept capture did not retain all hidden-state captures";
            return false;
        }
        std::string trajectory_ref;
        if (!daemon_flydelta_persist_concept_trajectory(
                provider, material, job.teaching_material_group_ref, job.id,
                baseline_result.capture_ref, conditioned_result.capture_ref,
                control_result.capture_ref, *baseline, *conditioned, *control,
                conditioned_result.host_evaluated &&
                    conditioned_result.verifier_known &&
                    conditioned_result.verifier_passed,
                baseline_result.host_evaluated &&
                    baseline_result.verifier_known &&
                    !baseline_result.verifier_passed &&
                    daemon_flydelta_negative_violation_allowed(
                        baseline_result.verifier_violation_kind),
                baseline_result.verifier_violation_kind,
                baseline_result.verifier_violation_dimensions,
                trajectory_ref, error)) return false;
        trajectory_refs.push_back(std::move(trajectory_ref));
    }
    return !trajectory_refs.empty();
}

} // namespace agent_daemon_flydelta_internal
