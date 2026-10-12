#include "agent-daemon-flydelta-internal.h"

#include "agent/agent-residual-patch.h"

#include <iterator>

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_run_concept_synthesis(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_concept_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (!provider || !provider->teaching_material_runtime ||
            job.teaching_material_group_ref.empty()) {
        error = "MODEL_ADAPTER_CAPABILITY_UNAVAILABLE: FlyDelta teaching material runtime is not bound";
        return false;
    }
    common_flydelta_teaching_material_group group;
    if (!provider->teaching_material_runtime->store().resolve_ready_group(
            job.teaching_material_group_ref, group, error)) return false;
    if (!group.trajectory_material_ready || group.trajectory_refs.size() < group.minimum_trajectories) {
        error = "FlyDelta concept synthesis requires complete trajectory material";
        return false;
    }
    if (group.relation_refs.empty()) {
        error = "FlyDelta concept synthesis has no source relation";
        return false;
    }
    daemon_flydelta_concept_relation_material material;
    if (!daemon_flydelta_parse_teaching_relation(
            *provider, group.relation_refs.front(), material, error)) return false;
    daemon_flydelta_concept_target target;
    if (!daemon_flydelta_resolve_concept_target(provider, job, target, error)) return false;
    material.layer_index = target.layer_index;
    material.parent_surface_ref = target.parent_surface_ref;
    material.parent_surface_revision = target.parent_surface_revision;
    material.parent_evidence_rank = target.parent_evidence_rank;
    const auto & relation = material.relation;
    common_flydelta_concept_spec spec;
    spec.concept_key = relation.teaching_key;
    spec.extraction_id = "flydelta://extraction/" +
        hash_sha256_hex(group.group_ref.data(), group.group_ref.size()).substr(0, 32);
    spec.behavior_key = relation.behavior_key;
    spec.source = relation.source;
    spec.source_ref = relation.id;
    spec.grounding_ref = relation.contrast_ref.empty()
        ? (relation.procedure_ref.empty()
            ? (relation.blueprint_ref.empty() ? relation.evidence_ref : relation.blueprint_ref)
            : relation.procedure_ref)
        : relation.contrast_ref;
    spec.procedure_ref = relation.procedure_ref;
    spec.verifier_ref = relation.verifier_ref;
    spec.model_profile_fingerprint = group.identity.model_profile_fingerprint;
    spec.tokenizer_fingerprint = group.identity.tokenizer_fingerprint;
    spec.template_fingerprint = group.identity.template_fingerprint;
    spec.capture_layout_revision = group.identity.capture_layout_revision;
    spec.scope_fingerprint = group.identity.scope_fingerprint;
    spec.host_approved = relation.host_approved;
    spec.redaction_attested = true;
    spec.require_control = true;
    if (!common_flydelta_concept_spec_validate(spec, error)) return false;

    std::vector<common_flydelta_concept_trajectory> trajectories;
    trajectories.reserve(group.trajectory_refs.size());
    for (const auto & trajectory_ref : group.trajectory_refs) {
        json value;
        if (!daemon_flydelta_read_json(*provider, trajectory_ref, value, error)) return false;
        try {
            common_flydelta_concept_trajectory trajectory;
            trajectory.schema_version = value.value("schema_version", 0);
            trajectory.id = value.value("id", trajectory_ref);
            trajectory.fixture_ref = value.value("fixture_ref", relation.verifier_ref);
            trajectory.baseline_capture_ref = value.value("baseline_capture_ref", "");
            trajectory.conditioned_capture_ref = value.value("conditioned_capture_ref", "");
            trajectory.control_capture_ref = value.value("control_capture_ref", "");
            trajectory.negative_capture_ref = value.value("negative_capture_ref", "");
            trajectory.semantic_anchor = value.value("semantic_anchor", material.semantic_anchor);
            trajectory.layer_index = value.value("layer_index", material.layer_index);
            trajectory.baseline = value.value("baseline", std::vector<float>{});
            trajectory.conditioned = value.value("conditioned", std::vector<float>{});
            trajectory.control = value.value("control", std::vector<float>{});
            trajectory.negative = value.value("negative", std::vector<float>{});
            trajectory.aligned = value.value("aligned", false);
            trajectory.conditioned_host_verified = value.value("conditioned_host_verified", false);
            trajectory.negative_host_verified = value.value("negative_host_verified", false);
            if (trajectory.layer_index != target.layer_index) {
                error = "FlyDelta concept trajectory layer does not match localized graft anchor";
                return false;
            }
            if (!common_flydelta_concept_trajectory_validate(
                    spec, trajectory, provider->model_n_embd, error)) return false;
            trajectories.push_back(std::move(trajectory));
        } catch (const std::exception & exception) {
            error = std::string("FlyDelta concept trajectory resource is malformed: ") + exception.what();
            return false;
        }
    }
    common_flydelta_concept_build_config build_config;
    build_config.dimension = provider->model_n_embd;
    build_config.min_trajectories = group.minimum_trajectories;
    build_config.max_trajectories = std::min<size_t>(64, trajectories.size());
    if (!common_flydelta_build_concept_candidates(
            spec, build_config, trajectories, candidates, error)) return false;

    // Reuse the host-verified conditioned and neutral/control captures for
    // positive-prototype synthesis. These candidates continue through the
    // existing evaluator, search and Oracle path.
    std::vector<common_flydelta_concept_prototype_sample> positive_samples;
    std::vector<common_flydelta_concept_prototype_sample> control_samples;
    std::vector<common_flydelta_concept_prototype_sample> negative_samples;
    positive_samples.reserve(trajectories.size());
    control_samples.reserve(trajectories.size());
    negative_samples.reserve(trajectories.size());
    for (const auto & trajectory : trajectories) {
        const auto make_sample = [&](const std::string & suffix,
                const std::string & capture_ref, const std::vector<float> & values) {
            common_flydelta_concept_prototype_sample sample;
            sample.id = trajectory.id + "/" + suffix;
            sample.capture_ref = capture_ref;
            sample.semantic_anchor = trajectory.semantic_anchor;
            sample.model_profile_fingerprint = spec.model_profile_fingerprint;
            sample.tokenizer_fingerprint = spec.tokenizer_fingerprint;
            sample.template_fingerprint = spec.template_fingerprint;
            sample.capture_layout_revision = spec.capture_layout_revision;
            sample.scope_fingerprint = spec.scope_fingerprint;
            sample.verifier_ref = spec.verifier_ref;
            sample.layer_index = trajectory.layer_index;
            sample.values = values;
            sample.host_verified = trajectory.conditioned_host_verified;
            sample.independent = true;
            return sample;
        };
        positive_samples.push_back(make_sample(
            "positive", trajectory.conditioned_capture_ref, trajectory.conditioned));
        control_samples.push_back(make_sample(
            "neutral-control", trajectory.control_capture_ref, trajectory.control));
        if (!trajectory.negative_capture_ref.empty()) {
            auto negative_sample = make_sample(
                "negative", trajectory.negative_capture_ref, trajectory.negative);
            negative_sample.host_verified = trajectory.negative_host_verified;
            negative_samples.push_back(std::move(negative_sample));
        }
    }
    common_flydelta_concept_prototype_build_config prototype_config;
    prototype_config.dimension = provider->model_n_embd;
    prototype_config.min_positive_samples = group.minimum_trajectories;
    prototype_config.min_control_samples = group.minimum_trajectories;
    prototype_config.max_samples = std::min<size_t>(64, trajectories.size());
    std::vector<common_flydelta_concept_candidate> prototype_candidates;
    if (!common_flydelta_build_positive_prototype_candidates(
            spec, prototype_config, positive_samples, control_samples,
            prototype_candidates, error)) return false;
    candidates.insert(candidates.end(),
        std::make_move_iterator(prototype_candidates.begin()),
        std::make_move_iterator(prototype_candidates.end()));
    if (!negative_samples.empty() && negative_samples.size() != trajectories.size()) {
        error = "FlyDelta negative concept material must be present for every trajectory";
        return false;
    }
    if (!negative_samples.empty()) {
        std::vector<common_flydelta_concept_candidate> negative_candidates;
        if (!common_flydelta_build_negative_repulsion_candidates(
                spec, prototype_config, negative_samples, control_samples,
                negative_candidates, error)) return false;
        candidates.insert(candidates.end(),
            std::make_move_iterator(negative_candidates.begin()),
            std::make_move_iterator(negative_candidates.end()));
    }
    return true;
}

} // namespace agent_daemon_flydelta_internal
