#include "agent-daemon-flydelta-internal.h"

#include "agent/agent-residual-patch.h"

#include <iterator>

namespace agent_daemon_flydelta_internal {

static bool daemon_flydelta_run_concept_synthesis_at_layer(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        int32_t requested_layer,
        std::vector<common_flydelta_concept_candidate> & candidates,
        common_flydelta_concept_layer_assessment & assessment,
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
    const bool challenge = !job.concept_localization.layers.empty();
    const auto & material_store = *provider->teaching_material_runtime;
    if (!(challenge ? material_store.resolve_layer_group(job.teaching_material_group_ref,
                requested_layer, group, error, common_flydelta_concept_localization_material_key(job))
            : material_store.resolve_ready_group(job.teaching_material_group_ref, group, error))) return false;
    if (!group.trajectory_material_ready || group.trajectory_refs.size() < group.minimum_trajectories) {
        error = "FlyDelta concept synthesis requires complete trajectory material";
        return false;
    }
    if (challenge && (group.compatibility_key != common_flydelta_teaching_material_compatibility_key(
                provider->teaching_material_runtime->identity()) ||
            group.identity.model_profile_fingerprint != job.seed.model_profile_fingerprint ||
            group.identity.tokenizer_fingerprint != job.seed.tokenizer_fingerprint ||
            group.identity.template_fingerprint != job.seed.template_fingerprint ||
            group.identity.execution_context_fingerprint != job.seed.execution_context_fingerprint ||
            group.identity.capture_layout_revision != provider->capture_layout_revision)) {
        error = "FlyDelta localization synthesis material has an incompatible runtime identity";
        return false;
    }
    if (group.relation_refs.empty()) {
        error = "FlyDelta concept synthesis has no source relation";
        return false;
    }
    if (challenge) {
        std::set<std::string> relation_ids;
        for (const auto & ref : group.relation_refs) {
            daemon_flydelta_concept_relation_material source;
            if (!daemon_flydelta_parse_teaching_relation(*provider, ref, source, error)) return false;
            if (!source.relation.host_approved || source.relation.teaching_key != group.teaching_key ||
                    source.relation.behavior_key != group.behavior_key ||
                    source.relation.scope.namespace_id != job.seed.scope.namespace_id ||
                    source.relation.scope.project_id != job.seed.scope.project_id ||
                    source.relation.scope.session_id != job.seed.scope.session_id ||
                    !relation_ids.insert(source.relation.id).second) {
                error = "FlyDelta localization relation is not approved in the job scope";
                return false;
            }
        }
    }
    daemon_flydelta_concept_relation_material material;
    if (!daemon_flydelta_parse_teaching_relation(
            *provider, group.relation_refs.front(), material, error)) return false;
    if (challenge && (!material.relation.host_approved ||
            material.relation.scope.namespace_id != job.seed.scope.namespace_id ||
            material.relation.scope.project_id != job.seed.scope.project_id ||
            material.relation.scope.session_id != job.seed.scope.session_id)) {
        error = "FlyDelta localization source relation is not approved in the job scope";
        return false;
    }
    daemon_flydelta_concept_target target;
    if (!daemon_flydelta_resolve_concept_target(provider, job, target, error)) return false;
    if (challenge) target.layer_index = requested_layer;
    material.layer_index = target.layer_index;
    material.parent_surface_ref = target.parent_surface_ref;
    material.parent_surface_revision = target.parent_surface_revision;
    material.parent_evidence_rank = target.parent_evidence_rank;
    const auto & relation = material.relation;
    common_flydelta_concept_spec spec;
    spec.concept_key = relation.teaching_key;
    spec.extraction_id = "flydelta://extraction/" +
        hash_sha256_hex(group.group_ref.data(), group.group_ref.size()).substr(0, 32);
    if (challenge) {
        const std::string identity = group.group_ref + "\n" + common_flydelta_concept_localization_material_key(job) +
            "\nlayer:" + std::to_string(requested_layer);
        spec.extraction_id = "flydelta://extraction/" + hash_sha256_hex(identity.data(), identity.size());
    }
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
            if (challenge) {
                const auto relation_ref = value.value("source_ref", "");
                const auto binding = std::find_if(group.layer_trajectories.begin(), group.layer_trajectories.end(),
                    [&](const auto & entry) {
                        return entry.layer_index == requested_layer && entry.trajectory_ref == trajectory_ref &&
                            entry.relation_ref == relation_ref &&
                            entry.capture_set_ref == common_flydelta_concept_localization_material_key(job);
                    });
                if (binding == group.layer_trajectories.end() ||
                        value.value("group_ref", "") != group.group_ref ||
                        value.value("capture_set_ref", "") != common_flydelta_concept_localization_material_key(job) ||
                        value.value("model_profile_fingerprint", "") != spec.model_profile_fingerprint ||
                        value.value("model_n_embd", size_t{0}) != provider->model_n_embd ||
                        value.value("model_n_layers", size_t{0}) != provider->model_n_layers ||
                        value.value("tokenizer_fingerprint", "") != group.identity.tokenizer_fingerprint ||
                        value.value("template_fingerprint", "") != group.identity.template_fingerprint ||
                        value.value("execution_context_fingerprint", "") != group.identity.execution_context_fingerprint ||
                        value.value("scope_fingerprint", "") != group.identity.scope_fingerprint ||
                        value.value("verifier_revision", "") != group.identity.verifier_revision ||
                        value.value("capture_layout_revision", "") != spec.capture_layout_revision ||
                        !value.contains("layer_index")) {
                    error = "FlyDelta localization trajectory provenance does not match its material binding";
                    return false;
                }
                daemon_flydelta_concept_relation_material bound_relation;
                if (!daemon_flydelta_parse_teaching_relation(*provider, relation_ref, bound_relation, error)) return false;
                if (value.value("source_relation_id", "") != bound_relation.relation.id ||
                        value.value("fixture_ref", "") != bound_relation.relation.verifier_ref ||
                        value.value("semantic_anchor", "") != bound_relation.semantic_anchor) {
                    error = "FlyDelta localization trajectory differs from its teaching relation";
                    return false;
                }
            }
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
    if (challenge) {
        if (!common_flydelta_assess_concept_layer(spec, build_config, trajectories,
                job.concept_localization.minimum_alignment, assessment, error)) return false;
        if (!assessment.usable) return true;
    }
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

bool daemon_flydelta_run_concept_synthesis(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_concept_candidate> & candidates,
        std::string & error, std::string * localization_assessment_ref) {
    candidates.clear();
    if (localization_assessment_ref) localization_assessment_ref->clear();
    if (!common_flydelta_concept_localization_request_validate(job.concept_localization, error)) return false;
    common_flydelta_concept_layer_assessment assessment;
    if (job.concept_localization.layers.empty()) {
        return daemon_flydelta_run_concept_synthesis_at_layer(provider, job, 0, candidates, assessment, error);
    }
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (!provider || !localization_assessment_ref) {
        error = "FlyDelta localization synthesis requires a scoped provider and an assessment sink";
        return false;
    }
    json scan = {{"kind", "flydelta_concept_localization_scan"}, {"schema_version", 1},
        {"capture_set_ref", job.concept_localization.capture_set_ref},
        {"origin_state_ref", job.concept_localization.origin_state_ref},
        {"origin_direction_ref", job.concept_localization.origin_direction_ref},
        {"material_key", common_flydelta_concept_localization_material_key(job)},
        {"material_group_ref", job.teaching_material_group_ref},
        {"model_profile_fingerprint", job.seed.model_profile_fingerprint},
        {"model_n_embd", provider->model_n_embd}, {"model_n_layers", provider->model_n_layers},
        {"capture_layout_revision", provider->capture_layout_revision},
        {"minimum_alignment", job.concept_localization.minimum_alignment},
        {"layers", json::array()}, {"experimental_only", true}};
    for (const auto layer : job.concept_localization.layers) {
        if (layer >= provider->model_n_layers) {
            error = "FlyDelta localization synthesis layer is outside the model";
            return false;
        }
        std::vector<common_flydelta_concept_candidate> layer_candidates;
        if (!daemon_flydelta_run_concept_synthesis_at_layer(provider, job,
                static_cast<int32_t>(layer), layer_candidates, assessment, error)) return false;
        common_flydelta_teaching_material_group group;
        if (!provider->teaching_material_runtime->resolve_layer_group(job.teaching_material_group_ref,
                static_cast<int32_t>(layer), group, error, common_flydelta_concept_localization_material_key(job))) return false;
        scan["layers"].push_back({{"layer_index", layer}, {"usable", assessment.usable},
            {"reason", assessment.reason}, {"median_signal_norm", assessment.median_signal_norm},
            {"median_alignment", assessment.median_alignment}, {"trajectories", group.trajectory_refs},
            {"candidate_count", layer_candidates.size()}});
        candidates.insert(candidates.end(), std::make_move_iterator(layer_candidates.begin()),
            std::make_move_iterator(layer_candidates.end()));
    }
    const auto serialized = scan.dump();
    if (!daemon_flydelta_put_json_resource(provider,
            "flydelta-localization-scan-" + hash_sha256_hex(serialized.data(), serialized.size()).substr(0, 24) + ".json",
            "flydelta_concept_localization_scan", job.concept_localization.capture_set_ref,
            serialized, *localization_assessment_ref, error)) return false;
    std::fprintf(stderr, "flydelta_concept_localization scan_ref=%s candidates=%zu layers=%zu\n",
        localization_assessment_ref->c_str(), candidates.size(), job.concept_localization.layers.size());
    return true;
}

} // namespace agent_daemon_flydelta_internal
