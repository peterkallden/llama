#include "agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {

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
            arm.request_host_verification = false;
            arm.max_capture_bytes = 4U * 1024U * 1024U;
            arm.max_generated_tokens = 1;
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
                trajectory_ref, error)) return false;
        trajectory_refs.push_back(std::move(trajectory_ref));
    }
    return !trajectory_refs.empty();
}

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
            trajectory.semantic_anchor = value.value("semantic_anchor", material.semantic_anchor);
            trajectory.layer_index = value.value("layer_index", material.layer_index);
            trajectory.baseline = value.value("baseline", std::vector<float>{});
            trajectory.conditioned = value.value("conditioned", std::vector<float>{});
            trajectory.control = value.value("control", std::vector<float>{});
            trajectory.aligned = value.value("aligned", false);
            trajectory.conditioned_host_verified = value.value("conditioned_host_verified", false);
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
    return common_flydelta_build_concept_candidates(
        spec, build_config, trajectories, candidates, error);
}

bool daemon_flydelta_fixture_from_job(
        const common_flydelta_experiment_job & job,
        common_flydelta_experiment_fixture & fixture,
        std::string & error) {
    fixture = {};
    fixture.id = job.seed.verifier_ref;
    fixture.task_fingerprint = job.seed.task_fingerprint;
    fixture.model_profile_fingerprint = job.seed.model_profile_fingerprint;
    fixture.tokenizer_fingerprint = job.seed.tokenizer_fingerprint;
    fixture.template_fingerprint = job.seed.template_fingerprint;
    fixture.execution_context_fingerprint = job.seed.execution_context_fingerprint;
    fixture.verifier_revision = job.seed.verifier_ref;
    return common_flydelta_experiment_fixture_validate(fixture, error);
}

bool daemon_flydelta_parse_depth(
        const std::string & value, common_flydelta_search_depth & depth) {
    if (value == "bootstrap") depth = common_flydelta_search_depth::bootstrap;
    else if (value == "shallow") depth = common_flydelta_search_depth::shallow;
    else if (value == "deep") depth = common_flydelta_search_depth::deep;
    else return false;
    return true;
}

bool daemon_flydelta_parse_phase(
        const std::string & value, common_flydelta_experiment_phase & phase) {
    if (value == "bootstrap") phase = common_flydelta_experiment_phase::bootstrap;
    else if (value == "shallow_controls") phase = common_flydelta_experiment_phase::shallow_controls;
    else if (value == "deep_controls") phase = common_flydelta_experiment_phase::deep_controls;
    else return false;
    return true;
}

bool daemon_flydelta_parse_refinement(
        const std::string & value, common_flydelta_bootstrap_refinement_kind & kind) {
    if (value == "bootstrap_zoom") kind = common_flydelta_bootstrap_refinement_kind::bootstrap_zoom;
    else if (value == "adaptive_alpha") kind = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
    else return false;
    return true;
}

json daemon_flydelta_orchestration_state_json(
        const common_flydelta_experiment_plan & plan,
        const common_flydelta_utility_history & history,
        const std::string & bootstrap_state_ref = {}) {
    const auto & region = plan.continuation.region;
    return {
        {"kind", "flydelta_daemon_orchestration_state"},
        {"continuation", {
            {"direction_index", plan.continuation.direction_index},
            {"region_trial_index", plan.continuation.region_trial_index},
            {"search_score", plan.continuation.search_score},
            {"host_helped", plan.continuation.host_helped},
            {"region", {
                {"layer_indices", region.layer_indices},
                {"anchor_layer_index", region.anchor_layer_index},
                {"total_scale", region.total_scale},
                {"per_layer_scale", region.per_layer_scale},
                {"source", common_flydelta_layer_search_candidate_source_name(region.source)},
            }},
        }},
        {"depth", common_flydelta_search_depth_name(plan.depth)},
        {"phase", common_flydelta_experiment_phase_name(plan.phase)},
        {"bootstrap_refinement", common_flydelta_bootstrap_refinement_kind_name(plan.bootstrap_refinement)},
        {"budget", {
            {"depth", common_flydelta_search_depth_name(plan.budget.depth)},
            {"max_region_trials", plan.budget.max_region_trials},
            {"max_coefficient_trials", plan.budget.max_coefficient_trials},
            {"full_generation_top_k", plan.budget.full_generation_top_k},
            {"build_aggregate_directions", plan.budget.build_aggregate_directions},
            {"require_decision_margin", plan.budget.require_decision_margin},
            {"allow_tfo_lite", plan.budget.allow_tfo_lite},
            {"include_opposite_control", plan.budget.include_opposite_control},
        }},
        {"required_compatible_directions", plan.required_compatible_directions},
        {"require_decision_margin", plan.require_decision_margin},
        {"run_rank_two_controls_first", plan.run_rank_two_controls_first},
        {"tfo_lite_permitted_by_evidence", plan.tfo_lite_permitted_by_evidence},
        {"tfo_lite_requires_utility_gate", plan.tfo_lite_requires_utility_gate},
        {"run_tfo_lite", plan.run_tfo_lite},
        {"run_orthogonal_search", plan.run_orthogonal_search},
        {"history", {
            {"qualifying_streak", history.qualifying_streak},
            {"nonqualifying_streak", history.nonqualifying_streak},
        }},
        {"bootstrap_state_ref", bootstrap_state_ref},
    };
}

std::string daemon_flydelta_continuation_key(const common_flydelta_experiment_plan & plan) {
    std::string identity = std::to_string(plan.continuation.region.anchor_layer_index);
    for (const auto layer : plan.continuation.region.layer_indices) {
        identity += "\n" + std::to_string(layer);
    }
    return hash_sha256_hex(identity.data(), identity.size());
}

std::string daemon_flydelta_bootstrap_key(const common_flydelta_bootstrap_zoom_state & state) {
    std::string identity = std::to_string(state.anchor_layer);
    for (const auto layer : state.local_layers) identity += "\n" + std::to_string(layer);
    return hash_sha256_hex(identity.data(), identity.size());
}

bool daemon_flydelta_orchestration_state_from_json(
        const std::string & text,
        common_flydelta_experiment_plan & plan,
        common_flydelta_utility_history & history,
        std::string & error) {
    try {
        const auto value = json::parse(text);
        if (!value.is_object() || value.value("kind", "") != "flydelta_daemon_orchestration_state") {
            error = "FlyDelta lifecycle record is not an orchestration state";
            return false;
        }
        plan = {};
        const auto continuation = value.at("continuation");
        const auto region = continuation.at("region");
        plan.continuation.direction_index = continuation.value("direction_index", size_t{0});
        plan.continuation.region_trial_index = continuation.value("region_trial_index", size_t{0});
        plan.continuation.search_score = continuation.value("search_score", 0.0f);
        plan.continuation.host_helped = continuation.value("host_helped", false);
        plan.continuation.region.layer_indices = region.at("layer_indices").get<std::vector<uint32_t>>();
        plan.continuation.region.anchor_layer_index = region.value("anchor_layer_index", 0U);
        plan.continuation.region.total_scale = region.value("total_scale", 0.0f);
        plan.continuation.region.per_layer_scale = region.value("per_layer_scale", 0.0f);
        const auto source = region.value("source", "diagnostic_singleton");
        if (source == "diagnostic_singleton") {
            plan.continuation.region.source = common_flydelta_layer_search_candidate_source::diagnostic_singleton;
        } else if (source == "neighborhood_expansion") {
            plan.continuation.region.source = common_flydelta_layer_search_candidate_source::neighborhood_expansion;
        } else {
            error = "FlyDelta orchestration region source is invalid";
            return false;
        }
        if (!daemon_flydelta_parse_depth(value.at("depth").get<std::string>(), plan.depth) ||
                !daemon_flydelta_parse_phase(value.at("phase").get<std::string>(), plan.phase) ||
                !daemon_flydelta_parse_refinement(value.at("bootstrap_refinement").get<std::string>(),
                    plan.bootstrap_refinement)) {
            error = "FlyDelta orchestration phase metadata is invalid";
            return false;
        }
        const auto budget = value.at("budget");
        if (!daemon_flydelta_parse_depth(budget.at("depth").get<std::string>(), plan.budget.depth)) {
            error = "FlyDelta orchestration budget depth is invalid";
            return false;
        }
        plan.budget.max_region_trials = budget.value("max_region_trials", size_t{0});
        plan.budget.max_coefficient_trials = budget.value("max_coefficient_trials", size_t{0});
        plan.budget.full_generation_top_k = budget.value("full_generation_top_k", size_t{0});
        plan.budget.build_aggregate_directions = budget.value("build_aggregate_directions", false);
        plan.budget.require_decision_margin = budget.value("require_decision_margin", false);
        plan.budget.allow_tfo_lite = budget.value("allow_tfo_lite", false);
        plan.budget.include_opposite_control = budget.value("include_opposite_control", false);
        plan.required_compatible_directions = value.value("required_compatible_directions", size_t{1});
        plan.require_decision_margin = value.value("require_decision_margin", false);
        plan.run_rank_two_controls_first = value.value("run_rank_two_controls_first", false);
        plan.tfo_lite_permitted_by_evidence = value.value("tfo_lite_permitted_by_evidence", false);
        plan.tfo_lite_requires_utility_gate = value.value("tfo_lite_requires_utility_gate", false);
        plan.run_tfo_lite = value.value("run_tfo_lite", false);
        plan.run_orthogonal_search = value.value("run_orthogonal_search", false);
        const auto state_history = value.at("history");
        history.qualifying_streak = state_history.value("qualifying_streak", size_t{0});
        history.nonqualifying_streak = state_history.value("nonqualifying_streak", size_t{0});
        std::string validation_error;
        if (!common_flydelta_search_budget_validate(plan.budget, validation_error)) {
            error = "FlyDelta orchestration state budget is invalid: " + validation_error;
            return false;
        }
        return true;
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta orchestration state JSON is invalid: ") + exception.what();
        return false;
    }
}

bool daemon_flydelta_run_bootstrap_zoom_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, directions, error)) return false;

    common_flydelta_bootstrap_zoom_config zoom_config;
    std::vector<common_flydelta_bootstrap_zoom_candidate> candidates;
    if (resume.phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom) {
        if (!common_flydelta_propose_bootstrap_alpha_zoom(
                resume.anchor_layer, resume.selected_scale, zoom_config, candidates, error)) return false;
    } else if (resume.phase == common_flydelta_bootstrap_zoom_phase::profile_zoom) {
        if (!common_flydelta_propose_bootstrap_profile_zoom(
                resume.local_layers, resume.anchor_layer, resume.selected_scale,
                zoom_config, candidates, error)) return false;
    } else {
        // Sign control is the final deterministic BootstrapZoom slice.  The
        // next orchestration decision, not this runner, decides whether a
        // later rank-one adaptive-alpha slice is warranted.
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = common_flydelta_bootstrap_zoom_phase::sign_control;
        candidate.layer_indices = {resume.anchor_layer};
        candidate.layer_weights = {1.0f};
        candidate.total_scale = resume.selected_scale;
        candidate.opposite_sign_control = true;
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty()) {
        error = "FlyDelta BootstrapZoom produced no bounded candidates";
        return false;
    }
    const size_t remaining = resume.extra_model_trials >= zoom_config.max_extra_model_trials
        ? 0 : zoom_config.max_extra_model_trials - resume.extra_model_trials;
    if (remaining == 0) {
        error = "FlyDelta BootstrapZoom budget is exhausted";
        return false;
    }
    if (candidates.size() > remaining) candidates.resize(remaining);

    common_flydelta_arm_batch_request batch;
    batch.batch_id = job.id + ":bootstrap-zoom:" + std::to_string(resume.next_candidate_index);
    batch.wave_id = common_flydelta_bootstrap_zoom_phase_name(resume.phase);
    common_flydelta_arm_request baseline;
    baseline.job_id = job.id;
    baseline.wave_id = batch.wave_id;
    baseline.proposal_index = 0;
    baseline.arm_id = batch.batch_id + ":baseline";
    baseline.context_ref = job.seed.baseline_ref;
    baseline.fixture_ref = fixture.id;
    baseline.intervention_ref = job.seed.candidate_ref;
    baseline.batch_compatibility_key = baseline.context_ref + "\n" + baseline.fixture_ref;
    baseline.fresh_context = true;
    baseline.request_capture = true;
    baseline.request_teacher_forced_margin = true;
    baseline.request_generation = true;
    baseline.request_host_verification = true;
    baseline.max_generated_tokens = 64;
    for (const auto & direction : directions) baseline.layer_indices.push_back(
        static_cast<uint32_t>(direction.layer_index));
    baseline.coefficients.assign(baseline.layer_indices.size(), 0.0f);
    if (!common_flydelta_arm_request_validate(baseline, error)) return false;
    batch.arms.push_back(baseline);
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto & candidate = candidates[index];
        common_flydelta_arm_request arm;
        arm.job_id = job.id;
        arm.wave_id = batch.wave_id;
        arm.proposal_index = index + 1;
        arm.arm_id = batch.batch_id + ":" + std::to_string(index);
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = fixture.id;
        arm.intervention_ref = job.seed.candidate_ref;
        arm.batch_compatibility_key = baseline.batch_compatibility_key;
        arm.apply_overlay = true;
        arm.fresh_context = true;
        arm.alpha = candidate.opposite_sign_control ? -candidate.total_scale : candidate.total_scale;
        arm.layer_indices = candidate.layer_indices;
        arm.coefficients = candidate.layer_weights;
        arm.request_capture = true;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = true;
        arm.request_host_verification = true;
        arm.max_generated_tokens = 64;
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_request diagnostic_batch = batch;
    for (auto & arm : diagnostic_batch.arms) {
        arm.request_generation = false;
        arm.request_host_verification = false;
        arm.max_generated_tokens = 0;
    }
    common_flydelta_arm_batch_result diagnostic_results;
    if (!daemon_flydelta_execute_batch(provider, diagnostic_batch, diagnostic_results, error) ||
            diagnostic_results.arms.size() != diagnostic_batch.arms.size()) {
        if (error.empty()) error = "FlyDelta BootstrapZoom diagnostic batch returned incomplete arms";
        return false;
    }

    // Diagnostics preserve the existing candidate order and ranking signal.
    // Only the bounded diagnostic frontier is re-run with generation and
    // host verification; this is an execution optimization, not a policy
    // or candidate-generation change.
    std::vector<size_t> frontier_indices;
    frontier_indices.reserve(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        frontier_indices.push_back(index);
    }
    std::stable_sort(frontier_indices.begin(), frontier_indices.end(),
        [&](const size_t left, const size_t right) {
            const auto & lhs = diagnostic_results.arms[left + 1];
            const auto & rhs = diagnostic_results.arms[right + 1];
            const float lhs_score = lhs.margin_available ? lhs.margin.normalized_delta() : 0.0f;
            const float rhs_score = rhs.margin_available ? rhs.margin.normalized_delta() : 0.0f;
            return lhs_score > rhs_score;
        });
    constexpr size_t full_generation_top_k = 2;
    if (frontier_indices.size() > full_generation_top_k) {
        frontier_indices.resize(full_generation_top_k);
    }
    common_flydelta_arm_batch_request full_batch;
    full_batch.batch_id = batch.batch_id + ":frontier";
    full_batch.wave_id = batch.wave_id;
    full_batch.arms.push_back(batch.arms.front());
    for (const size_t index : frontier_indices) full_batch.arms.push_back(batch.arms[index + 1]);
    common_flydelta_arm_batch_result full_results;
    if (!daemon_flydelta_execute_batch(provider, full_batch, full_results, error) ||
            full_results.arms.size() != full_batch.arms.size()) {
        if (error.empty()) error = "FlyDelta BootstrapZoom frontier batch returned incomplete arms";
        return false;
    }

    next = resume;
    next.state_ref.clear();
    std::vector<common_flydelta_bootstrap_zoom_trial> new_trials;
    common_flydelta_search_pipeline_direction_result direction_result;
    direction_result.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction_result.direction.layer_index = directions.front().layer_index;
    direction_result.direction.values = directions.front().values;
    direction_result.direction.origin = "runtime_resource";
    direction_result.direction.extraction_id = job.seed.candidate_ref;
    direction_result.direction.source_samples = 1;
    direction_result.direction.retained_samples = 1;
    direction_result.direction.experimental_only = true;
    const auto & baseline_result = full_results.arms.front();
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto & arm = diagnostic_batch.arms[index + 1];
        const auto frontier = std::find(frontier_indices.begin(), frontier_indices.end(), index);
        const bool full_execution = frontier != frontier_indices.end();
        const auto & arm_result = full_execution
            ? full_results.arms[1 + static_cast<size_t>(frontier - frontier_indices.begin())]
            : diagnostic_results.arms[index + 1];
        common_flydelta_representation_diagnostics diagnostics;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, diagnostic_batch.arms.front().arm_id, arm,
                candidates[index].layer_indices.front(), diagnostics, error)) return false;
        common_flydelta_bootstrap_zoom_trial trial;
        trial.candidate = candidates[index];
        trial.outcome = arm_result.host_outcome;
        trial.host_evaluated = full_execution && arm_result.host_evaluated;
        trial.verifier_known = full_execution && arm_result.verifier_known;
        trial.margin_available = arm_result.margin.available;
        trial.margin_delta = trial.margin_available
            ? (full_execution
                ? arm_result.margin.normalized_delta() - baseline_result.margin.normalized_delta()
                : arm_result.margin.normalized_delta()) : 0.0f;
        trial.diagnostics_available = diagnostics.layer_index != 0;
        trial.diagnostics = diagnostics;
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
        new_trials.push_back(trial);

        common_flydelta_intervention_region_trial region_trial;
        region_trial.candidate.layer_indices = candidates[index].layer_indices;
        region_trial.candidate.anchor_layer_index = candidates[index].layer_indices.front();
        region_trial.candidate.total_scale = candidates[index].total_scale;
        region_trial.candidate.per_layer_scale = candidates[index].total_scale /
            std::sqrt(static_cast<float>(candidates[index].layer_indices.size()));
        region_trial.requested_total_scale = candidates[index].total_scale;
        region_trial.executed_total_scale = candidates[index].total_scale;
        region_trial.outcome = full_execution
            ? arm_result.host_outcome : common_flydelta_counterfactual_outcome::unknown;
        region_trial.quality_delta = full_execution
            ? arm_result.quality - baseline_result.quality : 0.0f;
        region_trial.margin = arm_result.margin;
        region_trial.margin_comparison.available = trial.margin_available;
        region_trial.margin_comparison.baseline = baseline_result.margin;
        region_trial.margin_comparison.candidate = arm_result.margin;
        region_trial.executed = arm_result.executed;
        region_trial.verifier_known = full_execution && arm_result.verifier_known;
        region_trial.geometry_available = trial.diagnostics_available;
        region_trial.geometry = diagnostics;
        region_trial.safe_to_continue = arm_result.host_outcome !=
            common_flydelta_counterfactual_outcome::harmed &&
            trial.diagnostics_available &&
            diagnostics.leakage <= 1.0f && diagnostics.shift_norm <= 1.0f;
        region_trial.promising = trial.margin_available && trial.margin_delta > 0.0f;
        region_trial.search_score = trial.margin_delta;
        region_trial.evidence_ref = full_execution ? arm_result.generation_ref : arm_result.capture_ref;
        direction_result.region_trials.push_back(std::move(region_trial));
    }
    next.completed_trials.insert(next.completed_trials.end(), new_trials.begin(), new_trials.end());
    next.extra_model_trials += new_trials.size();
    next.next_candidate_index += new_trials.size();
    // A diagnostics-first slice may legitimately reject every newly probed
    // arm. Preserve the safety gate, but report that bounded branch as
    // no-useful-utility rather than turning it into a worker failure. This is
    // the same terminal semantics used by the ordinary search pipeline.
    bool has_safe_trial = false;
    for (const auto & trial : next.completed_trials) {
        if (trial.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
        const bool geometry_safe = !trial.diagnostics_available ||
            (trial.diagnostics.cosine >= 0.3f && trial.diagnostics.progress > 0.0f &&
             trial.diagnostics.leakage <= 1.0f && trial.diagnostics.shift_norm <= 1.0f);
        if (geometry_safe) {
            has_safe_trial = true;
            break;
        }
    }
    if (has_safe_trial && !common_flydelta_select_bootstrap_zoom_trial(
            next.completed_trials, next.selection, error)) return false;
    if (next.selection.selected) {
        const auto & selected = next.completed_trials[next.selection.trial_index];
        next.selected_scale = selected.candidate.total_scale;
        next.best_margin_delta = selected.margin_delta;
        next.best_search_score = next.selection.search_score;
    }
    if (resume.phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom) {
        next.phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
    } else if (resume.phase == common_flydelta_bootstrap_zoom_phase::profile_zoom &&
            zoom_config.include_opposite_sign_control) {
        next.phase = common_flydelta_bootstrap_zoom_phase::sign_control;
    } else {
        next.phase = common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
    }
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.search_status = output.directions.front().region_trials.empty()
        ? common_flydelta_search_status::no_useful_utility
        : common_flydelta_search_status::candidate_available;
    return true;
}

bool daemon_flydelta_run_search_pipeline(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_search_pipeline_config & config,
        common_flydelta_search_pipeline_result & result,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    fixture.id = job.seed.verifier_ref;
    fixture.task_fingerprint = job.seed.task_fingerprint;
    fixture.model_profile_fingerprint = job.seed.model_profile_fingerprint;
    fixture.tokenizer_fingerprint = job.seed.tokenizer_fingerprint;
    fixture.template_fingerprint = job.seed.template_fingerprint;
    fixture.execution_context_fingerprint = job.seed.execution_context_fingerprint;
    fixture.verifier_revision = job.seed.verifier_ref;
    if (!common_flydelta_experiment_fixture_validate(fixture, error)) return false;

    std::vector<common_flydelta_basis_direction> basis;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, basis, error)) return false;
    common_flydelta_search_pipeline_direction input;
    input.direction.kind = common_flydelta_direction_kind::raw_repair;
    input.direction.layer_index = basis.front().layer_index;
    input.direction.values = basis.front().values;
    input.direction.origin = "runtime_resource";
    input.direction.extraction_id = job.seed.candidate_ref;
    input.direction.source_samples = 1;
    input.direction.retained_samples = 1;
    input.direction.experimental_only = true;
    for (const auto & direction : basis) {
        input.available_layers.push_back(static_cast<uint32_t>(direction.layer_index));
        input.layer_anchors.push_back(static_cast<uint32_t>(direction.layer_index));
    }
    std::sort(input.available_layers.begin(), input.available_layers.end());
    std::sort(input.layer_anchors.begin(), input.layer_anchors.end());
    input.layer_anchors.erase(std::unique(input.layer_anchors.begin(), input.layer_anchors.end()),
        input.layer_anchors.end());
    if (!common_flydelta_search_pipeline_direction_validate(input, config, error)) return false;

    size_t arm_index = 0;
    const auto capture_layers = input.available_layers;
    std::string baseline_capture_arm_id;
    const auto make_arm = [provider, &job, &arm_index](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_layer_candidate * layer,
            const float scale,
            const bool apply_overlay,
            const std::string & wave_id,
            const size_t proposal_index,
            const bool full_execution,
            common_flydelta_arm_request & arm,
            std::string & runner_error) {
        arm = {};
        arm.job_id = job.id;
        arm.wave_id = wave_id;
        arm.proposal_index = proposal_index;
        arm.arm_id = "flydelta://runtime/" + job.id + "/" + std::to_string(arm_index++);
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = current_fixture.id;
        arm.intervention_ref = job.seed.candidate_ref;
        arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
        arm.apply_overlay = apply_overlay;
        arm.fresh_context = true;
        arm.alpha = apply_overlay ? scale : 0.0f;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = full_execution;
        arm.request_host_verification = full_execution;
        // Diagnostics are first-class search observations.  The production
        // host captures the small requested layer set and keeps it only for
        // this bounded comparison wave.
        arm.request_capture = true;
        arm.max_generated_tokens = full_execution ? 64 : 0;
        if (layer != nullptr && apply_overlay) {
            arm.layer_indices = layer->layer_indices;
            arm.coefficients.assign(layer->layer_indices.size(), 1.0f);
        }
        return common_flydelta_arm_request_validate(arm, runner_error);
    };
    common_flydelta_search_pipeline_runner runner = [provider, &job, &make_arm,
            &capture_layers, &baseline_capture_arm_id](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_direction_candidate &,
            const common_flydelta_layer_candidate * layer,
            float scale,
            bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_scale_geometry & geometry,
            std::string & runner_error) {
        common_flydelta_arm_request arm;
        if (!make_arm(current_fixture, layer, scale, apply_overlay,
                apply_overlay ? "region-diagnostic" : "baseline-diagnostic", 0, false,
                arm, runner_error)) return false;
        if (!apply_overlay) {
            // A baseline has no overlay layers of its own, but it must capture
            // the same searchable layers as later arms for aligned geometry.
            arm.layer_indices = capture_layers;
            arm.coefficients.assign(capture_layers.size(), 0.0f);
        }
        common_flydelta_arm_batch_request batch;
        batch.batch_id = job.id + ":runtime";
        batch.wave_id = arm.wave_id;
        batch.arms.push_back(arm);
        common_flydelta_arm_batch_result batch_result;
        if (!daemon_flydelta_execute_batch(provider, batch, batch_result, runner_error) ||
                batch_result.arms.size() != 1) return false;
        const auto & arm_result = batch_result.arms.front();
        if (!apply_overlay) baseline_capture_arm_id = arm.arm_id;
        trial.executed = arm_result.executed;
        trial.verifier_known = arm_result.verifier_known;
        trial.passed = daemon_flydelta_arm_semantic_passed(arm_result);
        trial.quality = arm_result.quality;
        trial.overlay_applied = apply_overlay;
        trial.intervention_count = arm.layer_indices.size();
        trial.evidence_ref = arm_result.generation_ref;
        margin = arm_result.margin;
        geometry = {};
        if (apply_overlay && !baseline_capture_arm_id.empty() &&
                !arm_result.capture_ref.empty()) {
            std::shared_ptr<const common_flydelta_hidden_state_capture> baseline_capture;
            std::shared_ptr<const common_flydelta_hidden_state_capture> overlay_capture;
            {
                std::lock_guard<std::mutex> lock(provider->capture_mutex);
                const auto baseline_it = provider->captures.find(baseline_capture_arm_id);
                const auto overlay_it = provider->captures.find(arm.arm_id);
                if (baseline_it != provider->captures.end()) baseline_capture = baseline_it->second;
                if (overlay_it != provider->captures.end()) overlay_capture = overlay_it->second;
            }
            std::vector<common_flydelta_basis_direction> available;
            if (baseline_capture && overlay_capture &&
                    daemon_flydelta_parse_directions(*provider, arm.intervention_ref,
                        available, runner_error)) {
                const uint32_t diagnostic_layer = layer == nullptr || layer->layer_indices.empty()
                    ? 0U : layer->layer_indices.front();
                const auto direction = std::find_if(available.begin(), available.end(),
                    [&](const auto & value) { return value.layer_index == static_cast<int32_t>(diagnostic_layer); });
                if (direction != available.end()) {
                    common_flydelta_behavior_delta delta;
                    delta.id = arm.arm_id + ":diagnostic";
                    delta.behavior_key = job.seed.behavior_key;
                    delta.capture_manifest_id = arm.context_ref;
                    delta.host_evidence_ref = job.seed.evidence_ref;
                    delta.model_profile_fingerprint = provider->model_profile_fingerprint;
                    delta.execution_context_fingerprint = job.seed.execution_context_fingerprint;
                    delta.capture_layout_revision = provider->capture_layout_revision;
                    delta.layer_index = direction->layer_index;
                    delta.values = direction->values;
                    common_flydelta_representation_diagnostics diagnostics;
                    if (!common_flydelta_representation_diagnostics_from_captures(
                            *baseline_capture, *overlay_capture, delta,
                            4U * 1024U * 1024U, diagnostics, runner_error)) return false;
                    geometry.available = true;
                    geometry.cosine = diagnostics.cosine;
                    geometry.progress = diagnostics.progress;
                    geometry.leakage = diagnostics.leakage;
                    geometry.shift_norm = diagnostics.shift_norm;
                }
            }
        }
        return arm_result.executed;
    };
    const common_flydelta_search_pipeline_batch_runner batch_runner = [provider, &job, &make_arm,
            &baseline_capture_arm_id](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_direction_candidate &,
            const std::vector<common_flydelta_intervention_region_candidate> & candidates,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_scale_geometry> & geometries,
            std::vector<bool> & geometry_available,
            std::string & runner_error) {
        common_flydelta_arm_batch_request batch;
        batch.batch_id = job.id + ":region";
        batch.wave_id = "whirlpool-region";
        batch.arms.reserve(candidates.size());
        for (size_t index = 0; index < candidates.size(); ++index) {
            common_flydelta_layer_candidate layer;
            layer.layer_indices = candidates[index].layer_indices;
            layer.total_scale = candidates[index].total_scale;
            common_flydelta_arm_request arm;
            if (!make_arm(current_fixture, &layer, candidates[index].total_scale, true,
                    batch.wave_id, index, false, arm, runner_error)) return false;
            batch.arms.push_back(std::move(arm));
        }
        common_flydelta_arm_batch_result batch_result;
        if (!daemon_flydelta_execute_batch(provider, batch, batch_result, runner_error) ||
                batch_result.arms.size() != candidates.size()) {
            if (runner_error.empty()) runner_error = "FlyDelta daemon region batch returned incomplete arms";
            return false;
        }
        trials.clear();
        margins.clear();
        geometries.clear();
        geometry_available.clear();
        trials.reserve(candidates.size());
        margins.reserve(candidates.size());
        geometries.reserve(candidates.size());
        geometry_available.reserve(candidates.size());
        for (size_t index = 0; index < candidates.size(); ++index) {
            const auto & arm = batch_result.arms[index];
            if (!arm.executed || arm.arm_id != batch.arms[index].arm_id) {
                runner_error = "FlyDelta daemon region batch returned an invalid arm";
                return false;
            }
            common_flydelta_counterfactual_trial trial;
            trial.executed = arm.executed;
            trial.verifier_known = arm.verifier_known;
            trial.passed = daemon_flydelta_arm_semantic_passed(arm);
            trial.quality = arm.quality;
            trial.overlay_applied = true;
            trial.intervention_count = batch.arms[index].layer_indices.size();
            trial.evidence_ref = arm.generation_ref;
            trials.push_back(std::move(trial));
            margins.push_back(arm.margin);
            common_flydelta_scale_geometry geometry;
            if (!baseline_capture_arm_id.empty() && !arm.capture_ref.empty()) {
                std::shared_ptr<const common_flydelta_hidden_state_capture> baseline_capture;
                std::shared_ptr<const common_flydelta_hidden_state_capture> overlay_capture;
                {
                    std::lock_guard<std::mutex> lock(provider->capture_mutex);
                    const auto baseline_it = provider->captures.find(baseline_capture_arm_id);
                    const auto overlay_it = provider->captures.find(arm.arm_id);
                    if (baseline_it != provider->captures.end()) baseline_capture = baseline_it->second;
                    if (overlay_it != provider->captures.end()) overlay_capture = overlay_it->second;
                }
                std::vector<common_flydelta_basis_direction> available;
                if (baseline_capture && overlay_capture &&
                        daemon_flydelta_parse_directions(*provider, batch.arms[index].intervention_ref,
                            available, runner_error)) {
                    const auto direction = std::find_if(available.begin(), available.end(),
                        [&](const auto & value) {
                            return value.layer_index == static_cast<int32_t>(
                                candidates[index].anchor_layer_index);
                        });
                    if (direction != available.end()) {
                        common_flydelta_behavior_delta delta;
                        delta.id = arm.arm_id + ":diagnostic";
                        delta.behavior_key = job.seed.behavior_key;
                        delta.capture_manifest_id = batch.arms[index].context_ref;
                        delta.host_evidence_ref = job.seed.evidence_ref;
                        delta.model_profile_fingerprint = provider->model_profile_fingerprint;
                        delta.execution_context_fingerprint = job.seed.execution_context_fingerprint;
                        delta.capture_layout_revision = provider->capture_layout_revision;
                        delta.layer_index = direction->layer_index;
                        delta.values = direction->values;
                        common_flydelta_representation_diagnostics diagnostics;
                        if (!common_flydelta_representation_diagnostics_from_captures(
                                *baseline_capture, *overlay_capture, delta,
                                4U * 1024U * 1024U, diagnostics, runner_error)) return false;
                        geometry.available = true;
                        geometry.cosine = diagnostics.cosine;
                        geometry.progress = diagnostics.progress;
                        geometry.leakage = diagnostics.leakage;
                        geometry.shift_norm = diagnostics.shift_norm;
                    }
                }
            }
            geometries.push_back(std::move(geometry));
            geometry_available.push_back(geometries.back().available);
        }
        return true;
    };
    if (!common_flydelta_run_search_pipeline_batched(
            fixture, config, {input}, runner, batch_runner, result, error)) return false;

    // Whirlpool/region search ranks compact diagnostic observations first.
    // Only the bounded frontier is re-run with generation and host
    // verification; the search algorithm and its candidate ordering remain
    // unchanged.
    struct frontier_arm {
        size_t direction_index = 0;
        size_t trial_index = 0;
    };
    std::vector<frontier_arm> frontier;
    for (size_t direction_index = 0; direction_index < result.directions.size(); ++direction_index) {
        const auto & direction = result.directions[direction_index];
        for (size_t trial_index = 0; trial_index < direction.region_trials.size(); ++trial_index) {
            if (direction.region_trials[trial_index].executed) {
                frontier.push_back({direction_index, trial_index});
            }
        }
    }
    std::stable_sort(frontier.begin(), frontier.end(), [&](const frontier_arm & left,
            const frontier_arm & right) {
        const auto & left_trial = result.directions[left.direction_index].region_trials[left.trial_index];
        const auto & right_trial = result.directions[right.direction_index].region_trials[right.trial_index];
        if (left_trial.search_score != right_trial.search_score) {
            return left_trial.search_score > right_trial.search_score;
        }
        if (left_trial.candidate.layer_indices.size() != right_trial.candidate.layer_indices.size()) {
            return left_trial.candidate.layer_indices.size() < right_trial.candidate.layer_indices.size();
        }
        return left_trial.candidate.total_scale < right_trial.candidate.total_scale;
    });
    constexpr size_t full_generation_top_k = 3;
    if (frontier.size() > full_generation_top_k) frontier.resize(full_generation_top_k);
    if (frontier.empty()) {
        result.search_status = common_flydelta_search_status::no_useful_utility;
        return true;
    }

    common_flydelta_arm_batch_request full_batch;
    full_batch.batch_id = job.id + ":region-full-generation";
    full_batch.wave_id = "whirlpool-region-full-generation";
    common_flydelta_arm_request baseline;
    if (!make_arm(fixture, nullptr, 0.0f, false, full_batch.wave_id, 0, true,
            baseline, error)) return false;
    baseline.layer_indices = capture_layers;
    baseline.coefficients.assign(capture_layers.size(), 0.0f);
    if (!common_flydelta_arm_request_validate(baseline, error)) return false;
    full_batch.arms.push_back(baseline);
    for (size_t index = 0; index < frontier.size(); ++index) {
        const auto & selected = result.directions[frontier[index].direction_index].region_trials[
            frontier[index].trial_index];
        common_flydelta_layer_candidate layer;
        layer.layer_indices = selected.candidate.layer_indices;
        layer.total_scale = selected.candidate.total_scale;
        common_flydelta_arm_request arm;
        if (!make_arm(fixture, &layer, selected.candidate.total_scale, true,
                full_batch.wave_id, index + 1, true, arm, error)) return false;
        full_batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result full_result;
    if (!daemon_flydelta_execute_batch(provider, full_batch, full_result, error) ||
            full_result.arms.size() != full_batch.arms.size()) {
        if (error.empty()) error = "FlyDelta region full-generation batch returned incomplete arms";
        return false;
    }
    const auto & baseline_arm = full_result.arms.front();
    common_flydelta_counterfactual_trial baseline_trial;
    baseline_trial.executed = baseline_arm.executed;
    baseline_trial.passed = daemon_flydelta_arm_semantic_passed(baseline_arm);
    baseline_trial.quality = baseline_arm.quality;
    baseline_trial.verifier_known = baseline_arm.verifier_known;
    baseline_trial.evidence_ref = baseline_arm.generation_ref;
    if (!common_flydelta_counterfactual_trial_validate(baseline_trial, error)) return false;
    for (size_t index = 0; index < frontier.size(); ++index) {
        const auto location = frontier[index];
        auto & trial = result.directions[location.direction_index].region_trials[location.trial_index];
        const auto & arm = full_result.arms[index + 1];
        common_flydelta_counterfactual_trial candidate_trial;
        candidate_trial.executed = arm.executed;
        candidate_trial.passed = daemon_flydelta_arm_semantic_passed(arm);
        candidate_trial.quality = arm.quality;
        candidate_trial.verifier_known = arm.verifier_known;
        candidate_trial.evidence_ref = arm.generation_ref;
        if (!common_flydelta_counterfactual_trial_validate(candidate_trial, error)) return false;
        trial.outcome = common_flydelta_classify_counterfactual(baseline_trial, candidate_trial);
        trial.quality_delta = candidate_trial.quality - baseline_trial.quality;
        trial.margin = arm.margin;
        trial.margin_comparison.available = baseline_arm.margin.available && arm.margin.available;
        trial.margin_comparison.baseline = baseline_arm.margin;
        trial.margin_comparison.candidate = arm.margin;
        trial.executed = candidate_trial.executed;
        trial.verifier_known = baseline_trial.verifier_known && candidate_trial.verifier_known;
        trial.evidence_ref = candidate_trial.evidence_ref;
    }
    result.selection = {};
    for (size_t direction_index = 0; direction_index < result.directions.size(); ++direction_index) {
        auto & direction = result.directions[direction_index];
        direction.region_selection = {};
        for (size_t trial_index = 0; trial_index < direction.region_trials.size(); ++trial_index) {
            const auto & trial = direction.region_trials[trial_index];
            if (trial.outcome != common_flydelta_counterfactual_outcome::helped ||
                    !trial.executed || !trial.verifier_known) continue;
            if (!direction.region_selection.selected ||
                    trial.quality_delta > direction.region_selection.score) {
                direction.region_selection = {true, trial_index, trial.quality_delta};
            }
            const bool better = !result.selection.selected ||
                trial.quality_delta > result.selection.score ||
                (trial.quality_delta == result.selection.score &&
                 trial.candidate.layer_indices.size() <
                    result.directions[result.selection.direction_index].region_trials[
                        result.selection.region_trial_index].candidate.layer_indices.size());
            if (better) {
                result.selection.selected = true;
                result.selection.intervention_region = true;
                result.selection.direction_index = direction_index;
                result.selection.region_trial_index = trial_index;
                result.selection.layer_result_index = 0;
                result.selection.scale = trial.candidate.total_scale;
                result.selection.score = trial.quality_delta;
            }
        }
    }
    result.search_status = result.selection.selected
        ? common_flydelta_search_status::candidate_available
        : common_flydelta_search_status::no_useful_utility;
    return true;
}

bool daemon_flydelta_run_adaptive_alpha_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, directions, error)) {
        return false;
    }
    const auto direction = std::find_if(directions.begin(), directions.end(), [&](const auto & value) {
        return value.layer_index == static_cast<int32_t>(resume.anchor_layer);
    });
    if (direction == directions.end()) {
        error = "FlyDelta AdaptiveAlpha anchor direction is absent from the intervention resource";
        return false;
    }
    common_flydelta_alpha_response_search_config config;
    const float prior_scale = resume.alpha_response_available && resume.alpha_response.selected
        ? resume.alpha_response.scale : resume.selected_scale;
    config.seed_scale = std::max(0.0001f, std::min(config.max_scale, prior_scale));
    std::string baseline_arm_id;
    size_t proposal_index = resume.next_candidate_index;
    const auto runner = [provider, &job, &fixture, layer = resume.anchor_layer,
            &baseline_arm_id, &proposal_index, config](const common_flydelta_experiment_fixture &,
            const float scale, const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & geometry_available, std::string & runner_error) mutable {
        std::string arm_id;
        const size_t arm_proposal = proposal_index++;
        if (!daemon_flydelta_run_rank1_alpha_arm(provider, job, fixture, layer, scale,
                apply_overlay, arm_proposal, !apply_overlay, trial, margin, geometry,
                geometry_available, arm_id, runner_error)) return false;
        if (!apply_overlay) {
            baseline_arm_id = std::move(arm_id);
            return true;
        }
        if (baseline_arm_id.empty()) {
            runner_error = "FlyDelta AdaptiveAlpha candidate ran before its baseline";
            return false;
        }
        common_flydelta_arm_request diagnostic_arm;
        diagnostic_arm.arm_id = arm_id;
        diagnostic_arm.context_ref = job.seed.baseline_ref;
        diagnostic_arm.intervention_ref = job.seed.candidate_ref;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, baseline_arm_id,
                diagnostic_arm, layer, geometry, runner_error)) return false;
        geometry_available = geometry.layer_index != 0;
        const bool safe_geometry = !geometry_available ||
            (geometry.cosine >= config.min_cosine &&
             geometry.leakage <= config.max_leakage &&
             geometry.shift_norm <= config.max_shift_norm);
        if (!safe_geometry) return true;
        if (!daemon_flydelta_run_rank1_alpha_arm(provider, job, fixture, layer, scale,
                true, arm_proposal, true, trial, margin, geometry,
                geometry_available, arm_id, runner_error)) return false;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, baseline_arm_id,
                diagnostic_arm, layer, geometry, runner_error)) return false;
        geometry_available = geometry.layer_index != 0;
        return true;
    };
    std::vector<common_flydelta_alpha_response_trial> trials;
    common_flydelta_alpha_response_selection selection;
    if (!common_flydelta_run_alpha_response_search(
            fixture, config, runner, trials, selection, error)) return false;

    next = resume;
    next.state_ref.clear();
    next.phase = common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
    next.refinement_kind = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
    next.alpha_response_available = true;
    next.alpha_response = selection;
    if (selection.selected) {
        next.selected_scale = selection.scale;
        next.best_search_score = selection.utility;
        if (selection.best_margin_available) {
            next.best_margin_delta = selection.best_margin_delta_normalized;
        }
    }
    next.next_candidate_index = proposal_index;

    common_flydelta_search_pipeline_direction_result direction_result;
    direction_result.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction_result.direction.layer_index = direction->layer_index;
    direction_result.direction.values = direction->values;
    direction_result.direction.origin = "runtime_adaptive_alpha";
    direction_result.direction.extraction_id = job.seed.candidate_ref;
    direction_result.direction.source_samples = 1;
    direction_result.direction.retained_samples = 1;
    direction_result.direction.experimental_only = true;
    for (const auto & alpha_trial : trials) {
        common_flydelta_intervention_region_trial region_trial;
        region_trial.candidate.layer_indices = {resume.anchor_layer};
        region_trial.candidate.anchor_layer_index = resume.anchor_layer;
        region_trial.candidate.total_scale = alpha_trial.scale;
        region_trial.candidate.per_layer_scale = alpha_trial.scale;
        region_trial.requested_total_scale = alpha_trial.requested_scale;
        region_trial.executed_total_scale = alpha_trial.scale;
        region_trial.outcome = alpha_trial.outcome;
        region_trial.quality_delta = alpha_trial.counterfactual.quality;
        region_trial.margin = alpha_trial.margin;
        region_trial.executed = alpha_trial.counterfactual.executed;
        region_trial.verifier_known = alpha_trial.counterfactual.verifier_known;
        region_trial.geometry_available = alpha_trial.geometry_available;
        region_trial.geometry = alpha_trial.geometry;
        region_trial.safe_to_continue = alpha_trial.outcome !=
            common_flydelta_counterfactual_outcome::harmed &&
            alpha_trial.safe_to_continue;
        region_trial.promising = alpha_trial.safe_to_continue &&
            alpha_trial.utility > config.utility_epsilon;
        region_trial.search_score = alpha_trial.utility;
        region_trial.evidence_ref = alpha_trial.counterfactual.evidence_ref;
        if (!common_flydelta_intervention_region_trial_validate(region_trial, error)) return false;
        direction_result.region_trials.push_back(std::move(region_trial));
    }
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.alpha_response_available = true;
    output.alpha_response = selection;
    output.search_status = selection.selected
        ? common_flydelta_search_status::candidate_available
        : common_flydelta_search_status::no_useful_utility;
    return true;
}

// Orthogonal remains a same-phase experimental escape.  The common layer
// derives the profile-space residual from persisted BootstrapZoom trials; the
// daemon owns only the fresh baseline/control wave that validates it.
bool daemon_flydelta_run_orthogonal_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    common_flydelta_orthogonal_search_config config;
    common_flydelta_orthogonal_search_input input;
    if (!common_flydelta_prepare_orthogonal_search_input(config, resume, input, error)) {
        return false;
    }
    common_flydelta_orthogonal_search_result orthogonal;
    if (!common_flydelta_build_orthogonal_search_direction(
            config, input.rank1_intervention, input.arms, orthogonal, error)) return false;
    if (!orthogonal.available) {
        output = {};
        output.search_status = common_flydelta_search_status::no_useful_utility;
        next = resume;
        next.state_ref.clear();
        return true;
    }
    std::vector<common_flydelta_basis_direction> available;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, available, error)) {
        return false;
    }
    const auto normalize = [](std::vector<float> values) {
        double energy = 0.0;
        for (const float value : values) energy += static_cast<double>(value) * value;
        if (energy > 0.0) {
            const float inverse = static_cast<float>(1.0 / std::sqrt(energy));
            for (float & value : values) value *= inverse;
        }
        return values;
    };
    struct orthogonal_control {
        std::string label;
        std::vector<float> profile;
        bool opposite = false;
    };
    std::vector<orthogonal_control> controls;
    controls.push_back({"rank1", input.rank1_intervention, false});
    controls.push_back({"orthogonal", orthogonal.direction, false});
    std::vector<float> sum(input.rank1_intervention.size(), 0.0f);
    std::vector<float> difference(input.rank1_intervention.size(), 0.0f);
    for (size_t index = 0; index < sum.size(); ++index) {
        sum[index] = input.rank1_intervention[index] + orthogonal.direction[index];
        difference[index] = input.rank1_intervention[index] - orthogonal.direction[index];
    }
    controls.push_back({"sum", normalize(std::move(sum)), false});
    controls.push_back({"difference", normalize(std::move(difference)), true});

    common_flydelta_arm_batch_request batch;
    batch.batch_id = job.id + ":orthogonal";
    batch.wave_id = "orthogonal-controls";
    common_flydelta_arm_request baseline;
    baseline.job_id = job.id;
    baseline.wave_id = batch.wave_id;
    baseline.proposal_index = 0;
    baseline.context_ref = job.seed.baseline_ref;
    baseline.fixture_ref = fixture.id;
    baseline.intervention_ref = job.seed.candidate_ref;
    baseline.batch_compatibility_key = baseline.context_ref + "\n" + baseline.fixture_ref;
    baseline.layer_indices = input.local_layers;
    baseline.coefficients.assign(input.local_layers.size(), 0.0f);
    baseline.fresh_context = true;
    baseline.request_capture = true;
    baseline.request_teacher_forced_margin = true;
    baseline.max_capture_bytes = 4U * 1024U * 1024U;
    baseline.max_generated_tokens = 64;
    baseline.arm_id = "flydelta://runtime/" + job.id + "/orthogonal/baseline";
    if (!common_flydelta_arm_request_validate(baseline, error)) return false;
    batch.arms.push_back(baseline);

    std::vector<common_flydelta_bootstrap_zoom_candidate> candidates;
    candidates.reserve(controls.size());
    for (size_t control_index = 0; control_index < controls.size(); ++control_index) {
        const auto & control = controls[control_index];
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = control.opposite
            ? common_flydelta_bootstrap_zoom_phase::sign_control
            : common_flydelta_bootstrap_zoom_phase::profile_zoom;
        candidate.total_scale = std::max(0.0001f, resume.selected_scale);
        candidate.opposite_sign_control = control.opposite;
        for (size_t layer = 0; layer < input.local_layers.size(); ++layer) {
            if (std::fabs(control.profile[layer]) <= 0.000001f) continue;
            const auto direction = std::find_if(available.begin(), available.end(),
                [&](const auto & value) {
                    return value.layer_index == static_cast<int32_t>(input.local_layers[layer]);
                });
            if (direction == available.end()) {
                error = "FlyDelta orthogonal control lacks a compatible source direction";
                return false;
            }
            candidate.layer_indices.push_back(input.local_layers[layer]);
            candidate.layer_weights.push_back(control.profile[layer]);
        }
        candidate.layer_weights = normalize(std::move(candidate.layer_weights));
        if (!common_flydelta_bootstrap_zoom_candidate_validate(candidate, error)) return false;
        common_flydelta_arm_request arm;
        arm.job_id = job.id;
        arm.wave_id = batch.wave_id;
        arm.proposal_index = control_index + 1;
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = fixture.id;
        arm.intervention_ref = job.seed.candidate_ref;
        arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
        arm.layer_indices = candidate.layer_indices;
        arm.coefficients = candidate.layer_weights;
        arm.alpha = candidate.total_scale;
        arm.apply_overlay = true;
        arm.fresh_context = true;
        arm.request_capture = true;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = false;
        arm.request_host_verification = false;
        arm.max_capture_bytes = 4U * 1024U * 1024U;
        arm.max_generated_tokens = 0;
        const std::string identity = job.id + "\n" + batch.wave_id + "\n" + control.label;
        arm.arm_id = "flydelta://runtime/" + job.id + "/orthogonal/" +
            hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        candidates.push_back(std::move(candidate));
        batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result diagnostic_results;
    if (!daemon_flydelta_execute_batch(provider, batch, diagnostic_results, error) ||
            diagnostic_results.arms.size() != batch.arms.size()) {
        if (error.empty()) error = "FlyDelta orthogonal diagnostic batch returned incomplete arms";
        return false;
    }

    std::vector<size_t> frontier_indices;
    for (size_t index = 0; index < candidates.size(); ++index) frontier_indices.push_back(index);
    std::stable_sort(frontier_indices.begin(), frontier_indices.end(),
        [&](const size_t left, const size_t right) {
            const auto & lhs = diagnostic_results.arms[left + 1];
            const auto & rhs = diagnostic_results.arms[right + 1];
            const float lhs_score = lhs.margin_available ? lhs.margin.normalized_delta() : 0.0f;
            const float rhs_score = rhs.margin_available ? rhs.margin.normalized_delta() : 0.0f;
            return lhs_score > rhs_score;
        });
    constexpr size_t full_generation_top_k = 2;
    if (frontier_indices.size() > full_generation_top_k) frontier_indices.resize(full_generation_top_k);
    common_flydelta_arm_batch_request full_batch;
    full_batch.batch_id = batch.batch_id + ":frontier";
    full_batch.wave_id = batch.wave_id;
    auto full_baseline = batch.arms.front();
    full_baseline.request_generation = true;
    full_baseline.request_host_verification = true;
    full_baseline.max_generated_tokens = 64;
    full_batch.arms.push_back(std::move(full_baseline));
    for (const size_t index : frontier_indices) {
        auto arm = batch.arms[index + 1];
        arm.request_generation = true;
        arm.request_host_verification = true;
        arm.max_generated_tokens = 64;
        full_batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result full_results;
    if (!daemon_flydelta_execute_batch(provider, full_batch, full_results, error) ||
            full_results.arms.size() != full_batch.arms.size()) {
        if (error.empty()) error = "FlyDelta orthogonal frontier batch returned incomplete arms";
        return false;
    }
    const auto & baseline_result = full_results.arms.front();
    if (!baseline_result.executed || baseline_result.arm_id != baseline.arm_id) {
        error = "FlyDelta orthogonal baseline arm is invalid";
        return false;
    }
    std::vector<common_flydelta_bootstrap_zoom_trial> trials;
    trials.reserve(candidates.size());
    const auto anchor_direction = std::find_if(available.begin(), available.end(),
        [&](const auto & value) {
            return value.layer_index == static_cast<int32_t>(resume.anchor_layer);
        });
    if (anchor_direction == available.end()) {
        error = "FlyDelta orthogonal surface lacks the rank-one anchor direction";
        return false;
    }
    common_flydelta_search_pipeline_direction_result direction_result;
    // A direction candidate is an embedding-space vector.  Orthogonal here
    // is a local layer-profile coordinate, so retain the compatible anchor
    // vector and record the new experimental surface exclusively in origin
    // and the persisted surface trials rather than fabricating a mismatched
    // embedding-space direction.
    direction_result.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction_result.direction.layer_index = anchor_direction->layer_index;
    direction_result.direction.values = anchor_direction->values;
    direction_result.direction.origin = "runtime_orthogonal_profile";
    direction_result.direction.extraction_id = job.seed.candidate_ref;
    direction_result.direction.source_samples = orthogonal.source_arm_count;
    direction_result.direction.retained_samples = orthogonal.source_arm_count;
    direction_result.direction.experimental_only = true;
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto frontier = std::find(frontier_indices.begin(), frontier_indices.end(), index);
        const bool full_execution = frontier != frontier_indices.end();
        const auto & arm = full_execution
            ? full_results.arms[1 + static_cast<size_t>(frontier - frontier_indices.begin())]
            : diagnostic_results.arms[index + 1];
        if (!arm.executed || arm.arm_id != batch.arms[index + 1].arm_id) {
            error = "FlyDelta orthogonal control arm is invalid";
            return false;
        }
        common_flydelta_bootstrap_zoom_trial trial;
        trial.candidate = candidates[index];
        trial.outcome = full_execution
            ? arm.host_outcome : common_flydelta_counterfactual_outcome::unknown;
        trial.host_evaluated = full_execution && arm.host_evaluated;
        trial.verifier_known = full_execution && arm.verifier_known;
        trial.margin_available = arm.margin.available;
        trial.margin_delta = trial.margin_available
            ? (full_execution
                ? arm.margin.normalized_delta() - baseline_result.margin.normalized_delta()
                : arm.margin.normalized_delta()) : 0.0f;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, baseline.arm_id,
                batch.arms[index + 1], trial.candidate.layer_indices.front(),
                trial.diagnostics, error)) return false;
        trial.diagnostics_available = trial.diagnostics.schema_version == 1 &&
            trial.diagnostics.layer_index != 0;
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
        trials.push_back(trial);
        common_flydelta_intervention_region_trial region_trial;
        region_trial.candidate.layer_indices = trial.candidate.layer_indices;
        region_trial.candidate.anchor_layer_index = resume.anchor_layer;
        region_trial.candidate.total_scale = trial.candidate.total_scale;
        region_trial.candidate.per_layer_scale = trial.candidate.total_scale /
            std::sqrt(static_cast<float>(std::max<size_t>(1, trial.candidate.layer_indices.size())));
        region_trial.requested_total_scale = trial.candidate.total_scale;
        region_trial.executed_total_scale = trial.candidate.total_scale;
        region_trial.outcome = trial.outcome;
        region_trial.quality_delta = full_execution
            ? arm.quality - baseline_result.quality : 0.0f;
        region_trial.margin = arm.margin;
        region_trial.executed = arm.executed;
        region_trial.verifier_known = full_execution && arm.verifier_known;
        region_trial.geometry_available = trial.diagnostics_available;
        region_trial.geometry = trial.diagnostics;
        region_trial.safe_to_continue = arm.host_outcome !=
            common_flydelta_counterfactual_outcome::harmed &&
            trial.diagnostics_available &&
            trial.diagnostics.cosine >= config.minimum_cosine &&
            trial.diagnostics.progress > 0.0f &&
            trial.diagnostics.leakage <= config.maximum_leakage &&
            trial.diagnostics.shift_norm <= config.maximum_shift_norm;
        region_trial.promising = region_trial.safe_to_continue &&
            trial.margin_available && trial.margin_delta > 0.0f;
        region_trial.search_score = trial.margin_available ? trial.margin_delta : 0.0f;
        region_trial.evidence_ref = full_execution ? arm.generation_ref : arm.capture_ref;
        if (!common_flydelta_intervention_region_trial_validate(region_trial, error)) return false;
        direction_result.region_trials.push_back(std::move(region_trial));
    }
    common_flydelta_bootstrap_zoom_selection selection;
    bool has_safe_trial = false;
    for (const auto & trial : trials) {
        if (trial.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
        const bool geometry_safe = !trial.diagnostics_available ||
            (trial.diagnostics.cosine >= config.minimum_cosine &&
             trial.diagnostics.progress > 0.0f &&
             trial.diagnostics.leakage <= config.maximum_leakage &&
             trial.diagnostics.shift_norm <= config.maximum_shift_norm);
        if (geometry_safe) {
            has_safe_trial = true;
            break;
        }
    }
    if (has_safe_trial && !common_flydelta_select_bootstrap_zoom_trial(
            trials, selection, error)) return false;
    direction_result.region_selection.selected = selection.selected;
    direction_result.region_selection.trial_index = selection.trial_index;
    direction_result.region_selection.score = selection.search_score;
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.search_status = selection.selected
        ? common_flydelta_search_status::candidate_available
        : common_flydelta_search_status::no_useful_utility;
    output.selection.selected = selection.selected;
    output.selection.intervention_region = selection.selected;
    output.selection.direction_index = 0;
    output.selection.region_trial_index = selection.trial_index;
    output.selection.score = selection.search_score;
    next = resume;
    next.state_ref.clear();
    next.parent_surface_revision = resume.surface_revision;
    next.surface_revision = resume.surface_revision + 1;
    next.search_rank = 2;
    next.surface_origin = "orthogonal_profile";
    next.parent_surface_ref = resume.state_ref;
    next.surface_trials = std::move(trials);
    return common_flydelta_bootstrap_zoom_state_validate(next, error);
}

bool daemon_flydelta_run_post_bootstrap_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_plan & plan,
        common_flydelta_search_pipeline_result & output,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    const int32_t layer_index = static_cast<int32_t>(plan.continuation.region.anchor_layer_index);
    std::vector<common_flydelta_direction_candidate> candidates;
    const size_t max_rank = plan.phase == common_flydelta_experiment_phase::shallow_controls
        ? 2 : 4;
    if (!daemon_flydelta_resolve_evidence_direction_candidates(
            provider, job, layer_index, 2, candidates, error)) return false;
    if (candidates.size() > max_rank) candidates.resize(max_rank);
    common_flydelta_low_rank_basis basis;
    if (!common_flydelta_build_low_rank_basis(
            candidates.front().values.size(), max_rank, candidates, basis, error)) return false;

    common_flydelta_coefficient_search_config coefficient_config;
    coefficient_config.max_candidates = plan.budget.max_coefficient_trials == 0
        ? 16 : std::min<size_t>(16, plan.budget.max_coefficient_trials);
    coefficient_config.strategy = plan.run_tfo_lite
        ? common_flydelta_coefficient_search_strategy::tfo_lite
        : common_flydelta_coefficient_search_strategy::coordinate;
    const auto run_mode = [provider, &job, &fixture](
            const common_flydelta_experiment_fixture &,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<float> & coefficients,
            const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & has_geometry, std::string & runner_error,
            const bool full_execution) {
        std::vector<common_flydelta_counterfactual_trial> trials;
        std::vector<common_flydelta_decision_margin> margins;
        std::vector<common_flydelta_representation_diagnostics> geometries;
        std::vector<bool> geometry_flags;
        if (!daemon_flydelta_run_coefficient_arm_batch(
                provider, job, fixture, current_basis, {coefficients}, apply_overlay,
                trials, margins, geometries, geometry_flags, runner_error,
                full_execution, full_execution ? "deep-full-single" : "deep-diagnostic-single")) {
            return false;
        }
        trial = std::move(trials.front());
        margin = std::move(margins.front());
        geometry = std::move(geometries.front());
        has_geometry = geometry_flags.front();
        return true;
    };
    const auto run_batch_mode = [provider, &job, &fixture] (
            const common_flydelta_experiment_fixture &,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_representation_diagnostics> & geometries,
            std::vector<bool> & geometry_flags, std::string & runner_error,
            const bool full_execution) {
        return daemon_flydelta_run_coefficient_arm_batch(
            provider, job, fixture, current_basis, coefficients, true,
            trials, margins, geometries, geometry_flags, runner_error,
            full_execution, full_execution ? "deep-full-batch" : "deep-diagnostic-batch");
    };
    const auto run_diagnostic = [run_mode](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<float> & coefficients,
            const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & has_geometry, std::string & runner_error) {
        return run_mode(current_fixture, current_basis, coefficients, apply_overlay,
            trial, margin, geometry, has_geometry, runner_error, false);
    };
    const auto run_diagnostic_batch = [run_batch_mode](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_representation_diagnostics> & geometries,
            std::vector<bool> & geometry_flags, std::string & runner_error) {
        return run_batch_mode(current_fixture, current_basis, coefficients, trials,
            margins, geometries, geometry_flags, runner_error, false);
    };
    const auto run_full = [run_mode](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<float> & coefficients,
            const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & has_geometry, std::string & runner_error) {
        return run_mode(current_fixture, current_basis, coefficients, apply_overlay,
            trial, margin, geometry, has_geometry, runner_error, true);
    };
    const auto run_full_batch = [run_batch_mode](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_representation_diagnostics> & geometries,
            std::vector<bool> & geometry_flags, std::string & runner_error) {
        return run_batch_mode(current_fixture, current_basis, coefficients, trials,
            margins, geometries, geometry_flags, runner_error, true);
    };
    std::vector<common_flydelta_coefficient_trial> trials;
    common_flydelta_coefficient_selection selection;
    if (plan.phase == common_flydelta_experiment_phase::deep_controls) {
        common_flydelta_deep_search_config deep_config;
        deep_config.max_rank = std::min<size_t>(max_rank, 4);
        deep_config.max_directions = std::min<size_t>(candidates.size(), 4);
        deep_config.full_generation_top_k = plan.budget.full_generation_top_k == 0
            ? 3 : std::min<size_t>(3, plan.budget.full_generation_top_k);
        deep_config.coefficients = coefficient_config;
        std::vector<common_flydelta_deep_search_direction> deep_directions;
        for (const auto & candidate : candidates) {
            common_flydelta_deep_search_direction value;
            value.direction = candidate;
            value.decision_score_available = true;
            value.decision_score = candidate.median_alignment;
            deep_directions.push_back(std::move(value));
        }
        common_flydelta_deep_search_result deep_result;
        if (!common_flydelta_run_deep_search_batched(
                fixture, deep_config, deep_directions, run_diagnostic, run_diagnostic_batch,
                run_full, run_full_batch, deep_result, error)) return false;
        basis = deep_result.basis;
        trials = deep_result.coefficient_trials;
        selection = deep_result.coefficient_selection;
    } else if (!common_flydelta_run_low_rank_coefficient_search_batched_staged(
            fixture, basis, coefficient_config,
            run_diagnostic, run_diagnostic_batch,
            run_full, run_full_batch,
            plan.budget.full_generation_top_k == 0
                ? 3 : std::min<size_t>(3, plan.budget.full_generation_top_k),
            trials, selection, error)) {
        return false;
    }

    common_flydelta_search_pipeline_direction_result direction_result;
    direction_result.direction = candidates.front();
    direction_result.direction.values = basis.vectors.front();
    direction_result.direction.layer_index = basis.layer_index;
    direction_result.direction.origin = "runtime_post_bootstrap_basis";
    direction_result.region_trials.reserve(trials.size());
    for (const auto & coefficient_trial : trials) {
        common_flydelta_intervention_region_trial trial;
        trial.candidate.layer_indices = {static_cast<uint32_t>(basis.layer_index)};
        trial.candidate.anchor_layer_index = static_cast<uint32_t>(basis.layer_index);
        trial.candidate.total_scale = coefficient_trial.executed_strength;
        trial.candidate.per_layer_scale = coefficient_trial.executed_strength;
        trial.requested_total_scale = coefficient_trial.requested_strength;
        trial.executed_total_scale = coefficient_trial.executed_strength;
        trial.outcome = coefficient_trial.outcome;
        trial.quality_delta = coefficient_trial.quality_delta;
        trial.margin = coefficient_trial.margin;
        trial.executed = coefficient_trial.executed;
        trial.verifier_known = coefficient_trial.verifier_known;
        trial.geometry_available = coefficient_trial.geometry_available;
        trial.geometry = coefficient_trial.geometry;
        trial.search_score = coefficient_trial.search_fitness;
        trial.promising = coefficient_trial.search_fitness > 0.0f;
        trial.safe_to_continue = !coefficient_trial.geometry_available ||
            (coefficient_trial.geometry.shift_norm <= 1.0f &&
             coefficient_trial.geometry.leakage <= 1.0f);
        trial.evidence_ref = "flydelta://runtime/coefficient-trial/" + job.id;
        if (!common_flydelta_intervention_region_trial_validate(trial, error)) return false;
        direction_result.region_trials.push_back(std::move(trial));
    }
    direction_result.region_selection.selected = selection.selected;
    direction_result.region_selection.trial_index = selection.trial_index;
    direction_result.region_selection.score = selection.score;
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.search_status = trials.empty()
        ? common_flydelta_search_status::no_useful_utility
        : common_flydelta_search_status::candidate_available;
    output.selection.selected = selection.selected;
    output.selection.intervention_region = true;
    output.selection.direction_index = 0;
    output.selection.region_trial_index = selection.trial_index;
    output.selection.score = selection.score;
    return true;
}

// Representation augmentation is a host-owned continuation over opaque
// donor resources.  The resource contains references to two already
// host-approved contexts: the target context and the target-plus-donor
// context.  No prompt or activation data is synthesized by this binding.
bool daemon_flydelta_run_donor_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_capture_manifest> & manifests,
        std::string & error) {
    error.clear();
    manifests.clear();
    const auto scoped_provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    for (const auto & reference : job.capture_candidate_ids) {
        json value;
        if (!daemon_flydelta_read_json(*scoped_provider, reference, value, error)) return false;
        common_flydelta_capture_candidate candidate;
        try {
            candidate.schema_version = value.value("schema_version", 0);
            candidate.id = value.value("id", reference);
            candidate.transaction_id = value.value("transaction_id", reference);
            const auto source = common_adaptation_evidence_source_from_name(
                value.value("source", ""));
            if (!source) {
                error = "FlyDelta donor capture candidate source is invalid";
                return false;
            }
            candidate.source = *source;
            const auto scope = value.value("scope", json::object());
            candidate.scope.namespace_id = scope.value("namespace_id", "");
            candidate.scope.project_id = scope.value("project_id", "");
            candidate.scope.session_id = scope.value("session_id", "");
            candidate.scope.turn_id = scope.value("turn_id", "");
            candidate.behavior_key = value.value("behavior_key", job.seed.behavior_key);
            candidate.task_fingerprint = value.value("task_fingerprint", job.seed.task_fingerprint);
            candidate.baseline_ref = value.value("baseline_ref", "");
            candidate.candidate_ref = value.value("candidate_ref", "");
            candidate.verifier_ref = value.value("verifier_ref", "");
            candidate.evidence_refs = value.value("evidence_refs", std::vector<std::string>{});
            candidate.model_profile_fingerprint = value.value(
                "model_profile_fingerprint", scoped_provider->model_profile_fingerprint);
            candidate.capture_layout_revision = value.value(
                "capture_layout_revision", scoped_provider->capture_layout_revision);
            candidate.candidate_ready = value.value("candidate_ready", true);
        } catch (const std::exception & exception) {
            error = std::string("FlyDelta donor capture candidate is malformed: ") + exception.what();
            return false;
        }
        if (!common_flydelta_capture_candidate_validate(candidate, error)) return false;
        const std::string baseline_ref = candidate.baseline_ref.empty()
            ? job.seed.baseline_ref : candidate.baseline_ref;
        const std::string candidate_ref = candidate.candidate_ref.empty()
            ? job.seed.candidate_ref : candidate.candidate_ref;
        const std::string fixture_ref = candidate.verifier_ref.empty()
            ? job.seed.verifier_ref : candidate.verifier_ref;
        if (baseline_ref.empty() || candidate_ref.empty() || fixture_ref.empty()) {
            error = "FlyDelta donor capture candidate must provide baseline_ref, candidate_ref and fixture_ref";
            return false;
        }
        common_flydelta_arm_batch_request batch;
        batch.batch_id = job.id + ":donor-capture:" + candidate.id;
        batch.wave_id = "donor-capture";
        for (size_t index = 0; index < 2; ++index) {
            common_flydelta_arm_request arm;
            arm.job_id = job.id;
            arm.wave_id = batch.wave_id;
            arm.proposal_index = index;
            arm.arm_id = batch.batch_id + (index == 0 ? ":baseline" : ":candidate");
            arm.context_ref = index == 0 ? baseline_ref : candidate_ref;
            arm.fixture_ref = fixture_ref;
            arm.intervention_ref = job.seed.candidate_ref;
            arm.fresh_context = true;
            arm.request_capture = true;
            arm.request_generation = true;
            arm.max_capture_bytes = 4U * 1024U * 1024U;
            arm.max_generated_tokens = 1;
            if (!common_flydelta_arm_request_validate(arm, error)) return false;
            batch.arms.push_back(std::move(arm));
        }
        common_flydelta_arm_batch_result result;
        if (!daemon_flydelta_execute_batch(scoped_provider, batch, result, error) ||
                result.arms.size() != batch.arms.size() ||
                result.arms[0].capture_ref.empty() || result.arms[1].capture_ref.empty()) {
            if (error.empty()) error = "FlyDelta donor capture did not return both manifests";
            return false;
        }
        common_flydelta_capture_manifest manifest;
        manifest.id = candidate.id + "/manifest";
        manifest.observation_id = candidate.transaction_id;
        manifest.source = candidate.source;
        manifest.behavior_key = candidate.behavior_key.empty()
            ? job.seed.behavior_key : candidate.behavior_key;
        manifest.model_profile_fingerprint = candidate.model_profile_fingerprint;
        manifest.template_fingerprint = job.seed.template_fingerprint.empty()
            ? scoped_provider->model_profile_fingerprint + ":template"
            : job.seed.template_fingerprint;
        manifest.execution_context_fingerprint = job.seed.execution_context_fingerprint;
        manifest.positive_execution_ref = candidate_ref;
        manifest.negative_execution_ref = baseline_ref;
        manifest.capture_layout_revision = candidate.capture_layout_revision;
        const std::string evidence_identity = baseline_ref + "\n" + candidate_ref + "\n" +
            candidate.id;
        manifest.evidence_hash = "sha256:" + hash_sha256_hex(
            evidence_identity.data(), evidence_identity.size());
        manifest.redaction_attested = true;
        manifest.captured_bytes = 1;
        if (!common_flydelta_capture_manifest_validate(
                manifest, 4U * 1024U * 1024U, error)) return false;
        std::shared_ptr<const common_flydelta_hidden_state_capture> baseline_capture;
        std::shared_ptr<const common_flydelta_hidden_state_capture> candidate_capture;
        if (!daemon_flydelta_capture_from_reference(
                scoped_provider, result.arms[0].capture_ref, baseline_capture, error) ||
                !daemon_flydelta_capture_from_reference(
                scoped_provider, result.arms[1].capture_ref, candidate_capture, error)) return false;
        manifest.captured_bytes = (baseline_capture->values.size() +
            candidate_capture->values.size()) * sizeof(float);
        if (!common_flydelta_capture_manifest_validate(
                manifest, 4U * 1024U * 1024U, error)) return false;
        std::string manifest_ref;
        if (!daemon_flydelta_put_json_resource(
                scoped_provider, "flydelta-donor-manifest-" +
                    hash_sha256_hex(manifest.id.data(), manifest.id.size()).substr(0, 24) + ".json",
                "flydelta_donor_capture_manifest", candidate.id,
                common_flydelta_capture_manifest_to_json(manifest), manifest_ref, error)) return false;
        manifests.push_back(std::move(manifest));
    }
    return !manifests.empty();
}

bool daemon_flydelta_run_representation_augmentation(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_search_pipeline_config & pipeline_config,
        const common_flydelta_representation_augmentation_state * resume,
        common_flydelta_search_pipeline_result & result,
        common_flydelta_representation_augmentation_state & next,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (resume == nullptr) {
        error = "FlyDelta representation augmentation requires a resumed typed state";
        return false;
    }
    next = *resume;
    daemon_flydelta_augmentation_material material;
    if (!daemon_flydelta_load_augmentation_material(
            provider, resume->state_ref, material, error)) return false;
    if (material.parent_direction_ref.empty()) material.parent_direction_ref = job.seed.candidate_ref;

    const auto seed_result = [&]() {
        common_flydelta_search_pipeline_result value;
        std::string seed_error;
        if (!daemon_flydelta_augmentation_seed_result(provider, job, next, value, seed_error)) {
            error = std::move(seed_error);
        }
        return value;
    };

    switch (resume->phase) {
        case common_flydelta_representation_augmentation_phase::discover_donors:
            if (next.donor_candidate_refs.empty()) {
                error = "FlyDelta representation augmentation has no host-owned donor candidates";
                return false;
            }
            next.phase = common_flydelta_representation_augmentation_phase::qualify_donor;
            next.next_action = "run_representation_augmentation";
            result = seed_result();
            return error.empty();
        case common_flydelta_representation_augmentation_phase::qualify_donor:
            for (const auto & reference : next.donor_candidate_refs) {
                if (std::none_of(material.donors.begin(), material.donors.end(),
                        [&](const auto & donor) { return donor.candidate.donor_id == reference; })) {
                    daemon_flydelta_augmentation_donor_material donor;
                    if (!daemon_flydelta_parse_augmentation_donor(
                            provider, reference, next, job, donor, error)) return false;
                    material.donors.push_back(std::move(donor));
                }
            }
            next.phase = common_flydelta_representation_augmentation_phase::capture_donor;
            next.next_action = "run_representation_augmentation";
            if (!daemon_flydelta_persist_augmentation_material(provider, material, error)) return false;
            result = seed_result();
            return error.empty();
        case common_flydelta_representation_augmentation_phase::capture_donor: {
            auto donor = std::find_if(material.donors.begin(), material.donors.end(),
                [](const auto & value) { return value.donor_capture_ref.empty(); });
            if (donor == material.donors.end()) {
                next.phase = common_flydelta_representation_augmentation_phase::build_latent_delta;
                next.next_action = "run_representation_augmentation";
            } else {
                if (!daemon_flydelta_capture_augmentation_pair(provider, job, *donor, error)) return false;
                common_flydelta_representation_donor_qualification qualified;
                if (!common_flydelta_qualify_representation_donor(
                        donor->candidate, donor->qualification,
                        common_flydelta_representation_augmentation_config{}.minimum_margin_gain,
                        common_flydelta_representation_augmentation_config{}.max_leakage,
                        common_flydelta_representation_augmentation_config{}.max_shift_norm,
                        qualified, error)) return false;
                donor->qualification = qualified;
                if (qualified.search_qualified) {
                    if (std::find(next.qualified_donor_refs.begin(), next.qualified_donor_refs.end(),
                            donor->candidate.donor_id) == next.qualified_donor_refs.end()) {
                        next.qualified_donor_refs.push_back(donor->candidate.donor_id);
                    }
                    next.phase = common_flydelta_representation_augmentation_phase::build_latent_delta;
                } else {
                    next.phase = common_flydelta_representation_augmentation_phase::capture_donor;
                }
                next.next_action = "run_representation_augmentation";
            }
            if (!daemon_flydelta_persist_augmentation_material(provider, material, error)) return false;
            result = seed_result();
            return error.empty();
        }
        case common_flydelta_representation_augmentation_phase::build_latent_delta: {
            auto donor = std::find_if(material.donors.begin(), material.donors.end(),
                [&](const auto & value) {
                    return value.qualification.search_qualified &&
                        !value.target_capture_ref.empty() && !value.donor_capture_ref.empty() &&
                        std::find(next.evaluated_donor_refs.begin(), next.evaluated_donor_refs.end(),
                            value.candidate.donor_id) == next.evaluated_donor_refs.end();
                });
            if (donor == material.donors.end()) {
                next.phase = common_flydelta_representation_augmentation_phase::done;
                next.remaining_budget = 0;
                next.next_action = "retain";
                result = seed_result();
                return error.empty();
            }
            std::shared_ptr<const common_flydelta_hidden_state_capture> target;
            std::shared_ptr<const common_flydelta_hidden_state_capture> donor_capture;
            if (!daemon_flydelta_capture_from_reference(
                    provider, donor->target_capture_ref, target, error) ||
                    !daemon_flydelta_capture_from_reference(
                    provider, donor->donor_capture_ref, donor_capture, error)) return false;
            std::vector<float> target_values;
            std::vector<float> donor_values;
            if (!daemon_flydelta_capture_layer(*target, donor->layer_index, target_values, error) ||
                    !daemon_flydelta_capture_layer(*donor_capture, donor->layer_index, donor_values, error)) return false;
            std::vector<common_flydelta_basis_direction> parent_directions;
            if (!daemon_flydelta_parse_directions(
                    *provider, material.parent_direction_ref, parent_directions, error)) return false;
            const auto parent = std::find_if(parent_directions.begin(), parent_directions.end(),
                [&](const auto & value) { return value.layer_index == donor->layer_index; });
            if (parent == parent_directions.end()) {
                error = "FlyDelta augmentation donor layer is absent from the parent surface";
                return false;
            }
            common_flydelta_representation_latent_delta latent;
            if (!common_flydelta_build_residualized_latent_delta(
                    donor->candidate.donor_id, donor->layer_index, donor_values, target_values,
                    {parent->values}, common_flydelta_representation_augmentation_config{}.minimum_residual_norm,
                    latent, error)) return false;
            donor->latent = latent;
            if (!latent.available || !common_flydelta_apply_representation_augmentation(next, latent, error)) {
                next.phase = common_flydelta_representation_augmentation_phase::done;
                next.remaining_budget = 0;
                next.next_action = "retain";
            } else {
                next.best_donor_ref = donor->candidate.donor_id;
                next.phase = common_flydelta_representation_augmentation_phase::run_controls;
                next.next_action = "run_representation_augmentation";
            }
            if (!daemon_flydelta_persist_augmentation_material(provider, material, error)) return false;
            result = seed_result();
            return error.empty();
        }
        case common_flydelta_representation_augmentation_phase::run_controls: {
            auto donor = std::find_if(material.donors.begin(), material.donors.end(),
                [&](const auto & value) { return value.candidate.donor_id == next.best_donor_ref; });
            if (donor == material.donors.end() || !donor->latent.available) {
                error = "FlyDelta augmentation controls have no persisted latent delta";
                return false;
            }
            std::vector<common_flydelta_basis_direction> parent_directions;
            if (!daemon_flydelta_parse_directions(
                    *provider, material.parent_direction_ref, parent_directions, error)) return false;
            const auto parent = std::find_if(parent_directions.begin(), parent_directions.end(),
                [&](const auto & value) { return value.layer_index == donor->latent.layer_index; });
            if (parent == parent_directions.end()) {
                error = "FlyDelta augmentation controls lack the parent direction";
                return false;
            }
            common_flydelta_direction_candidate parent_candidate;
            parent_candidate.layer_index = parent->layer_index;
            parent_candidate.values = parent->values;
            parent_candidate.origin = "runtime_augmentation_parent";
            parent_candidate.experimental_only = true;
            common_flydelta_direction_candidate latent_candidate;
            latent_candidate.layer_index = donor->latent.layer_index;
            latent_candidate.values = donor->latent.values;
            latent_candidate.origin = "runtime_augmentation_residual";
            latent_candidate.experimental_only = true;
            if (!common_flydelta_direction_candidate_validate(
                    parent_candidate, provider->model_n_embd, error) ||
                    !common_flydelta_direction_candidate_validate(
                    latent_candidate, provider->model_n_embd, error)) return false;
            common_flydelta_low_rank_basis basis;
            if (!common_flydelta_build_low_rank_basis(
                    provider->model_n_embd, 2, {parent_candidate, latent_candidate}, basis, error)) return false;
            common_flydelta_experiment_fixture fixture;
            if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
            std::vector<std::vector<float>> coefficients;
            if (!common_flydelta_propose_representation_augmentation_controls(
                    common_flydelta_representation_augmentation_config{}, coefficients, error)) return false;
            std::vector<common_flydelta_counterfactual_trial> baseline_trials;
            std::vector<common_flydelta_decision_margin> baseline_margins;
            std::vector<common_flydelta_representation_diagnostics> baseline_geometry;
            std::vector<bool> baseline_geometry_available;
            if (!daemon_flydelta_run_coefficient_arm_batch(
                    provider, job, fixture, basis, {{0.0f, 0.0f}}, false,
                    baseline_trials, baseline_margins, baseline_geometry,
                    baseline_geometry_available, error, false, "augmentation-diagnostics")) return false;
            std::vector<common_flydelta_counterfactual_trial> trials;
            std::vector<common_flydelta_decision_margin> margins;
            std::vector<common_flydelta_representation_diagnostics> geometries;
            std::vector<bool> geometry_available;
            if (!daemon_flydelta_run_coefficient_arm_batch(
                    provider, job, fixture, basis, coefficients, true,
                    trials, margins, geometries, geometry_available, error,
                    false, "augmentation-diagnostics")) return false;
            common_flydelta_search_pipeline_direction_result direction;
            direction.direction.kind = common_flydelta_direction_kind::raw_repair;
            direction.direction.layer_index = basis.layer_index;
            direction.direction.values = basis.vectors.front();
            direction.direction.origin = "runtime_representation_augmentation";
            direction.direction.extraction_id = donor->candidate.donor_id;
            direction.direction.source_samples = 2;
            direction.direction.retained_samples = 2;
            direction.direction.experimental_only = true;
            bool positive = false;
            float best_gain = next.best_margin_gain;
            size_t best_index = 0;
            for (size_t index = 0; index < trials.size(); ++index) {
                auto geometry = geometries[index];
                if (geometry_available[index]) {
                    geometry.layer_index = static_cast<uint32_t>(basis.layer_index);
                    geometry.schema_version = 1;
                }
                common_flydelta_intervention_region_trial trial;
                trial.candidate.layer_indices = {static_cast<uint32_t>(basis.layer_index)};
                trial.candidate.anchor_layer_index = static_cast<uint32_t>(basis.layer_index);
                trial.candidate.total_scale = 1.0f;
                trial.candidate.per_layer_scale = trial.candidate.total_scale;
                trial.requested_total_scale = trial.candidate.total_scale;
                trial.executed_total_scale = trial.candidate.total_scale;
                trial.outcome = trials[index].passed
                    ? common_flydelta_counterfactual_outcome::helped
                    : common_flydelta_counterfactual_outcome::unknown;
                trial.quality_delta = trials[index].quality;
                trial.margin = margins[index];
                trial.margin_comparison.available = baseline_margins.front().available && margins[index].available;
                trial.margin_comparison.baseline = baseline_margins.front();
                trial.margin_comparison.candidate = margins[index];
                trial.executed = trials[index].executed;
                trial.verifier_known = trials[index].verifier_known;
                trial.geometry_available = geometry_available[index];
                trial.geometry = geometry;
                trial.safe_to_continue = !trial.geometry_available ||
                    (geometry.leakage <= 1.0f && geometry.shift_norm <= 1.0f);
                trial.search_score = trial.margin_comparison.available
                    ? trial.margin_comparison.normalized_delta() : 0.0f;
                trial.promising = trial.safe_to_continue && trial.search_score > 0.0f;
                if (trial.promising && (!positive || trial.search_score > best_gain)) {
                    positive = true;
                    best_gain = trial.search_score;
                    best_index = index;
                }
                if (!common_flydelta_intervention_region_trial_validate(trial, error)) return false;
                direction.region_trials.push_back(std::move(trial));
            }
            // Diagnostics establish the frontier using compact geometry and
            // teacher-forced margin only.  Spend generation/verifier budget
            // on the selected control after ranking, through the same batch
            // host and the same immutable arm identity contract.
            if (positive) {
                std::vector<common_flydelta_counterfactual_trial> frontier_trials;
                std::vector<common_flydelta_decision_margin> frontier_margins;
                std::vector<common_flydelta_representation_diagnostics> frontier_geometry;
                std::vector<bool> frontier_geometry_available;
                if (!daemon_flydelta_run_coefficient_arm_batch(
                        provider, job, fixture, basis, {coefficients[best_index]}, true,
                        frontier_trials, frontier_margins, frontier_geometry,
                        frontier_geometry_available, error, true, "augmentation-frontier")) {
                    return false;
                }
                if (frontier_trials.size() != 1 || frontier_margins.size() != 1 ||
                        frontier_geometry.size() != 1 || frontier_geometry_available.size() != 1) {
                    error = "FlyDelta augmentation frontier returned an incomplete arm";
                    return false;
                }
                auto & selected_trial = direction.region_trials[best_index];
                selected_trial.outcome = frontier_trials.front().passed
                    ? common_flydelta_counterfactual_outcome::helped
                    : frontier_trials.front().verifier_known
                        ? frontier_trials.front().quality > 0.0f
                            ? common_flydelta_counterfactual_outcome::neutral
                            : common_flydelta_counterfactual_outcome::harmed
                        : common_flydelta_counterfactual_outcome::unknown;
                selected_trial.quality_delta = frontier_trials.front().quality;
                selected_trial.margin = frontier_margins.front();
                selected_trial.executed = frontier_trials.front().executed;
                selected_trial.verifier_known = frontier_trials.front().verifier_known;
                selected_trial.geometry_available = frontier_geometry_available.front();
                selected_trial.geometry = frontier_geometry.front();
                selected_trial.evidence_ref = frontier_trials.front().evidence_ref;
                selected_trial.safe_to_continue = selected_trial.outcome !=
                    common_flydelta_counterfactual_outcome::harmed &&
                    selected_trial.safe_to_continue;
                selected_trial.margin_comparison.candidate = frontier_margins.front();
                selected_trial.search_score = selected_trial.margin_comparison.available
                    ? selected_trial.margin_comparison.normalized_delta() : selected_trial.search_score;
                selected_trial.promising = selected_trial.safe_to_continue &&
                    selected_trial.search_score > 0.0f;
                if (!common_flydelta_intervention_region_trial_validate(selected_trial, error)) {
                    return false;
                }
            }
            result = {};
            result.directions.push_back(std::move(direction));
            result.search_status = positive
                ? common_flydelta_search_status::candidate_available
                : common_flydelta_search_status::no_useful_utility;
            next.best_margin_gain = positive ? best_gain : next.best_margin_gain;
            if (positive) {
                next.phase = common_flydelta_representation_augmentation_phase::localize_surface;
                next.next_action = "recenter_augmented_surface";
                next.remaining_budget = next.remaining_budget > coefficients.size()
                    ? next.remaining_budget - coefficients.size() : 1;
                // The selected control is a normal immutable direction resource;
                // the next slice reuses Whirlpool on that localized surface.
                const auto selected = coefficients[best_index];
                if (!daemon_flydelta_register_composed_direction(
                        provider, material.parent_direction_ref, basis, selected,
                        material.selected_control_ref, error)) return false;
            } else {
                next.phase = common_flydelta_representation_augmentation_phase::done;
                next.remaining_budget = 0;
                next.next_action = "retain";
            }
            if (!daemon_flydelta_persist_augmentation_material(provider, material, error)) return false;
            return true;
        }
        case common_flydelta_representation_augmentation_phase::localize_surface: {
            if (material.selected_control_ref.empty()) {
                error = "FlyDelta augmentation recenter has no selected control surface";
                return false;
            }
            auto localized_job = job;
            localized_job.seed.candidate_ref = material.selected_control_ref;
            if (!daemon_flydelta_run_search_pipeline(
                    provider, localized_job, pipeline_config, result, error)) return false;
            const bool useful = result.selection.selected && result.selection.score > 0.0f;
            next.phase = common_flydelta_representation_augmentation_phase::done;
            next.remaining_budget = 0;
            next.next_action = useful ? "allow_tfo_lite" : "retain";
            return true;
        }
        case common_flydelta_representation_augmentation_phase::full_generation:
        case common_flydelta_representation_augmentation_phase::verify:
            next.phase = common_flydelta_representation_augmentation_phase::done;
            next.remaining_budget = 0;
            next.next_action = "retain";
            result = seed_result();
            return error.empty();
        case common_flydelta_representation_augmentation_phase::done:
            next.next_action = "retain";
            result = seed_result();
            return error.empty();
    }
    error = "FlyDelta representation augmentation phase is unsupported by the daemon binding";
    return false;
}

bool daemon_flydelta_run_counterfactual(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_counterfactual_report> & reports,
        std::string & error) {
    error.clear();
    reports.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    if (job.seed.baseline_ref.empty() || job.seed.candidate_ref.empty() ||
            job.alpha_search.candidates.empty()) {
        error = "FlyDelta counterfactual requires baseline, candidate and alpha references";
        return false;
    }

    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref,
            directions, error) || directions.empty()) return false;
    // Counterfactual jobs are the existing rank-one refinement contract. A
    // multi-direction candidate must go through its existing search phase;
    // silently selecting one direction here would change experiment meaning.
    if (directions.size() != 1) {
        error = "FlyDelta counterfactual requires one direction; use search_pipeline for a basis";
        return false;
    }
    const uint32_t layer = static_cast<uint32_t>(directions.front().layer_index);

    for (const float alpha : job.alpha_search.candidates) {
        const std::string alpha_id = job.seed.candidate_ref + ":alpha:" +
            std::to_string(alpha);
        const common_flydelta_counterfactual_runner runner =
            [provider, &job, layer, alpha](
                    const common_flydelta_experiment_fixture &,
                    bool apply_overlay,
                    common_flydelta_counterfactual_trial & trial,
                    std::string & runner_error) {
                common_flydelta_arm_batch_request batch;
                batch.batch_id = job.id + ":counterfactual:" + std::to_string(alpha);
                batch.wave_id = "counterfactual";
                common_flydelta_arm_request arm;
                arm.job_id = job.id;
                arm.wave_id = batch.wave_id;
                arm.proposal_index = apply_overlay ? 1 : 0;
                arm.arm_id = batch.batch_id + (apply_overlay ? ":candidate" : ":baseline");
                arm.context_ref = job.seed.baseline_ref;
                arm.fixture_ref = job.seed.verifier_ref;
                arm.intervention_ref = job.seed.candidate_ref;
                arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
                arm.layer_indices = {layer};
                arm.coefficients = {1.0f};
                arm.alpha = apply_overlay ? alpha : 0.0f;
                arm.apply_overlay = apply_overlay;
                arm.fresh_context = true;
                arm.request_generation = true;
                arm.request_host_verification = true;
                arm.max_capture_bytes = 4U * 1024U * 1024U;
                const size_t runtime_token_limit = provider->n_predict > 0
                    ? static_cast<size_t>(provider->n_predict) : 256U;
                arm.max_generated_tokens = job.kind ==
                        common_flydelta_experiment_job_kind::evaluation
                    ? std::min(runtime_token_limit, job.evaluation_limits.max_generated_tokens)
                    : runtime_token_limit;
                if (!common_flydelta_arm_request_validate(arm, runner_error)) return false;
                batch.arms.push_back(std::move(arm));
                common_flydelta_arm_batch_result executed;
                if (!daemon_flydelta_execute_batch(provider, batch, executed, runner_error) ||
                        executed.arms.size() != 1) return false;
                const auto & arm_result = executed.arms.front();
                trial = {};
                trial.executed = arm_result.executed;
                trial.verifier_known = arm_result.verifier_known;
                // finalize_arm deliberately reports a single semantic pass as
                // NEUTRAL. At this layer that pass is the trial predicate;
                // common_flydelta_classify_counterfactual() then derives the
                // relational HELPED outcome from baseline versus candidate.
                trial.passed = daemon_flydelta_arm_semantic_passed(arm_result);
                trial.quality = trial.passed ? 1.0f : 0.0f;
                trial.overlay_applied = apply_overlay;
                trial.intervention_count = apply_overlay ? 1 : 0;
                trial.evidence_ref = arm_result.generation_ref;
                return common_flydelta_counterfactual_trial_validate(trial, runner_error);
            };
        common_flydelta_counterfactual_report report;
        const std::string report_candidate_id = job.evaluation_candidate_id.empty()
            ? alpha_id : job.evaluation_candidate_id;
        if (!common_flydelta_run_counterfactual(
                job.id, report_candidate_id, job.seed.baseline_ref,
                report_candidate_id, fixture, runner, report, error)) return false;
        reports.push_back(std::move(report));
    }
    return !reports.empty();
}

bool daemon_flydelta_run_evaluation(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        common_flydelta_evaluation_report & report,
        std::vector<common_flydelta_evaluation_fixture_result> & fixture_results,
        std::string & error) {
    error.clear();
    report = {};
    fixture_results.clear();
    const auto scoped_provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (!common_flydelta_evaluation_limits_validate(job.evaluation_limits, error)) return false;
    json candidate_artifact;
    if (!daemon_flydelta_read_json_bounded(
            *scoped_provider, job.seed.candidate_ref, 4U * 1024U * 1024U,
            candidate_artifact, error) || candidate_artifact.value("kind", "") != "flydelta") {
        error = "FlyDelta evaluation candidate reference must resolve to a flydelta artifact";
        return false;
    }
    common_flydelta_artifact artifact;
    if (!common_flydelta_artifact_from_json(
            candidate_artifact.dump(), 1U << 20, 4U * 1024U * 1024U,
            artifact, error) || artifact.id != job.evaluation_candidate_id ||
            (!job.seed.evidence_ref.empty() && artifact.content_hash != job.seed.evidence_ref)) {
        if (error.empty()) error = "FlyDelta evaluation candidate artifact identity is incompatible";
        return false;
    }
    json suite;
    if (!daemon_flydelta_read_json(*scoped_provider, job.evaluation_suite_ref, suite, error)) return false;
    if (suite.value("kind", "") != "flydelta_evaluation_suite" ||
            suite.value("revision", "") != job.evaluation_revision ||
            !suite.contains("fixtures") || !suite["fixtures"].is_array() ||
            suite["fixtures"].empty() ||
            suite["fixtures"].size() > job.evaluation_limits.max_fixtures) {
        error = "FlyDelta evaluation suite is missing a matching revision or bounded fixtures";
        return false;
    }
    report.schema_version = 1;
    report.revision_id = job.evaluation_revision;
    report.candidate_id = job.evaluation_candidate_id;
    report.baseline_profile_id = job.seed.model_profile_fingerprint + ":baseline";
    report.candidate_profile_id = job.evaluation_candidate_id;
    report.test_suite_revision = job.evaluation_revision;
    report.intended_behavior_passed = true;
    report.retention_passed = true;
    report.agent_regression_passed = true;
    std::set<common_flydelta_evaluation_suite_kind> seen_kinds;
    size_t model_calls = 0;
    std::set<std::string> seen_fixture_refs;
    for (const auto & item : suite["fixtures"]) {
        if (!item.is_object()) {
            error = "FlyDelta evaluation suite contains a malformed fixture entry";
            return false;
        }
        common_flydelta_evaluation_suite_kind suite_kind;
        if (!common_flydelta_evaluation_suite_kind_from_name(
                item.value("suite_kind", ""), suite_kind)) {
            error = "FlyDelta evaluation fixture has an unknown suite kind";
            return false;
        }
        const std::string fixture_ref = item.value("fixture_ref", "");
        const std::string context_ref = item.value("context_ref", "");
        if (fixture_ref.empty() || context_ref.empty()) {
            error = "FlyDelta evaluation fixture requires fixture_ref and context_ref";
            return false;
        }
        if (!seen_fixture_refs.insert(fixture_ref).second) {
            error = "FlyDelta evaluation suite contains a duplicate fixture reference";
            return false;
        }
        auto fixture_job = job;
        fixture_job.id = job.id + ":fixture:" + std::to_string(fixture_results.size());
        fixture_job.seed.id = fixture_job.id;
        fixture_job.seed.baseline_ref = context_ref;
        fixture_job.seed.verifier_ref = fixture_ref;
        fixture_job.alpha_search.candidates = {1.0f};
        fixture_job.alpha_search.max_candidates = 1;
        std::vector<common_flydelta_counterfactual_report> reports;
        bool completed = false;
        std::string last_error;
        for (size_t attempt = 0; attempt <= job.evaluation_limits.max_retries; ++attempt) {
            if (model_calls + 2 > job.evaluation_limits.max_model_calls) {
                error = "FlyDelta evaluation model-call limit exhausted";
                return false;
            }
            reports.clear();
            std::string attempt_error;
            if (daemon_flydelta_run_counterfactual(
                    scoped_provider, fixture_job, reports, attempt_error) &&
                    reports.size() == 1) {
                completed = true;
                model_calls += 2;
                break;
            }
            model_calls += 2;
            last_error = attempt_error;
        }
        if (!completed) {
            error = last_error.empty()
                ? "FlyDelta evaluation fixture returned no report" : last_error;
            return false;
        }
        const auto & counterfactual = reports.front();
        common_flydelta_evaluation_fixture_result fixture_result;
        fixture_result.candidate_id = job.evaluation_candidate_id;
        fixture_result.suite_kind = suite_kind;
        fixture_result.fixture_ref = fixture_ref;
        fixture_result.verifier_revision = job.evaluation_revision;
        fixture_result.baseline_known = counterfactual.baseline.verifier_known;
        fixture_result.baseline_passed = counterfactual.baseline.passed;
        fixture_result.candidate_known = counterfactual.candidate.verifier_known;
        fixture_result.candidate_passed = counterfactual.candidate.passed;
        fixture_result.passed = fixture_result.baseline_known &&
            fixture_result.candidate_known && fixture_result.candidate_passed;
        fixture_result.report_ref = job.id + ":fixture:" + std::to_string(fixture_results.size());
        fixture_result.counterfactual = counterfactual;
        if (!common_flydelta_evaluation_fixture_result_validate(fixture_result, error)) return false;
        seen_kinds.insert(suite_kind);
        ++report.evaluated_turns;
        report.baseline_successes += fixture_result.baseline_passed ? 1 : 0;
        report.candidate_successes += fixture_result.candidate_passed ? 1 : 0;
        ++report.candidate_interventions;
        if (fixture_result.baseline_passed && !fixture_result.candidate_passed) ++report.false_interventions;
        if (suite_kind == common_flydelta_evaluation_suite_kind::intended ||
                suite_kind == common_flydelta_evaluation_suite_kind::holdout) {
            report.intended_behavior_passed = report.intended_behavior_passed && fixture_result.passed;
        } else if (suite_kind == common_flydelta_evaluation_suite_kind::retention) {
            report.retention_passed = report.retention_passed && fixture_result.passed;
        } else {
            report.agent_regression_passed = report.agent_regression_passed && fixture_result.passed;
        }
        fixture_results.push_back(std::move(fixture_result));
    }
    if (seen_kinds.size() != 4) {
        error = "FlyDelta evaluation suite must contain intended, holdout, retention and agent_regression";
        return false;
    }
    report.status = report.intended_behavior_passed && report.retention_passed &&
        report.agent_regression_passed ? "passed" : "failed";
    return common_flydelta_evaluation_report_validate(report, error);
}

std::function<bool(
        const std::shared_ptr<common_agent_server_context_host> &,
        common_agent_server_flydelta_binding &,
        std::string &)>
make_daemon_flydelta_resource_binding_factory(
        const daemon_options & options,
        agent_resource_store * resources,
        std::shared_ptr<common_flydelta_teaching_material_runtime> teaching_material_runtime,
        std::shared_ptr<common_learning_lifecycle_store> lifecycle_store) {
    return [options, resources, teaching_material_runtime, lifecycle_store](
            const std::shared_ptr<common_agent_server_context_host> & host,
            common_agent_server_flydelta_binding & binding, std::string & error) {
        error.clear();
        if (!host || resources == nullptr) {
            error = "FlyDelta resource provider requires a resident host and resource store";
            return false;
        }
        const auto * context = host->server().get_llama_context();
        const auto * model = context == nullptr ? nullptr : llama_get_model(context);
        if (model == nullptr || llama_model_n_embd(model) <= 0 || llama_model_n_layer(model) <= 2) {
            error = "FlyDelta resource provider requires a loaded compatible model";
            return false;
        }
        auto provider = std::make_shared<daemon_flydelta_resource_provider>();
        provider->host = host;
        provider->resources = resources;
        // The provider starts with the portable resource-contract defaults.
        // Worker callbacks replace this view with job.seed.scope before
        // resolving job-owned resources.
        provider->authority = {};
        provider->model_profile_fingerprint = options.adaptation_flydelta_model_profile_fingerprint.empty()
            ? "model:" + std::filesystem::path(options.model).filename().string()
            : options.adaptation_flydelta_model_profile_fingerprint;
        provider->capture_layout_revision = options.adaptation_flydelta_capture_layout_revision;
        provider->teaching_material_runtime = teaching_material_runtime;
        provider->n_predict = options.n_predict;
        provider->n_threads = options.n_threads;
        provider->model_n_embd = static_cast<size_t>(llama_model_n_embd(model));
        provider->model_n_layers = static_cast<size_t>(llama_model_n_layer(model));
        provider->lifecycle_store = lifecycle_store;
        if (!provider->lifecycle_store) {
            auto lifecycle = make_agent_learning_lifecycle_store(
                options.adaptation_flydelta_lifecycle_backend,
                options.adaptation_flydelta_lifecycle_path, error);
            if (!lifecycle) return false;
            provider->lifecycle_store = std::shared_ptr<common_learning_lifecycle_store>(
                std::move(lifecycle));
        }

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
        binding.register_evaluator = [provider](common_flydelta_evaluator_config & config,
                common_flydelta_evaluator_callbacks & callbacks, std::string & callback_error) {
            config = {};
            config.pipeline.dimension = provider->model_n_embd;
            config.pipeline.max_directions = 1;
            config.pipeline.whirlpool_max_rounds = 1;
            config.pipeline.whirlpool_probes_per_round = 4;
            config.pipeline.whirlpool_max_trials = 4;
            config.pipeline.region_max_trials = 4;
            config.direction.dimension = provider->model_n_embd;
            config.direction.layer_index = 1;
            config.direction.behavior_key = "runtime/resource";
            config.direction.model_profile_fingerprint = provider->model_profile_fingerprint;
            config.direction.execution_context_fingerprint = "runtime-resource-v0";
            config.direction.capture_layout_revision = provider->capture_layout_revision;
            callbacks = {};
            callbacks.run_evaluation = [provider](
                    const common_flydelta_experiment_job & job,
                    common_flydelta_evaluation_report & report,
                    std::vector<common_flydelta_evaluation_fixture_result> & fixtures,
                    std::string & callback_error) {
                return daemon_flydelta_run_evaluation(
                    provider, job, report, fixtures, callback_error);
            };
            const auto pipeline = config.pipeline;
            callbacks.resolve_behavior_delta = [provider](const std::string & reference,
                    common_flydelta_behavior_delta & delta,
                    common_flydelta_intervention_credit & credit,
                    std::string & resolver_error) {
                return daemon_flydelta_parse_behavior_delta(
                    *provider, reference, delta, credit, resolver_error);
            };
            // ConceptSynthesis produces ordinary experimental direction
            // material. Persist it in the same immutable direction registry
            // used by the normal search path; the scheduler passes the
            // returned reference as the next job's candidate_ref. This seam
            // never approves, activates or promotes the concept direction.
            callbacks.persist_experimental_direction = [provider](
                    const common_flydelta_direction_candidate & candidate,
                    std::string & direction_ref,
                    std::string & persist_error) {
                if (!common_flydelta_direction_candidate_validate(
                        candidate, provider->model_n_embd, persist_error)) return false;
                common_flydelta_basis_direction direction;
                direction.layer_index = candidate.layer_index;
                direction.values = candidate.values;
                const std::string source_ref = candidate.extraction_id.empty()
                    ? "flydelta://concept-graft/anonymous"
                    : candidate.extraction_id;
                return daemon_flydelta_register_composed_directions(
                    provider, source_ref, {std::move(direction)},
                    "flydelta:concept-graft", direction_ref, persist_error);
            };
            callbacks.persist_experimental_direction_for_job = [provider](
                    const common_flydelta_experiment_job & job,
                    const common_flydelta_direction_candidate & candidate,
                    std::string & direction_ref,
                    std::string & persist_error) {
                const auto scoped_provider = daemon_flydelta_provider_for_scope(
                    provider, job.seed.scope);
                if (!scoped_provider) {
                    persist_error = "FlyDelta concept graft requires a scoped provider";
                    return false;
                }
                if (!common_flydelta_direction_candidate_validate(
                        candidate, scoped_provider->model_n_embd, persist_error)) return false;
                common_flydelta_basis_direction direction;
                direction.layer_index = candidate.layer_index;
                direction.values = candidate.values;
                const std::string source_ref = candidate.extraction_id.empty()
                    ? "flydelta://concept-graft/anonymous"
                    : candidate.extraction_id;
                return daemon_flydelta_register_composed_directions(
                    scoped_provider, source_ref, {std::move(direction)},
                    "flydelta:concept-graft", direction_ref, persist_error);
            };
            callbacks.run_concept_capture = [provider](
                    const common_flydelta_experiment_job & job,
                    std::vector<std::string> & trajectory_refs,
                    std::string & capture_error) {
                return daemon_flydelta_run_concept_capture(
                    provider, job, trajectory_refs, capture_error);
            };
            callbacks.run_concept_synthesis = [provider](
                    const common_flydelta_experiment_job & job,
                    std::vector<common_flydelta_concept_candidate> & candidates,
                    std::string & synthesis_error) {
                return daemon_flydelta_run_concept_synthesis(
                    provider, job, candidates, synthesis_error);
            };
            callbacks.run_donor_capture = [provider](
                    const common_flydelta_experiment_job & job,
                    std::vector<common_flydelta_capture_manifest> & manifests,
                    std::string & capture_error) {
                return daemon_flydelta_run_donor_capture(
                    provider, job, manifests, capture_error);
            };
            callbacks.run_representation_augmentation_with_state = [provider, pipeline](
                    const common_flydelta_experiment_job & job,
                    const common_flydelta_representation_augmentation_state * resume,
                    common_flydelta_search_pipeline_result & result,
                    common_flydelta_representation_augmentation_state & next,
                    std::string & augmentation_error) {
                return daemon_flydelta_run_representation_augmentation(
                    provider, job, pipeline, resume, result, next, augmentation_error);
            };
            callbacks.run_search_pipeline = [provider, pipeline](
                    const common_flydelta_experiment_job & job,
                    common_flydelta_search_pipeline_result & result,
                    std::string & runner_error) {
                return daemon_flydelta_run_search_pipeline(provider, job, pipeline, result, runner_error);
            };
            callbacks.run_search_pipeline_with_state = [provider, pipeline](
                    const common_flydelta_experiment_job & job,
                    const common_flydelta_bootstrap_zoom_state * resume,
                    common_flydelta_search_pipeline_result & result,
                    common_flydelta_bootstrap_zoom_state & next,
                    std::string & runner_error) {
                const auto scoped_provider = daemon_flydelta_provider_for_scope(
                    provider, job.seed.scope);
                if (!scoped_provider) {
                    runner_error = "FlyDelta search runner could not create a scoped provider";
                    return false;
                }
                if (resume != nullptr) {
                    return daemon_flydelta_run_bootstrap_zoom_slice(
                        scoped_provider, job, *resume, result, next, runner_error);
                }
                if (!daemon_flydelta_run_search_pipeline(
                        scoped_provider, job, pipeline, result, runner_error)) {
                    return false;
                }
                std::vector<common_flydelta_basis_direction> directions;
                if (!daemon_flydelta_parse_directions(*scoped_provider, job.seed.candidate_ref,
                        directions, runner_error)) return false;
                common_flydelta_search_continuation continuation;
                if (!common_flydelta_select_search_continuation(result, continuation, runner_error)) {
                    return false;
                }
                next = {};
                next.behavior_key = job.seed.behavior_key;
                next.model_profile_fingerprint = job.seed.model_profile_fingerprint;
                next.capture_layout_revision = scoped_provider->capture_layout_revision;
                next.anchor_layer = continuation.region.anchor_layer_index == 0
                    ? static_cast<uint32_t>(directions.front().layer_index)
                    : continuation.region.anchor_layer_index;
                next.selected_scale = continuation.region.total_scale > 0.0f
                    ? continuation.region.total_scale : pipeline.scale.initial_scale;
                next.best_search_score = continuation.search_score;
                next.best_margin_delta = continuation.search_score;
                next.local_layers = continuation.region.layer_indices;
                if (next.local_layers.empty()) next.local_layers = {
                    static_cast<uint32_t>(directions.front().layer_index)};
                next.surface_origin = "bootstrap_rank1";
                return common_flydelta_bootstrap_zoom_state_validate(next, runner_error);
            };
            // Reuse the append-only lifecycle journal already used by the
            // adaptation subsystem.  The worker receives opaque refs only;
            // model material stays in the daemon resource provider.
            common_flydelta_bootstrap_zoom_lifecycle_context bootstrap_context;
            bootstrap_context.namespace_id = provider->authority.namespace_id;
            bootstrap_context.session_id = provider->authority.session_id;
            bootstrap_context.source_id = "daemon-flydelta";
            bootstrap_context.created_at = "daemon-runtime-v1";
            if (!common_flydelta_configure_bootstrap_zoom_lifecycle_callbacks(
                    *provider->lifecycle_store, bootstrap_context, callbacks, callback_error)) {
                return false;
            }
            provider->resolve_bootstrap_zoom_state = callbacks.resolve_bootstrap_zoom_state;
            provider->resolve_bootstrap_zoom_state_for_job =
                callbacks.resolve_bootstrap_zoom_state_for_job;
            const auto lifecycle_persist_bootstrap_state = callbacks.persist_bootstrap_zoom_state;
            const auto lifecycle_persist_bootstrap_state_for_job =
                callbacks.persist_bootstrap_zoom_state_for_job;
            callbacks.persist_bootstrap_zoom_state = [provider, lifecycle_persist_bootstrap_state](
                    const common_flydelta_bootstrap_zoom_state & state,
                    std::string & state_ref,
                    std::string & state_error) {
                if (!lifecycle_persist_bootstrap_state(state, state_ref, state_error)) return false;
                std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                provider->bootstrap_state_by_continuation[
                    daemon_flydelta_bootstrap_key(state)] = state_ref;
                return true;
            };
            callbacks.persist_bootstrap_zoom_state_for_job = [
                    provider, lifecycle_persist_bootstrap_state_for_job] (
                    const common_flydelta_experiment_job & job,
                    const common_flydelta_bootstrap_zoom_state & state,
                    std::string & state_ref,
                    std::string & state_error) {
                if (!lifecycle_persist_bootstrap_state_for_job ||
                        !lifecycle_persist_bootstrap_state_for_job(
                            job, state, state_ref, state_error)) return false;
                std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                provider->bootstrap_state_by_continuation[
                    daemon_flydelta_bootstrap_key(state)] = state_ref;
                return true;
            };
            common_flydelta_lifecycle_event_context augmentation_context;
            augmentation_context.source_id = "daemon-flydelta";
            augmentation_context.scope.namespace_id = provider->authority.namespace_id;
            augmentation_context.scope.session_id = provider->authority.session_id;
            augmentation_context.created_at = "daemon-runtime-v1";
            if (!common_flydelta_configure_representation_augmentation_lifecycle_callbacks(
                    *provider->lifecycle_store, augmentation_context, callbacks, callback_error)) {
                return false;
            }
            provider->resolve_representation_augmentation_state =
                callbacks.resolve_representation_augmentation_state;
            provider->resolve_representation_augmentation_state_for_job =
                callbacks.resolve_representation_augmentation_state_for_job;
            const auto resolve_bootstrap_state = callbacks.resolve_bootstrap_zoom_state;
            callbacks.resolve_search_orchestration_state = [provider, resolve_bootstrap_state](
                    const std::string & state_ref,
                    common_flydelta_experiment_plan & plan,
                    common_flydelta_utility_history & history,
                    std::string & state_error) {
                {
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    const auto cached = provider->orchestration_states.find(state_ref);
                    if (cached != provider->orchestration_states.end()) {
                        plan = cached->second.first;
                        history = cached->second.second;
                        return true;
                    }
                }
                if (state_ref.rfind("flydelta://state/bootstrap-zoom/", 0) == 0) {
                    common_flydelta_bootstrap_zoom_state bootstrap;
                    if (!resolve_bootstrap_state || !resolve_bootstrap_state(
                            state_ref, bootstrap, state_error)) return false;
                    plan = {};
                    plan.continuation.region.layer_indices = bootstrap.local_layers;
                    plan.continuation.region.anchor_layer_index = bootstrap.anchor_layer;
                    plan.continuation.region.total_scale = bootstrap.selected_scale;
                    plan.continuation.region.per_layer_scale = bootstrap.selected_scale /
                        std::sqrt(static_cast<float>(std::max<size_t>(1, bootstrap.local_layers.size())));
                    plan.continuation.search_score = bootstrap.best_search_score;
                    plan.depth = common_flydelta_search_depth::bootstrap;
                    plan.phase = common_flydelta_experiment_phase::bootstrap;
                    plan.budget = common_flydelta_search_budget_for_depth(plan.depth);
                    plan.required_compatible_directions = 1;
                    history = {};
                    return common_flydelta_search_budget_validate(plan.budget, state_error);
                }
                const auto records = provider->lifecycle_store->list(state_error);
                if (!state_error.empty()) return false;
                for (auto it = records.rbegin(); it != records.rend(); ++it) {
                    if (it->kind != common_learning_lifecycle_kind::flydelta_experiment ||
                            it->subject_id != state_ref) continue;
                    if (!daemon_flydelta_orchestration_state_from_json(
                            it->payload_json, plan, history, state_error)) return false;
                    std::string bootstrap_ref;
                    try {
                        bootstrap_ref = json::parse(it->payload_json).value("bootstrap_state_ref", "");
                    } catch (...) {
                        state_error = "FlyDelta orchestration bootstrap reference is invalid";
                        return false;
                    }
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    provider->orchestration_states[state_ref] = {plan, history};
                    if (!bootstrap_ref.empty()) {
                        provider->bootstrap_state_by_orchestration_ref[state_ref] = bootstrap_ref;
                        provider->bootstrap_state_by_continuation[
                            daemon_flydelta_continuation_key(plan)] = bootstrap_ref;
                    }
                    return true;
                }
                state_error = "FlyDelta orchestration state was not found";
                return false;
            };
            callbacks.persist_search_orchestration_state = [provider](
                    const common_flydelta_experiment_plan & plan,
                    const common_flydelta_utility_history & history,
                    std::string & state_ref,
                    std::string & state_error) {
                std::string bootstrap_ref;
                {
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    const auto it = provider->bootstrap_state_by_continuation.find(
                        daemon_flydelta_continuation_key(plan));
                    if (it != provider->bootstrap_state_by_continuation.end()) bootstrap_ref = it->second;
                }
                const auto payload = daemon_flydelta_orchestration_state_json(
                    plan, history, bootstrap_ref).dump();
                const auto digest = hash_sha256_hex(payload.data(), payload.size()).substr(0, 32);
                state_ref = "flydelta://state/orchestration/" + digest;
                common_learning_lifecycle_record record;
                record.event_id = "flydelta://event/orchestration/" + digest;
                record.subject_id = state_ref;
                record.kind = common_learning_lifecycle_kind::flydelta_experiment;
                record.status = common_learning_lifecycle_status::running;
                record.idempotency_key = "flydelta/orchestration/" + digest;
                record.source_id = "daemon-flydelta";
                record.namespace_id = provider->authority.namespace_id;
                record.session_id = provider->authority.session_id;
                record.content_hash = "sha256:" + hash_sha256_hex(payload.data(), payload.size());
                record.created_at = "daemon-runtime-v1";
                record.payload_json = payload;
                if (!provider->lifecycle_store->append(record, state_error)) return false;
                std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                provider->orchestration_states[state_ref] = {plan, history};
                if (!bootstrap_ref.empty()) {
                    provider->bootstrap_state_by_orchestration_ref[state_ref] = bootstrap_ref;
                }
                return true;
            };
            callbacks.resolve_search_orchestration_state_for_job = [provider](
                    const common_flydelta_experiment_job & job,
                    const std::string & state_ref,
                    common_flydelta_experiment_plan & plan,
                    common_flydelta_utility_history & history,
                    std::string & state_error) {
                if (state_ref.rfind("flydelta://state/bootstrap-zoom/", 0) == 0) {
                    common_flydelta_bootstrap_zoom_state bootstrap;
                    if (!provider->resolve_bootstrap_zoom_state_for_job ||
                            !provider->resolve_bootstrap_zoom_state_for_job(
                                job, state_ref, bootstrap, state_error)) return false;
                    plan = {};
                    plan.continuation.region.layer_indices = bootstrap.local_layers;
                    plan.continuation.region.anchor_layer_index = bootstrap.anchor_layer;
                    plan.continuation.region.total_scale = bootstrap.selected_scale;
                    plan.continuation.region.per_layer_scale = bootstrap.selected_scale /
                        std::sqrt(static_cast<float>(std::max<size_t>(1, bootstrap.local_layers.size())));
                    plan.continuation.search_score = bootstrap.best_search_score;
                    plan.depth = common_flydelta_search_depth::bootstrap;
                    plan.phase = common_flydelta_experiment_phase::bootstrap;
                    plan.budget = common_flydelta_search_budget_for_depth(plan.depth);
                    plan.required_compatible_directions = 1;
                    history = {};
                    return common_flydelta_search_budget_validate(plan.budget, state_error);
                }
                const auto records = provider->lifecycle_store->list(state_error);
                if (!state_error.empty()) return false;
                for (auto it = records.rbegin(); it != records.rend(); ++it) {
                    if (it->kind != common_learning_lifecycle_kind::flydelta_experiment ||
                            it->subject_id != state_ref ||
                            !daemon_flydelta_lifecycle_record_matches_scope(
                                *it, job.seed.scope)) continue;
                    if (!daemon_flydelta_orchestration_state_from_json(
                            it->payload_json, plan, history, state_error)) return false;
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    provider->orchestration_states[state_ref] = {plan, history};
                    return true;
                }
                state_error = "FlyDelta orchestration state was not found in job scope";
                return false;
            };
            callbacks.persist_search_orchestration_state_for_job = [provider](
                    const common_flydelta_experiment_job & job,
                    const common_flydelta_experiment_plan & plan,
                    const common_flydelta_utility_history & history,
                    std::string & state_ref,
                    std::string & state_error) {
                const auto scoped_provider = daemon_flydelta_provider_for_scope(
                    provider, job.seed.scope);
                if (!scoped_provider) {
                    state_error = "FlyDelta orchestration persistence has no scoped provider";
                    return false;
                }
                std::string bootstrap_ref;
                {
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    const auto it = provider->bootstrap_state_by_continuation.find(
                        daemon_flydelta_continuation_key(plan));
                    if (it != provider->bootstrap_state_by_continuation.end()) bootstrap_ref = it->second;
                }
                const auto payload = daemon_flydelta_orchestration_state_json(
                    plan, history, bootstrap_ref).dump();
                const auto digest = hash_sha256_hex(payload.data(), payload.size()).substr(0, 32);
                state_ref = "flydelta://state/orchestration/" + digest;
                common_learning_lifecycle_record record;
                record.event_id = "flydelta://event/orchestration/" + digest;
                record.subject_id = state_ref;
                record.kind = common_learning_lifecycle_kind::flydelta_experiment;
                record.status = common_learning_lifecycle_status::running;
                record.idempotency_key = "flydelta/orchestration/" + digest;
                record.source_id = "daemon-flydelta";
                record.namespace_id = scoped_provider->authority.namespace_id;
                record.project_id = scoped_provider->authority.project_id;
                record.session_id = scoped_provider->authority.session_id;
                record.content_hash = "sha256:" + hash_sha256_hex(payload.data(), payload.size());
                record.created_at = "daemon-runtime-v1";
                record.payload_json = payload;
                if (!scoped_provider->lifecycle_store->append(record, state_error)) return false;
                std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                provider->orchestration_states[state_ref] = {plan, history};
                if (!bootstrap_ref.empty()) {
                    provider->bootstrap_state_by_orchestration_ref[state_ref] = bootstrap_ref;
                }
                return true;
            };
            provider->resolve_search_orchestration_state =
                callbacks.resolve_search_orchestration_state;
            provider->resolve_search_orchestration_state_for_job =
                callbacks.resolve_search_orchestration_state_for_job;
            const auto resolve_bootstrap_state_for_job =
                callbacks.resolve_bootstrap_zoom_state_for_job;
            const auto resolve_orchestration_state_for_job =
                callbacks.resolve_search_orchestration_state_for_job;
            const auto persist_bootstrap_state_for_job =
                callbacks.persist_bootstrap_zoom_state_for_job;
            callbacks.run_search_pipeline_with_search_state = [provider,
                    resolve_bootstrap_state_for_job, resolve_orchestration_state_for_job,
                    persist_bootstrap_state_for_job](const common_flydelta_experiment_job & job,
                    const std::string & state_ref,
                    common_flydelta_search_pipeline_result & result,
                    std::string & next_state_ref,
                    std::string & state_error) {
                common_flydelta_experiment_plan plan;
                common_flydelta_utility_history history;
                if (!resolve_orchestration_state_for_job ||
                        !resolve_orchestration_state_for_job(
                            job, state_ref, plan, history, state_error)) return false;
                if (plan.phase != common_flydelta_experiment_phase::bootstrap) {
                    if (plan.phase == common_flydelta_experiment_phase::shallow_controls ||
                            plan.phase == common_flydelta_experiment_phase::deep_controls) {
                        if (!daemon_flydelta_run_post_bootstrap_slice(
                                provider, job, plan, result, state_error)) return false;
                        next_state_ref = state_ref;
                        return true;
                    }
                    state_error = "FlyDelta daemon received an unsupported post-Bootstrap phase";
                    return false;
                }
                std::string bootstrap_ref = job.bootstrap_zoom_state_ref;
                if (bootstrap_ref.empty()) {
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    const auto it = provider->bootstrap_state_by_orchestration_ref.find(state_ref);
                    if (it != provider->bootstrap_state_by_orchestration_ref.end()) bootstrap_ref = it->second;
                }
                if (bootstrap_ref.empty()) {
                    state_error = "FlyDelta Bootstrap continuation requires a BootstrapZoom state ref";
                    return false;
                }
                common_flydelta_bootstrap_zoom_state bootstrap;
                if (!resolve_bootstrap_state_for_job ||
                        !resolve_bootstrap_state_for_job(
                            job, bootstrap_ref, bootstrap, state_error)) return false;
                if (plan.run_orthogonal_search) {
                    if (!daemon_flydelta_run_orthogonal_slice(
                            provider, job, bootstrap, result, bootstrap, state_error)) return false;
                } else if (bootstrap.phase == common_flydelta_bootstrap_zoom_phase::adaptive_alpha) {
                    if (!daemon_flydelta_run_adaptive_alpha_slice(
                            provider, job, bootstrap, result, bootstrap, state_error)) return false;
                } else if (!daemon_flydelta_run_bootstrap_zoom_slice(
                        provider, job, bootstrap, result, bootstrap, state_error)) {
                    return false;
                }
                std::string persisted_bootstrap_ref;
                if (!persist_bootstrap_state_for_job || !persist_bootstrap_state_for_job(
                        job, bootstrap, persisted_bootstrap_ref, state_error)) return false;
                {
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    provider->bootstrap_state_by_continuation[
                        daemon_flydelta_continuation_key(plan)] = persisted_bootstrap_ref;
                }
                // The typed Bootstrap state is persisted by the evaluator's
                // regular state callback.  Preserve the orchestration ref for
                // the pure policy decision that follows this slice.
                next_state_ref = state_ref;
                return true;
            };
            callback_error.clear();
            return true;
        };
        return true;
    };
}

bool flydelta_batch_mode_valid(const std::string & mode) {
    return mode == "disabled" || mode == "auto" || mode == "required";
}

bool flydelta_native_batch_requested(const daemon_options & options) {
    return options.adaptation_flydelta_enabled &&
        options.adaptation_flydelta_batch_mode != "disabled" &&
        options.n_gpu_layers > 0;
}

void apply_flydelta_auto_batch_capacity(
        const daemon_options & options,
        common_agent_model_catalog & catalog) {
    if (!flydelta_native_batch_requested(options)) return;
    const std::string profile_id = options.model_profile.empty()
        ? catalog.default_profile
        : options.model_profile;
    const auto profile = catalog.profiles.find(profile_id);
    if (profile == catalog.profiles.end()) return;
    const auto base = catalog.bases.find(profile->second.base_model_id);
    if (base == catalog.bases.end() || base->second.backend != "server-context") return;

    // A profile can request a larger capacity explicitly. The ordinary 1/1
    // compatibility default is upgraded only for the selected FlyDelta
    // profile; disabling the batch policy keeps a deliberately scalar profile.
    if (profile->second.n_parallel == 1 && profile->second.n_sequences == 1) {
        const auto capacity = static_cast<int>(options.adaptation_flydelta_batch_parallelism);
        profile->second.n_parallel = capacity;
        profile->second.n_sequences = capacity;
    }
}



} // namespace agent_daemon_flydelta_internal


namespace agent_daemon_flydelta_internal {

bool configure_daemon_flydelta_teaching_material(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    error.clear();
    if (!options.adaptation_capture) return true;
    // The existing transaction backend may intentionally be JSONL/SQLite.
    // Do not silently reinterpret those paths as teaching-material storage;
    // only the shared Cozo relation is wired in this V0.
    if (options.adaptation_transaction_backend != "auto" &&
            options.adaptation_transaction_backend != "cozo") return true;

    common_flydelta_teaching_material_identity identity;
    identity.model_profile_fingerprint =
        options.adaptation_flydelta_model_profile_fingerprint.empty()
            ? (options.model_profile.empty()
                ? "model:" + std::filesystem::path(options.model).filename().string()
                : "profile:" + options.model_profile)
            : options.adaptation_flydelta_model_profile_fingerprint;
    identity.tokenizer_fingerprint = "daemon:tokenizer-unspecified-v1";
    identity.template_fingerprint = "daemon:template-unspecified-v1";
    identity.capture_layout_revision = options.adaptation_flydelta_capture_layout_revision;
    identity.execution_context_fingerprint = "daemon:server-context-v1";
    identity.scope_fingerprint = "daemon:teaching-material-v1";
    identity.verifier_revision = "daemon:host-verifier-v1";

    // The adaptation transaction path is already the host's durable Cozo
    // location. Cozo stores teaching material in its own relation, so this
    // does not mix ledger rows with material rows. An empty path deliberately
    // keeps the existing in-memory behavior for ephemeral hosts/tests.
    auto material_runtime = make_agent_flydelta_teaching_material_runtime(
        options.adaptation_transaction_backend,
        options.adaptation_transaction_path,
        std::move(identity),
        error);
    if (!material_runtime) return false;
    runtime.flydelta_teaching_material_runtime = material_runtime;
    // Keep the existing runtime observer seam, but make the daemon's
    // reference-only index restartable by persisting each resolved relation
    // in the same resource store that the FlyDelta worker already reads.
    // The material store receives the durable URI, so queued concept jobs do
    // not depend on an in-process relation map.
    const auto configured_observer = runtime.flydelta_teaching_material_observer;
    agent_resource_store * resource_store = runtime.resource_store.get();
    runtime.flydelta_teaching_material_observer = [
            material_runtime, configured_observer, resource_store](
            const common_flydelta_teaching_relation & relation,
            std::string & observer_error) {
        if (configured_observer && !configured_observer(relation, observer_error)) {
            return false;
        }
        agent_resource_read_authority authority;
        authority.namespace_id = relation.scope.namespace_id;
        authority.project_id = relation.scope.project_id;
        authority.session_id = relation.scope.session_id;
        authority.turn_id = relation.scope.turn_id;
        authority.now = std::time(nullptr);
        common_flydelta_teaching_relation persisted_relation;
        if (!daemon_flydelta_persist_teaching_relation(
                resource_store, authority, relation, persisted_relation, observer_error)) {
            return false;
        }
        return material_runtime->observe_relation(persisted_relation, observer_error);
    };
    return true;
}

bool configure_daemon_flydelta_review_store(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    error.clear();
    if (!options.adaptation_flydelta_enabled) return true;
    if (runtime.flydelta_sideband_review_store ||
            runtime.flydelta_sideband_registry ||
            runtime.flydelta_review_lifecycle_store) {
        error = "FlyDelta review store is already registered";
        return false;
    }

    // Reuse the lifecycle backend selected for FlyDelta. The review journal
    // is an additional typed record in that append-only store, not a second
    // persistence system. The same journal is passed to the resource provider
    // below so replay and bounded worker state share one durable history.
    auto lifecycle = make_agent_learning_lifecycle_store(
        options.adaptation_flydelta_lifecycle_backend,
        options.adaptation_flydelta_lifecycle_path,
        error);
    if (!lifecycle) return false;
    runtime.flydelta_review_lifecycle_store =
        std::shared_ptr<common_learning_lifecycle_store>(std::move(lifecycle));
    runtime.flydelta_sideband_registry =
        std::make_shared<common_flydelta_sideband_registry>();
    runtime.flydelta_sideband_review_store =
        std::make_shared<common_flydelta_sideband_review_store>(
            *runtime.flydelta_review_lifecycle_store);

    if (!runtime.flydelta_sideband_review_store->replay(
            *runtime.flydelta_sideband_registry, error)) {
        runtime.flydelta_sideband_review_store.reset();
        runtime.flydelta_sideband_registry.reset();
        runtime.flydelta_review_lifecycle_store.reset();
        return false;
    }
    runtime.flydelta_sideband_reviews_replayed = true;
    return true;
}

bool configure_daemon_flydelta_server_binding(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    error.clear();
    if (!options.adaptation_flydelta_enabled) return true;
    if (!runtime.flydelta_server_binding_factory &&
            runtime.flydelta_server_binding_callbacks.has_value()) {
        runtime.flydelta_server_binding_factory =
            common_agent_server_flydelta_binding_factory_from_callbacks(
                std::move(*runtime.flydelta_server_binding_callbacks));
        runtime.flydelta_server_binding_callbacks.reset();
    }
    // A generic daemon may be configured for FlyDelta before an embedding
    // host has supplied semantic callbacks. Keep that state explicitly idle;
    // never invent prompts, repairs or host outcomes here.
    if (!runtime.flydelta_server_binding_factory) return true;
    if (!runtime.model_residency) {
        error = "FlyDelta startup binding requires a model catalog/residency manager";
        return false;
    }

    const std::string profile_id = options.model_profile.empty()
        ? runtime.model_residency->catalog().default_profile
        : options.model_profile;
    common_agent_runtime_model_resident_handle handle;
    if (!runtime.model_residency->acquire(profile_id, handle, error)) return false;
    const auto loaded = common_agent_runtime_loaded_model_cast(handle.model);
    if (!loaded || !loaded->server_context_host) {
        std::string release_error;
        runtime.model_residency->release(handle, release_error);
        error = "FlyDelta startup binding requires a resident server-context model";
        return false;
    }

    common_agent_server_flydelta_binding binding;
    if (!runtime.flydelta_server_binding_factory(
            loaded->server_context_host, binding, error)) {
        std::string release_error;
        runtime.model_residency->release(handle, release_error);
        return false;
    }
    if (!binding.teaching_material_runtime) {
        binding.teaching_material_runtime = runtime.flydelta_teaching_material_runtime;
    }
    if (!common_agent_daemon_register_flydelta_server_binding(
            runtime, loaded->server_context_host, std::move(binding), error)) {
        std::string release_error;
        runtime.model_residency->release(handle, release_error);
        return false;
    }
    if (options.adaptation_flydelta_batch_mode == "required" &&
            (!runtime.flydelta_model_adapter ||
                !runtime.flydelta_model_adapter->capabilities.bounded_arm_batch)) {
        std::string release_error;
        runtime.model_residency->release(handle, release_error);
        runtime.flydelta_model_adapter.reset();
        runtime.flydelta_model_host.reset();
        runtime.flydelta_model_capabilities = {};
        error = "FlyDelta native batch mode is required but unavailable for the selected resident model";
        return false;
    }
    runtime.flydelta_model_handle = std::move(handle);
    return true;
}


} // namespace agent_daemon_flydelta_internal

namespace agent_daemon_flydelta_internal {

bool configure_daemon_flydelta_model_residency(
        const daemon_options & options,
        common_agent_model_catalog & catalog,
        bool & native_batch,
        bool & allow_scalar_fallback,
        std::string & error) {
    native_batch = false;
    allow_scalar_fallback = true;
    if (!flydelta_batch_mode_valid(options.adaptation_flydelta_batch_mode)) {
        error = "runtime.adaptation.flydelta.batch_mode must be disabled, auto or required";
        return false;
    }
    if (options.adaptation_flydelta_batch_parallelism == 0 ||
            options.adaptation_flydelta_batch_parallelism > 8) {
        error = "runtime.adaptation.flydelta.batch_parallelism must be between 1 and 8";
        return false;
    }
    apply_flydelta_auto_batch_capacity(options, catalog);
    native_batch = flydelta_native_batch_requested(options);
    allow_scalar_fallback = options.adaptation_flydelta_batch_mode != "required";
    error.clear();
    return true;
}

bool configure_daemon_flydelta_runtime(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    if (!configure_daemon_flydelta_teaching_material(options, runtime, error)) {
        error = "FlyDelta teaching-material runtime initialization failed: " + error;
        return false;
    }
    if (!configure_daemon_flydelta_review_store(options, runtime, error)) {
        error = "FlyDelta review-store initialization failed: " + error;
        return false;
    }
    if (options.adaptation_flydelta_enabled &&
            !runtime.flydelta_server_binding_factory &&
            !runtime.flydelta_server_binding_callbacks.has_value()) {
        runtime.flydelta_server_binding_factory =
            make_daemon_flydelta_resource_binding_factory(
                options, runtime.resource_store.get(),
                runtime.flydelta_teaching_material_runtime,
                runtime.flydelta_review_lifecycle_store);
    }
    if (!configure_daemon_flydelta_server_binding(options, runtime, error)) {
        error = "FlyDelta server binding initialization failed: " + error;
        return false;
    }
    return true;
}

void configure_daemon_flydelta_runtime_config(
        const daemon_options & options,
        const common_agent_daemon_runtime & runtime,
        common_agent_runtime_config & config) {
    config.enable_flydelta_capture_candidates =
        options.adaptation_flydelta_capture_candidates;
    config.enable_flydelta_candidate_lifecycle =
        options.adaptation_flydelta_capture_candidates &&
        !options.adaptation_flydelta_lifecycle_path.empty();
    config.flydelta_lifecycle_backend = options.adaptation_flydelta_lifecycle_backend;
    config.flydelta_lifecycle_path = options.adaptation_flydelta_lifecycle_path;
    config.flydelta_model_profile_fingerprint =
        options.adaptation_flydelta_model_profile_fingerprint.empty()
            ? (options.model_profile.empty()
                ? "model:" + options.model
                : "profile:" + options.model_profile)
            : options.adaptation_flydelta_model_profile_fingerprint;
    config.flydelta_capture_layout_revision =
        options.adaptation_flydelta_capture_layout_revision;
    config.flydelta_max_capture_candidates =
        options.adaptation_flydelta_max_capture_candidates;
    config.flydelta_capture_job_enqueue = runtime.flydelta_capture_job_enqueue;
    config.flydelta_teaching_material_observer =
        runtime.flydelta_teaching_material_observer;
    config.flydelta_teaching_material_runtime =
        runtime.flydelta_teaching_material_runtime;
}

} // namespace agent_daemon_flydelta_internal
