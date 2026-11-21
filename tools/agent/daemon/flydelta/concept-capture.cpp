#include "agent-daemon-flydelta-internal.h"

#include "agent/agent-residual-patch.h"
#include "agent/adaptation/flydelta/flydelta-search-budget.h"

#include <iterator>
#include <iostream>

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

// A SemiOracle relation already carries a host-proved rejected/preferred
// contrast.  Its three context resources end with the corresponding
// continuation, so capturing their prompt state with n_predict=0 is a
// deterministic, teacher-forced representation capture.  Do not sample the
// model again and accidentally turn a verified relation into a different
// interaction.  Other evidence sources retain the established free-generation
// and verifier path.
bool daemon_flydelta_uses_forced_context_capture(
        const common_flydelta_teaching_relation & relation) {
    return relation.source == common_adaptation_evidence_source::semi_oracle_preference &&
        relation.host_approved &&
        relation.conditioned_origin == common_flydelta_teaching_origin::host_derived;
}

} // namespace

bool daemon_flydelta_validate_concept_admission(
        const daemon_flydelta_resource_provider & provider,
        const common_flydelta_experiment_job & job, std::string & error) {
    if (job.seed.schema_version != 2) return true;
    if (!common_flydelta_experiment_seed_validate(job.seed, error) || !provider.teaching_material_runtime) return false;
    json admission;
    if (!daemon_flydelta_read_json_bounded(provider, job.seed.admission_ref, 1024U * 1024U, admission, error)) return false;
    common_flydelta_teaching_material_group group;
    if (!provider.teaching_material_runtime->resolve_relation_set(job.teaching_material_group_ref, group, error)) return false;
    try {
        const auto selection = admission.at("selection_mode").get<std::string>();
        if (selection != "auto" && selection != "contrast" && selection != "positive_only") {
            error = "FlyDelta concept admission selection mode is invalid";
            return false;
        }
        if (selection != "positive_only") {
            common_flydelta_teaching_material_group selected;
            bool available = false;
            const auto mode = selection == "auto" ? common_flydelta_teaching_material_mode::automatic
                : common_flydelta_teaching_material_mode::contrast;
            if (!provider.teaching_material_runtime->resolve_group_for_family(job.seed.teaching_key,
                    job.seed.behavior_key, selected, available, error, mode, &job.seed.scope)) return false;
            if (!available || selected.group_ref != job.teaching_material_group_ref) {
                error = "FlyDelta concept material selection changed; a new host admission is required";
                return false;
            }
        }
        const auto & scope = admission.at("scope");
        if (admission.at("kind") != "flydelta_concept_experiment_admission" || admission.at("schema_version") != 1 ||
                !admission.at("host_authorized").get<bool>() || !admission.at("experimental_only").get<bool>() ||
                provider.model_profile_fingerprint != job.seed.model_profile_fingerprint ||
                admission.at("learning_eligible").get<bool>() || group.group_ref.empty() ||
                admission.at("material_group_ref") != job.teaching_material_group_ref ||
                admission.at("material_revision") != common_flydelta_teaching_material_revision(group) ||
                admission.at("positive_only").get<bool>() != group.positive_only ||
                (selection == "positive_only" && !group.positive_only) ||
                admission.at("teaching_key") != job.seed.teaching_key || admission.at("behavior_key") != job.seed.behavior_key ||
                admission.at("task_fingerprint") != job.seed.task_fingerprint ||
                admission.at("model_profile_fingerprint") != job.seed.model_profile_fingerprint ||
                admission.at("tokenizer_fingerprint") != job.seed.tokenizer_fingerprint ||
                admission.at("template_fingerprint") != job.seed.template_fingerprint ||
                admission.at("execution_context_fingerprint") != job.seed.execution_context_fingerprint ||
                admission.at("capture_layout_revision") != provider.capture_layout_revision ||
                admission.at("baseline_ref") != job.seed.baseline_ref || admission.at("verifier_ref") != job.seed.verifier_ref ||
                admission.at("verifier_revision") != job.seed.verifier_contract_revision ||
                scope.at("namespace_id") != job.seed.scope.namespace_id || scope.at("project_id") != job.seed.scope.project_id ||
                scope.at("session_id") != job.seed.scope.session_id || scope.at("turn_id") != job.seed.scope.turn_id ||
                group.scope.namespace_id != job.seed.scope.namespace_id || group.scope.project_id != job.seed.scope.project_id ||
                group.scope.session_id != job.seed.scope.session_id) {
            error = "FlyDelta concept admission does not match the immutable evaluation or material identity";
            return false;
        }
        if (!common_flydelta_teaching_material_execution_matches(group.identity,
                job.seed.model_profile_fingerprint, job.seed.tokenizer_fingerprint, job.seed.template_fingerprint,
                job.seed.execution_context_fingerprint, provider.capture_layout_revision, error)) return false;
    } catch (const std::exception & e) {
        error = std::string("FlyDelta concept admission is malformed: ") + e.what();
        return false;
    }
    return true;
}

bool daemon_flydelta_admit_concept_experiment(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        common_flydelta_experiment_job context,
        common_flydelta_teaching_material_mode mode,
        common_flydelta_experiment_collection_result & result, std::string & error) {
    result = common_flydelta_experiment_collection_result::disabled;
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, context.seed.scope);
    if (!provider || !provider->teaching_material_runtime || !provider->resources || !provider->lifecycle_store ||
            !context.capture_manifest_ids.empty() || !context.behavior_delta_ids.empty() ||
            !context.training_example_ids.empty() || !context.seed.transaction_ids.empty() ||
            !context.seed.admission_ref.empty()) {
        error = "FlyDelta concept admission requires a host evaluation context and existing resource/journal stores";
        return false;
    }
    common_flydelta_teaching_material_group group;
    bool available = false;
    if (!provider->teaching_material_runtime->resolve_group_for_family(context.seed.teaching_key,
            context.seed.behavior_key, group, available, error, mode, &context.seed.scope)) return false;
    if (!available) return true;
    const auto & execution = provider->teaching_material_runtime->identity();
    // Empty identities are filled only on this new host-produced job. Explicit
    // identities must already match; old resources are never rewritten.
    const auto bind = [&](std::string & field, const std::string & actual, const char * name) {
        if (!field.empty() && field != actual) {
            error = std::string("FlyDelta concept admission ") + name + " does not match the resident profile";
            return false;
        }
        field = actual;
        return true;
    };
    if (!bind(context.seed.model_profile_fingerprint, execution.model_profile_fingerprint, "model") ||
            !bind(context.seed.tokenizer_fingerprint, execution.tokenizer_fingerprint, "tokenizer") ||
            !bind(context.seed.template_fingerprint, execution.template_fingerprint, "template") ||
            !bind(context.seed.execution_context_fingerprint, execution.execution_context_fingerprint, "context")) return false;
    context.teaching_material_group_ref = group.group_ref;
    context.kind = common_flydelta_experiment_job_kind::search_pipeline;
    common_agent_generation_request baseline;
    json verifier;
    if (!daemon_flydelta_parse_context(*provider, context.seed.baseline_ref, baseline, error) ||
            !daemon_flydelta_read_json(*provider, context.seed.verifier_ref, verifier, error)) return false;
    if (context.seed.verifier_contract_revision.empty()) {
        const auto verifier_bytes = verifier.dump();
        context.seed.verifier_contract_revision = "sha256:" + hash_sha256_hex(verifier_bytes.data(), verifier_bytes.size());
    }
    daemon_flydelta_concept_target target;
    if (!daemon_flydelta_resolve_concept_target(provider, context, target, error)) return false;
    json admission = {{"kind", "flydelta_concept_experiment_admission"}, {"schema_version", 1},
        {"host_authorized", true}, {"experimental_only", true}, {"learning_eligible", false},
        {"selection_mode", mode == common_flydelta_teaching_material_mode::automatic ? "auto" :
            mode == common_flydelta_teaching_material_mode::contrast ? "contrast" : "positive_only"},
        {"positive_only", group.positive_only}, {"material_group_ref", group.group_ref},
        {"material_revision", common_flydelta_teaching_material_revision(group)},
        {"teaching_key", context.seed.teaching_key}, {"behavior_key", context.seed.behavior_key},
        {"task_fingerprint", context.seed.task_fingerprint}, {"baseline_ref", context.seed.baseline_ref},
        {"where_origin_ref", context.bootstrap_zoom_state_ref},
        {"verifier_ref", context.seed.verifier_ref}, {"verifier_revision", context.seed.verifier_contract_revision},
        {"model_profile_fingerprint", execution.model_profile_fingerprint},
        {"tokenizer_fingerprint", execution.tokenizer_fingerprint}, {"template_fingerprint", execution.template_fingerprint},
        {"execution_context_fingerprint", execution.execution_context_fingerprint},
        {"capture_layout_revision", execution.capture_layout_revision},
        {"scope", {{"namespace_id", context.seed.scope.namespace_id}, {"project_id", context.seed.scope.project_id},
            {"session_id", context.seed.scope.session_id}, {"turn_id", context.seed.scope.turn_id}}}};
    const auto bytes = admission.dump();
    const auto key = hash_sha256_hex(bytes.data(), bytes.size());
    context.seed.schema_version = 2;
    context.seed.source = common_adaptation_evidence_source::user_taught_concept;
    context.seed.id = "flydelta://concept-experiment/" + key;
    context.id = context.seed.id;
    if (context.code_revision.empty()) context.code_revision = "host-concept-admission-v1";
    if (!daemon_flydelta_put_json_resource(provider, "concept-admission-" + key + ".json",
            "flydelta_concept_experiment_admission", group.group_ref, bytes, context.seed.admission_ref, error)) return false;
    context.seed.evidence_ref = context.seed.admission_ref;
    if (!daemon_flydelta_validate_concept_admission(*provider, context, error)) return false;
    return common_flydelta_collect_concept_experiment_job(queue_root, limits, context,
        *provider->teaching_material_runtime, mode, result, error);
}

// Save a single authorized representation in the existing trajectory resource
// channel. No control, synthetic baseline or behavioral verdict is introduced.
bool daemon_flydelta_persist_positive_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_teaching_material_group & group,
        const daemon_flydelta_concept_relation_material & material,
        const common_flydelta_arm_result & result,
        const common_flydelta_hidden_state_capture & capture,
        std::string & reference, std::string & error) {
    error.clear();
    reference.clear();
    std::vector<float> values;
    if (!common_flydelta_teaching_relation_validate(material.relation, error) ||
            !group.positive_only || !material.relation.positive_only || !result.executed ||
            !material.relation.host_approved || result.capture_ref.empty() ||
            std::find(group.relation_refs.begin(), group.relation_refs.end(), material.relation_resource_ref) == group.relation_refs.end() ||
            !common_flydelta_hidden_state_capture_validate(capture, 4U * 1024U * 1024U, error) ||
            !capture.captured || capture.n_embd != provider->model_n_embd ||
            capture.model_profile_fingerprint != group.identity.model_profile_fingerprint ||
            capture.capture_layout_revision != group.identity.capture_layout_revision ||
            capture.position != common_flydelta_capture_position::generation_boundary ||
            !daemon_flydelta_capture_layer(capture, material.layer_index, values, error)) {
        if (error.empty()) error = "FlyDelta positive capture identity or boundary is invalid";
        return false;
    }
    common_agent_generation_request request;
    if (!daemon_flydelta_parse_context(*provider, material.relation.conditioned_ref, request, error)) return false;
    // Hash logical model input, excluding wrapper IDs and acquisition times.
    const auto input = common_chat_msgs_to_json_oaicompat(request.messages).dump();
    const auto fingerprint = hash_sha256_hex(input.data(), input.size());
    const auto identity = group.group_ref + "\n" + fingerprint + "\n" +
        material.capture_set_ref + "\n" + std::to_string(material.layer_index);
    const auto id = "flydelta://positive-capture/" + hash_sha256_hex(identity.data(), identity.size());
    json payload = {{"schema_version", 2}, {"kind", "flydelta_positive_capture"},
        {"id", id}, {"positive_only", true}, {"group_ref", group.group_ref},
        {"source_ref", material.relation_resource_ref}, {"source_relation_id", material.relation.id},
        {"task_fingerprint", material.relation.task_fingerprint},
        {"observation_fingerprint", fingerprint}, {"capture_ref", result.capture_ref},
        {"semantic_anchor", material.semantic_anchor}, {"layer_index", material.layer_index},
        {"values", values}, {"capture_set_ref", material.capture_set_ref},
        {"model_profile_fingerprint", group.identity.model_profile_fingerprint},
        {"tokenizer_fingerprint", group.identity.tokenizer_fingerprint},
        {"template_fingerprint", group.identity.template_fingerprint},
        {"execution_context_fingerprint", group.identity.execution_context_fingerprint},
        {"scope_fingerprint", group.identity.scope_fingerprint},
        {"capture_layout_revision", group.identity.capture_layout_revision},
        {"verifier_revision", group.identity.verifier_revision},
        {"host_authorized", true}, {"experimental_only", true}, {"learning_eligible", false}};
    return daemon_flydelta_put_json_resource(provider, "positive-capture-" +
        hash_sha256_hex(identity.data(), identity.size()).substr(0, 24) + ".json",
        "flydelta_positive_capture", material.relation.id, payload.dump(), reference, error);
}

static bool daemon_flydelta_run_concept_capture_at_layer(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        int32_t requested_layer,
        const std::string & requested_relation,
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
    if (!daemon_flydelta_validate_concept_admission(*provider, job, error)) return false;
    common_flydelta_teaching_material_group group;
    if (!provider->teaching_material_runtime->resolve_relation_set(
            job.teaching_material_group_ref, group, error)) return false;
    if (group.positive_only && job.seed.schema_version == 1) {
        common_flydelta_teaching_material_group selected;
        bool available = false;
        if (!provider->teaching_material_runtime->resolve_group_for_family(group.teaching_key, group.behavior_key,
                selected, available, error, common_flydelta_teaching_material_mode::automatic, &job.seed.scope)) return false;
        if (!available || selected.group_ref != group.group_ref) {
            error = "FlyDelta automatic concept material selection changed";
            return false;
        }
    }
    if (group.positive_only) {
        if (group.compatibility_key != common_flydelta_teaching_material_compatibility_key(
                provider->teaching_material_runtime->identity())) {
            error = "FlyDelta positive capture material has an incompatible host identity";
            return false;
        }
        if (!common_flydelta_teaching_material_execution_matches(group.identity,
                job.seed.model_profile_fingerprint, job.seed.tokenizer_fingerprint,
                job.seed.template_fingerprint, job.seed.execution_context_fingerprint,
                provider->capture_layout_revision, error)) return false;
    }
    if (!provider->teaching_material_runtime->contrast_admissible(group, error, job.seed.verifier_contract_revision)) return false;
    if (!group.relation_set_ready || group.relation_refs.size() < group.minimum_relations) {
        error = "FlyDelta concept capture requires a relation-ready teaching material group";
        return false;
    }

    daemon_flydelta_concept_target target;
    if (!daemon_flydelta_resolve_concept_target(provider, job, target, error)) return false;
    const bool challenge = !job.concept_localization.layers.empty();
    if (challenge) target.layer_index = requested_layer;

    std::vector<daemon_flydelta_concept_relation_material> materials;
    materials.reserve(group.relation_refs.size());
    for (const auto & relation_ref : group.relation_refs) {
        if (!requested_relation.empty() && relation_ref != requested_relation) continue;
        if (challenge) {
            const auto saved = std::find_if(group.layer_trajectories.begin(), group.layer_trajectories.end(),
                [&](const auto & entry) {
                    return entry.layer_index == requested_layer && entry.relation_ref == relation_ref &&
                        entry.capture_set_ref == common_flydelta_concept_localization_material_key(job);
                });
            if (saved != group.layer_trajectories.end()) {
                trajectory_refs.push_back(saved->trajectory_ref);
                continue;
            }
        }
        daemon_flydelta_concept_relation_material material;
        bool source_invalid = false;
        if (!daemon_flydelta_parse_teaching_relation(*provider, relation_ref, material, error, &source_invalid)) {
            if (!group.positive_only && source_invalid && provider->lifecycle_store) {
                const auto diagnostic = error;
                if (!provider->teaching_material_runtime->assess_contrast(group.group_ref,
                        job.seed.verifier_ref, {relation_ref, job.seed.evidence_ref}, true, error,
                        common_flydelta_teaching_material_revision(group), job.seed.verifier_contract_revision.empty()
                            ? group.identity.verifier_revision : job.seed.verifier_contract_revision)) return false;
                error = diagnostic;
            }
            return false;
        }
        material.relation_resource_ref = relation_ref;
        if (material.relation.behavior_key != group.behavior_key ||
                material.relation.teaching_key != group.teaching_key ||
                material.relation.positive_only != group.positive_only ||
                (!group.positive_only && material.relation.control_ref.empty()) || !material.relation.host_approved ||
                ((challenge || group.positive_only) && (material.relation.scope.namespace_id != job.seed.scope.namespace_id ||
                    material.relation.scope.project_id != job.seed.scope.project_id ||
                    material.relation.scope.session_id != job.seed.scope.session_id))) {
            const std::string diagnostic = "FlyDelta concept relation is incompatible with its material group";
            if (!group.positive_only && provider->lifecycle_store) {
                if (!provider->teaching_material_runtime->assess_contrast(group.group_ref,
                        job.seed.verifier_ref, {relation_ref, job.seed.evidence_ref}, true, error,
                        common_flydelta_teaching_material_revision(group), job.seed.verifier_contract_revision.empty()
                            ? group.identity.verifier_revision : job.seed.verifier_contract_revision)) return false;
            }
            error = diagnostic;
            return false;
        }
        common_flydelta_concept_capture_plan plan;
        const std::string plan_identity = job.id + "\n" + relation_ref +
            (challenge ? "\nlayer:" + std::to_string(target.layer_index) : std::string{});
        plan.positive_only = group.positive_only;
        plan.schema_version = group.positive_only ? 2 : 1;
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
        material.capture_set_ref = common_flydelta_concept_localization_material_key(job);
        if (challenge) material.capture_identity = group.identity;
        materials.push_back(std::move(material));
    }

    if (materials.empty()) return !trajectory_refs.empty();
    common_flydelta_arm_batch_request batch;
    batch.batch_id = job.id + ":concept-capture" +
        (challenge ? ":layer:" + std::to_string(requested_layer) + ":relation:" +
            hash_sha256_hex(requested_relation.data(), requested_relation.size()).substr(0, 24)
            : std::string{});
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
        const bool forced_context_capture = group.positive_only ||
            daemon_flydelta_uses_forced_context_capture(material.relation);
        capture_binding binding;
        binding.material_index = index;
        const std::array<std::string, 3> contexts = {
            material.relation.baseline_ref,
            material.relation.conditioned_ref,
            material.relation.control_ref,
        };
        for (size_t side = 0; side < contexts.size(); ++side) {
            if (group.positive_only && side != 1) continue;
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
            arm.request_generation = !forced_context_capture;
            arm.request_host_verification = !forced_context_capture;
            arm.max_capture_bytes = 4U * 1024U * 1024U;
            arm.max_generated_tokens = std::max<size_t>(
                64U, provider->n_predict > 0 ? provider->n_predict : 64U);
            const std::string identity = job.id + "\n" + material.relation.id + "\n" +
                std::to_string(side) + "\n" + contexts[side] +
                (challenge ? "\nlayer:" + std::to_string(material.layer_index) +
                    "\nrelation:" + material.relation_resource_ref : std::string{});
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
    std::string attempt_ref;
    common_flydelta_search_consumption consumption;
    if (challenge) {
        if (!provider->lifecycle_store) {
            error = "FlyDelta localization capture requires a durable execution journal";
            return false;
        }
        std::vector<std::string> logical_ids;
        for (const auto & arm : batch.arms) logical_ids.push_back(arm.arm_id);
        if (!common_flydelta_begin_search_execution(*provider->lifecycle_store, job, 0,
                job.concept_localization.max_capture_model_calls, logical_ids, batch.batch_id,
                3, attempt_ref, consumption, error)) return false;
    }
    const bool executed = daemon_flydelta_execute_batch(provider, batch, batch_result, error);
    if (challenge) {
        size_t completed = 0;
        for (const auto & arm : batch_result.arms) if (arm.executed) ++completed;
        std::string journal_error;
        if (!common_flydelta_finish_search_execution(*provider->lifecycle_store, job,
                attempt_ref, completed, executed, journal_error)) {
            error = journal_error;
            return false;
        }
        if (!common_flydelta_load_search_consumption(*provider->lifecycle_store, job, 0,
                job.concept_localization.max_capture_model_calls, consumption, journal_error)) {
            error = journal_error;
            return false;
        }
        if (provider->host && provider->host->config().agent_trace) {
            std::clog << "[flydelta] concept_localization_capture " << json{
                {"job_id", job.id}, {"capture_set_ref", job.concept_localization.capture_set_ref},
                {"layer_index", requested_layer}, {"relation_ref", requested_relation},
                {"attempt_ref", attempt_ref}, {"logical_used", consumption.logical_used},
                {"logical_limit", job.concept_localization.max_capture_model_calls},
                {"physical_rows_started", consumption.physical_rows_started},
                {"physical_rows_completed", consumption.physical_rows_completed},
                {"physical_rows_unresolved", consumption.physical_rows_unresolved},
            }.dump() << '\n';
        }
    }
    if (!executed || batch_result.arms.size() != batch.arms.size()) {
        if (error.empty()) error = "FlyDelta concept capture batch returned incomplete arms";
        return false;
    }
    for (const auto & binding : bindings) {
        const auto & material = materials[binding.material_index];
        if (group.positive_only) {
            const auto & result = batch_result.arms[binding.conditioned_arm];
            std::shared_ptr<const common_flydelta_hidden_state_capture> capture;
            {
                std::lock_guard<std::mutex> lock(provider->capture_mutex);
                const auto it = provider->captures.find(batch.arms[binding.conditioned_arm].arm_id);
                if (it != provider->captures.end()) capture = it->second;
            }
            std::string reference;
            if (!capture || !daemon_flydelta_persist_positive_capture(provider, group, material, result, *capture, reference, error)) {
                if (error.empty()) error = "FlyDelta positive capture was not retained";
                return false;
            }
            if (challenge ? !provider->teaching_material_runtime->observe_layer_trajectory(
                    group.group_ref, material.layer_index, material.relation_resource_ref,
                    reference, error, material.capture_set_ref)
                : !provider->teaching_material_runtime->observe_trajectory(group.group_ref, reference, error)) return false;
            trajectory_refs.push_back(std::move(reference));
            continue;
        }
        const auto & baseline_result = batch_result.arms[binding.baseline_arm];
        const auto & conditioned_result = batch_result.arms[binding.conditioned_arm];
        const auto & control_result = batch_result.arms[binding.control_arm];
        if (!baseline_result.executed || !conditioned_result.executed || !control_result.executed) {
            error = "FlyDelta concept capture contains an unexecuted arm";
            return false;
        }
        const bool forced_context_capture =
            daemon_flydelta_uses_forced_context_capture(material.relation);
        const bool conditioned_verified = forced_context_capture ||
            (conditioned_result.host_evaluated &&
                conditioned_result.verifier_known && conditioned_result.verifier_passed);
        if (!conditioned_verified) {
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
            error = "FlyDelta concept capture did not retain all hidden-state captures"
                " baseline=" + (baseline ? std::string("yes") : "no")
                + " conditioned=" + (conditioned ? "yes" : "no")
                + " control=" + (control ? "yes" : "no");
            return false;
        }
        std::string trajectory_ref;
        if (!daemon_flydelta_persist_concept_trajectory(
                provider, material, job.teaching_material_group_ref, job.id,
                baseline_result.capture_ref, conditioned_result.capture_ref,
                control_result.capture_ref, *baseline, *conditioned, *control,
                conditioned_verified,
                baseline_result.host_evaluated &&
                    baseline_result.verifier_known &&
                    !baseline_result.verifier_passed &&
                    daemon_flydelta_negative_violation_allowed(
                        baseline_result.verifier_violation_kind),
                baseline_result.verifier_violation_kind,
                baseline_result.verifier_violation_dimensions,
                trajectory_ref, error)) return false;
        if (challenge ? !provider->teaching_material_runtime->observe_layer_trajectory(
                job.teaching_material_group_ref, material.layer_index, material.relation_resource_ref,
                trajectory_ref, error, material.capture_set_ref)
            : !provider->teaching_material_runtime->observe_trajectory(group.group_ref, trajectory_ref, error)) return false;
        trajectory_refs.push_back(std::move(trajectory_ref));
    }
    return !trajectory_refs.empty();
}

bool daemon_flydelta_run_concept_capture(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        std::vector<std::string> & trajectory_refs, std::string & error) {
    trajectory_refs.clear();
    if (!common_flydelta_concept_localization_request_validate(job.concept_localization, error)) return false;
    if (job.concept_localization.layers.empty()) {
        return daemon_flydelta_run_concept_capture_at_layer(provider, job, 0, {}, trajectory_refs, error);
    }
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (!provider || !provider->teaching_material_runtime) {
        error = "FlyDelta localization requires a teaching-material runtime";
        return false;
    }
    common_flydelta_teaching_material_group group;
    if (!provider->teaching_material_runtime->resolve_relation_set(
            job.teaching_material_group_ref, group, error)) return false;
    if (!group.relation_set_ready || group.relation_refs.size() > 64 ||
            group.compatibility_key != common_flydelta_teaching_material_compatibility_key(
                provider->teaching_material_runtime->identity()) ||
            group.identity.model_profile_fingerprint != job.seed.model_profile_fingerprint ||
            group.identity.tokenizer_fingerprint != job.seed.tokenizer_fingerprint ||
            group.identity.template_fingerprint != job.seed.template_fingerprint ||
            group.identity.execution_context_fingerprint != job.seed.execution_context_fingerprint ||
            group.identity.capture_layout_revision != provider->capture_layout_revision ||
            group.relation_refs.size() * job.concept_localization.layers.size() * (group.positive_only ? 1 : 3) >
                job.concept_localization.max_capture_model_calls) {
        error = "FlyDelta localization material is incompatible or exceeds its capture budget";
        return false;
    }
    std::set<std::string> relation_ids;
    for (const auto & ref : group.relation_refs) {
        daemon_flydelta_concept_relation_material material;
        if (!daemon_flydelta_parse_teaching_relation(*provider, ref, material, error)) return false;
        if (!material.relation.host_approved || material.relation.teaching_key != group.teaching_key ||
                material.relation.behavior_key != group.behavior_key ||
                material.relation.scope.namespace_id != job.seed.scope.namespace_id ||
                material.relation.scope.project_id != job.seed.scope.project_id ||
                material.relation.scope.session_id != job.seed.scope.session_id ||
                !relation_ids.insert(material.relation.id).second) {
            error = "FlyDelta localization requires distinct approved relations in its job scope";
            return false;
        }
    }
    for (const auto layer : job.concept_localization.layers) {
        if (layer >= provider->model_n_layers) {
            error = "FlyDelta localization layer is outside the model";
            return false;
        }
    }
    for (const auto layer : job.concept_localization.layers) {
        for (const auto & relation_ref : group.relation_refs) {
            std::vector<std::string> refs;
            if (!daemon_flydelta_run_concept_capture_at_layer(provider, job,
                    static_cast<int32_t>(layer), relation_ref, refs, error)) return false;
            trajectory_refs.insert(trajectory_refs.end(), refs.begin(), refs.end());
        }
    }
    return true;
}

} // namespace agent_daemon_flydelta_internal
