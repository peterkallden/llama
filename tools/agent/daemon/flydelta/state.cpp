#include "../agent-daemon-flydelta-internal.h"
#include "agent/adaptation/flydelta/flydelta-search-budget.h"

#include "agent/agent-residual-patch.h"
#include "agent/adaptation/flydelta/experiment/flydelta-guidance-qualification.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "state.h"

namespace agent_daemon_flydelta_internal {

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
    fixture.verifier_revision = job.seed.verifier_contract_revision.empty()
        ? job.seed.verifier_ref : job.seed.verifier_contract_revision;
    fixture.guidance_fixture_group_id = job.seed.fixture_group_id;
    fixture.outcome_guidance_verifier_revision = job.seed.verifier_contract_revision;
    return common_flydelta_experiment_fixture_validate(fixture, error);
}

std::string daemon_flydelta_verifier_contract_revision(
        const common_flydelta_oracle_registry & registry,
        const std::string & verifier_ref) {
    std::set<std::string> revisions;
    for (const auto & registration : registry.evaluators) {
        if (registration.evaluator_ref == verifier_ref ||
                registration.expected_contract_ref == verifier_ref) {
            const std::string & revision = !registration.expected_contract_revision.empty()
                ? registration.expected_contract_revision : registration.evaluator_revision;
            if (!revision.empty()) revisions.insert(revision);
        }
    }
    return revisions.size() == 1 ? *revisions.begin() : std::string();
}



json daemon_flydelta_anchor_observation_json(
        const common_flydelta_semantic_progress_observation & o) {
    return {{"available", o.available}, {"coverage", o.regression_coverage_complete},
        {"operation", static_cast<int>(o.operation_state)}, {"dataset", static_cast<int>(o.dataset_state)},
        {"grouping", static_cast<int>(o.grouping_state)}, {"measure", static_cast<int>(o.measure_state)},
        {"contract", static_cast<int>(o.contract_state)}, {"score", o.score},
        {"objective", json::parse(common_flydelta_progress_assessment_to_json(o.objective))}};
}

bool daemon_flydelta_anchor_observation_from_json(
        const json & value, common_flydelta_semantic_progress_observation & o, std::string & error) {
    try {
        o = {};
        o.available = value.at("available").get<bool>();
        o.regression_coverage_complete = value.at("coverage").get<bool>();
        if (value.contains("objective")) {
            if (!common_flydelta_progress_assessment_from_json(value.at("objective").dump(), o.objective, error)) return false;
            if (o.objective.assessment_known) {
                o.score = static_cast<int>(o.objective.satisfied_goals.size());
                if (!o.available || !o.regression_coverage_complete || !o.objective.regression_known || o.objective.has_regression) {
                    error = "FlyDelta anchor objective coverage is incomplete"; return false;
                }
                return true;
            }
        }
        auto state = [&](const char * name) {
            const int raw = value.at(name).get<int>();
            if (raw < static_cast<int>(common_flydelta_semantic_progress_field_state::unknown) ||
                    raw > static_cast<int>(common_flydelta_semantic_progress_field_state::matched)) {
                throw std::runtime_error("invalid progress state");
            }
            return static_cast<common_flydelta_semantic_progress_field_state>(raw);
        };
        o.operation_state = state("operation"); o.dataset_state = state("dataset");
        o.grouping_state = state("grouping"); o.measure_state = state("measure");
        o.contract_state = state("contract"); o.score = value.at("score").get<int>();
        if (!o.available || !o.regression_coverage_complete ||
                o.operation_state == common_flydelta_semantic_progress_field_state::unknown ||
                o.dataset_state == common_flydelta_semantic_progress_field_state::unknown ||
                o.grouping_state == common_flydelta_semantic_progress_field_state::unknown ||
                o.measure_state == common_flydelta_semantic_progress_field_state::unknown ||
                o.contract_state == common_flydelta_semantic_progress_field_state::unknown) {
            error = "FlyDelta anchor progress coverage is incomplete"; return false;
        }
        return true;
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta anchor observation is invalid: ") + exception.what();
        return false;
    }
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
        const std::string & bootstrap_state_ref) {
    const auto & region = plan.continuation.region;
    return {
        {"kind", "flydelta_daemon_orchestration_state"},
        {"search_anchor_ref", plan.search_anchor_ref},
        {"search_anchor_round", plan.search_anchor_round},
        {"search_anchor_pending", plan.search_anchor_pending},
        {"search_model_call_limit", plan.search_model_call_limit},
        {"search_model_calls_used", plan.search_model_calls_used},
        {"continuation", {
            {"direction_index", plan.continuation.direction_index},
            {"region_trial_index", plan.continuation.region_trial_index},
            {"search_score", plan.continuation.search_score},
            {"host_helped", plan.continuation.host_helped},
            {"intervention_ref", plan.continuation.intervention_ref},
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

std::string daemon_flydelta_scope_fingerprint(const common_agent_scope & scope) {
    const std::string identity =
        std::to_string(static_cast<int>(scope.memory_scope)) + "\n" +
        std::to_string(static_cast<int>(scope.plan_scope)) + "\n" +
        scope.namespace_id + "\n" + scope.session_id + "\n" +
        scope.project_id + "\n" + scope.turn_id + "\n" +
        (scope.memory_global_opt_in ? "1" : "0");
    return "sha256:" + hash_sha256_hex(identity.data(), identity.size());
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
        plan.search_anchor_ref = value.value("search_anchor_ref", "");
        plan.search_anchor_round = value.value("search_anchor_round", size_t{0});
        plan.search_anchor_pending = value.value("search_anchor_pending", false);
        plan.search_model_call_limit = value.value("search_model_call_limit", size_t{256});
        plan.search_model_calls_used = value.value("search_model_calls_used", size_t{0});
        if (plan.search_anchor_ref.size() > 512 || plan.search_anchor_round > 1024 ||
                plan.search_model_call_limit == 0 || plan.search_model_call_limit > 65536 ||
                plan.search_model_calls_used > plan.search_model_call_limit) {
            error = "FlyDelta orchestration search anchor is invalid";
            return false;
        }
        const auto continuation = value.at("continuation");
        const auto region = continuation.at("region");
        plan.continuation.direction_index = continuation.value("direction_index", size_t{0});
        plan.continuation.region_trial_index = continuation.value("region_trial_index", size_t{0});
        plan.continuation.search_score = continuation.value("search_score", 0.0f);
        plan.continuation.host_helped = continuation.value("host_helped", false);
        plan.continuation.intervention_ref = continuation.value("intervention_ref", "");
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

bool daemon_flydelta_bind_state_callbacks(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        common_flydelta_evaluator_callbacks & callbacks, std::string & callback_error) {
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
        auto state = daemon_flydelta_orchestration_state_json(plan, history, bootstrap_ref);
        const auto payload = state.dump();
        const auto digest = hash_sha256_hex(payload.data(), payload.size()).substr(0, 32);
        state_ref = "flydelta://state/orchestration/" + digest;
        common_flydelta_lifecycle_event_context lifecycle_context;
        lifecycle_context.event_id = "flydelta://event/orchestration/" + digest;
        lifecycle_context.idempotency_key = "flydelta/orchestration/" + digest;
        lifecycle_context.source_id = "daemon-flydelta";
        lifecycle_context.scope.namespace_id = provider->authority.namespace_id;
        lifecycle_context.scope.project_id = provider->authority.project_id;
        lifecycle_context.scope.session_id = provider->authority.session_id;
        lifecycle_context.content_hash = "sha256:" + hash_sha256_hex(payload.data(), payload.size());
        lifecycle_context.created_at = "daemon-runtime-v1";
        if (!common_flydelta_append_orchestration_state_lifecycle(
                *provider->lifecycle_store, lifecycle_context, state_ref,
                payload, state_error)) return false;
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
        auto state = daemon_flydelta_orchestration_state_json(plan, history, bootstrap_ref);
        if (plan.search_anchor_pending && !plan.search_anchor_ref.empty()) {
            // The immutable state and the intent to resume are one
            // lifecycle publication, before the worker can enqueue.
            state["anchor_continuation"] = {
                {"parent_job", json::parse(common_flydelta_experiment_job_to_json(job))},
                {"bootstrap_state_ref", job.bootstrap_zoom_state_ref},
                {"representation_state_ref", job.representation_augmentation_state_ref},
            };
        }
        const auto payload = state.dump();
        const auto digest = hash_sha256_hex(payload.data(), payload.size()).substr(0, 32);
        state_ref = "flydelta://state/orchestration/" + digest;
        common_flydelta_lifecycle_event_context lifecycle_context;
        lifecycle_context.event_id = "flydelta://event/orchestration/" + digest;
        lifecycle_context.idempotency_key = "flydelta/orchestration/" + digest;
        lifecycle_context.source_id = "daemon-flydelta";
        lifecycle_context.scope.namespace_id = scoped_provider->authority.namespace_id;
        lifecycle_context.scope.project_id = scoped_provider->authority.project_id;
        lifecycle_context.scope.session_id = scoped_provider->authority.session_id;
        lifecycle_context.content_hash = "sha256:" + hash_sha256_hex(payload.data(), payload.size());
        lifecycle_context.created_at = "daemon-runtime-v1";
        if (!common_flydelta_append_orchestration_state_lifecycle(
                *scoped_provider->lifecycle_store, lifecycle_context, state_ref,
                payload, state_error)) return false;
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
    return true;
}

} // namespace agent_daemon_flydelta_internal
