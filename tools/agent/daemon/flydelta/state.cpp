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
