#include "agent-daemon-flydelta-internal.h"
#include "flydelta/state.h"
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

namespace agent_daemon_flydelta_internal {

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
        auto provider = daemon_flydelta_make_resource_provider(
            options, host, resources, teaching_material_runtime,
            lifecycle_store, error);
        if (!provider) return false;

        daemon_flydelta_bind_execution_callbacks(
            provider, teaching_material_runtime, binding);
        binding.register_evaluator = [provider](common_flydelta_evaluator_config & config,
                common_flydelta_evaluator_callbacks & callbacks, std::string & callback_error) {
            return daemon_flydelta_register_evaluator(provider, config, callbacks, callback_error);
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
        (options.n_gpu_layers > 0 || options.adaptation_flydelta_factorized_cvec_batch);
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

    runtime.flydelta_concept_candidate_index =
        std::make_shared<common_agent_concept_candidate_index>(
            runtime.flydelta_review_lifecycle_store.get());
    if (!runtime.flydelta_concept_candidate_index->load(error)) {
        runtime.flydelta_concept_candidate_index.reset();
        runtime.flydelta_sideband_review_store.reset();
        runtime.flydelta_sideband_registry.reset();
        runtime.flydelta_review_lifecycle_store.reset();
        return false;
    }

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
    if (options.adaptation_flydelta_factorized_cvec_batch) {
        const std::string profile_id = options.model_profile.empty()
            ? catalog.default_profile : options.model_profile;
        const auto profile = catalog.profiles.find(profile_id);
        const auto base = profile == catalog.profiles.end()
            ? catalog.bases.end() : catalog.bases.find(profile->second.base_model_id);
        if (profile == catalog.profiles.end() || base == catalog.bases.end() ||
                base->second.backend != "server-context") {
            error = "factorized cvec batching requires a server-context model profile; set runtime.adaptation.flydelta.disable_factorized_cvec_batch=true to use sparse/dense overlays";
            return false;
        }
    }
    apply_flydelta_auto_batch_capacity(options, catalog);
    if (options.adaptation_flydelta_factorized_cvec_batch) {
        const std::string profile_id = options.model_profile.empty()
            ? catalog.default_profile : options.model_profile;
        const auto profile = catalog.profiles.find(profile_id);
        if (profile == catalog.profiles.end() || profile->second.n_parallel < 2 ||
                profile->second.n_sequences < 2) {
            error = "factorized cvec batching requires a model profile with parallel and sequence capacity of at least 2; set runtime.adaptation.flydelta.disable_factorized_cvec_batch=true to use sparse/dense overlays";
            return false;
        }
    }
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
    config.concept_candidate_index = runtime.flydelta_concept_candidate_index;
    config.flydelta_teaching_material_observer =
        runtime.flydelta_teaching_material_observer;
    config.flydelta_reuse_first_observer = runtime.flydelta_reuse_first_observer;
    config.flydelta_teaching_material_runtime =
        runtime.flydelta_teaching_material_runtime;
}

} // namespace agent_daemon_flydelta_internal
