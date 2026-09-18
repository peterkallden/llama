#pragma once

#include "agent/adaptation/flydelta/flydelta-worker.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

struct common_flydelta_evaluator_config;
struct common_flydelta_evaluator_callbacks;

// Host/runtime capabilities for one concrete model-facing FlyDelta adapter.
// These flags describe what the adapter can execute; they do not grant search
// permission, evidence rank, learning credit or promotion authority.
struct common_flydelta_model_capabilities {
    bool bootstrap_zoom = false;
    bool adaptive_alpha = false;
    bool teacher_forced_margin = false;
    bool orthogonal_search = false;
    bool representation_augmentation = false;
    bool host_verification = false;
};

// Runtime-owned registration point for model-facing FlyDelta execution. The
// host owns model contexts, opaque reference resolution, fresh inference and
// host verification; the common worker only receives the resulting callback.
struct common_flydelta_model_host {
    common_flydelta_model_capabilities capabilities;
    std::function<bool(
            common_flydelta_evaluator_config & config,
            common_flydelta_evaluator_callbacks & callbacks,
            std::string & error)> register_evaluator;
};

bool common_flydelta_model_host_validate(
        const common_flydelta_model_host & host,
        std::string & error);

struct common_flydelta_model_adapter {
    common_flydelta_model_capabilities capabilities;
    common_flydelta_experiment_worker_callback worker_callback;
};

bool common_flydelta_model_adapter_validate(
        const common_flydelta_model_adapter & adapter,
        std::string & error);

// A capability is useful only when the adapter callback is registered too.
bool common_flydelta_model_adapter_supports_search(
        const common_flydelta_model_adapter & adapter);

// Builds the production-facing bridge around the existing bounded evaluator.
// The supplied callbacks remain host-owned: they resolve opaque references,
// create fresh inference contexts and perform host verification. This factory
// is model/backend-neutral and only composes evaluator execution with the
// generic worker callback.
std::shared_ptr<const common_flydelta_model_adapter>
common_flydelta_model_adapter_from_evaluator(
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_model_capabilities capabilities,
        std::string & error);

// Production composition seam. This remains backend- and model-neutral; the
// supplied host decides how to resolve refs and execute fresh model work.
std::shared_ptr<const common_flydelta_model_adapter>
common_flydelta_model_adapter_from_host(
        const common_flydelta_model_host & host,
        std::string & error);
