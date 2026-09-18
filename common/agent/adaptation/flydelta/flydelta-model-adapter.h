#pragma once

#include "agent/adaptation/flydelta/flydelta-worker.h"

#include <cstddef>
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
