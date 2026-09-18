#pragma once

#include "agent/adaptation/flydelta/flydelta-worker.h"

#include <cstddef>
#include <memory>
#include <string>

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

