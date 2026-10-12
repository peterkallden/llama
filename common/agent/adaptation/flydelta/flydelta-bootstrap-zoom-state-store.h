#pragma once

#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/lifecycle-store.h"

#include <string>

// The lifecycle journal owns durable BootstrapZoom resume state. The worker
// only sees opaque references and invokes the callbacks configured here.
// A host supplies its identity/timestamp policy rather than the adapter
// inventing runtime scope or wall-clock data.
struct common_flydelta_bootstrap_zoom_lifecycle_context {
    std::string namespace_id = "local";
    std::string project_id;
    std::string session_id;
    std::string source_id;
    std::string created_at;
};

bool common_flydelta_bootstrap_zoom_lifecycle_context_validate(
        const common_flydelta_bootstrap_zoom_lifecycle_context & context,
        std::string & error);

// Adds resolve/persist callbacks to an evaluator callback bundle. It leaves
// all model execution callbacks host-owned and does not replace them.
bool common_flydelta_configure_bootstrap_zoom_lifecycle_callbacks(
        common_learning_lifecycle_store & store,
        const common_flydelta_bootstrap_zoom_lifecycle_context & context,
        common_flydelta_evaluator_callbacks & callbacks,
        std::string & error);
