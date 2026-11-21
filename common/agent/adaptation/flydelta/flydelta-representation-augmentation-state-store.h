#pragma once

#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/lifecycle-store.h"

#include <string>

// Uses the existing append-only FlyDelta lifecycle journal. The callbacks
// expose only opaque refs to the worker; state JSON remains reference-only.
bool common_flydelta_configure_representation_augmentation_lifecycle_callbacks(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        common_flydelta_evaluator_callbacks & callbacks,
        std::string & error);
