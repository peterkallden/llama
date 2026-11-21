#pragma once

#include "plan/plan-types.h"

#include <cstddef>

struct common_plan_blueprint_validation_config {
    size_t maximum_steps = 128;
    size_t maximum_dependencies_per_step = 32;
    size_t maximum_text_bytes = 4096;
};

// Validates the immutable blueprint template before selection or
// instantiation. The validator is intentionally independent of a store so
// bootstrap, learned and imported blueprints share one structural boundary.
bool common_plan_validate_blueprint(
        const common_plan_state & blueprint,
        const common_plan_blueprint_validation_config & config,
        std::string & error);

// Instantiates a persisted blueprint as an independently mutable task plan.
bool common_plan_instantiate_blueprint(
    const common_plan_state & blueprint,
    const std::string & instance_id,
    const std::string & session_id,
    common_plan_state & instance,
    std::string & error,
    common_plan_scope scope = common_plan_scope::turn,
    int64_t now = 0);
