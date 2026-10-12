#pragma once

#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

// Experiment routes are host-neutral control-plane adapters. They consume one
// completed slice and return the existing orchestration result; they do not
// execute a model, write lifecycle state, or grant evidence/promotion credit.
// The orchestration header remains the public facade for existing callers.

bool common_flydelta_route_bootstrap(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error);

bool common_flydelta_route_shallow_controls(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error);

bool common_flydelta_route_deep_controls(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error);

// Dispatches one completed slice to the route for the current experiment
// phase. Phase-specific route names are intentionally explicit even though
// V1 shares the same policy implementation; this keeps route structure
// separate from the existing gate and transition semantics.
bool common_flydelta_route_experiment_slice(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error);
