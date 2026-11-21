#include "agent/adaptation/flydelta/flydelta-worker-budget.h"

bool common_flydelta_worker_budget_compute(
        const size_t total_workers,
        const bool enabled,
        const size_t requested_flydelta_workers,
        common_flydelta_worker_budget & budget,
        std::string & error) {
    error.clear();
    budget = {};
    budget.total_workers = total_workers;
    budget.enabled = enabled;
    budget.requested_flydelta_workers = requested_flydelta_workers;

    if (total_workers == 0) {
        error = "total worker count must be greater than zero";
        return false;
    }
    if (!enabled && requested_flydelta_workers != 0) {
        error = "FlyDelta worker count must be zero when FlyDelta workers are disabled";
        return false;
    }
    if (enabled && requested_flydelta_workers == 0) {
        error = "enabled FlyDelta workers require a positive worker count";
        return false;
    }
    if (requested_flydelta_workers > total_workers) {
        error = "FlyDelta worker count cannot exceed the total worker count";
        return false;
    }

    budget.flydelta_workers = enabled ? requested_flydelta_workers : 0;
    budget.agent_workers = total_workers - budget.flydelta_workers;
    return true;
}
