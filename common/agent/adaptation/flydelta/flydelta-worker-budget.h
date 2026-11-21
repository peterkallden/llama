#pragma once

#include <cstddef>
#include <string>

// Describes the worker capacity split without owning threads or queues.
// The daemon/host scheduler consumes this value when a FlyDelta lane is
// actually installed; the arithmetic is kept separate so configuration and
// scheduling can be tested without starting inference.
struct common_flydelta_worker_budget {
    size_t total_workers = 1;
    bool enabled = false;
    size_t requested_flydelta_workers = 0;
    size_t flydelta_workers = 0;
    size_t agent_workers = 1;
};

bool common_flydelta_worker_budget_compute(
        size_t total_workers,
        bool enabled,
        size_t requested_flydelta_workers,
        common_flydelta_worker_budget & budget,
        std::string & error);
