#pragma once

#include "agent/adaptation/flydelta/flydelta-worker.h"

namespace common_flydelta_trace_detail {
// Reporting only: never reclassifies evidence or chooses a continuation.
void append_worker_result(const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_worker_result & result, common_flydelta_trace & trace);
} // namespace common_flydelta_trace_detail
