#pragma once

#include "agent/adaptation/flydelta/flydelta-queue.h"

#include <functional>
#include <string>
#include <vector>

struct common_flydelta_experiment_worker_result {
    std::string safe_summary;
    std::vector<common_flydelta_counterfactual_report> counterfactual_reports;
};

struct common_flydelta_experiment_worker_report {
    common_flydelta_experiment_queue_state state = common_flydelta_experiment_queue_state::pending;
    std::string job_id;
    std::string safe_summary;
    size_t report_count = 0;
};

// The worker owns queue lifecycle and result validation. The callback owns
// resolving reference IDs and performing bounded host-side work; it must not
// return raw prompts, tool output, credentials or activation tensors here.
using common_flydelta_experiment_worker_callback = std::function<bool(
        const common_flydelta_experiment_job & job,
        common_flydelta_experiment_worker_result & result,
        std::string & error)>;

// Claims and processes at most one job. Callback failures become a terminal
// failed job without exposing callback diagnostics in queue state. An empty
// queue returns true with state=pending and no job ID.
bool common_flydelta_experiment_worker_run_once(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        const common_flydelta_experiment_worker_callback & callback,
        common_flydelta_experiment_worker_report & report,
        std::string & error);

