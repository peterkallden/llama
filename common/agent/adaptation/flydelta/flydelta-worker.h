#pragma once

#include "agent/adaptation/flydelta/flydelta-queue.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"
#include "agent/adaptation/flydelta/flydelta-aggregation.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"
#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#include <functional>
#include <string>
#include <vector>

struct common_flydelta_evaluator_config;
struct common_flydelta_evaluator_callbacks;

struct common_flydelta_experiment_worker_result {
    std::string safe_summary;
    std::vector<common_flydelta_counterfactual_report> counterfactual_reports;
    std::vector<common_flydelta_direction_candidate> direction_candidates;
    std::vector<common_flydelta_basis_direction> basis_directions;
    std::vector<common_flydelta_search_pipeline_result> search_pipeline_results;
    std::vector<common_flydelta_search_continuation> search_continuations;
    std::vector<float> delta_memory_weights;
    common_flydelta_aggregation_snapshot aggregation;
    common_flydelta_evidence_depth_result evidence_depth;
    common_flydelta_search_budget search_budget;
    bool has_bootstrap_zoom_state = false;
    common_flydelta_bootstrap_zoom_state bootstrap_zoom_state;
    std::string bootstrap_zoom_state_ref;
};

struct common_flydelta_experiment_worker_report {
    common_flydelta_experiment_queue_state state = common_flydelta_experiment_queue_state::pending;
    std::string job_id;
    std::string safe_summary;
    size_t report_count = 0;
    common_flydelta_evidence_depth_result evidence_depth;
    common_flydelta_search_budget search_budget;
    bool has_bootstrap_zoom_state = false;
    common_flydelta_bootstrap_zoom_state bootstrap_zoom_state;
    std::string bootstrap_zoom_state_ref;
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

// Adapts the queue worker to the common evaluator. The evaluator callbacks
// still own reference resolution, fresh model contexts and host verification;
// this bridge only transports typed evaluator results into the worker result.
bool common_flydelta_experiment_worker_run_evaluator_once(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_experiment_worker_report & report,
        std::string & error);
