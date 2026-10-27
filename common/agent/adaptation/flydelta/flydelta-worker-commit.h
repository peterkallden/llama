#pragma once

#include "agent/adaptation/flydelta/flydelta-worker.h"
#include "agent/adaptation/lifecycle-store.h"

// Persist only the existing host commit/scheduling inputs, not a second
// evaluator state. Replay invokes the same idempotent host callbacks.
bool common_flydelta_append_worker_commit_intent(common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_worker_report & report, std::string & error,
        const common_flydelta_experiment_job * scheduling_parent = nullptr);

bool common_flydelta_resolve_worker_commit_parent(const common_learning_lifecycle_store & store,
        const common_flydelta_experiment_job & job,
        common_flydelta_experiment_job & parent, std::string & error);

bool common_flydelta_recover_worker_commits(common_learning_lifecycle_store & store,
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        const common_flydelta_worker_commit_callback & persist,
        const common_flydelta_worker_commit_callback & schedule, std::string & error);
