#pragma once

#include "agent/adaptation/flydelta/flydelta-job.h"

#include <cstddef>
#include <filesystem>
#include <string>

enum class common_flydelta_experiment_queue_state {
    pending,
    running,
    succeeded,
    failed,
    cancelled,
};

const char * common_flydelta_experiment_queue_state_name(
        common_flydelta_experiment_queue_state state);

struct common_flydelta_experiment_queue_limits {
    size_t max_job_bytes = 64 * 1024;
    size_t max_summary_bytes = 4 * 1024;
};

struct common_flydelta_claimed_experiment_job {
    std::string queue_key;
    common_flydelta_experiment_job job;
};

// The queue stores only the typed, reference-only job envelope. Captures,
// activations and verifier payloads stay in their respective stores.
bool common_flydelta_experiment_queue_enqueue(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_queue_limits & limits,
        std::string & error);

// Claims the lexicographically first pending job. An empty queue is not an
// error and returns true with claimed.queue_key empty.
bool common_flydelta_experiment_queue_claim_next(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        common_flydelta_claimed_experiment_job & claimed,
        std::string & error);

// Completes a claimed job by writing a bounded safe status and moving it to a
// terminal state. Result artifacts remain referenceable data, not queue JSON.
bool common_flydelta_experiment_queue_complete(
        const std::filesystem::path & queue_root,
        const common_flydelta_claimed_experiment_job & claimed,
        common_flydelta_experiment_queue_state state,
        const std::string & safe_summary,
        const common_flydelta_experiment_queue_limits & limits,
        std::string & error);
