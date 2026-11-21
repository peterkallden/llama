#pragma once

#include "agent/adaptation/flydelta/flydelta-job.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

struct common_flydelta_queue_owner;

// Keep this owner alive for the entire daemon/worker lifetime. All queue
// writers also acquire the same process-local owner, so an external writer
// cannot bypass an active daemon's advisory lock through these APIs.
// Supported durability/locking scope: a local filesystem, not a shared NFS queue.
std::shared_ptr<common_flydelta_queue_owner> common_flydelta_acquire_queue_owner(
        const std::filesystem::path & queue_root, std::string & error);

// Call under exclusive ownership before starting any worker. Reconcile a
// durable terminal acknowledgement first; otherwise retain the logical ID
// and return an abandoned running job to pending.
bool common_flydelta_experiment_queue_recover(
        const std::filesystem::path & queue_root,
        const struct common_flydelta_experiment_queue_limits & limits,
        std::string & error);

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
    // Bounds crash replays of the complete worker job, including phases
    // which precede the detailed coefficient-execution ledger.
    size_t max_execution_attempts = 3;
};

struct common_flydelta_claimed_experiment_job {
    std::string queue_key;
    common_flydelta_experiment_job job;
};

// Read-only idempotency check used by host bridges before enqueue. It scans
// only the bounded queue state directories and never reads payload content.
bool common_flydelta_experiment_queue_contains(
        const std::filesystem::path & queue_root,
        const std::string & job_id,
        bool & contains,
        std::string & error);

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

// Inspect the unique queue state and verify the stored job envelope before
// recovery performs any follow-up writes.
bool common_flydelta_experiment_queue_job_state(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_queue_limits & limits,
        common_flydelta_experiment_queue_state & state,
        std::string & error);

// A verified, journaled worker result may precede its queue acknowledgement.
// Reconcile it without executing the evaluator again. Missing/mismatched jobs
// are errors; a matching already terminal acknowledgement is idempotent.
bool common_flydelta_experiment_queue_acknowledge_recovered_result(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_job & job,
        const std::string & safe_summary,
        const common_flydelta_experiment_queue_limits & limits,
        std::string & error);
