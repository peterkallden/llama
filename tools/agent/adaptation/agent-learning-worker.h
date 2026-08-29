#pragma once

#include "agent/adaptation/trainer-protocol.h"

#include <cstddef>
#include <filesystem>
#include <string>

enum class agent_learning_worker_job_state {
    idle,
    running,
    succeeded,
    failed,
    cancelled,
};

const char * agent_learning_worker_job_state_name(agent_learning_worker_job_state state);

struct agent_learning_worker_limits {
    size_t max_artifact_bytes = 64 * 1024 * 1024;
};

struct agent_learning_worker_report {
    agent_learning_worker_job_state state = agent_learning_worker_job_state::idle;
    std::string job_id;
    std::string safe_summary;
};

// Writes a complete, immutable job bundle through a staging directory and an
// atomic rename into queue_root/pending. A job id may be queued only once.
bool agent_learning_enqueue_job(
        const std::filesystem::path & queue_root,
        const common_learning_training_job & job,
        const common_learning_corpus_revision & corpus,
        std::string & error);

// Claims at most one pending job. It returns true with an idle report when the
// queue is empty and false only for a queue/IO error.
bool agent_learning_worker_run_once(
        const std::filesystem::path & queue_root,
        const agent_learning_worker_limits & limits,
        agent_learning_worker_report & report,
        std::string & error);
