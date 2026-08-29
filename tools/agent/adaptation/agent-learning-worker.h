#pragma once

#include "agent/adaptation/trainer-protocol.h"

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

enum class agent_learning_worker_job_state {
    idle,
    running,
    succeeded,
    failed,
    cancelled,
};

const char * agent_learning_worker_job_state_name(agent_learning_worker_job_state state);

struct agent_learning_worker_limits {
    size_t max_job_bytes = 64 * 1024;
    size_t max_artifact_bytes = 64 * 1024 * 1024;
    size_t max_corpus_bytes = 64 * 1024 * 1024;
    size_t max_manifest_bytes = 1 * 1024 * 1024;
    size_t max_result_bytes = 1 * 1024 * 1024;
    size_t max_job_runtime_seconds = 24 * 60 * 60;
    std::string worker_id = "local-worker";
    // Each command is an operator-owned argv prefix. The worker appends only
    // fixed file arguments for the claimed immutable bundle; it never expands
    // model- or corpus-provided text into a shell command.
    std::map<std::string, std::vector<std::string>> external_trainer_commands;
};

struct agent_learning_worker_capabilities {
    bool cuda = false;
    std::vector<std::string> trainer_kinds = {"fake-sft"};
    size_t max_parallel_jobs = 1;
};

std::string agent_learning_worker_capabilities_json();
std::string agent_learning_worker_capabilities_json(const agent_learning_worker_limits & limits);

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
