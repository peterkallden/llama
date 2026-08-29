#pragma once

#include "agent-learning-adapter-import.h"
#include "agent-learning-lifecycle-store.h"
#include "agent-learning-transaction-store.h"
#include "agent-learning-worker.h"
#include "agent/adaptation/adapter-registry.h"

#include <filesystem>
#include <functional>
#include <string>

// Host-level composition of the offline adaptation path.  It deliberately
// does not run inside a user turn and it never treats a trainer artifact as
// evaluated until the injected host evaluator returns a passed report.
struct agent_learning_adaptation_orchestrator_config {
    common_learning_transaction_store & transaction_store;
    common_learning_lifecycle_store & lifecycle_store;
    std::filesystem::path queue_root;
    std::string base_architecture = "qwen2";
    std::string serving_model_fingerprint = "serve:qwen-small";
    std::string tokenizer_fingerprint = "tokenizer:v1";
    std::string chat_template_fingerprint = "chat:v1";
    std::string code_revision = "local-adaptation-smoke";
    std::string trainer_kind = "fake-sft";
    std::string trainer_version = "fake-trainer-1";
    size_t max_artifact_bytes = 64 * 1024 * 1024;
    size_t max_lifecycle_payload_bytes = 4 * 1024 * 1024;
    agent_learning_worker_limits worker_limits;
    std::function<bool(
            const common_learning_training_job & job,
            const common_learning_training_result & result,
            common_learning_evaluation_report & report,
            std::string & error)> evaluator;
};

struct agent_learning_adaptation_orchestrator_result {
    common_learning_corpus_revision corpus;
    common_learning_training_job job;
    common_learning_training_result training_result;
    common_learning_evaluation_report evaluation;
    common_learning_adapter_manifest manifest;
    common_agent_model_profile profile;
    std::filesystem::path verified_artifact_path;
    agent_learning_worker_report worker_report;
    size_t lifecycle_records = 0;
};

bool agent_learning_run_adaptation(
        const common_training_candidate & candidate,
        const agent_learning_adaptation_orchestrator_config & config,
        agent_learning_adaptation_orchestrator_result & result,
        std::string & error);
