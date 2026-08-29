#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct common_learning_training_job {
    int schema_version = 1;
    std::string id;
    std::string corpus_revision_id;
    std::string corpus_bundle_hash;
    std::string base_training_fingerprint;
    std::string trainer_kind = "qlora-sft";
    std::string trainer_version;
    std::string code_revision;
    uint64_t seed = 42;
    size_t deadline_seconds = 3600;
};

struct common_learning_training_result {
    int schema_version = 1;
    std::string job_id;
    std::string adapter_id;
    std::string artifact_path;
    std::string artifact_sha256;
    std::string corpus_bundle_hash;
    std::string base_training_fingerprint;
    std::string trainer_kind;
    std::string trainer_version;
    std::string evaluation_revision;
    std::string evaluation_status;
};

bool common_learning_validate_training_job(
        const common_learning_training_job & job,
        std::string & error);

bool common_learning_validate_training_result(
        const common_learning_training_job & job,
        const common_learning_training_result & result,
        std::string & error);
