#include "agent/adaptation/trainer-protocol.h"

#include <cctype>
#include <filesystem>

static bool nonempty_bounded(const std::string & value, size_t max = 512) {
    return !value.empty() && value.size() <= max;
}

static bool sha256_shape(const std::string & value) {
    if (value.rfind("sha256:", 0) != 0 || value.size() != 71) return false;
    for (size_t i = 7; i < value.size(); ++i) if (!std::isxdigit(static_cast<unsigned char>(value[i]))) return false;
    return true;
}

static bool safe_artifact_path(const std::string & value) {
    const std::filesystem::path path(value);
    return !path.empty() && !path.is_absolute() && path.lexically_normal() == path &&
        value.find("..") == std::string::npos;
}

bool common_learning_validate_training_job(const common_learning_training_job & job, std::string & error) {
    error.clear();
    if (job.schema_version != 1) { error = "unsupported training job schema"; return false; }
    if (!nonempty_bounded(job.id) || !nonempty_bounded(job.corpus_revision_id) ||
            !nonempty_bounded(job.corpus_bundle_hash) ||
            !nonempty_bounded(job.base_training_fingerprint) || !nonempty_bounded(job.trainer_kind) ||
            !nonempty_bounded(job.trainer_version) || !nonempty_bounded(job.code_revision)) {
        error = "training job has missing or oversized identity"; return false;
    }
    if (job.deadline_seconds == 0 || job.deadline_seconds > 7 * 24 * 60 * 60) {
        error = "training job deadline is outside bounds"; return false;
    }
    return true;
}

bool common_learning_validate_training_result(
        const common_learning_training_job & job,
        const common_learning_training_result & result,
        std::string & error) {
    error.clear();
    if (!common_learning_validate_training_job(job, error)) return false;
    if (result.schema_version != 1 || result.job_id != job.id) { error = "training result does not match job"; return false; }
    if (!nonempty_bounded(result.adapter_id) || !nonempty_bounded(result.artifact_path) ||
            !nonempty_bounded(result.base_training_fingerprint) || !sha256_shape(result.artifact_sha256)) {
        error = "training result has invalid artifact identity"; return false;
    }
    if (!safe_artifact_path(result.artifact_path) ||
            result.corpus_bundle_hash != job.corpus_bundle_hash ||
            result.base_training_fingerprint != job.base_training_fingerprint ||
            result.trainer_kind != job.trainer_kind || result.trainer_version != job.trainer_version) {
        error = "training result does not match training inputs"; return false;
    }
    if (!nonempty_bounded(result.evaluation_revision) || result.evaluation_status != "passed") {
        error = "training result lacks a passed evaluation"; return false;
    }
    return true;
}
