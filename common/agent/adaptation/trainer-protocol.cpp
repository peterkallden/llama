#include "agent/adaptation/trainer-protocol.h"

#include <cctype>
#include <filesystem>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::ordered_json;

static std::string identity_hash(const std::string & text) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << "identity:fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

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

static bool json_string(const json & value, const char * name, std::string & output, std::string & error) {
    if (!value.contains(name) || !value.at(name).is_string()) {
        error = std::string("training JSON requires string field: ") + name;
        return false;
    }
    output = value.at(name).get<std::string>();
    return true;
}

static bool json_size(const json & value, const char * name, size_t & output, std::string & error) {
    if (!value.contains(name) || !value.at(name).is_number_unsigned()) {
        error = std::string("training JSON requires non-negative integer field: ") + name;
        return false;
    }
    output = value.at(name).get<size_t>();
    return true;
}

bool common_learning_make_training_job(
        const common_learning_corpus_revision & corpus,
        const std::string & base_training_fingerprint,
        const std::string & code_revision,
        const std::string & trainer_version,
        common_learning_training_job & job,
        std::string & error) {
    error.clear();
    if (corpus.id.empty() || corpus.bundle_hash.empty() || base_training_fingerprint.empty() ||
            code_revision.empty() || trainer_version.empty()) {
        error = "training job requires a complete corpus and trainer identity";
        return false;
    }
    job = {};
    const auto identity = corpus.bundle_hash + "\n" + base_training_fingerprint + "\n" +
        "qlora-sft\n" + trainer_version + "\n" + code_revision + "\n" + std::to_string(corpus.seed);
    job.id = "learning://job/" + identity_hash(identity);
    job.corpus_revision_id = corpus.id;
    job.corpus_bundle_hash = corpus.bundle_hash;
    job.base_training_fingerprint = base_training_fingerprint;
    job.code_revision = code_revision;
    job.trainer_version = trainer_version;
    job.seed = corpus.seed;
    return common_learning_validate_training_job(job, error);
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
    if (!nonempty_bounded(result.evaluation_revision) ||
            (result.evaluation_status != "not_run" &&
             result.evaluation_status != "passed" &&
             result.evaluation_status != "failed")) {
        error = "training result has invalid evaluation status"; return false;
    }
    return true;
}

bool common_learning_training_job_to_json(
        const common_learning_training_job & job,
        std::string & output,
        std::string & error) {
    error.clear();
    if (!common_learning_validate_training_job(job, error)) return false;
    output = json{
        {"schema_version", job.schema_version},
        {"id", job.id},
        {"corpus_revision_id", job.corpus_revision_id},
        {"corpus_bundle_hash", job.corpus_bundle_hash},
        {"base_training_fingerprint", job.base_training_fingerprint},
        {"trainer_kind", job.trainer_kind},
        {"trainer_version", job.trainer_version},
        {"code_revision", job.code_revision},
        {"seed", job.seed},
        {"deadline_seconds", job.deadline_seconds},
    }.dump();
    return true;
}

bool common_learning_training_job_from_json(
        const std::string & input,
        common_learning_training_job & job,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(input);
        if (!value.is_object() || !value.contains("schema_version") ||
                !value.at("schema_version").is_number_integer()) {
            error = "training job JSON requires integer schema_version";
            return false;
        }
        job = {};
        job.schema_version = value.at("schema_version").get<int>();
        if (!json_string(value, "id", job.id, error) ||
                !json_string(value, "corpus_revision_id", job.corpus_revision_id, error) ||
                !json_string(value, "corpus_bundle_hash", job.corpus_bundle_hash, error) ||
                !json_string(value, "base_training_fingerprint", job.base_training_fingerprint, error) ||
                !json_string(value, "trainer_kind", job.trainer_kind, error) ||
                !json_string(value, "trainer_version", job.trainer_version, error) ||
                !json_string(value, "code_revision", job.code_revision, error) ||
                !json_size(value, "seed", job.seed, error) ||
                !json_size(value, "deadline_seconds", job.deadline_seconds, error)) {
            return false;
        }
        return common_learning_validate_training_job(job, error);
    } catch (const std::exception & exception) {
        error = std::string("invalid training job JSON: ") + exception.what();
        return false;
    }
}

bool common_learning_training_result_to_json(
        const common_learning_training_result & result,
        std::string & output,
        std::string & error) {
    error.clear();
    output = json{
        {"schema_version", result.schema_version},
        {"job_id", result.job_id},
        {"adapter_id", result.adapter_id},
        {"artifact_path", result.artifact_path},
        {"artifact_sha256", result.artifact_sha256},
        {"corpus_bundle_hash", result.corpus_bundle_hash},
        {"base_training_fingerprint", result.base_training_fingerprint},
        {"trainer_kind", result.trainer_kind},
        {"trainer_version", result.trainer_version},
        {"evaluation_revision", result.evaluation_revision},
        {"evaluation_status", result.evaluation_status},
    }.dump();
    return true;
}

bool common_learning_training_result_from_json(
        const std::string & input,
        common_learning_training_result & result,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(input);
        if (!value.is_object() || !value.contains("schema_version") ||
                !value.at("schema_version").is_number_integer()) {
            error = "training result JSON requires integer schema_version";
            return false;
        }
        result = {};
        result.schema_version = value.at("schema_version").get<int>();
        if (!json_string(value, "job_id", result.job_id, error) ||
                !json_string(value, "adapter_id", result.adapter_id, error) ||
                !json_string(value, "artifact_path", result.artifact_path, error) ||
                !json_string(value, "artifact_sha256", result.artifact_sha256, error) ||
                !json_string(value, "corpus_bundle_hash", result.corpus_bundle_hash, error) ||
                !json_string(value, "base_training_fingerprint", result.base_training_fingerprint, error) ||
                !json_string(value, "trainer_kind", result.trainer_kind, error) ||
                !json_string(value, "trainer_version", result.trainer_version, error) ||
                !json_string(value, "evaluation_revision", result.evaluation_revision, error) ||
                !json_string(value, "evaluation_status", result.evaluation_status, error)) {
            return false;
        }
        return true;
    } catch (const std::exception & exception) {
        error = std::string("invalid training result JSON: ") + exception.what();
        return false;
    }
}
