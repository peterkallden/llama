#include "agent-learning-worker.h"

#include "hash/hash.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

static void require(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::abort();
    }
}

static common_learning_training_job job(const std::string & id, const std::string & trainer_kind) {
    common_learning_training_job value;
    value.id = "learning://job/" + id;
    value.corpus_revision_id = "learning://corpus/revision-1";
    value.corpus_bundle_hash = "identity:fnv1a64:0123456789abcdef";
    value.base_training_fingerprint = "base:qwen-small:training-v1";
    value.trainer_kind = trainer_kind;
    value.trainer_version = "fake-trainer-1";
    value.code_revision = "worker-test-1";
    value.seed = 42;
    value.deadline_seconds = 60;
    return value;
}

static common_learning_corpus_revision corpus() {
    common_learning_corpus_revision value;
    value.id = "learning://corpus/revision-1";
    value.bundle_hash = "identity:fnv1a64:0123456789abcdef";
    value.jsonl = "{\"input\":\"fixture\",\"target\":\"valid\"}\n";
    value.manifest_json = "{\"schema_version\":1,\"jsonl_hash\":\"identity:fnv1a64:fixture\"}";
    return value;
}

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

static std::string queue_key(const std::string & job_id) {
    return "job-" + hash_sha256_hex(job_id.data(), job_id.size()).substr(0, 32);
}

int main() {
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-learning-worker-test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);

    const auto first_job = job("fake-1", "fake-sft");
    const auto first_corpus = corpus();
    std::string error;
    require(agent_learning_enqueue_job(root, first_job, first_corpus, error),
            "valid worker job was not enqueued: " + error);
    const auto first_key = queue_key(first_job.id);
    require(std::filesystem::exists(root / "pending" / first_key / "job.json"),
            "job bundle was not atomically published");
    require(!agent_learning_enqueue_job(root, first_job, first_corpus, error),
            "duplicate worker job was accepted");

    agent_learning_worker_report report;
    require(agent_learning_worker_run_once(root, {}, report, error), "worker run failed: " + error);
    require(report.state == agent_learning_worker_job_state::succeeded, "fake worker job did not succeed");
    const auto succeeded_directory = root / "succeeded" / first_key;
    require(std::filesystem::exists(succeeded_directory / "state.json"), "worker state was not persisted");
    common_learning_training_result result;
    require(common_learning_training_result_from_json(read_file(succeeded_directory / "result.json"), result, error),
            "worker result was not valid JSON: " + error);
    require(common_learning_validate_training_result(first_job, result, error),
            "worker result did not satisfy the trainer contract: " + error);
    const auto artifact = succeeded_directory / "artifacts" / result.artifact_path;
    require(std::filesystem::exists(artifact), "worker artifact is missing");
    const auto artifact_bytes = read_file(artifact);
    require(result.artifact_sha256 == "sha256:" + hash_sha256_hex(artifact_bytes.data(), artifact_bytes.size()),
            "worker artifact hash was not reproducible");
    const auto state = json::parse(read_file(succeeded_directory / "state.json"));
    require(state.at("state") == "succeeded", "terminal worker state is incorrect");

    require(agent_learning_worker_run_once(root, {}, report, error), "empty worker run failed");
    require(report.state == agent_learning_worker_job_state::idle, "empty worker queue was not idle");

    const auto cancelled_job = job("fake-2", "fake-sft");
    require(agent_learning_enqueue_job(root, cancelled_job, first_corpus, error), "cancel job was not enqueued");
    std::ofstream(root / "pending" / queue_key(cancelled_job.id) / "cancel") << "cancel";
    require(agent_learning_worker_run_once(root, {}, report, error), "cancelled worker run failed");
    require(report.state == agent_learning_worker_job_state::cancelled, "cancel request was ignored");
    require(std::filesystem::exists(root / "cancelled" / queue_key(cancelled_job.id) / "state.json"),
            "cancelled job was not finalized");

    const auto unsupported_job = job("unsupported-1", "qlora-sft");
    require(agent_learning_enqueue_job(root, unsupported_job, first_corpus, error),
            "unsupported job was not enqueued");
    require(agent_learning_worker_run_once(root, {}, report, error), "unsupported worker run failed at queue level");
    require(report.state == agent_learning_worker_job_state::failed, "unsupported trainer was not rejected");
    require(std::filesystem::exists(root / "failed" / queue_key(unsupported_job.id) / "state.json"),
            "failed job was not finalized");

    std::filesystem::remove_all(root, ec);
    return 0;
}
