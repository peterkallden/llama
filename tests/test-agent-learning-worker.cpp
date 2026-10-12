#include "agent-learning-worker.h"

#include "hash/hash.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

using plain_json = nlohmann::json;

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

static void write_file(const std::filesystem::path & path, const std::string & value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "could not create worker test file");
    output << value;
    require(static_cast<bool>(output), "could not write worker test file");
}

static std::string queue_key(const std::string & job_id) {
    return "job-" + hash_sha256_hex(job_id.data(), job_id.size()).substr(0, 32);
}

static int run_external_fixture(int argc, char ** argv) {
    std::filesystem::path job_path;
    std::filesystem::path artifacts;
    std::filesystem::path result_path;
    for (int index = 2; index + 1 < argc; index += 2) {
        const std::string argument = argv[index];
        if (argument == "--job") job_path = argv[index + 1];
        else if (argument == "--artifacts") artifacts = argv[index + 1];
        else if (argument == "--result") result_path = argv[index + 1];
    }
    if (job_path.empty() || artifacts.empty() || result_path.empty()) return 2;
    std::string error;
    common_learning_training_job input;
    if (!common_learning_training_job_from_json(read_file(job_path), input, error)) return 3;
    const auto artifact_relative = std::filesystem::path("adapters") / "external-fixture.gguf";
    const auto artifact = artifacts / artifact_relative;
    std::error_code ec;
    std::filesystem::create_directories(artifact.parent_path(), ec);
    if (ec) return 4;
    write_file(artifact, "external fixture adapter\n");
    const auto bytes = read_file(artifact);
    common_learning_training_result output;
    output.job_id = input.id;
    output.adapter_id = "external-fixture";
    output.artifact_path = artifact_relative.generic_string();
    output.artifact_sha256 = "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
    output.corpus_bundle_hash = input.corpus_bundle_hash;
    output.base_training_fingerprint = input.base_training_fingerprint;
    output.trainer_kind = input.trainer_kind;
    output.trainer_version = input.trainer_version;
    output.evaluation_revision = "external-fixture:not-run";
    output.evaluation_status = "not_run";
    std::string output_json;
    if (!common_learning_training_result_to_json(output, output_json, error)) return 5;
    write_file(result_path, output_json);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--external-fixture") {
        return run_external_fixture(argc, argv);
    }
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
    require(result.evaluation_status == "not_run",
            "fake trainer incorrectly claimed that evaluation passed");
    const auto artifact = succeeded_directory / "artifacts" / result.artifact_path;
    require(std::filesystem::exists(artifact), "worker artifact is missing");
    const auto artifact_bytes = read_file(artifact);
    require(result.artifact_sha256 == "sha256:" + hash_sha256_hex(artifact_bytes.data(), artifact_bytes.size()),
            "worker artifact hash was not reproducible");
    const auto state = plain_json::parse(read_file(succeeded_directory / "state.json"));
    require(state.at("state") == "succeeded", "terminal worker state is incorrect");
    require(state.at("worker_id") == "local-worker", "worker identity was not persisted");
    require(state.at("lease_expires_at_epoch_seconds") == 0, "terminal worker lease was not cleared");

    const auto capabilities = plain_json::parse(agent_learning_worker_capabilities_json());
    require(!capabilities.at("cuda").get<bool>(), "fake worker unexpectedly advertises CUDA");
    require(capabilities.at("trainer_kinds").at(0) == "fake-sft", "worker capability contract is incorrect");
    require(capabilities.at("max_parallel_jobs") == 1, "worker parallelism contract is incorrect");
    agent_learning_worker_limits configured_limits;
    configured_limits.external_trainer_commands["qlora-sft"] = {"operator-owned-qlora-entrypoint"};
    const auto configured_capabilities = plain_json::parse(agent_learning_worker_capabilities_json(configured_limits));
    require(configured_capabilities.at("trainer_kinds").at(1) == "qlora-sft",
            "configured external trainer is absent from worker capabilities");

    const auto external_job = job("external-1", "qlora-sft");
    require(agent_learning_enqueue_job(root, external_job, first_corpus, error),
            "external trainer job was not enqueued");
    agent_learning_worker_limits external_limits;
    external_limits.external_trainer_commands["qlora-sft"] = {
        std::filesystem::absolute(argv[0]).string(), "--external-fixture"};
    require(agent_learning_worker_run_once(root, external_limits, report, error),
            "external worker run failed at queue level: " + error);
    require(report.state == agent_learning_worker_job_state::succeeded,
            "external trainer fixture did not complete: " + report.safe_summary);
    require(report.safe_summary == "external trainer completed",
            "external trainer was reported with the wrong completion summary");
    const auto external_result_path = root / "succeeded" / queue_key(external_job.id) / "result.json";
    common_learning_training_result external_result;
    require(common_learning_training_result_from_json(read_file(external_result_path), external_result, error) &&
            common_learning_validate_training_result(external_job, external_result, error),
            "external trainer result did not satisfy the trainer contract: " + error);
    require(std::filesystem::exists(root / "succeeded" / queue_key(external_job.id) / "trainer.log"),
            "external worker did not persist bounded trainer output");

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

    const auto bounded_job = job("bounded-1", "fake-sft");
    require(agent_learning_enqueue_job(root, bounded_job, first_corpus, error),
            "bounded job was not enqueued");
    agent_learning_worker_limits bounded_limits;
    bounded_limits.max_corpus_bytes = 1;
    require(agent_learning_worker_run_once(root, bounded_limits, report, error),
            "bounded worker run failed at queue level");
    require(report.state == agent_learning_worker_job_state::failed, "corpus byte bound was ignored");
    require(plain_json::parse(read_file(root / "failed" / queue_key(bounded_job.id) / "state.json")).at("safe_summary") ==
                "corpus bundle exceeds worker byte bound", "bounded job failure was not explained");

    const auto stale_key = "job-stale-running";
    const auto stale_directory = root / "running" / stale_key;
    std::filesystem::create_directories(stale_directory);
    common_learning_training_job stale_job = job("stale-1", "fake-sft");
    std::string stale_job_json;
    require(common_learning_training_job_to_json(stale_job, stale_job_json, error),
            "stale job could not be serialized");
    write_file(stale_directory / "job.json", stale_job_json);
    write_file(stale_directory / "state.json", plain_json{
        {"state", "running"},
        {"job_id", stale_job.id},
        {"safe_summary", "job claimed"},
        {"worker_id", "dead-worker"},
        {"updated_at_epoch_seconds", 1},
        {"lease_expires_at_epoch_seconds", 1},
    }.dump());
    require(agent_learning_worker_run_once(root, {}, report, error), "stale worker run failed");
    const auto stale_state = plain_json::parse(read_file(root / "failed" / stale_key / "state.json"));
    require(stale_state.at("safe_summary") == "worker lease expired; job may be retried",
            "stale running job was not recovered");

    std::filesystem::remove_all(root, ec);
    return 0;
}
