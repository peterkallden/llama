#include "agent-learning-worker.h"

#include "hash/hash.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <system_error>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

std::string queue_key(const std::string & job_id, std::string & error) {
    static constexpr const char * prefix = "learning://job/";
    if (job_id.rfind(prefix, 0) != 0 || job_id.size() == std::char_traits<char>::length(prefix)) {
        error = "training job id is not a queue-safe learning id";
        return {};
    }
    return "job-" + hash_sha256_hex(job_id.data(), job_id.size()).substr(0, 32);
}

bool write_text(const std::filesystem::path & path, const std::string & text, std::string & error) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot write worker file: " + path.string();
        return false;
    }
    output << text;
    if (!output) {
        error = "failed writing worker file: " + path.string();
        return false;
    }
    return true;
}

bool read_text(const std::filesystem::path & path, std::string & text, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot read worker file: " + path.string();
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        error = "failed reading worker file: " + path.string();
        return false;
    }
    text = buffer.str();
    return true;
}

bool write_state(
        const std::filesystem::path & directory,
        const agent_learning_worker_job_state state,
        const std::string & job_id,
        const std::string & summary,
        std::string & error) {
    const auto text = json{
        {"state", agent_learning_worker_job_state_name(state)},
        {"job_id", job_id},
        {"safe_summary", summary},
    }.dump();
    return write_text(directory / "state.json", text, error);
}

bool create_queue_directories(const std::filesystem::path & root, std::string & error) {
    std::error_code ec;
    for (const auto & name : {"pending", "running", "succeeded", "failed", "cancelled"}) {
        std::filesystem::create_directories(root / name, ec);
        if (ec) {
            error = "cannot create worker queue directory: " + (root / name).string();
            return false;
        }
    }
    return true;
}

bool move_terminal(
        const std::filesystem::path & running,
        const std::filesystem::path & terminal_root,
        const std::string & key,
        std::string & error) {
    std::error_code ec;
    std::filesystem::create_directories(terminal_root, ec);
    if (ec) {
        error = "cannot create worker terminal directory";
        return false;
    }
    std::filesystem::rename(running, terminal_root / key, ec);
    if (ec) {
        error = "cannot finalize worker job: " + ec.message();
        return false;
    }
    return true;
}

bool fail_claimed_job(
        const std::filesystem::path & running,
        const std::filesystem::path & queue_root,
        const std::string & key,
        const std::string & job_id,
        const std::string & summary,
        agent_learning_worker_report & report,
        std::string & error) {
    report = {agent_learning_worker_job_state::failed, job_id, summary};
    if (!write_state(running, report.state, job_id, summary, error)) return false;
    return move_terminal(running, queue_root / "failed", key, error);
}

bool cancellation_requested(const std::filesystem::path & directory) {
    std::error_code ec;
    return std::filesystem::exists(directory / "cancel", ec) && !ec;
}

bool run_fake_trainer(
        const std::filesystem::path & running,
        const common_learning_training_job & job,
        const std::string & corpus_jsonl,
        const agent_learning_worker_limits & limits,
        common_learning_training_result & result,
        std::string & error) {
    if (job.trainer_kind != "fake-sft") {
        error = "worker has no enabled trainer for kind: " + job.trainer_kind;
        return false;
    }
    const std::string artifact = "fake-adapter\njob=" + job.id + "\ncorpus=" +
        job.corpus_bundle_hash + "\n" + corpus_jsonl;
    if (artifact.size() > limits.max_artifact_bytes) {
        error = "fake trainer artifact exceeds worker byte bound";
        return false;
    }
    const std::string adapter_id = "agent-adaptation-fake-" + hash_sha256_hex(job.id.data(), job.id.size()).substr(0, 16);
    const std::filesystem::path artifact_relative = std::filesystem::path("adapters") / (adapter_id + ".gguf");
    const auto artifact_path = running / "artifacts" / artifact_relative;
    std::error_code ec;
    std::filesystem::create_directories(artifact_path.parent_path(), ec);
    if (ec || !write_text(artifact_path, artifact, error)) return false;
    result = {};
    result.job_id = job.id;
    result.adapter_id = adapter_id;
    result.artifact_path = artifact_relative.generic_string();
    result.artifact_sha256 = "sha256:" + hash_sha256_hex(artifact.data(), artifact.size());
    result.corpus_bundle_hash = job.corpus_bundle_hash;
    result.base_training_fingerprint = job.base_training_fingerprint;
    result.trainer_kind = job.trainer_kind;
    result.trainer_version = job.trainer_version;
    result.evaluation_revision = "fake-evaluation:" + job.id;
    result.evaluation_status = "passed";
    return true;
}

} // namespace

const char * agent_learning_worker_job_state_name(agent_learning_worker_job_state state) {
    switch (state) {
        case agent_learning_worker_job_state::idle: return "idle";
        case agent_learning_worker_job_state::running: return "running";
        case agent_learning_worker_job_state::succeeded: return "succeeded";
        case agent_learning_worker_job_state::failed: return "failed";
        case agent_learning_worker_job_state::cancelled: return "cancelled";
    }
    return "failed";
}

bool agent_learning_enqueue_job(
        const std::filesystem::path & queue_root,
        const common_learning_training_job & job,
        const common_learning_corpus_revision & corpus,
        std::string & error) {
    error.clear();
    if (!common_learning_validate_training_job(job, error)) return false;
    if (corpus.id != job.corpus_revision_id || corpus.bundle_hash != job.corpus_bundle_hash ||
            corpus.jsonl.empty() || corpus.manifest_json.empty()) {
        error = "worker job bundle does not match corpus identity";
        return false;
    }
    const auto key = queue_key(job.id, error);
    if (key.empty()) return false;
    if (!create_queue_directories(queue_root, error)) return false;
    std::error_code ec;
    for (const auto & state : {"pending", "running", "succeeded", "failed", "cancelled"}) {
        if (std::filesystem::exists(queue_root / state / key, ec) && !ec) {
            error = "training job is already queued or finalized";
            return false;
        }
    }
    const auto staging = queue_root / "pending" / ("." + key + ".staging");
    std::filesystem::remove_all(staging, ec);
    std::filesystem::create_directories(staging, ec);
    if (ec) {
        error = "cannot create worker staging directory";
        return false;
    }
    std::string job_json;
    if (!common_learning_training_job_to_json(job, job_json, error) ||
            !write_text(staging / "job.json", job_json, error) ||
            !write_text(staging / "corpus.jsonl", corpus.jsonl, error) ||
            !write_text(staging / "corpus-manifest.json", corpus.manifest_json, error)) {
        std::filesystem::remove_all(staging, ec);
        return false;
    }
    std::filesystem::rename(staging, queue_root / "pending" / key, ec);
    if (ec) {
        std::filesystem::remove_all(staging, ec);
        error = "cannot atomically enqueue worker job: " + ec.message();
        return false;
    }
    return true;
}

bool agent_learning_worker_run_once(
        const std::filesystem::path & queue_root,
        const agent_learning_worker_limits & limits,
        agent_learning_worker_report & report,
        std::string & error) {
    error.clear();
    report = {};
    if (limits.max_artifact_bytes == 0) {
        error = "worker artifact byte bound must be positive";
        return false;
    }
    if (!create_queue_directories(queue_root, error)) return false;
    std::vector<std::filesystem::path> pending;
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(queue_root / "pending", ec)) {
        if (ec) break;
        if (entry.is_directory(ec) && !ec && entry.path().filename().string().front() != '.') {
            pending.push_back(entry.path());
        }
    }
    if (ec) {
        error = "cannot inspect pending worker jobs: " + ec.message();
        return false;
    }
    std::sort(pending.begin(), pending.end());
    for (const auto & candidate : pending) {
        const auto key = candidate.filename().string();
        const auto running = queue_root / "running" / key;
        std::filesystem::rename(candidate, running, ec);
        if (ec) {
            ec.clear();
            continue;
        }
        std::string job_text;
        common_learning_training_job job;
        if (!read_text(running / "job.json", job_text, error) ||
                !common_learning_training_job_from_json(job_text, job, error)) {
            const auto job_id = job.id.empty() ? std::string("unknown") : job.id;
            if (!fail_claimed_job(running, queue_root, key, job_id, "invalid training job bundle", report, error)) return false;
            return true;
        }
        report = {agent_learning_worker_job_state::running, job.id, "job claimed"};
        if (!write_state(running, report.state, job.id, report.safe_summary, error)) return false;
        if (cancellation_requested(running)) {
            report = {agent_learning_worker_job_state::cancelled, job.id, "cancel requested before trainer start"};
            if (!write_state(running, report.state, job.id, report.safe_summary, error) ||
                    !move_terminal(running, queue_root / "cancelled", key, error)) return false;
            return true;
        }
        std::string corpus_jsonl;
        if (!read_text(running / "corpus.jsonl", corpus_jsonl, error)) {
            if (!fail_claimed_job(running, queue_root, key, job.id, "corpus bundle is unreadable", report, error)) return false;
            return true;
        }
        common_learning_training_result result;
        if (!run_fake_trainer(running, job, corpus_jsonl, limits, result, error)) {
            const std::string summary = error;
            error.clear();
            if (!fail_claimed_job(running, queue_root, key, job.id, summary, report, error)) return false;
            return true;
        }
        if (cancellation_requested(running)) {
            report = {agent_learning_worker_job_state::cancelled, job.id, "cancel requested after trainer completion"};
            if (!write_state(running, report.state, job.id, report.safe_summary, error) ||
                    !move_terminal(running, queue_root / "cancelled", key, error)) return false;
            return true;
        }
        std::string result_json;
        if (!common_learning_validate_training_result(job, result, error) ||
                !common_learning_training_result_to_json(result, result_json, error) ||
                !write_text(running / "result.json", result_json, error)) {
            const std::string summary = error.empty() ? "worker produced an invalid training result" : error;
            error.clear();
            if (!fail_claimed_job(running, queue_root, key, job.id, summary, report, error)) return false;
            return true;
        }
        report = {agent_learning_worker_job_state::succeeded, job.id, "fake trainer completed"};
        if (!write_state(running, report.state, job.id, report.safe_summary, error) ||
                !move_terminal(running, queue_root / "succeeded", key, error)) return false;
        return true;
    }
    return true;
}
