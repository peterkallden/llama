#include "agent-learning-worker.h"

#include "hash/hash.h"
#include "subproc.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>
#include <system_error>
#include <utility>
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

bool write_text_atomic(const std::filesystem::path & path, const std::string & text, std::string & error) {
    const auto temporary = path.string() + ".tmp";
    std::error_code ec;
    std::filesystem::remove(temporary, ec);
    if (!write_text(temporary, text, error)) return false;
    std::filesystem::rename(temporary, path, ec);
#if defined(_WIN32)
    if (ec) {
        ec.clear();
        std::filesystem::remove(path, ec);
        if (!ec) std::filesystem::rename(temporary, path, ec);
    }
#endif
    if (ec) {
        std::filesystem::remove(temporary, ec);
        error = "cannot atomically replace worker file: " + path.string();
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
        const std::string & worker_id,
        const uint64_t lease_expires_at_epoch_seconds,
        std::string & error) {
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto text = json{
        {"state", agent_learning_worker_job_state_name(state)},
        {"job_id", job_id},
        {"safe_summary", summary},
        {"worker_id", worker_id},
        {"updated_at_epoch_seconds", now},
        {"lease_expires_at_epoch_seconds", lease_expires_at_epoch_seconds},
    }.dump();
    return write_text_atomic(directory / "state.json", text, error);
}

bool regular_file_within_bound(
        const std::filesystem::path & path,
        const size_t max_bytes,
        const std::string & description,
    std::string & error) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec || status.type() != std::filesystem::file_type::regular) {
        error = description + " is not a regular file";
        return false;
    }
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        error = "cannot inspect " + description + " size";
        return false;
    }
    if (size > max_bytes) {
        error = description + " exceeds worker byte bound";
        return false;
    }
    return true;
}

bool resolve_artifact_inside_root(
        const std::filesystem::path & root_path,
        const std::filesystem::path & relative_path,
        std::filesystem::path & resolved,
        std::string & error) {
    std::error_code ec;
    const auto root = std::filesystem::canonical(root_path, ec);
    if (ec) {
        error = "trainer artifact root is not available";
        return false;
    }
    resolved = std::filesystem::canonical(root / relative_path, ec);
    if (ec || !std::filesystem::is_regular_file(resolved, ec) || ec) {
        error = "trainer artifact is not a regular file";
        return false;
    }
    const auto relative = resolved.lexically_relative(root);
    if (relative.empty() || relative.is_absolute() || relative.string().find("..") == 0) {
        error = "trainer artifact escapes its root";
        return false;
    }
    return true;
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
        const agent_learning_worker_limits & limits,
        agent_learning_worker_report & report,
        std::string & error) {
    report = {agent_learning_worker_job_state::failed, job_id, summary};
    if (!write_state(running, report.state, job_id, summary, limits.worker_id, 0, error)) return false;
    return move_terminal(running, queue_root / "failed", key, error);
}

bool cancellation_requested(const std::filesystem::path & directory) {
    std::error_code ec;
    return std::filesystem::exists(directory / "cancel", ec) && !ec;
}

uint64_t now_epoch_seconds();

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
    // The deterministic trainer creates an artifact but performs no model
    // evaluation. A separate evaluator must produce a passed report before
    // the artifact can enter the registry canary state.
    result.evaluation_status = "not_run";
    return true;
}

bool run_external_trainer(
        const std::filesystem::path & running,
        const common_learning_training_job & job,
        const agent_learning_worker_limits & limits,
        const uint64_t deadline_epoch_seconds,
        common_learning_training_result & result,
        std::string & error) {
    const auto configured = limits.external_trainer_commands.find(job.trainer_kind);
    if (configured == limits.external_trainer_commands.end() || configured->second.empty()) {
        error = "worker has no enabled trainer for kind: " + job.trainer_kind;
        return false;
    }
    if (!common_subproc::is_supported()) {
        error = "worker was built without subprocess support; external trainers are unavailable";
        return false;
    }

    const auto result_path = running / "result.json";
    std::error_code ec;
    if (std::filesystem::exists(result_path, ec) && !ec) {
        error = "claimed job already contains a trainer result";
        return false;
    }
    if (!regular_file_within_bound(running / "job.json", limits.max_job_bytes, "training job", error) ||
            !regular_file_within_bound(running / "corpus-manifest.json", limits.max_manifest_bytes,
                "corpus manifest", error)) return false;
    std::vector<std::string> argv = configured->second;
    const auto append_path = [&argv](const char * name, const std::filesystem::path & path) {
        argv.emplace_back(name);
        argv.push_back(path.string());
    };
    append_path("--job", running / "job.json");
    append_path("--corpus", running / "corpus.jsonl");
    append_path("--corpus-manifest", running / "corpus-manifest.json");
    append_path("--artifacts", running / "artifacts");
    append_path("--result", result_path);

    common_subproc process;
    const int options = subprocess_option_no_window |
        subprocess_option_combined_stdout_stderr |
        subprocess_option_enable_async |
        subprocess_option_inherit_environment |
        subprocess_option_search_user_path;
    if (!process.create(argv, options, {}, running.string().c_str())) {
        error = "could not start external trainer for kind: " + job.trainer_kind;
        return false;
    }

    // The trainer protocol is file-based.  This avoids passing sensitive corpus
    // material over a command line and keeps result validation entirely host
    // owned.  A periodic state update doubles as a renewable crash lease.
    const auto trainer_log = running / "trainer.log";
    std::ofstream trainer_log_output(trainer_log, std::ios::binary | std::ios::trunc);
    if (!trainer_log_output) {
        process.terminate();
        process.join();
        error = "cannot open bounded external trainer log";
        return false;
    }
    static constexpr size_t max_trainer_log_bytes = 64 * 1024;
    size_t trainer_log_bytes = 0;
    const auto drain_output = [&process, &trainer_log_output, &trainer_log_bytes]() {
        char buffer[4096];
        while (const auto count = process.read_stdout(buffer, sizeof(buffer))) {
            if (trainer_log_bytes >= max_trainer_log_bytes) continue;
            const auto writable = std::min<size_t>(count, max_trainer_log_bytes - trainer_log_bytes);
            trainer_log_output.write(buffer, static_cast<std::streamsize>(writable));
            trainer_log_bytes += writable;
        }
    };
    uint64_t last_state_update = 0;
    while (process.alive()) {
        drain_output();
        const auto now = now_epoch_seconds();
        if (cancellation_requested(running)) {
            process.terminate();
            process.join();
            error = "cancel requested while external trainer was running";
            return false;
        }
        if (now >= deadline_epoch_seconds) {
            process.terminate();
            process.join();
            error = "external trainer exceeded worker runtime bound";
            return false;
        }
        if (now != last_state_update) {
            std::string state_error;
            if (!write_state(running, agent_learning_worker_job_state::running, job.id,
                    "external trainer running", limits.worker_id, deadline_epoch_seconds, state_error)) {
                process.terminate();
                process.join();
                error = state_error;
                return false;
            }
            last_state_update = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    drain_output();
    trainer_log_output.close();
    if (process.join() != 0) {
        error = "external trainer exited unsuccessfully";
        return false;
    }

    std::string result_json;
    if (!regular_file_within_bound(result_path, limits.max_result_bytes, "trainer result", error) ||
            !read_text(result_path, result_json, error) ||
            !common_learning_training_result_from_json(result_json, result, error) ||
            !common_learning_validate_training_result(job, result, error)) {
        if (error.empty()) error = "external trainer produced an invalid result";
        return false;
    }
    const auto artifact_root = running / "artifacts";
    std::filesystem::path artifact_path;
    if (!resolve_artifact_inside_root(artifact_root, result.artifact_path, artifact_path, error) ||
            !regular_file_within_bound(artifact_path, limits.max_artifact_bytes, "trainer artifact", error)) return false;
    std::string artifact_bytes;
    if (!read_text(artifact_path, artifact_bytes, error) ||
            result.artifact_sha256 != "sha256:" + hash_sha256_hex(artifact_bytes.data(), artifact_bytes.size())) {
        error = "trainer artifact SHA-256 does not match result";
        return false;
    }
    return true;
}

uint64_t now_epoch_seconds() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

bool recover_stale_running_jobs(
        const std::filesystem::path & queue_root,
        const agent_learning_worker_limits & limits,
        std::string & error) {
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(queue_root / "running", ec)) {
        if (ec) break;
        const bool is_directory = entry.is_directory(ec);
        if (ec) break;
        if (!is_directory) continue;

        const auto state_path = entry.path() / "state.json";
        std::string state_text;
        if (!read_text(state_path, state_text, error)) return false;
        json state;
        try {
            state = json::parse(state_text);
        } catch (const std::exception &) {
            error = "cannot parse running worker state: " + state_path.string();
            return false;
        }
        if (state.value("state", std::string{}) != "running") continue;
        const auto lease = state.value("lease_expires_at_epoch_seconds", uint64_t{0});
        if (lease == 0 || lease > now_epoch_seconds()) continue;

        const auto job_id = state.value("job_id", std::string{"unknown"});
        if (!write_state(entry.path(), agent_learning_worker_job_state::failed, job_id,
                "worker lease expired; job may be retried", limits.worker_id, 0, error)) return false;
        if (!move_terminal(entry.path(), queue_root / "failed", entry.path().filename().string(), error)) return false;
    }
    if (ec) {
        error = "cannot inspect running worker jobs: " + ec.message();
        return false;
    }
    return true;
}

} // namespace

std::string agent_learning_worker_capabilities_json(const agent_learning_worker_limits & limits) {
    agent_learning_worker_capabilities capabilities;
    for (const auto & item : limits.external_trainer_commands) {
        if (!item.first.empty() && !item.second.empty()) capabilities.trainer_kinds.push_back(item.first);
    }
    json trainer_kinds = json::array();
    for (const auto & trainer_kind : capabilities.trainer_kinds) {
        trainer_kinds.push_back(trainer_kind);
    }
    return json{
        {"cuda", capabilities.cuda},
        {"trainer_kinds", std::move(trainer_kinds)},
        {"max_parallel_jobs", capabilities.max_parallel_jobs},
    }.dump();
}

std::string agent_learning_worker_capabilities_json() {
    return agent_learning_worker_capabilities_json({});
}

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
    if (limits.max_job_bytes == 0 || limits.max_manifest_bytes == 0 || limits.max_result_bytes == 0) {
        error = "worker metadata byte bounds must be positive";
        return false;
    }
    if (limits.max_corpus_bytes == 0) {
        error = "worker corpus byte bound must be positive";
        return false;
    }
    if (limits.max_job_runtime_seconds == 0) {
        error = "worker runtime bound must be positive";
        return false;
    }
    if (limits.worker_id.empty()) {
        error = "worker id must not be empty";
        return false;
    }
    if (!create_queue_directories(queue_root, error)) return false;
    if (!recover_stale_running_jobs(queue_root, limits, error)) return false;
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
        if (!regular_file_within_bound(running / "job.json", limits.max_job_bytes,
                "training job", error) || !read_text(running / "job.json", job_text, error) ||
                !common_learning_training_job_from_json(job_text, job, error)) {
            const auto job_id = job.id.empty() ? std::string("unknown") : job.id;
            if (!fail_claimed_job(running, queue_root, key, job_id, "invalid training job bundle", limits, report, error)) return false;
            return true;
        }
        report = {agent_learning_worker_job_state::running, job.id, "job claimed"};
        const auto job_runtime_seconds = std::min(job.deadline_seconds, limits.max_job_runtime_seconds);
        const auto lease_expires_at = now_epoch_seconds() + job_runtime_seconds;
        if (!write_state(running, report.state, job.id, report.safe_summary, limits.worker_id, lease_expires_at, error)) return false;
        if (cancellation_requested(running)) {
            report = {agent_learning_worker_job_state::cancelled, job.id, "cancel requested before trainer start"};
            if (!write_state(running, report.state, job.id, report.safe_summary, limits.worker_id, 0, error) ||
                    !move_terminal(running, queue_root / "cancelled", key, error)) return false;
            return true;
        }
        std::string corpus_jsonl;
        if (!regular_file_within_bound(running / "corpus.jsonl", limits.max_corpus_bytes,
                "corpus bundle", error) || !read_text(running / "corpus.jsonl", corpus_jsonl, error)) {
            if (!fail_claimed_job(running, queue_root, key, job.id, "corpus bundle is unreadable", limits, report, error)) return false;
            return true;
        }
        if (!regular_file_within_bound(running / "corpus-manifest.json", limits.max_manifest_bytes,
                "corpus manifest", error)) {
            const auto summary = error;
            error.clear();
            if (!fail_claimed_job(running, queue_root, key, job.id, summary, limits, report, error)) return false;
            return true;
        }
        common_learning_training_result result;
        const bool trainer_ok = job.trainer_kind == "fake-sft"
            ? run_fake_trainer(running, job, corpus_jsonl, limits, result, error)
            : run_external_trainer(running, job, limits, lease_expires_at, result, error);
        if (!trainer_ok) {
            const std::string summary = error;
            error.clear();
            if (cancellation_requested(running)) {
                report = {agent_learning_worker_job_state::cancelled, job.id,
                    summary.empty() ? "cancel requested while trainer was running" : summary};
                if (!write_state(running, report.state, job.id, report.safe_summary, limits.worker_id, 0, error) ||
                        !move_terminal(running, queue_root / "cancelled", key, error)) return false;
                return true;
            }
            if (!fail_claimed_job(running, queue_root, key, job.id, summary, limits, report, error)) return false;
            return true;
        }
        if (cancellation_requested(running)) {
            report = {agent_learning_worker_job_state::cancelled, job.id, "cancel requested after trainer completion"};
            if (!write_state(running, report.state, job.id, report.safe_summary, limits.worker_id, 0, error) ||
                    !move_terminal(running, queue_root / "cancelled", key, error)) return false;
            return true;
        }
        std::string result_json;
        if (!common_learning_validate_training_result(job, result, error) ||
                !common_learning_training_result_to_json(result, result_json, error) ||
                !write_text_atomic(running / "result.json", result_json, error)) {
            const std::string summary = error.empty() ? "worker produced an invalid training result" : error;
            error.clear();
            if (!fail_claimed_job(running, queue_root, key, job.id, summary, limits, report, error)) return false;
            return true;
        }
        report = {agent_learning_worker_job_state::succeeded, job.id,
            job.trainer_kind == "fake-sft" ? "fake trainer completed" : "external trainer completed"};
        if (!write_state(running, report.state, job.id, report.safe_summary, limits.worker_id, 0, error) ||
                !move_terminal(running, queue_root / "succeeded", key, error)) return false;
        return true;
    }
    return true;
}
