#include "agent/adaptation/flydelta/flydelta-queue.h"

#include "hash/hash.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <system_error>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

const char * state_directory(common_flydelta_experiment_queue_state state) {
    return common_flydelta_experiment_queue_state_name(state);
}

std::string queue_key(const std::string & id) {
    return "job-" + hash_sha256_hex(id.data(), id.size()).substr(0, 32);
}

bool write_text(const std::filesystem::path & path, const std::string & text, std::string & error) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot write FlyDelta queue file: " + path.string();
        return false;
    }
    output << text;
    if (!output) {
        error = "failed writing FlyDelta queue file: " + path.string();
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
    if (ec) {
        std::filesystem::remove(temporary, ec);
        error = "cannot atomically write FlyDelta queue file: " + path.string();
        return false;
    }
    return true;
}

bool read_text(const std::filesystem::path & path, std::string & text, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot read FlyDelta queue file: " + path.string();
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        error = "failed reading FlyDelta queue file: " + path.string();
        return false;
    }
    text = buffer.str();
    return true;
}

bool ensure_directories(const std::filesystem::path & root, std::string & error) {
    std::error_code ec;
    for (const auto state : {
            common_flydelta_experiment_queue_state::pending,
            common_flydelta_experiment_queue_state::running,
            common_flydelta_experiment_queue_state::succeeded,
            common_flydelta_experiment_queue_state::failed,
            common_flydelta_experiment_queue_state::cancelled}) {
        std::filesystem::create_directories(root / state_directory(state), ec);
        if (ec) {
            error = "cannot create FlyDelta queue directory: " +
                (root / state_directory(state)).string();
            return false;
        }
    }
    return true;
}

bool regular_file_within_bound(const std::filesystem::path & path, size_t max_bytes,
        const char * description, std::string & error) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec || status.type() != std::filesystem::file_type::regular) {
        error = std::string(description) + " is not a regular file";
        return false;
    }
    if (std::filesystem::file_size(path, ec) > max_bytes || ec) {
        error = std::string(description) + " exceeds FlyDelta queue byte bound";
        return false;
    }
    return true;
}

bool already_queued(const std::filesystem::path & root, const std::string & key) {
    std::error_code ec;
    for (const auto state : {
            common_flydelta_experiment_queue_state::pending,
            common_flydelta_experiment_queue_state::running,
            common_flydelta_experiment_queue_state::succeeded,
            common_flydelta_experiment_queue_state::failed,
            common_flydelta_experiment_queue_state::cancelled}) {
        if (std::filesystem::exists(root / state_directory(state) / key, ec) && !ec) return true;
    }
    return false;
}

} // namespace

const char * common_flydelta_experiment_queue_state_name(
        common_flydelta_experiment_queue_state state) {
    switch (state) {
        case common_flydelta_experiment_queue_state::pending: return "pending";
        case common_flydelta_experiment_queue_state::running: return "running";
        case common_flydelta_experiment_queue_state::succeeded: return "succeeded";
        case common_flydelta_experiment_queue_state::failed: return "failed";
        case common_flydelta_experiment_queue_state::cancelled: return "cancelled";
    }
    return "failed";
}

bool common_flydelta_experiment_queue_enqueue(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_queue_limits & limits,
        std::string & error) {
    error.clear();
    if (limits.max_job_bytes == 0 || limits.max_summary_bytes == 0) {
        error = "FlyDelta queue byte bounds must be positive";
        return false;
    }
    if (!common_flydelta_experiment_job_validate(job, 128, error)) return false;
    const auto text = common_flydelta_experiment_job_to_json(job);
    if (text.size() > limits.max_job_bytes) {
        error = "FlyDelta experiment job exceeds queue byte bound";
        return false;
    }
    if (!ensure_directories(queue_root, error)) return false;
    const auto key = queue_key(job.id);
    if (already_queued(queue_root, key)) {
        error = "FlyDelta experiment job is already queued or finalized";
        return false;
    }
    const auto staging = queue_root / "pending" / ("." + key + ".staging");
    std::error_code ec;
    std::filesystem::remove_all(staging, ec);
    std::filesystem::create_directories(staging, ec);
    if (ec || !write_text(staging / "job.json", text, error)) {
        std::filesystem::remove_all(staging, ec);
        if (error.empty()) error = "cannot create FlyDelta queue staging directory";
        return false;
    }
    std::filesystem::rename(staging, queue_root / "pending" / key, ec);
    if (ec) {
        std::filesystem::remove_all(staging, ec);
        error = "cannot atomically enqueue FlyDelta experiment job: " + ec.message();
        return false;
    }
    return true;
}

bool common_flydelta_experiment_queue_claim_next(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        common_flydelta_claimed_experiment_job & claimed,
        std::string & error) {
    error.clear();
    claimed = {};
    if (limits.max_job_bytes == 0 || !ensure_directories(queue_root, error)) return false;
    std::vector<std::filesystem::path> pending;
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(queue_root / "pending", ec)) {
        if (ec) break;
        const auto name = entry.path().filename().string();
        if (entry.is_directory(ec) && !ec && !name.empty() && name.front() != '.') {
            pending.push_back(entry.path());
        }
    }
    if (ec) {
        error = "cannot inspect FlyDelta pending queue: " + ec.message();
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
        std::string text;
        common_flydelta_experiment_job job;
        if (!regular_file_within_bound(running / "job.json", limits.max_job_bytes,
                "FlyDelta experiment job", error) ||
                !read_text(running / "job.json", text, error) ||
                !common_flydelta_experiment_job_from_json(text, job, error)) {
            return false;
        }
        claimed.queue_key = key;
        claimed.job = std::move(job);
        return true;
    }
    return true;
}

bool common_flydelta_experiment_queue_complete(
        const std::filesystem::path & queue_root,
        const common_flydelta_claimed_experiment_job & claimed,
        common_flydelta_experiment_queue_state state,
        const std::string & safe_summary,
        const common_flydelta_experiment_queue_limits & limits,
        std::string & error) {
    error.clear();
    if (claimed.queue_key.empty() || claimed.job.id.empty() ||
            state == common_flydelta_experiment_queue_state::pending ||
            state == common_flydelta_experiment_queue_state::running) {
        error = "FlyDelta completion requires a claimed job and terminal state";
        return false;
    }
    if (safe_summary.size() > limits.max_summary_bytes) {
        error = "FlyDelta completion summary exceeds queue byte bound";
        return false;
    }
    if (!ensure_directories(queue_root, error)) return false;
    const auto running = queue_root / "running" / claimed.queue_key;
    std::error_code ec;
    if (!std::filesystem::is_directory(running, ec) || ec) {
        error = "FlyDelta claimed job is not in running queue";
        return false;
    }
    const auto status = json{
        {"state", common_flydelta_experiment_queue_state_name(state)},
        {"job_id", claimed.job.id},
        {"safe_summary", safe_summary},
    }.dump();
    if (!write_text_atomic(running / "state.json", status, error)) return false;
    std::filesystem::rename(running, queue_root / state_directory(state) / claimed.queue_key, ec);
    if (ec) {
        error = "cannot finalize FlyDelta experiment job: " + ec.message();
        return false;
    }
    return true;
}
