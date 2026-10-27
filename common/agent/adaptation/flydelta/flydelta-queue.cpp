#include "agent/adaptation/flydelta/flydelta-queue.h"

#include "hash/hash.h"
#include "agent/durable-file.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <system_error>
#include <vector>
#include <map>
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <pthread.h>
#ifdef __linux__
#include <sys/vfs.h>
#endif
#endif

using json = nlohmann::ordered_json;

struct common_flydelta_queue_owner {
    std::recursive_mutex writes;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    DWORD pid = GetCurrentProcessId();
    ~common_flydelta_queue_owner() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#else
    int fd = -1;
    pid_t pid = ::getpid();
    ~common_flydelta_queue_owner() { if (fd >= 0) ::close(fd); }
#endif
};

namespace {
std::mutex owners_mutex;
std::map<std::filesystem::path, std::weak_ptr<common_flydelta_queue_owner>> owners;
#ifndef _WIN32
void lock_owners_before_fork() { owners_mutex.lock(); }
void unlock_owners_after_fork() { owners_mutex.unlock(); }
void close_inherited_owners_after_fork() {
    // Closing the child's copy does not unlock the parent's remaining file
    // description. Do not explicitly LOCK_UN a shared flock after fork.
    for (const auto & entry : owners) {
        if (const auto owner = entry.second.lock()) {
            if (owner->fd >= 0) ::close(owner->fd);
            owner->fd = -1;
        }
    }
    owners_mutex.unlock();
}
#endif
}

std::shared_ptr<common_flydelta_queue_owner> common_flydelta_acquire_queue_owner(
        const std::filesystem::path & root, std::string & error) {
    error.clear();
    if (root.empty()) { error = "FlyDelta queue ownership requires a directory"; return {}; }
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) { error = "cannot create FlyDelta queue: " + ec.message(); return {}; }
    const auto canonical = std::filesystem::canonical(root, ec);
    if (ec) { error = "cannot resolve FlyDelta queue: " + ec.message(); return {}; }
#ifndef _WIN32
    static std::once_flag fork_handlers;
    static int fork_handler_error = 0;
    std::call_once(fork_handlers, [] {
        fork_handler_error = ::pthread_atfork(lock_owners_before_fork,
            unlock_owners_after_fork, close_inherited_owners_after_fork);
    });
    if (fork_handler_error != 0) { error = "cannot protect FlyDelta queue ownership across fork"; return {}; }
#endif
    std::lock_guard<std::mutex> guard(owners_mutex);
#ifdef _WIN32
    const auto pid = GetCurrentProcessId();
#else
    const auto pid = ::getpid();
#endif
    if (const auto existing = owners[canonical].lock()) {
        if (existing->pid == pid) return existing;
    }
#ifdef __linux__
    struct statfs filesystem{};
    if (::statfs(canonical.c_str(), &filesystem) != 0) {
        error = "cannot inspect FlyDelta queue filesystem"; return {};
    }
    // Reject known network filesystems; local-filesystem semantics remain a
    // deployment requirement for other mount types (including FUSE mounts).
    if (filesystem.f_type == 0x6969 || filesystem.f_type == 0xFF534D42 ||
            filesystem.f_type == 0x517B || filesystem.f_type == 0x73757245) {
        error = "FlyDelta queue ownership requires a local filesystem"; return {};
    }
#endif
    auto owner = std::make_shared<common_flydelta_queue_owner>();
    const auto lock_path = canonical / ".owner.lock";
#ifdef _WIN32
    owner->handle = CreateFileW(lock_path.c_str(), GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    OVERLAPPED overlapped{};
    if (owner->handle == INVALID_HANDLE_VALUE || !LockFileEx(owner->handle,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlapped)) {
        error = "FlyDelta queue LOCK_BUSY or lock unavailable"; return {};
    }
#else
    owner->fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (owner->fd < 0) { error = "cannot open FlyDelta queue owner lock"; return {}; }
    if (::flock(owner->fd, LOCK_EX | LOCK_NB) != 0) {
        error = "FlyDelta queue LOCK_BUSY or lock unavailable"; return {};
    }
#endif
    owners[canonical] = owner;
    return owner;
}

namespace {

const char * state_directory(common_flydelta_experiment_queue_state state) {
    return common_flydelta_experiment_queue_state_name(state);
}

std::string queue_key(const std::string & id) {
    return "job-" + hash_sha256_hex(id.data(), id.size()).substr(0, 32);
}

bool write_text(const std::filesystem::path & path, const std::string & text, std::string & error) {
    return common_agent_write_durable_file(path, text, false, error);
}

bool write_text_atomic(const std::filesystem::path & path, const std::string & text, std::string & error) {
    return common_agent_write_durable_file(path, text, false, error);
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
    bool created = false;
    for (const auto state : {
            common_flydelta_experiment_queue_state::pending,
            common_flydelta_experiment_queue_state::running,
            common_flydelta_experiment_queue_state::succeeded,
            common_flydelta_experiment_queue_state::failed,
            common_flydelta_experiment_queue_state::cancelled}) {
        created = std::filesystem::create_directories(root / state_directory(state), ec) || created;
        if (ec) {
            error = "cannot create FlyDelta queue directory: " +
                (root / state_directory(state)).string();
            return false;
        }
    }
    return !created || common_agent_sync_directory_ancestors(root, error);
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

bool unique_queue_state(const std::filesystem::path & root, const std::string & key, std::string & error) {
    size_t states = 0;
    std::error_code ec;
    for (const char * name : {"pending", "running", "succeeded", "failed", "cancelled"}) {
        if (std::filesystem::exists(root / name / key, ec)) ++states;
        if (ec) { error = "cannot inspect FlyDelta queue state: " + ec.message(); return false; }
    }
    if (states != 1) { error = "FlyDelta recovery requires exactly one authoritative queue state"; return false; }
    return true;
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

bool common_flydelta_experiment_queue_contains(
        const std::filesystem::path & queue_root,
        const std::string & job_id,
        bool & contains,
        std::string & error) {
    error.clear();
    contains = false;
    const auto owner = common_flydelta_acquire_queue_owner(queue_root, error);
    if (!owner) return false;
    std::lock_guard<std::recursive_mutex> guard(owner->writes);
    if (job_id.empty()) {
        error = "FlyDelta queue lookup requires a job id";
        return false;
    }
    if (!ensure_directories(queue_root, error)) return false;
    contains = already_queued(queue_root, queue_key(job_id));
    return true;
}

bool common_flydelta_experiment_queue_enqueue(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_queue_limits & limits,
        std::string & error) {
    error.clear();
    const auto owner = common_flydelta_acquire_queue_owner(queue_root, error);
    if (!owner) return false;
    std::lock_guard<std::recursive_mutex> guard(owner->writes);
    if (limits.max_job_bytes == 0 || limits.max_summary_bytes == 0 ||
            limits.max_execution_attempts == 0 || limits.max_execution_attempts > 16) {
        error = "FlyDelta queue byte or execution-attempt bounds are invalid";
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
    return common_agent_sync_directory(queue_root / "pending", error);
}

bool common_flydelta_experiment_queue_claim_next(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        common_flydelta_claimed_experiment_job & claimed,
        std::string & error) {
    error.clear();
    claimed = {};
    const auto owner = common_flydelta_acquire_queue_owner(queue_root, error);
    if (!owner) return false;
    std::lock_guard<std::recursive_mutex> guard(owner->writes);
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
        if (!common_agent_sync_directory(queue_root / "pending", error) ||
                !common_agent_sync_directory(queue_root / "running", error)) return false;
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
        if (limits.max_execution_attempts == 0 || limits.max_execution_attempts > 16) {
            error = "FlyDelta queue execution-attempt limit is invalid"; return false;
        }
        size_t attempts = 0;
        if (std::filesystem::exists(running / "execution.json", ec)) {
            if (!regular_file_within_bound(running / "execution.json", 4096, "FlyDelta execution attempts", error) ||
                    !read_text(running / "execution.json", text, error)) return false;
            const auto previous = json::parse(text, nullptr, false);
            if (!previous.is_object() || !previous.contains("job_id") || previous.at("job_id") != claimed.job.id ||
                    !previous.contains("attempts") || !previous.at("attempts").is_number_unsigned() ||
                    !previous.contains("limit") || previous.at("limit") != limits.max_execution_attempts) {
                error = "FlyDelta job execution attempts are incompatible"; return false;
            }
            attempts = previous.at("attempts").get<size_t>();
        }
        if (ec) { error = "cannot inspect FlyDelta execution attempts"; return false; }
        if (attempts >= limits.max_execution_attempts) {
            std::string completion_error;
            if (!common_flydelta_experiment_queue_complete(queue_root, claimed,
                    common_flydelta_experiment_queue_state::failed, "FlyDelta physical job retry limit exhausted",
                    limits, completion_error)) { error = completion_error; return false; }
            error = "FlyDelta physical job retry limit exhausted";
            return false;
        }
        if (!write_text_atomic(running / "execution.json", json{{"schema_version", 1},
                {"job_id", claimed.job.id}, {"attempts", attempts + 1},
                {"limit", limits.max_execution_attempts}, {"state", "started"}}.dump(), error)) return false;
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
    const auto owner = common_flydelta_acquire_queue_owner(queue_root, error);
    if (!owner) return false;
    std::lock_guard<std::recursive_mutex> guard(owner->writes);
    if (claimed.queue_key.empty() || claimed.job.id.empty() ||
            claimed.queue_key != queue_key(claimed.job.id) ||
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
    return common_agent_sync_directory(queue_root / "running", error) &&
        common_agent_sync_directory(queue_root / state_directory(state), error);
}

bool common_flydelta_experiment_queue_recover(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        std::string & error) {
    if (limits.max_job_bytes == 0 || limits.max_summary_bytes == 0 ||
            limits.max_execution_attempts == 0 || limits.max_execution_attempts > 16) {
        error = "FlyDelta recovery byte or execution-attempt bounds are invalid"; return false;
    }
    const auto owner = common_flydelta_acquire_queue_owner(queue_root, error);
    if (!owner) return false;
    std::lock_guard<std::recursive_mutex> guard(owner->writes);
    if (!ensure_directories(queue_root, error)) return false;
    std::error_code ec;
    std::vector<std::filesystem::path> abandoned;
    for (const auto & entry : std::filesystem::directory_iterator(queue_root / "running", ec)) {
        if (ec) break;
        if (entry.symlink_status(ec).type() != std::filesystem::file_type::directory || ec) {
            error = "FlyDelta running queue contains a non-directory entry"; return false;
        }
        abandoned.push_back(entry.path());
    }
    if (ec) { error = "cannot inspect FlyDelta running queue: " + ec.message(); return false; }
    std::sort(abandoned.begin(), abandoned.end());
    for (const auto & entry : abandoned) {
        std::string text;
        common_flydelta_experiment_job job;
        if (!regular_file_within_bound(entry / "job.json", limits.max_job_bytes, "FlyDelta job", error) ||
                !read_text(entry / "job.json", text, error) ||
                !common_flydelta_experiment_job_from_json(text, job, error)) return false;
        if (entry.filename() != queue_key(job.id)) { error = "FlyDelta recovery job identity differs"; return false; }
        if (!unique_queue_state(queue_root, queue_key(job.id), error)) return false;
        const char * destination = "pending";
        if (std::filesystem::exists(entry / "state.json", ec)) {
            if (!regular_file_within_bound(entry / "state.json", limits.max_summary_bytes + 2048,
                    "FlyDelta terminal acknowledgement", error) ||
                    !read_text(entry / "state.json", text, error)) return false;
            const auto state = json::parse(text, nullptr, false);
            if (!state.is_object() || !state.contains("job_id") || !state.at("job_id").is_string() ||
                    state.at("job_id") != job.id || !state.contains("state") || !state.at("state").is_string()) {
                error = "FlyDelta recovery terminal acknowledgement is invalid"; return false;
            }
            const auto status = state.at("state").get<std::string>();
            if (status == "succeeded") destination = "succeeded";
            else if (status == "failed") destination = "failed";
            else if (status == "cancelled") destination = "cancelled";
            else { error = "FlyDelta recovery acknowledgement is not terminal"; return false; }
        }
        if (ec) { error = "cannot inspect FlyDelta acknowledgement: " + ec.message(); return false; }
        const auto target = queue_root / destination / entry.filename();
        if (std::filesystem::exists(target, ec) || ec) {
            error = "FlyDelta recovery found conflicting queue states"; return false;
        }
        std::filesystem::rename(entry, target, ec);
        if (ec) { error = "cannot recover FlyDelta queue job: " + ec.message(); return false; }
        if (!common_agent_sync_directory(queue_root / "running", error) ||
                !common_agent_sync_directory(queue_root / destination, error)) return false;
    }
    return true;
}

bool common_flydelta_experiment_queue_job_state(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_queue_limits & limits,
        common_flydelta_experiment_queue_state & state,
        std::string & error) {
    error.clear();
    const auto owner = common_flydelta_acquire_queue_owner(queue_root, error);
    if (!owner || !ensure_directories(queue_root, error)) return false;
    std::lock_guard<std::recursive_mutex> guard(owner->writes);
    const auto key = queue_key(job.id);
    if (!unique_queue_state(queue_root, key, error)) return false;
    std::error_code ec;
    for (const auto candidate : {common_flydelta_experiment_queue_state::pending,
            common_flydelta_experiment_queue_state::running,
            common_flydelta_experiment_queue_state::succeeded,
            common_flydelta_experiment_queue_state::failed,
            common_flydelta_experiment_queue_state::cancelled}) {
        const auto directory = queue_root / state_directory(candidate) / key;
        if (!std::filesystem::exists(directory, ec)) {
            if (ec) { error = "cannot inspect FlyDelta queue state"; return false; }
            continue;
        }
        std::string text;
        common_flydelta_experiment_job stored;
        if (!regular_file_within_bound(directory / "job.json", limits.max_job_bytes, "FlyDelta job", error) ||
                !read_text(directory / "job.json", text, error) ||
                !common_flydelta_experiment_job_from_json(text, stored, error)) return false;
        if (common_flydelta_experiment_job_to_json(stored) != common_flydelta_experiment_job_to_json(job)) {
            error = "FlyDelta queue job identity differs"; return false;
        }
        const bool terminal_directory = candidate != common_flydelta_experiment_queue_state::pending &&
            candidate != common_flydelta_experiment_queue_state::running;
        const bool has_terminal_record = std::filesystem::exists(directory / "state.json", ec);
        if (ec) { error = "cannot inspect FlyDelta terminal record"; return false; }
        if (terminal_directory || has_terminal_record) {
            if (!regular_file_within_bound(directory / "state.json", limits.max_summary_bytes + 2048,
                    "FlyDelta terminal state", error) || !read_text(directory / "state.json", text, error)) return false;
            const auto terminal = json::parse(text, nullptr, false);
            if (!terminal.is_object() || !terminal.contains("job_id") ||
                    !terminal.at("job_id").is_string() || terminal.at("job_id") != job.id ||
                    !terminal.contains("state") || !terminal.at("state").is_string()) {
                error = "FlyDelta terminal state does not match its job"; return false;
            }
            const auto status = terminal.at("state").get<std::string>();
            auto terminal_state = common_flydelta_experiment_queue_state::succeeded;
            if (status == "succeeded") terminal_state = common_flydelta_experiment_queue_state::succeeded;
            else if (status == "failed") terminal_state = common_flydelta_experiment_queue_state::failed;
            else if (status == "cancelled") terminal_state = common_flydelta_experiment_queue_state::cancelled;
            else { error = "FlyDelta terminal state is invalid"; return false; }
            if (terminal_directory && terminal_state != candidate) {
                error = "FlyDelta terminal directory conflicts with its record"; return false;
            }
            state = terminal_state;
            return true;
        }
        state = candidate;
        return true;
    }
    error = "FlyDelta queue job has no authoritative state";
    return false;
}

bool common_flydelta_experiment_queue_acknowledge_recovered_result(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_job & job,
        const std::string & safe_summary,
        const common_flydelta_experiment_queue_limits & limits,
        std::string & error) {
    const auto owner = common_flydelta_acquire_queue_owner(queue_root, error);
    if (!owner || !ensure_directories(queue_root, error)) return false;
    std::lock_guard<std::recursive_mutex> guard(owner->writes);
    const auto key = queue_key(job.id);
    if (!unique_queue_state(queue_root, key, error)) return false;
    std::error_code ec;
    for (const char * state : {"succeeded", "running", "pending"}) {
        const auto directory = queue_root / state / key;
        if (!std::filesystem::exists(directory, ec)) {
            if (ec) { error = "cannot inspect recovered FlyDelta result"; return false; }
            continue;
        }
        std::string text;
        common_flydelta_experiment_job stored;
        if (!regular_file_within_bound(directory / "job.json", limits.max_job_bytes, "FlyDelta job", error) ||
                !read_text(directory / "job.json", text, error) ||
                !common_flydelta_experiment_job_from_json(text, stored, error)) return false;
        if (common_flydelta_experiment_job_to_json(stored) != common_flydelta_experiment_job_to_json(job)) {
            error = "FlyDelta durable result differs from queued job"; return false;
        }
        if (std::string(state) == "succeeded") {
            if (!regular_file_within_bound(directory / "state.json", limits.max_summary_bytes + 2048,
                    "FlyDelta acknowledgement", error) || !read_text(directory / "state.json", text, error)) return false;
            const auto acknowledgement = json::parse(text, nullptr, false);
            if (!acknowledgement.is_object() || !acknowledgement.contains("job_id") ||
                    acknowledgement.at("job_id") != job.id || !acknowledgement.contains("state") ||
                    acknowledgement.at("state") != "succeeded") {
                error = "FlyDelta recovered result has an invalid terminal acknowledgement"; return false;
            }
            return true;
        }
        if (std::string(state) == "pending") {
            std::filesystem::rename(directory, queue_root / "running" / key, ec);
            if (ec) { error = "cannot reconcile recovered FlyDelta result"; return false; }
            if (!common_agent_sync_directory(queue_root / "pending", error) ||
                    !common_agent_sync_directory(queue_root / "running", error)) return false;
        }
        return common_flydelta_experiment_queue_complete(queue_root, {key, job},
            common_flydelta_experiment_queue_state::succeeded, safe_summary, limits, error);
    }
    error = "FlyDelta durable result has no compatible queued job";
    return false;
}
