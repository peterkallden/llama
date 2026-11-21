#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

enum class agent_project_file_change_kind { created, changed, removed, moved, rescan_required };

struct agent_project_file_change {
    agent_project_file_change_kind kind = agent_project_file_change_kind::changed;
    std::string path;
    std::string previous_path;
    uint64_t generation = 0;
};

struct agent_project_file_watch_config {
    std::string root;
    std::vector<std::string> explicit_files;
    std::vector<std::string> file_extensions = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".m", ".mm", ".cmake", ".mk", ".make"};
    std::vector<std::string> file_names = {
        "CMakeLists.txt", "Makefile", "makefile", "GNUmakefile", "meson.build", "meson.options",
        "CMakePresets.json", "CMakeUserPresets.json", "BUILD", "BUILD.bazel", "WORKSPACE", "configure", "configure.ac", "Makefile.am"};
    std::chrono::milliseconds poll_interval{250};
    size_t max_pending_events = 4096;
    // Also permits deterministic testing of the portable backend.
    bool force_polling = false;
};

// Host-owned, backend-neutral project file notifications. Multiple consumers
// may subscribe to the same canonical root and share the underlying watches.
class agent_project_file_watch_service {
public:
    // Return true only after the consumer has accepted the entire batch.
    // Callbacks run independently and must finish in bounded time; unsubscribe
    // and service destruction wait for a callback already in progress.
    using callback = std::function<bool(const std::vector<agent_project_file_change> &)>;

    agent_project_file_watch_service();
    ~agent_project_file_watch_service();
    agent_project_file_watch_service(const agent_project_file_watch_service &) = delete;
    agent_project_file_watch_service & operator=(const agent_project_file_watch_service &) = delete;

    bool subscribe(const agent_project_file_watch_config & config, callback notify,
            uint64_t & subscription_id, std::string & error);
    void unsubscribe(uint64_t subscription_id);
    bool reconcile(uint64_t subscription_id, std::string & error);
    bool needs_reconciliation(uint64_t subscription_id) const;
    uint64_t latest_generation(uint64_t subscription_id) const;
    uint64_t delivered_generation(uint64_t subscription_id) const;
    bool wait_for_delivery(uint64_t subscription_id, std::chrono::milliseconds timeout) const;
    std::string backend(uint64_t subscription_id) const;

private:
    struct impl;
    std::unique_ptr<impl> state_;
};
