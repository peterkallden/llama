#include "agent-project-file-watch.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <system_error>
#include <thread>
#include <utility>

#ifdef __linux__
#include <cerrno>
#include <poll.h>
#include <sys/inotify.h>
#include <unistd.h>
#endif

namespace {
namespace fs = std::filesystem;

bool excluded_directory(const fs::path & path) {
    const auto name = path.filename().string();
    static const std::set<std::string> excluded = {
        ".git", ".hg", ".svn", ".cache", ".codex-tmp", "node_modules",
        "build", "build-agent", "build-agent-vulkan", "build-agent-cuda", "dist",
        "target", "out", "CMakeFiles", "Testing", "agent-data", "coverage",
    };
    return excluded.count(name) != 0 || name.rfind("build-", 0) == 0;
}

bool watched_file(const fs::path & path, const std::vector<std::string> & extensions,
        const std::vector<std::string> & names) {
    const auto name = path.filename().string();
    const auto ext = path.extension().string();
    return std::find(extensions.begin(), extensions.end(), ext) != extensions.end() ||
        std::find(names.begin(), names.end(), name) != names.end();
}

struct file_stamp {
    fs::file_time_type modified{};
    uintmax_t size = 0;
    friend bool operator==(const file_stamp & a, const file_stamp & b) {
        return a.modified == b.modified && a.size == b.size;
    }
};

bool path_within(const fs::path & root, const fs::path & candidate) {
    auto rel = candidate.lexically_relative(root);
    return !rel.empty() && !rel.is_absolute() && *rel.begin() != "..";
}

bool project_path_allowed(const fs::path & root, const fs::path & candidate) {
    if (!path_within(root, candidate)) return false;
    auto relative = candidate.lexically_relative(root);
    auto last = relative.end();
    if (relative.begin() != last) --last;
    for (auto it = relative.begin(); it != last; ++it)
        if (excluded_directory(*it)) return false;
    return true;
}
}

struct agent_project_file_watch_service::impl {
    struct subscription {
        uint64_t id = 0;
        std::string root;
        std::set<std::string> explicit_files;
        std::vector<std::string> file_extensions;
        std::vector<std::string> file_names;
        callback notify;
        uint64_t latest_generation = 0;
        uint64_t delivered_generation = 0;
        bool needs_reconciliation = true;
        size_t max_pending_events = 4096;
        std::chrono::milliseconds poll_interval{2000};
        std::deque<std::vector<agent_project_file_change>> pending;
        size_t pending_events = 0;
        bool closing = false;
        std::thread delivery_worker;
    };
    struct root_group {
        std::string root;
        std::set<uint64_t> subscriptions;
        std::map<std::string, file_stamp> snapshot;
        std::chrono::steady_clock::time_point last_scan{};
        bool force_polling = false;
#ifdef __linux__
        int fd = -1;
        std::map<int, std::string> directories;
        std::vector<agent_project_file_change> native_changes;
        bool degraded = false;
#endif
        size_t queued_events = 0;
        bool rescan_required = false;
    };

    mutable std::mutex mutex;
    std::condition_variable wake;
    std::map<uint64_t, subscription> subscriptions;
    std::map<std::string, root_group> roots;
    std::thread worker;
    bool stopping = false;
    uint64_t next_id = 1;
    uint64_t generation = 0;

    void deliver(uint64_t id) {
        for (;;) {
            callback notify;
            std::vector<agent_project_file_change> batch;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] {
                    const auto sub = subscriptions.find(id);
                    return stopping || sub == subscriptions.end() || sub->second.closing || !sub->second.pending.empty();
                });
                auto sub = subscriptions.find(id);
                if (stopping || sub == subscriptions.end() || sub->second.closing) return;
                batch = std::move(sub->second.pending.front());
                sub->second.pending.pop_front();
                sub->second.pending_events -= batch.size();
                notify = sub->second.notify;
            }
            bool accepted = false;
            try { accepted = notify && notify(batch); } catch (...) { accepted = false; }
            std::lock_guard<std::mutex> lock(mutex);
            const auto sub = subscriptions.find(id);
            if (sub == subscriptions.end()) return;
            if (!accepted) sub->second.needs_reconciliation = true;
            else if (!batch.empty() && !sub->second.needs_reconciliation)
                sub->second.delivered_generation = std::max(sub->second.delivered_generation, batch.back().generation);
            wake.notify_all();
        }
    }

    static bool stamp(const fs::path & path, file_stamp & result) {
        std::error_code ec;
        if (fs::is_symlink(fs::symlink_status(path, ec)) || ec || !fs::is_regular_file(path, ec) || ec) return false;
        result.modified = fs::last_write_time(path, ec);
        if (ec) return false;
        result.size = fs::file_size(path, ec);
        return !ec;
    }

    static std::map<std::string, file_stamp> scan(const root_group & group,
            const std::map<uint64_t, subscription> & subscriptions, bool & complete) {
        std::map<std::string, file_stamp> result;
        complete = true;
        std::error_code ec;
        fs::recursive_directory_iterator it(group.root, fs::directory_options::none, ec), end;
        if (ec) complete = false;
        size_t visited = 0;
        for (; !ec && it != end; it.increment(ec)) {
            if (++visited > 250000 || result.size() >= 100000) { complete = false; break; }
            const auto status = it->symlink_status(ec);
            if (ec) { complete = false; ec.clear(); continue; }
            if (fs::is_symlink(status)) {
                if (fs::is_directory(status)) it.disable_recursion_pending();
                continue;
            }
            if (fs::is_directory(status)) {
                if (excluded_directory(it->path())) it.disable_recursion_pending();
                continue;
            }
            if (!fs::is_regular_file(status)) continue;
            bool selected = false;
            const auto abs = it->path().lexically_normal().string();
            for (auto id : group.subscriptions) {
                const auto sub = subscriptions.find(id);
                if (sub != subscriptions.end() && (sub->second.explicit_files.count(abs) ||
                        watched_file(it->path(), sub->second.file_extensions, sub->second.file_names))) {
                    selected = true; break;
                }
            }
            if (!selected) continue;
            file_stamp value;
            if (stamp(it->path(), value)) result.emplace(it->path().lexically_normal().string(), value);
        }
        if (ec) complete = false;
        for (auto id : group.subscriptions) {
            const auto sub = subscriptions.find(id);
            if (sub == subscriptions.end()) continue;
            for (const auto & file : sub->second.explicit_files) {
                fs::path path(file);
                file_stamp value;
                if (stamp(path, value)) result.emplace(path.string(), value);
            }
        }
        return result;
    }

#ifdef __linux__
    static bool add_watch_tree(root_group & group, const fs::path & root) {
        if (group.fd < 0) return false;
        bool tree_complete = true;
        auto add = [&group, &tree_complete](const fs::path & dir) {
            const int wd = inotify_add_watch(group.fd, dir.c_str(),
                IN_CREATE | IN_CLOSE_WRITE | IN_MODIFY | IN_DELETE | IN_MOVED_FROM |
                IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF | IN_ATTRIB | IN_ONLYDIR);
            if (wd < 0) { group.degraded = true; tree_complete = false; return false; }
            group.directories[wd] = dir.lexically_normal().string();
            return true;
        };
        if (!add(root)) return false;
        std::error_code ec;
        fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
        for (; !ec && it != end; it.increment(ec)) {
            const auto status = it->symlink_status(ec);
            if (ec) { ec.clear(); continue; }
            if (fs::is_symlink(status)) { it.disable_recursion_pending(); continue; }
            if (!fs::is_directory(status)) continue;
            if (excluded_directory(it->path())) { it.disable_recursion_pending(); continue; }
            add(it->path());
        }
        if (ec) { group.degraded = true; tree_complete = false; }
        return tree_complete;
    }

    static void drain_events(root_group & group) {
        if (group.fd < 0) return;
        alignas(inotify_event) char buffer[64 * 1024];
        std::map<uint32_t, std::string> moved_from;
        for (;;) {
            const auto bytes = ::read(group.fd, buffer, sizeof(buffer));
            if (bytes < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                group.rescan_required = true;
                break;
            }
            if (bytes == 0) break;
            size_t offset = 0;
            while (offset + sizeof(inotify_event) <= static_cast<size_t>(bytes)) {
                const auto * event = reinterpret_cast<const inotify_event *>(buffer + offset);
                if (event->mask & IN_Q_OVERFLOW) group.rescan_required = true;
                if (event->mask & (IN_IGNORED | IN_DELETE_SELF | IN_MOVE_SELF)) group.rescan_required = true;
                if ((event->mask & IN_ISDIR) && (event->mask & (IN_MOVED_FROM | IN_MOVED_TO))) group.rescan_required = true;
                if ((event->mask & IN_ISDIR) && event->len && (event->mask & (IN_CREATE | IN_MOVED_TO))) {
                    const auto parent = group.directories.find(event->wd);
                    if (parent != group.directories.end()) {
                        const fs::path path = fs::path(parent->second) / event->name;
                        if (path_within(group.root, path) && !excluded_directory(path) && !add_watch_tree(group, path))
                            group.rescan_required = true;
                    }
                }
                if (!(event->mask & IN_ISDIR) && event->len) {
                    const auto parent = group.directories.find(event->wd);
                    if (parent != group.directories.end()) {
                        const auto path = (fs::path(parent->second) / event->name).lexically_normal().string();
                        if (event->mask & IN_MOVED_FROM) moved_from[event->cookie] = path;
                        else if (event->mask & IN_MOVED_TO) {
                            const auto old = moved_from.find(event->cookie);
                            if (old != moved_from.end() && event->cookie != 0) {
                                group.native_changes.push_back({agent_project_file_change_kind::moved, path, old->second, 0});
                                moved_from.erase(old);
                            } else group.native_changes.push_back({agent_project_file_change_kind::created, path, {}, 0});
                        } else if (event->mask & IN_DELETE)
                            group.native_changes.push_back({agent_project_file_change_kind::removed, path, {}, 0});
                        else if (event->mask & IN_CREATE)
                            group.native_changes.push_back({agent_project_file_change_kind::created, path, {}, 0});
                        else if (event->mask & (IN_CLOSE_WRITE | IN_ATTRIB))
                            group.native_changes.push_back({agent_project_file_change_kind::changed, path, {}, 0});
                    }
                }
                if (event->mask & (IN_CREATE | IN_CLOSE_WRITE | IN_MODIFY | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_ATTRIB))
                    ++group.queued_events;
                if (group.native_changes.size() > 65536 || moved_from.size() > 65536) {
                    group.native_changes.clear();
                    moved_from.clear();
                    group.rescan_required = true;
                }
                offset += sizeof(inotify_event) + event->len;
            }
        }
        for (const auto & item : moved_from)
            group.native_changes.push_back({agent_project_file_change_kind::removed, item.second, {}, 0});
    }
#endif

    void run() {
        auto next_poll = std::chrono::steady_clock::now();
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait_until(lock, next_poll, [this] { return stopping; });
                if (stopping) return;
                const auto now = std::chrono::steady_clock::now();
                next_poll = now + std::chrono::milliseconds(100);
        for (auto & entry : roots) {
                    auto & group = entry.second;
#ifdef __linux__
                    drain_events(group);
                    if (group.rescan_required && group.fd >= 0) {
                        add_watch_tree(group, group.root);
                        for (auto id : group.subscriptions) {
                            const auto sub = subscriptions.find(id);
                            if (sub == subscriptions.end()) continue;
                            for (const auto & file : sub->second.explicit_files) {
                                const fs::path path(file);
                                const auto parent = path.parent_path();
                                const int wd = inotify_add_watch(group.fd, parent.c_str(),
                                    IN_CREATE | IN_CLOSE_WRITE | IN_MODIFY | IN_DELETE | IN_MOVED_FROM |
                                    IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF | IN_ATTRIB | IN_ONLYDIR);
                                if (wd >= 0) group.directories[wd] = parent.string();
                                else group.degraded = true;
                            }
                        }
                    }
#endif
                    auto interval = std::chrono::milliseconds(2000);
                    for (auto id : group.subscriptions) {
                        const auto sub = subscriptions.find(id);
                        if (sub != subscriptions.end()) interval = std::min(interval, sub->second.poll_interval);
                    }
                    const bool event_scan = group.queued_events != 0 || group.rescan_required;
                    const bool polling_scan = group.last_scan.time_since_epoch().count() == 0 ||
                        now - group.last_scan >= interval;
                    if (!event_scan && !polling_scan) continue;
                    bool scan_complete = false;
                    auto current = scan(group, subscriptions, scan_complete);
                    if (!scan_complete) {
                        // Never fabricate deletions from an incomplete tree.
                        current = group.snapshot;
                        group.rescan_required = true;
                    }
                    group.last_scan = now;
                    std::vector<agent_project_file_change> changes;
                    std::set<std::string> removed, created;
                    for (const auto & prior : group.snapshot)
                        if (!current.count(prior.first)) removed.insert(prior.first);
                    for (const auto & item : current) {
                        const auto old = group.snapshot.find(item.first);
                        if (old == group.snapshot.end()) created.insert(item.first);
                        else if (!(old->second == item.second)) changes.push_back({agent_project_file_change_kind::changed, item.first, {}, 0});
                    }
                    for (const auto & path : created) changes.push_back({agent_project_file_change_kind::created, path, {}, 0});
                    for (const auto & path : removed) changes.push_back({agent_project_file_change_kind::removed, path, {}, 0});
#ifdef __linux__
                    std::set<std::string> native_paths;
                    for (const auto & change : group.native_changes) {
                        native_paths.insert(change.path);
                        if (!change.previous_path.empty()) native_paths.insert(change.previous_path);
                    }
                    changes.erase(std::remove_if(changes.begin(), changes.end(), [&](const auto & change) {
                        return native_paths.count(change.path) != 0;
                    }), changes.end());
                    changes.insert(changes.begin(), group.native_changes.begin(), group.native_changes.end());
                    group.native_changes.clear();
#endif
                    bool overflow = group.rescan_required;
                    if (overflow) changes = {{agent_project_file_change_kind::rescan_required, group.root, {}, 0}};
                    if (!changes.empty()) {
                        for (auto & change : changes) change.generation = ++generation;
                        for (auto id : group.subscriptions) {
                            auto sub = subscriptions.find(id);
                            if (sub == subscriptions.end()) continue;
                            auto delivered = changes;
                            delivered.erase(std::remove_if(delivered.begin(), delivered.end(), [&](const auto & change) {
                                if (change.kind == agent_project_file_change_kind::rescan_required) return false;
                                auto matches = [&](const std::string & path) {
                                    const fs::path file(path);
                                    return sub->second.explicit_files.count(file.lexically_normal().string()) ||
                                        (project_path_allowed(group.root, file) &&
                                         watched_file(file, sub->second.file_extensions, sub->second.file_names));
                                };
                                return !matches(change.path) && (change.previous_path.empty() || !matches(change.previous_path));
                            }), delivered.end());
                            if (delivered.empty()) continue;
                            sub->second.latest_generation = delivered.back().generation;
                            if (overflow) sub->second.needs_reconciliation = true;
                            if (sub->second.pending_events + delivered.size() > sub->second.max_pending_events) {
                                sub->second.pending.clear();
                                delivered = {{agent_project_file_change_kind::rescan_required, group.root, {}, ++generation}};
                                sub->second.latest_generation = delivered.back().generation;
                                sub->second.pending_events = 0;
                                sub->second.needs_reconciliation = true;
                            }
                            sub->second.pending_events += delivered.size();
                            sub->second.pending.push_back(std::move(delivered));
                            wake.notify_all();
                        }
                    }
                    group.snapshot = current;
                    group.queued_events = 0;
                    group.rescan_required = false;
                }
            }
        }
    }

    void ensure_worker() {
        if (!worker.joinable()) worker = std::thread([this] { run(); });
    }
};

agent_project_file_watch_service::agent_project_file_watch_service() : state_(std::make_unique<impl>()) {}

agent_project_file_watch_service::~agent_project_file_watch_service() {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->stopping = true;
        state_->wake.notify_all();
    }
    if (state_->worker.joinable()) state_->worker.join();
    for (auto & entry : state_->subscriptions)
        if (entry.second.delivery_worker.joinable()) entry.second.delivery_worker.join();
#ifdef __linux__
    for (auto & entry : state_->roots) if (entry.second.fd >= 0) ::close(entry.second.fd);
#endif
}

bool agent_project_file_watch_service::subscribe(const agent_project_file_watch_config & config,
        callback notify, uint64_t & subscription_id, std::string & error) {
    if (!notify || config.root.empty() || config.max_pending_events == 0 || config.max_pending_events > 65536) {
        error = "file watch requires a root, callback, and queue limit in the range 1..65536";
        return false;
    }
    auto valid_filter = [](const std::vector<std::string> & values, size_t max_length, bool extension) {
        for (const auto & value : values) {
            if (value.empty() || value.size() > max_length || value.find('\0') != std::string::npos ||
                    value.find('/') != std::string::npos || value.find('\\') != std::string::npos ||
                    (extension && value.front() != '.')) return false;
        }
        return true;
    };
    if (config.explicit_files.size() > 128 || config.file_extensions.size() > 64 || config.file_names.size() > 128 ||
            !valid_filter(config.file_extensions, 32, true) || !valid_filter(config.file_names, 128, false)) {
        error = "file watch filters are invalid or exceed their bounds";
        return false;
    }
    std::error_code ec;
    const auto root = fs::canonical(config.root, ec);
    if (ec || !fs::is_directory(root, ec) || ec) { error = "file watch root is unavailable"; return false; }
    std::set<std::string> explicit_files;
    for (const auto & item : config.explicit_files) {
        const auto file = fs::weakly_canonical(item, ec);
        if (ec) { error = "explicit file watch path could not be resolved"; return false; }
        // Explicit paths are host configuration (for example a compilation
        // database in a separate build tree), never tool-supplied paths.
        explicit_files.insert(file.string());
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->stopping) { error = "file watch service is stopping"; return false; }
    if (state_->subscriptions.size() >= 128) { error = "file watch subscription limit reached"; return false; }
    const auto root_name = root.string();
    auto found = state_->roots.find(root_name);
    const bool new_root = found == state_->roots.end();
    if (found == state_->roots.end()) {
        impl::root_group group;
        group.root = root_name;
        group.force_polling = config.force_polling;
#ifdef __linux__
        if (!group.force_polling) group.fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
        if (group.fd >= 0) impl::add_watch_tree(group, root);
#endif
        found = state_->roots.emplace(root_name, std::move(group)).first;
    }
    if (found->second.force_polling != config.force_polling) {
        error = "subscribers sharing a root must use the same watch backend";
        return false;
    }
    auto id = state_->next_id++;
    impl::subscription sub;
    sub.id = id;
    sub.root = root_name;
    sub.explicit_files = std::move(explicit_files);
    sub.file_extensions = config.file_extensions;
    sub.file_names = config.file_names;
    sub.notify = std::move(notify);
    sub.max_pending_events = config.max_pending_events;
    sub.poll_interval = std::max(config.poll_interval, std::chrono::milliseconds(100));
    sub.needs_reconciliation = false;
    state_->subscriptions.emplace(id, std::move(sub));
    found->second.subscriptions.insert(id);
#ifdef __linux__
    if (found->second.fd >= 0) {
        for (const auto & file : state_->subscriptions.at(id).explicit_files) {
            const fs::path path(file);
            const auto parent = path.parent_path();
            const int wd = inotify_add_watch(found->second.fd, parent.c_str(),
                IN_CREATE | IN_CLOSE_WRITE | IN_MODIFY | IN_DELETE | IN_MOVED_FROM |
                IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF | IN_ATTRIB | IN_ONLYDIR);
            if (wd >= 0) found->second.directories[wd] = parent.string();
            else {
                found->second.degraded = true;
            }
        }
    }
#endif
    // Only the first subscriber establishes the shared baseline. Adding a
    // subscriber must never discard changes pending for existing consumers.
    if (new_root) {
        bool complete = false;
        found->second.snapshot = impl::scan(found->second, state_->subscriptions, complete);
        if (!complete) state_->subscriptions.at(id).needs_reconciliation = true;
    }
    state_->subscriptions.at(id).delivery_worker = std::thread([state = state_.get(), id] { state->deliver(id); });
    state_->ensure_worker();
    state_->wake.notify_all();
    subscription_id = id;
    error.clear();
    return true;
}

void agent_project_file_watch_service::unsubscribe(uint64_t subscription_id) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    auto sub = state_->subscriptions.find(subscription_id);
    if (sub == state_->subscriptions.end()) return;
    sub->second.closing = true;
    state_->wake.notify_all();
    // A consumer may cancel itself from its callback. The service retains the
    // stopped subscription until an external unsubscribe or destruction joins it.
    if (sub->second.delivery_worker.get_id() == std::this_thread::get_id()) return;
    auto delivery_worker = std::move(sub->second.delivery_worker);
    lock.unlock();
    if (delivery_worker.joinable()) delivery_worker.join();
    lock.lock();
    sub = state_->subscriptions.find(subscription_id);
    if (sub == state_->subscriptions.end()) return;
    const auto root = sub->second.root;
    state_->subscriptions.erase(sub);
    auto group = state_->roots.find(root);
    if (group == state_->roots.end()) return;
    group->second.subscriptions.erase(subscription_id);
    if (group->second.subscriptions.empty()) {
#ifdef __linux__
        if (group->second.fd >= 0) ::close(group->second.fd);
#endif
        state_->roots.erase(group);
    }
}

bool agent_project_file_watch_service::reconcile(uint64_t subscription_id, std::string & error) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto sub = state_->subscriptions.find(subscription_id);
    if (sub == state_->subscriptions.end()) { error = "file watch subscription is unavailable"; return false; }
    auto group = state_->roots.find(sub->second.root);
    if (group == state_->roots.end()) { error = "file watch root is unavailable"; return false; }
    // Do not replace the shared baseline or drain another subscriber's events.
    // The next scan will deliver any outstanding differences to all consumers.
    group->second.last_scan = {};
#ifdef __linux__
    if (group->second.fd < 0 && !group->second.force_polling) {
        group->second.fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    }
    if (group->second.fd >= 0) {
        group->second.degraded = false;
        impl::add_watch_tree(group->second, group->second.root);
        for (const auto id : group->second.subscriptions) {
          const auto other = state_->subscriptions.find(id);
          if (other == state_->subscriptions.end()) continue;
          for (const auto & file : other->second.explicit_files) {
            const fs::path path(file);
            const auto parent = path.parent_path();
            const int wd = inotify_add_watch(group->second.fd, parent.c_str(),
                IN_CREATE | IN_CLOSE_WRITE | IN_MODIFY | IN_DELETE | IN_MOVED_FROM |
                IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF | IN_ATTRIB | IN_ONLYDIR);
            if (wd >= 0) group->second.directories[wd] = parent.string();
            else group->second.degraded = true;
          }
        }
    }
#endif
    // The consumer has reset its own state before acknowledging this point.
    // Older queued batches are superseded for this subscriber only.
    sub->second.pending.clear();
    sub->second.pending_events = 0;
    sub->second.delivered_generation = sub->second.latest_generation;
    sub->second.needs_reconciliation = false;
    state_->wake.notify_all();
    error.clear();
    return true;
}

bool agent_project_file_watch_service::needs_reconciliation(uint64_t id) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto sub = state_->subscriptions.find(id);
    return sub == state_->subscriptions.end() || sub->second.needs_reconciliation;
}

uint64_t agent_project_file_watch_service::latest_generation(uint64_t id) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto sub = state_->subscriptions.find(id);
    return sub == state_->subscriptions.end() ? 0 : sub->second.latest_generation;
}

uint64_t agent_project_file_watch_service::delivered_generation(uint64_t id) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto sub = state_->subscriptions.find(id);
    return sub == state_->subscriptions.end() ? 0 : sub->second.delivered_generation;
}

bool agent_project_file_watch_service::wait_for_delivery(uint64_t id, std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->wake.wait_for(lock, timeout, [&] {
        const auto sub = state_->subscriptions.find(id);
        return state_->stopping || sub == state_->subscriptions.end() || sub->second.closing ||
            sub->second.needs_reconciliation || sub->second.delivered_generation >= sub->second.latest_generation;
    });
    const auto sub = state_->subscriptions.find(id);
    return sub != state_->subscriptions.end() && !sub->second.closing && !sub->second.needs_reconciliation &&
        sub->second.delivered_generation >= sub->second.latest_generation;
}

std::string agent_project_file_watch_service::backend(uint64_t id) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto sub = state_->subscriptions.find(id);
    if (sub == state_->subscriptions.end()) return "unavailable";
    const auto group = state_->roots.find(sub->second.root);
    if (group == state_->roots.end()) return "unavailable";
#ifdef __linux__
    if (group->second.fd >= 0) return group->second.degraded ? "inotify_degraded_polling" : "inotify_with_polling_fallback";
#endif
    return "polling";
}
