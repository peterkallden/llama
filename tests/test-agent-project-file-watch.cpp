#include "agent-project-file-watch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void write_file(const std::filesystem::path & path, const std::string & content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << content;
    if (!output) throw std::runtime_error("could not write watched fixture");
}

bool wait_for(std::condition_variable & cv, std::unique_lock<std::mutex> & lock,
        const std::function<bool()> & predicate) {
    return cv.wait_for(lock, std::chrono::seconds(8), predicate);
}
}

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() /
        ("llama-project-file-watch-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto outside = root.parent_path() / (root.filename().string() + "-compile-db");
    try {
        fs::create_directories(root);
        write_file(outside, "[]");
        agent_project_file_watch_service watcher;
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<agent_project_file_change> first_events, second_events;
        auto make_callback = [&](auto & sink) {
            return [&mutex, &cv, &sink](const auto & changes) {
                std::lock_guard<std::mutex> lock(mutex);
                sink.insert(sink.end(), changes.begin(), changes.end());
                cv.notify_all();
                return true;
            };
        };
        agent_project_file_watch_config config;
        config.root = root.string();
        config.explicit_files = {outside.string()};
        config.poll_interval = std::chrono::milliseconds(100);
        uint64_t first = 0, second = 0;
        std::string error;
        require(watcher.subscribe(config, make_callback(first_events), first, error), "first watch subscription must succeed");
        config.root = (root / ".." / root.filename()).string();
        require(watcher.subscribe(config, make_callback(second_events), second, error), "canonical duplicate root must share its watch");

        const auto source = root / "src" / "first.cpp";
        write_file(source, "int first;\n");
        {
            std::unique_lock<std::mutex> lock(mutex);
            require(wait_for(cv, lock, [&] {
                return std::any_of(first_events.begin(), first_events.end(), [&](const auto & event) {
                    return event.path == source.string() && event.kind == agent_project_file_change_kind::created;
                });
            }), "new nested source file must be observed");
        }
        const auto delivery_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!watcher.delivered_generation(first) && std::chrono::steady_clock::now() < delivery_deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        require(watcher.delivered_generation(first) > 0, "delivered generation must advance after callback acceptance");

        const auto renamed = root / "src" / "renamed.cpp";
        fs::rename(source, renamed);
        {
            std::unique_lock<std::mutex> lock(mutex);
            require(wait_for(cv, lock, [&] {
#ifdef __linux__
                return std::any_of(first_events.begin(), first_events.end(), [&](const auto & event) {
                    return event.kind == agent_project_file_change_kind::moved &&
                        event.path == renamed.string() && event.previous_path == source.string();
                });
#else
                const auto removed = std::any_of(first_events.begin(), first_events.end(), [&](const auto & event) {
                    return event.path == source.string() && event.kind == agent_project_file_change_kind::removed;
                });
                const auto created = std::any_of(first_events.begin(), first_events.end(), [&](const auto & event) {
                    return event.path == renamed.string() && event.kind == agent_project_file_change_kind::created;
                });
                return removed && created;
#endif
            }), "atomic rename must be reported as a move");
        }

        write_file(root / "build" / "ignored.cpp", "int ignored;\n");
        write_file(root / ".git" / "ignored.cpp", "int ignored;\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        {
            std::lock_guard<std::mutex> lock(mutex);
            require(std::none_of(first_events.begin(), first_events.end(), [](const auto & event) {
                return event.path.find("ignored.cpp") != std::string::npos;
            }), "build and Git directories must be excluded");
            require(!second_events.empty(), "duplicate root subscriber must receive changes");
        }

        write_file(outside, "[{\"file\":\"x.cpp\"}]");
        {
            std::unique_lock<std::mutex> lock(mutex);
            require(wait_for(cv, lock, [&] {
                return std::any_of(first_events.begin(), first_events.end(), [&](const auto & event) {
                    return event.path == outside.string() && event.kind == agent_project_file_change_kind::changed;
                });
            }), "explicit host-configured compilation database outside root must be watched");
        }

        fs::remove(renamed);
        {
            std::unique_lock<std::mutex> lock(mutex);
            require(wait_for(cv, lock, [&] {
                return std::any_of(first_events.begin(), first_events.end(), [&](const auto & event) {
                    return event.path == renamed.string() && event.kind == agent_project_file_change_kind::removed;
                });
            }), "file deletion must be observed");
        }

        require(watcher.reconcile(first, error), "manual reconciliation must succeed");
        require(!watcher.needs_reconciliation(first), "successful reconciliation clears stale status");

        const auto during_reconcile = root / "src" / "during-reconcile.cpp";
        write_file(during_reconcile, "int during_reconcile;\n");
        require(watcher.reconcile(first, error), "one consumer can reconcile independently");
        {
            std::unique_lock<std::mutex> lock(mutex);
            require(wait_for(cv, lock, [&] {
                return std::any_of(second_events.begin(), second_events.end(), [&](const auto & event) {
                    return event.path == during_reconcile.string();
                });
            }), "reconciling one consumer must not discard another consumer's pending change");
        }

#ifdef __linux__
        // An atomic replacement can preserve both size and mtime; its inotify
        // rename event must still reach subscribers.
        const auto replace_target = root / "src" / "replacement.cpp";
        write_file(replace_target, "int old;\n");
        {
            std::unique_lock<std::mutex> lock(mutex);
            require(wait_for(cv, lock, [&] {
                return std::any_of(first_events.begin(), first_events.end(), [&](const auto & event) {
                    return event.path == replace_target.string();
                });
            }), "replacement fixture must be observed before the atomic swap");
        }
        const auto original_time = fs::last_write_time(replace_target);
        const auto temporary = root / "src" / "replacement.tmp";
        write_file(temporary, "int new;\n");
        fs::last_write_time(temporary, original_time);
        fs::rename(temporary, replace_target);
        {
            std::unique_lock<std::mutex> lock(mutex);
            require(wait_for(cv, lock, [&] {
                return std::any_of(first_events.begin(), first_events.end(), [&](const auto & event) {
                    return event.path == replace_target.string() && event.kind == agent_project_file_change_kind::moved;
                });
            }), "inotify must report an atomic replacement even with unchanged size and mtime");
        }
#endif

        std::atomic<bool> slow_entered{false}, release_slow{false};
        std::atomic<int> failed_calls{0};
        uint64_t slow = 0, failing = 0;
        require(watcher.subscribe(config, [&](const auto &) {
            slow_entered = true;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!release_slow && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return true;
        }, slow, error), "slow consumer subscription must succeed");
        require(watcher.subscribe(config, [&](const auto &) {
            ++failed_calls;
            return false;
        }, failing, error), "failing consumer subscription must succeed");
        const auto isolated = root / "src" / "isolated.cpp";
        write_file(isolated, "int isolated;\n");
        bool observed_by_first = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            observed_by_first = wait_for(cv, lock, [&] {
                return std::any_of(first_events.begin(), first_events.end(), [&](const auto & event) {
                    return event.path == isolated.string();
                });
            });
        }
        const auto failure_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while ((!slow_entered || !failed_calls || !watcher.needs_reconciliation(failing)) &&
                std::chrono::steady_clock::now() < failure_deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        release_slow = true;
        require(observed_by_first && slow_entered, "slow consumer must not block another consumer");
        require(failed_calls && watcher.needs_reconciliation(failing), "failed delivery must require reconciliation");
        require(watcher.delivered_generation(failing) == 0, "failed delivery must not be acknowledged");
        watcher.unsubscribe(failing);
        watcher.unsubscribe(slow);
        watcher.unsubscribe(second);
        watcher.unsubscribe(first);

        const auto polling_root = root / "polling-root";
        fs::create_directories(polling_root);
        agent_project_file_watch_config polling;
        polling.root = polling_root.string();
        polling.force_polling = true;
        polling.poll_interval = std::chrono::milliseconds(150);
        std::atomic<bool> polled{false};
        uint64_t polling_id = 0;
        require(watcher.subscribe(polling, [&](const auto & changes) {
            for (const auto & change : changes)
                if (change.path == (polling_root / "poll.cpp").string()) polled = true;
            return true;
        }, polling_id, error), "polling backend subscription must succeed");
        require(watcher.backend(polling_id) == "polling", "forced portable backend must be reported");
        write_file(polling_root / "poll.cpp", "int poll;\n");
        const auto polling_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!polled && std::chrono::steady_clock::now() < polling_deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        require(polled, "polling backend must discover file changes");
        watcher.unsubscribe(polling_id);

        const auto overflow_root = root / "overflow-root";
        fs::create_directories(overflow_root);
        agent_project_file_watch_config overflow_config;
        overflow_config.root = overflow_root.string();
        overflow_config.force_polling = true;
        overflow_config.poll_interval = std::chrono::milliseconds(100);
        overflow_config.max_pending_events = 1;
        std::atomic<bool> overflow_entered{false}, overflow_release{false};
        uint64_t overflow_id = 0;
        require(watcher.subscribe(overflow_config, [&](const auto &) {
            overflow_entered = true;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!overflow_release && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return true;
        }, overflow_id, error), "bounded queue subscription must succeed");
        write_file(overflow_root / "a.cpp", "int a;\n");
        const auto enter_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!overflow_entered && std::chrono::steady_clock::now() < enter_deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        write_file(overflow_root / "b.cpp", "int b;\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        write_file(overflow_root / "c.cpp", "int c;\n");
        const auto overflow_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!watcher.needs_reconciliation(overflow_id) && std::chrono::steady_clock::now() < overflow_deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const bool overflow_detected = watcher.needs_reconciliation(overflow_id);
        overflow_release = true;
        require(overflow_entered && overflow_detected, "bounded callback queue overflow must require reconciliation");
        watcher.unsubscribe(overflow_id);
        fs::remove_all(root);
        fs::remove(outside);
        return 0;
    } catch (const std::exception & exception) {
        std::error_code ignored;
        fs::remove_all(root, ignored);
        fs::remove(outside, ignored);
        std::fprintf(stderr, "%s\n", exception.what());
        return 1;
    }
}
