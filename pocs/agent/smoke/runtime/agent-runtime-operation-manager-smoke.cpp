#include "common/runtime/runtime-operation.h"

#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>

int main() {
    common_runtime_operation_manager manager;
    int polls = 0;
    std::string error;
    common_runtime_operation operation;
    operation.operation_id = "op-1";
    operation.kind = common_runtime_operation_kind::tool;
    operation.detail = "smoke";

    if (!manager.begin(
            operation,
            [&polls](bool & ready, std::string &) {
                ready = ++polls >= 2;
                return true;
            },
            {},
            error)) {
        std::fprintf(stderr, "begin failed: %s\n", error.c_str());
        return 1;
    }

    bool ready = false;
    const bool first_poll_ok = manager.poll("op-1", ready, error);
    if (!first_poll_ok || ready) {
        std::fprintf(stderr, "first poll did not remain pending: ok=%d ready=%d polls=%d error=%s\n", first_poll_ok, ready, polls, error.c_str());
        return 1;
    }
    if (!manager.poll("op-1", ready, error) || !ready) {
        std::fprintf(stderr, "second poll did not complete\n");
        return 1;
    }

    common_runtime_operation_status status;
    if (!manager.describe("op-1", status) ||
            status.state != common_runtime_operation_state::completed) {
        std::fprintf(stderr, "completed operation was not retained as completed\n");
        return 1;
    }
    if (manager.cleanup_terminal() != 1 || manager.describe("op-1", status)) {
        std::fprintf(stderr, "terminal operation cleanup failed\n");
        return 1;
    }

    common_runtime_operation expiring;
    expiring.operation_id = "op-expire";
    expiring.deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    bool timeout_cancel_called = false;
    if (!manager.begin(
            expiring,
            [](bool &, std::string &) { return true; },
            [&timeout_cancel_called](std::string &) {
                timeout_cancel_called = true;
                return true;
            },
            error)) {
        std::fprintf(stderr, "expiring begin failed: %s\n", error.c_str());
        return 1;
    }
    if (manager.poll("op-expire", ready, error) ||
            error != "operation deadline exceeded" ||
            !timeout_cancel_called ||
            !manager.describe("op-expire", status) ||
            status.state != common_runtime_operation_state::timed_out) {
        std::fprintf(stderr, "deadline transition/cancellation failed: %s\n", error.c_str());
        return 1;
    }

    common_runtime_operation cancellable;
    cancellable.operation_id = "op-cancel";
    bool cancel_called = false;
    if (!manager.begin(
            cancellable,
            [](bool &, std::string &) { return true; },
            [&cancel_called](std::string &) {
                cancel_called = true;
                return true;
            },
            error) ||
            !manager.cancel("op-cancel", error) ||
            !cancel_called ||
            !manager.describe("op-cancel", status) ||
            status.state != common_runtime_operation_state::cancelled) {
            std::fprintf(stderr, "cancellation transition failed: %s\n", error.c_str());
        return 1;
    }

    common_runtime_operation race_operation;
    race_operation.operation_id = "op-poll-cancel-race";
    auto poll_started = std::make_shared<std::promise<void>>();
    auto poll_started_future = poll_started->get_future();
    auto release_poll = std::make_shared<std::promise<void>>();
    const auto release_poll_future = release_poll->get_future().share();
    if (!manager.begin(
            race_operation,
            [poll_started, release_poll_future](bool & poll_ready, std::string &) {
                poll_started->set_value();
                release_poll_future.wait();
                poll_ready = true;
                return true;
            },
            [](std::string &) { return true; },
            error)) {
        std::fprintf(stderr, "race operation setup failed: %s\n", error.c_str());
        return 1;
    }
    bool race_ready = false;
    std::string race_error;
    auto poll_future = std::async(std::launch::async, [&manager, &race_ready, &race_error]() {
        return manager.poll("op-poll-cancel-race", race_ready, race_error);
    });
    if (poll_started_future.wait_for(std::chrono::seconds(1)) != std::future_status::ready ||
            !manager.cancel("op-poll-cancel-race", error)) {
        release_poll->set_value();
        poll_future.wait();
        std::fprintf(stderr, "poll/cancel race setup failed: %s\n", error.c_str());
        return 1;
    }
    release_poll->set_value();
    if (!poll_future.get() || !race_ready ||
            !manager.describe("op-poll-cancel-race", status) ||
            status.state != common_runtime_operation_state::cancelled) {
        std::fprintf(stderr, "poll overwrote cancellation with completion\n");
        return 1;
    }

    struct blocking_cleanup_probe {
        std::shared_ptr<std::promise<void>> destruction_started;

        explicit blocking_cleanup_probe(
                std::shared_ptr<std::promise<void>> destruction_started_value)
            : destruction_started(std::move(destruction_started_value)) {}

        ~blocking_cleanup_probe() {
            destruction_started->set_value();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    };

    auto destruction_started = std::make_shared<std::promise<void>>();
    auto destruction_started_future = destruction_started->get_future();
    auto probe = std::make_shared<blocking_cleanup_probe>(destruction_started);
    common_runtime_operation cleanup_operation;
    cleanup_operation.operation_id = "op-cleanup";
    if (!manager.begin(
            cleanup_operation,
            [probe](bool & ready, std::string &) {
                ready = true;
                return true;
            },
            {},
            error) || !manager.poll("op-cleanup", ready, error)) {
        std::fprintf(stderr, "cleanup operation setup failed: %s\n", error.c_str());
        return 1;
    }
    probe.reset();

    common_runtime_operation live_operation;
    live_operation.operation_id = "op-live";
    if (!manager.begin(
            live_operation,
            [](bool & ready, std::string &) {
                ready = false;
                return true;
            },
            {},
            error)) {
        std::fprintf(stderr, "live operation setup failed: %s\n", error.c_str());
        return 1;
    }

    auto cleanup_future = std::async(std::launch::async, [&manager]() {
        return manager.cleanup_terminal();
    });
    if (destruction_started_future.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
        cleanup_future.wait();
        std::fprintf(stderr, "terminal cleanup did not begin destroying its detached entry\n");
        return 1;
    }
    common_runtime_operation_status live_status;
    auto describe_future = std::async(std::launch::async, [&manager, &live_status]() {
        return manager.describe("op-live", live_status);
    });
    if (describe_future.wait_for(std::chrono::milliseconds(250)) != std::future_status::ready ||
            !describe_future.get()) {
        cleanup_future.wait();
        std::fprintf(stderr, "terminal cleanup held the operation manager mutex\n");
        return 1;
    }
    if (cleanup_future.get() != 4) {
        std::fprintf(stderr, "detached terminal cleanup removed an unexpected count\n");
        return 1;
    }

    return 0;
}
