#include "common/runtime/runtime-operation.h"

#include <chrono>
#include <cstdio>
#include <string>

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
    if (!manager.begin(expiring, [](bool &, std::string &) { return true; }, {}, error)) {
        std::fprintf(stderr, "expiring begin failed: %s\n", error.c_str());
        return 1;
    }
    if (manager.poll("op-expire", ready, error) ||
            error != "operation deadline exceeded" ||
            !manager.describe("op-expire", status) ||
            status.state != common_runtime_operation_state::timed_out) {
        std::fprintf(stderr, "deadline transition failed: %s\n", error.c_str());
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

    return 0;
}
