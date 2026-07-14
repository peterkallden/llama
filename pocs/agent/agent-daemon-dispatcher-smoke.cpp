#include "agent-daemon-adapter.h"
#include "agent-daemon-dispatcher.h"

#include "log.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

namespace {

common_agent_daemon_command make_turn_command(
        std::string request_id,
        std::string prompt,
        std::string turn_id,
        int n_predict) {
    common_agent_daemon_command command;
    command.request_id = std::move(request_id);
    command.type = common_agent_daemon_command_type::run_turn;
    command.turn = common_agent_daemon_turn_payload{};
    command.turn->request.request_id = command.request_id;
    command.turn->request.turn.mode = common_agent_runtime_host_mode::chat;
    command.turn->request.turn.prompt = std::move(prompt);
    command.turn->request.turn.session_id = "dispatcher-smoke-session";
    command.turn->request.turn.namespace_id = "dispatcher-smoke";
    command.turn->request.turn.project_id = "dispatcher-smoke-project";
    command.turn->request.turn.turn_id = std::move(turn_id);
    command.turn->request.turn.memory_scope = common_memory_scope::project;
    command.turn->request.turn.plan_scope = common_plan_scope::project;
    command.turn->request.turn.n_predict = n_predict;
    return command;
}

common_agent_daemon_command make_reset_command() {
    common_agent_daemon_command command;
    command.request_id = "reset-1";
    command.type = common_agent_daemon_command_type::reset_session;
    command.session = common_agent_daemon_session_payload{
        {"dispatcher-smoke", "dispatcher-smoke-session"},
    };
    return command;
}

common_agent_daemon_command make_close_command() {
    common_agent_daemon_command command;
    command.request_id = "close-1";
    command.type = common_agent_daemon_command_type::close_session;
    command.session = common_agent_daemon_session_payload{
        {"dispatcher-smoke", "dispatcher-smoke-session"},
    };
    return command;
}

} // namespace

int main(int argc, char ** argv) {
    daemon_options options;
    if (!parse_agent_daemon_args(argc, argv, options)) {
        print_agent_daemon_usage(argv[0]);
        return 2;
    }

    common_log_set_verbosity_thold(LOG_LEVEL_WARN);

    common_agent_daemon_runtime runtime;
    std::string error;
    if (!initialize_agent_daemon_environment(options, runtime, error)) {
        std::fprintf(stderr, "failed to initialize daemon environment: %s\n", error.c_str());
        return 2;
    }

    common_agent_daemon_dispatcher dispatcher(std::move(runtime), 8);

    common_agent_daemon_command_result first_result;
    common_agent_daemon_command_result second_result;
    std::string first_error;
    std::string second_error;
    bool first_ok = false;
    bool second_ok = false;

    auto first_command = make_turn_command(
        "turn-1",
        "Write the numbers 1 through 80, separated by spaces.",
        "dispatcher-turn-1",
        96);
    auto second_command = make_turn_command(
        "turn-2",
        "Reply with OK only.",
        "dispatcher-turn-2",
        8);

    std::thread first_thread([&]() {
        first_ok = dispatcher.execute(first_command, first_result, first_error);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::thread second_thread([&]() {
        second_ok = dispatcher.execute(second_command, second_result, second_error);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (dispatcher.queued_command_count() == 0 &&
            std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    if (dispatcher.queued_command_count() == 0) {
        std::fprintf(stderr, "second turn never reached the dispatcher queue\n");
        if (first_thread.joinable()) first_thread.join();
        if (second_thread.joinable()) second_thread.join();
        return 1;
    }

    common_agent_daemon_command cancel_command;
    cancel_command.request_id = "cancel-1";
    cancel_command.type = common_agent_daemon_command_type::cancel_turn;
    cancel_command.cancel = common_agent_daemon_cancel_payload{"turn-2", {}};

    common_agent_daemon_command_result cancel_result;
    std::string cancel_error;
    const bool cancel_ok = dispatcher.execute(cancel_command, cancel_result, cancel_error);

    if (first_thread.joinable()) first_thread.join();
    if (second_thread.joinable()) second_thread.join();

    if (!cancel_ok) {
        std::fprintf(stderr, "cancel command failed: %s\n", cancel_error.c_str());
        return 1;
    }
    if (cancel_result.event != "turn_cancelled" || cancel_result.target_request_id != "turn-2") {
        std::fprintf(stderr, "unexpected cancel result\n");
        return 1;
    }
    if (cancel_result.status.active_request_id != "turn-1" ||
            cancel_result.status.active_turn_id != "dispatcher-turn-1" ||
            cancel_result.status.active_turn_phase.empty() ||
            cancel_result.status.active_turn_disposition.empty() ||
            cancel_result.status.queued_command_count != 0) {
        std::fprintf(stderr, "cancel result missing consistent status snapshot\n");
        return 1;
    }
    if (cancel_result.daemon_event_count < 1 ||
            cancel_result.events.empty() ||
            cancel_result.events.back().type != "turn.cancelled") {
        std::fprintf(stderr, "cancel result missing daemon cancellation event\n");
        return 1;
    }
    if (!first_ok || first_result.turn_result.response.empty()) {
        std::fprintf(stderr, "first turn failed: %s\n", first_error.c_str());
        return 1;
    }
    if (first_result.daemon_event_count < 2) {
        std::fprintf(stderr, "first turn missing daemon lifecycle events\n");
        return 1;
    }
    if (second_ok) {
        std::fprintf(stderr, "second queued turn unexpectedly ran to completion\n");
        return 1;
    }
    if (!second_result.turn_result.cancelled ||
            second_result.turn_result.error != "turn cancelled before execution") {
        std::fprintf(stderr, "second queued turn was not marked cancelled correctly\n");
        return 1;
    }
    if (second_result.daemon_event_count < 1 ||
            second_result.events.empty() ||
            second_result.events.back().type != "turn.cancelled") {
        std::fprintf(stderr, "second queued turn missing daemon cancellation event\n");
        return 1;
    }

    std::printf("dispatcher_cancelled_request=%s\n", cancel_result.target_request_id.c_str());
    std::printf("dispatcher_active_turn=%s/%s:%s\n",
        cancel_result.status.active_turn_id.c_str(),
        cancel_result.status.active_turn_phase.c_str(),
        cancel_result.status.active_turn_disposition.c_str());
    std::printf("first_turn_response=%s\n", first_result.turn_result.response.c_str());
    std::printf("second_turn_cancelled=%s\n", second_result.turn_result.cancelled ? "yes" : "no");
    std::printf("second_turn_error=%s\n", second_result.turn_result.error.c_str());

    common_agent_daemon_runtime reset_runtime;
    if (!initialize_agent_daemon_environment(options, reset_runtime, error)) {
        std::fprintf(stderr, "failed to initialize reset daemon environment: %s\n", error.c_str());
        return 2;
    }
    common_agent_daemon_dispatcher reset_dispatcher(std::move(reset_runtime), 8);

    common_agent_daemon_command_result reset_first_result;
    common_agent_daemon_command_result reset_second_result;
    common_agent_daemon_command_result reset_command_result;
    std::string reset_first_error;
    std::string reset_second_error;
    std::string reset_error;
    bool reset_first_ok = false;
    bool reset_second_ok = false;

    auto reset_first_command = make_turn_command(
        "turn-reset-1",
        "Write the numbers 1 through 80, separated by spaces.",
        "dispatcher-reset-turn-1",
        96);
    auto reset_second_command = make_turn_command(
        "turn-reset-2",
        "Reply with OK only.",
        "dispatcher-reset-turn-2",
        8);
    auto reset_command = make_reset_command();

    std::thread reset_first_thread([&]() {
        reset_first_ok = reset_dispatcher.execute(reset_first_command, reset_first_result, reset_first_error);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::thread reset_second_thread([&]() {
        reset_second_ok = reset_dispatcher.execute(reset_second_command, reset_second_result, reset_second_error);
    });

    const auto reset_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (reset_dispatcher.queued_command_count() == 0 &&
            std::chrono::steady_clock::now() < reset_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    if (reset_dispatcher.queued_command_count() == 0) {
        std::fprintf(stderr, "reset scenario never queued the second turn\n");
        if (reset_first_thread.joinable()) reset_first_thread.join();
        if (reset_second_thread.joinable()) reset_second_thread.join();
        return 1;
    }

    const bool reset_ok = reset_dispatcher.execute(reset_command, reset_command_result, reset_error);

    if (reset_first_thread.joinable()) reset_first_thread.join();
    if (reset_second_thread.joinable()) reset_second_thread.join();

    if (!reset_ok || reset_command_result.event != "session_reset") {
        std::fprintf(stderr, "reset command failed: %s\n", reset_error.c_str());
        return 1;
    }
    if (!reset_first_ok || reset_first_result.turn_result.response.empty()) {
        std::fprintf(stderr, "reset scenario first turn failed: %s\n", reset_first_error.c_str());
        return 1;
    }
    if (reset_second_ok) {
        std::fprintf(stderr, "reset scenario queued turn unexpectedly ran to completion\n");
        return 1;
    }
    if (reset_second_result.turn_result.cancelled ||
            reset_second_result.event != "turn_rejected" ||
            reset_second_result.turn_result.error != "session reset before queued turn reached session lane") {
        std::fprintf(stderr, "reset scenario queued turn did not receive the expected rejection\n");
        return 1;
    }
    if (reset_second_result.daemon_event_count < 1 ||
            reset_second_result.events.empty() ||
            reset_second_result.events.back().type != "turn.rejected") {
        std::fprintf(stderr, "reset scenario queued turn missing rejection event\n");
        return 1;
    }
    if (reset_command_result.status.queued_command_count != 0 ||
            reset_command_result.status.session_count < 1) {
        std::fprintf(stderr, "reset scenario lifecycle result missing consistent status snapshot\n");
        return 1;
    }

    common_agent_daemon_runtime close_runtime;
    if (!initialize_agent_daemon_environment(options, close_runtime, error)) {
        std::fprintf(stderr, "failed to initialize close daemon environment: %s\n", error.c_str());
        return 2;
    }
    common_agent_daemon_dispatcher close_dispatcher(std::move(close_runtime), 8);

    common_agent_daemon_command_result close_first_result;
    common_agent_daemon_command_result close_second_result;
    common_agent_daemon_command_result close_command_result;
    std::string close_first_error;
    std::string close_second_error;
    std::string close_error;
    bool close_first_ok = false;
    bool close_second_ok = false;

    auto close_first_command = make_turn_command(
        "turn-close-1",
        "Write the numbers 1 through 80, separated by spaces.",
        "dispatcher-close-turn-1",
        96);
    auto close_second_command = make_turn_command(
        "turn-close-2",
        "Reply with OK only.",
        "dispatcher-close-turn-2",
        8);
    auto close_command = make_close_command();

    std::thread close_first_thread([&]() {
        close_first_ok = close_dispatcher.execute(close_first_command, close_first_result, close_first_error);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::thread close_second_thread([&]() {
        close_second_ok = close_dispatcher.execute(close_second_command, close_second_result, close_second_error);
    });

    const auto close_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (close_dispatcher.queued_command_count() == 0 &&
            std::chrono::steady_clock::now() < close_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    if (close_dispatcher.queued_command_count() == 0) {
        std::fprintf(stderr, "close scenario never queued the second turn\n");
        if (close_first_thread.joinable()) close_first_thread.join();
        if (close_second_thread.joinable()) close_second_thread.join();
        return 1;
    }

    const bool close_ok = close_dispatcher.execute(close_command, close_command_result, close_error);

    if (close_first_thread.joinable()) close_first_thread.join();
    if (close_second_thread.joinable()) close_second_thread.join();

    if (!close_ok || close_command_result.event != "session_closed") {
        std::fprintf(stderr, "close command failed: %s\n", close_error.c_str());
        return 1;
    }
    if (!close_first_ok || close_first_result.turn_result.response.empty()) {
        std::fprintf(stderr, "close scenario first turn failed: %s\n", close_first_error.c_str());
        return 1;
    }
    if (close_second_ok) {
        std::fprintf(stderr, "close scenario queued turn unexpectedly ran to completion\n");
        return 1;
    }
    if (close_second_result.turn_result.cancelled ||
            close_second_result.event != "turn_rejected" ||
            close_second_result.turn_result.error != "session closed before queued turn reached session lane") {
        std::fprintf(stderr, "close scenario queued turn did not receive the expected rejection\n");
        return 1;
    }
    if (close_second_result.daemon_event_count < 1 ||
            close_second_result.events.empty() ||
            close_second_result.events.back().type != "turn.rejected") {
        std::fprintf(stderr, "close scenario queued turn missing rejection event\n");
        return 1;
    }
    if (close_command_result.status.queued_command_count != 0 ||
            close_command_result.status.session_count != 0) {
        std::fprintf(stderr, "close scenario lifecycle result missing consistent status snapshot\n");
        return 1;
    }

    std::printf("reset_second_turn_event=%s\n", reset_second_result.event.c_str());
    std::printf("reset_second_turn_error=%s\n", reset_second_result.turn_result.error.c_str());
    std::printf("close_second_turn_event=%s\n", close_second_result.event.c_str());
    std::printf("close_second_turn_error=%s\n", close_second_result.turn_result.error.c_str());
    return 0;
}
