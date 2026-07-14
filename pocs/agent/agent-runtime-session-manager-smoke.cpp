#include "agent-runtime-session-manager.h"

#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cstdio>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>

int main() {
    common_memory_in_memory_store memory_store;
    common_plan_in_memory_store plan_store;
    common_memory_policy_pack session_policy_pack;
    session_policy_pack.id = "session-policy-a";
    session_policy_pack.purpose = "Keep this session on the host-owned policy path.";

    common_agent_runtime_session_manager_build_config manager_build_config = {
        memory_store,
        plan_store,
    };
    manager_build_config.resident_request = {
        "",
        "",
        "",
        "",
        std::nullopt,
        "fake.gguf",
        32,
        0,
        false,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
    };

    auto call_count = std::make_shared<int>(0);
    manager_build_config.tooling_resolver =
        [call_count](
                const common_agent_runtime_resident_runtime *,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            tooling = {};
            ++(*call_count);
            error = "resolver-call=" + std::to_string(*call_count) + " turn=" + request.turn_id;
            return false;
        };
    auto manager_config = make_agent_runtime_session_manager_config(std::move(manager_build_config));
    if (!manager_config.tooling_resolver) {
        std::fprintf(stderr, "session manager smoke lost the tooling resolver while building config\n");
        return 1;
    }
    common_agent_runtime_session_manager manager(std::move(manager_config));

    std::string error;
    common_agent_runtime_session_manager_turn_result first_result;
    if (manager.run_turn({
            "request-1",
            {
                common_agent_runtime_host_mode::chat,
                "hello",
                "session-a",
                "namespace-a",
                "",
                "turn-1",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                session_policy_pack,
                make_common_agent_runtime_execution_control({}),
            },
        }, first_result, error)) {
        std::fprintf(stderr, "session manager smoke unexpectedly succeeded on first resolver failure\n");
        return 1;
    }
    if (error.find("resolver-call=1 turn=turn-1") == std::string::npos) {
        std::fprintf(
            stderr,
            "first resolver failure did not preserve diagnostics: error='%s' result.error='%s' call_count=%d\n",
            error.c_str(),
            first_result.error.c_str(),
            *call_count);
        return 1;
    }

    auto sessions = manager.list_sessions();
    if (sessions.size() != 1) {
        std::fprintf(stderr, "session manager did not retain the failed session lane\n");
        return 1;
    }
    if (sessions[0].policy_pack_id != "session-policy-a") {
        std::fprintf(stderr, "session manager did not retain session policy pack id: %s\n",
            sessions[0].policy_pack_id.c_str());
        return 1;
    }
    if (sessions[0].queued_turn_count != 0 || sessions[0].has_active_turn) {
        std::fprintf(stderr, "session manager left stale active or queued state after failed turn\n");
        return 1;
    }
    if (!sessions[0].active_request_id.empty()) {
        std::fprintf(stderr, "session manager retained stale active request identity\n");
        return 1;
    }
    if (sessions[0].last_turn_id != "turn-1" || sessions[0].last_turn_phase != "failed") {
        std::fprintf(stderr, "session manager did not record failed turn state correctly\n");
        return 1;
    }

    auto cancelled_control = make_common_agent_runtime_execution_control({});
    cancelled_control.cancellation->request_cancel("manager smoke cancelled");

    common_agent_runtime_session_manager_turn_result cancelled_result;
    if (manager.run_turn({
            "request-2",
            {
                common_agent_runtime_host_mode::chat,
                "world",
                "session-a",
                "namespace-a",
                "",
                "turn-2",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                cancelled_control,
            },
        }, cancelled_result, error)) {
        std::fprintf(stderr, "session manager smoke unexpectedly succeeded on cancelled turn\n");
        return 1;
    }
    if (!cancelled_result.cancelled || cancelled_result.error != "manager smoke cancelled") {
        std::fprintf(stderr, "session manager did not preserve cancellation result\n");
        return 1;
    }

    sessions = manager.list_sessions();
    if (sessions.size() != 1) {
        std::fprintf(stderr, "session manager lost the session lane after cancellation\n");
        return 1;
    }
    if (sessions[0].last_turn_id != "turn-2" || sessions[0].last_turn_phase != "cancelled") {
        std::fprintf(stderr, "session manager did not record cancelled turn state correctly\n");
        return 1;
    }

    if (!manager.reset_session({"namespace-a", "session-a"}, error)) {
        std::fprintf(stderr, "session manager reset failed: %s\n", error.c_str());
        return 1;
    }
    sessions = manager.list_sessions();
    if (sessions.size() != 1 || !sessions[0].last_turn_id.empty() || !sessions[0].last_turn_phase.empty()) {
        std::fprintf(stderr, "session manager reset did not clear lane state\n");
        return 1;
    }

    if (!manager.close_session({"namespace-a", "session-a"}, error)) {
        std::fprintf(stderr, "session manager close failed: %s\n", error.c_str());
        return 1;
    }
    if (!manager.list_sessions().empty()) {
        std::fprintf(stderr, "session manager close did not remove the lane\n");
        return 1;
    }

    auto active_entered = std::make_shared<std::promise<void>>();
    auto active_entered_future = active_entered->get_future();
    auto active_signal_sent = std::make_shared<bool>(false);

    common_agent_runtime_session_manager_build_config active_manager_build_config = {
        memory_store,
        plan_store,
    };
    active_manager_build_config.resident_request = {
        "",
        "",
        "",
        "",
        std::nullopt,
        "fake.gguf",
        32,
        0,
        false,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
    };
    active_manager_build_config.tooling_resolver =
        [active_entered, active_signal_sent](
                const common_agent_runtime_resident_runtime *,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            if (!*active_signal_sent) {
                *active_signal_sent = true;
                active_entered->set_value();
            }
            while (!request.execution_control.is_cancel_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            tooling = {};
            error.clear();
            return true;
        };
    auto active_manager_config = make_agent_runtime_session_manager_config(std::move(active_manager_build_config));
    if (!active_manager_config.tooling_resolver) {
        std::fprintf(stderr, "session manager active-turn smoke lost the tooling resolver while building config\n");
        return 1;
    }
    common_agent_runtime_session_manager active_manager(std::move(active_manager_config));

    auto active_control = make_common_agent_runtime_execution_control({});
    common_agent_runtime_session_manager_turn_result active_result;
    std::string active_error;
    auto active_future = std::async(std::launch::async, [&]() {
        return active_manager.run_turn({
            "request-3",
            {
                common_agent_runtime_host_mode::chat,
                "blocked",
                "session-b",
                "namespace-b",
                "",
                "turn-3",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                active_control,
            },
        }, active_result, active_error);
    });

    if (active_entered_future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        std::fprintf(stderr, "session manager active-turn smoke did not enter the resolver in time\n");
        return 1;
    }

    const auto active_turn = active_manager.describe_active_turn();
    if (!active_turn.has_value() ||
            active_turn->request_id != "request-3" ||
            active_turn->turn_id != "turn-3" ||
            (active_turn->phase != "preparing" &&
             active_turn->phase != "awaiting_inference") ||
            active_turn->disposition != "continue_immediately") {
        std::fprintf(
            stderr,
            "session manager did not surface the active turn descriptor: request='%s' turn='%s' phase='%s' disposition='%s'\n",
            active_turn.has_value() ? active_turn->request_id.c_str() : "",
            active_turn.has_value() ? active_turn->turn_id.c_str() : "",
            active_turn.has_value() ? active_turn->phase.c_str() : "",
            active_turn.has_value() ? active_turn->disposition.c_str() : "");
        return 1;
    }

    common_agent_runtime_active_turn_descriptor cancelled_active_turn;
    error.clear();
    if (!active_manager.request_cancel_active_turn("request-3", "", cancelled_active_turn, error) ||
            cancelled_active_turn.request_id != "request-3" ||
            cancelled_active_turn.turn_id != "turn-3" ||
            !cancelled_active_turn.cancellation_requested) {
        std::fprintf(stderr, "session manager failed to cancel the active turn through the session lane: %s\n", error.c_str());
        return 1;
    }

    if (active_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        std::fprintf(stderr, "session manager active-turn smoke did not finish after cancellation\n");
        return 1;
    }
    if (active_future.get()) {
        std::fprintf(stderr, "session manager active-turn smoke unexpectedly succeeded\n");
        return 1;
    }
    if (!active_result.cancelled || active_result.error != "turn cancelled by host") {
        std::fprintf(stderr, "session manager active-turn smoke did not preserve cancellation result\n");
        return 1;
    }

    const auto active_sessions = active_manager.list_sessions();
    if (active_sessions.size() != 1 ||
            active_sessions[0].active_turn_disposition != "" ||
            active_sessions[0].last_turn_id != "turn-3" ||
            active_sessions[0].last_turn_phase != "cancelled" ||
            active_sessions[0].last_turn_disposition != "cancelled") {
        std::fprintf(stderr, "session manager active-turn smoke did not retain cancelled last-turn diagnostics\n");
        return 1;
    }

    auto queued_entered = std::make_shared<std::promise<void>>();
    auto queued_entered_future = queued_entered->get_future();
    auto queued_release = std::make_shared<std::promise<void>>();
    auto queued_release_future = queued_release->get_future().share();
    auto queued_call_count = std::make_shared<int>(0);

    common_agent_runtime_session_manager_build_config queued_manager_build_config = {
        memory_store,
        plan_store,
    };
    queued_manager_build_config.resident_request = {
        "",
        "",
        "",
        "",
        std::nullopt,
        "fake.gguf",
        32,
        0,
        false,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
    };
    queued_manager_build_config.tooling_resolver =
        [queued_entered, queued_release_future, queued_call_count](
                const common_agent_runtime_resident_runtime *,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            ++(*queued_call_count);
            if (*queued_call_count == 1) {
                queued_entered->set_value();
                queued_release_future.wait();
            }
            tooling = {};
            error = "queued-resolver turn=" + request.turn_id;
            return false;
        };
    auto queued_manager_config = make_agent_runtime_session_manager_config(std::move(queued_manager_build_config));
    common_agent_runtime_session_manager queued_manager(std::move(queued_manager_config));

    common_agent_runtime_session_manager_turn_result queued_first_result;
    std::string queued_first_error;
    auto queued_first_future = std::async(std::launch::async, [&]() {
        return queued_manager.run_turn({
            "request-4",
            {
                common_agent_runtime_host_mode::chat,
                "queued-first",
                "session-c",
                "namespace-c",
                "",
                "turn-4",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, queued_first_result, queued_first_error);
    });

    if (queued_entered_future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        std::fprintf(stderr, "queued session smoke did not enter the first resolver in time\n");
        return 1;
    }

    common_agent_runtime_session_manager_turn_result queued_second_result;
    std::string queued_second_error;
    auto queued_second_future = std::async(std::launch::async, [&]() {
        return queued_manager.run_turn({
            "request-5",
            {
                common_agent_runtime_host_mode::chat,
                "queued-second",
                "session-c",
                "namespace-c",
                "",
                "turn-5",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, queued_second_result, queued_second_error);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (queued_second_future.wait_for(std::chrono::milliseconds(1)) != std::future_status::timeout) {
        std::fprintf(stderr, "queued session smoke let the second turn finish before the lane was released\n");
        return 1;
    }

    const auto queued_sessions = queued_manager.list_sessions();
    if (queued_sessions.size() != 1 ||
            !queued_sessions[0].has_active_turn ||
            queued_sessions[0].queued_turn_count < 1 ||
            queued_sessions[0].active_request_id != "request-4" ||
            queued_sessions[0].active_turn_id != "turn-4") {
        std::fprintf(stderr, "queued session smoke did not expose active+queued lane state\n");
        return 1;
    }

    queued_release->set_value();

    if (queued_first_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready ||
            queued_second_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        std::fprintf(stderr, "queued session smoke did not finish after lane release\n");
        return 1;
    }
    if (queued_first_future.get() || queued_second_future.get()) {
        std::fprintf(stderr, "queued session smoke unexpectedly succeeded\n");
        return 1;
    }
    if (queued_first_error.find("turn=turn-4") == std::string::npos ||
            queued_second_error.find("turn=turn-5") == std::string::npos) {
        std::fprintf(stderr, "queued session smoke did not preserve per-message resolver diagnostics\n");
        return 1;
    }

    std::printf("session_manager_tooling_calls=%d\n", *call_count);
    std::printf("session_manager_queued_calls=%d\n", *queued_call_count);
    return 0;
}
