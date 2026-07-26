#include "tools/agent/runtime/agent-runtime-session-manager.h"
#include "tools/agent/runtime/agent-inference-capacity-gate.h"
#include "tools/agent/runtime/agent-runtime-turn-driver.h"

#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cstdio>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

class lease_failure_task final : public common_agent_runtime_inference_task {
public:
    lease_failure_task(
            std::shared_ptr<common_agent_inference_capacity_gate> gate,
            std::string lease_id,
            std::shared_ptr<int> cancel_calls)
        : gate(std::move(gate))
        , lease_id(std::move(lease_id))
        , cancel_calls(std::move(cancel_calls)) {}

    bool poll(
            bool & ready,
            common_agent_runtime_session_host_turn_result &, std::string & error) override {
        ready = false;
        error.clear();
        return true;
    }

    bool cancel(std::string & error) override {
        ++*cancel_calls;
        gate->release(lease_id);
        error.clear();
        return true;
    }

private:
    std::shared_ptr<common_agent_inference_capacity_gate> gate;
    std::string lease_id;
    std::shared_ptr<int> cancel_calls;
};

class lease_failure_executor final : public common_agent_runtime_inference_executor {
public:
    lease_failure_executor(
            std::shared_ptr<common_agent_inference_capacity_gate> gate,
            std::shared_ptr<int> cancel_calls)
        : gate(std::move(gate)), cancel_calls(std::move(cancel_calls)) {}

    std::shared_ptr<common_agent_runtime_inference_task> submit(
            common_agent_runtime_session_host *,
            common_agent_runtime_session_host_turn_request,
            const std::shared_ptr<common_agent_inference_capacity_gate> &,
            std::string lease_id,
            std::string & error) override {
        error.clear();
        return std::make_shared<lease_failure_task>(gate, std::move(lease_id), cancel_calls);
    }

private:
    std::shared_ptr<common_agent_inference_capacity_gate> gate;
    std::shared_ptr<int> cancel_calls;
};

bool contains_event_type(
        const std::vector<common_agent_daemon_event> & events,
        const char * type) {
    for (const auto & event : events) {
        if (event.type == type) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    common_agent_inference_capacity_gate capacity_gate(1);
    if (!capacity_gate.try_acquire() || capacity_gate.try_acquire() ||
            capacity_gate.active() != 1 || capacity_gate.capacity() != 1) {
        std::fprintf(stderr, "inference capacity gate did not enforce a single active lease\n");
        return 1;
    }
    capacity_gate.release();
    if (!capacity_gate.try_acquire()) {
        std::fprintf(stderr, "inference capacity gate did not release its lease\n");
        return 1;
    }
    capacity_gate.release();
    if (!capacity_gate.try_acquire()) {
        std::fprintf(stderr, "inference capacity gate setup failed\n");
        return 1;
    }
    auto capacity_wait_started = std::make_shared<std::promise<void>>();
    auto capacity_wait_started_future = capacity_wait_started->get_future();
    auto capacity_waiter = std::async(std::launch::async, [capacity_wait_started, &capacity_gate]() {
        capacity_wait_started->set_value();
        while (!capacity_gate.try_acquire()) {
            std::this_thread::yield();
        }
        return true;
    });
    if (capacity_wait_started_future.wait_for(std::chrono::seconds(1)) != std::future_status::ready ||
            capacity_gate.active() != 1) {
        std::fprintf(stderr, "inference capacity gate waiter did not remain bounded\n");
        return 1;
    }
    capacity_gate.release();
    if (capacity_waiter.wait_for(std::chrono::seconds(1)) != std::future_status::ready ||
            !capacity_waiter.get() || capacity_gate.active() != 1) {
        std::fprintf(stderr, "inference capacity gate waiter did not acquire after release\n");
        return 1;
    }
    capacity_gate.release();

    std::string admission_error;
    if (!capacity_gate.try_acquire() ||
            !capacity_gate.enqueue({
                "inference-old",
                common_agent_inference_priority::background,
                {},
            }, admission_error) ||
            !capacity_gate.enqueue({
                "inference-new",
                common_agent_inference_priority::normal,
                {},
            }, admission_error) ||
            capacity_gate.try_acquire("inference-new")) {
        std::fprintf(stderr, "inference admission queue did not preserve capacity ordering\n");
        return 1;
    }
    capacity_gate.release();
    if (!capacity_gate.try_acquire("inference-new")) {
        std::fprintf(stderr, "inference admission queue did not honor priority\n");
        return 1;
    }
    capacity_gate.release("inference-new");
    if (!capacity_gate.try_acquire("inference-old")) {
        std::fprintf(stderr, "inference admission queue did not retain the lower priority waiter\n");
        return 1;
    }
    capacity_gate.release("inference-old");

    if (!capacity_gate.enqueue({
                "inference-first",
                common_agent_inference_priority::normal,
                {},
            }, admission_error) ||
            !capacity_gate.enqueue({
                "inference-second",
                common_agent_inference_priority::normal,
                {},
            }, admission_error) ||
            !capacity_gate.try_acquire("inference-first")) {
        std::fprintf(stderr, "inference admission queue did not preserve FIFO within priority\n");
        return 1;
    }
    capacity_gate.release("inference-first");
    if (!capacity_gate.try_acquire("inference-second")) {
        std::fprintf(stderr, "inference admission queue did not grant the next waiter\n");
        return 1;
    }
    capacity_gate.release("inference-second");
    if (!capacity_gate.enqueue({"inference-cancelled"}, admission_error) ||
            !capacity_gate.cancel("inference-cancelled", admission_error) ||
            capacity_gate.waiting() != 0) {
        std::fprintf(stderr, "inference admission queue did not cancel a waiter\n");
        return 1;
    }
    std::printf("inference_admission=priority_fifo_cancel\n");

    auto lease_failure_gate = std::make_shared<common_agent_inference_capacity_gate>(1);
    auto lease_failure_cancel_calls = std::make_shared<int>(0);
    common_runtime_operation_manager lease_failure_operations;
    common_runtime_operation duplicate_inference;
    duplicate_inference.operation_id = "inference:lease-failure-request";
    if (!lease_failure_operations.begin(
            duplicate_inference,
            [](bool & ready, std::string &) {
                ready = false;
                return true;
            },
            {},
            admission_error)) {
        std::fprintf(stderr, "lease failure duplicate operation setup failed\n");
        return 1;
    }
    std::optional<common_agent_runtime_session_manager_pending_operation> lease_failure_pending;
    auto lease_failure_active = std::make_optional(
        make_common_agent_runtime_turn_execution(
            "lease-failure-request",
            "lease-failure-turn",
            common_agent_runtime_host_mode::chat,
            false,
            std::make_shared<common_agent_runtime_cancellation_state>()));
    lease_failure_active->phase = common_agent_runtime_turn_phase::awaiting_inference;
    common_agent_runtime_session_host_turn_request lease_failure_request;
    lease_failure_request.turn_id = "lease-failure-turn";
    lease_failure_request.execution_control = make_common_agent_runtime_execution_control({});
    common_agent_runtime_session_host_turn_result lease_failure_result;
    std::string lease_failure_error;
    std::mutex lease_failure_mutex;
    const auto lease_failure_disposition = advance_common_agent_runtime_turn(
        nullptr,
        lease_failure_pending,
        lease_failure_active,
        lease_failure_mutex,
        lease_failure_operations,
        {},
        lease_failure_gate,
        std::make_shared<lease_failure_executor>(lease_failure_gate, lease_failure_cancel_calls),
        "lease-failure-request",
        lease_failure_request,
        lease_failure_result,
        lease_failure_error,
        common_agent_event_emitter());
    if (lease_failure_disposition != common_agent_runtime_turn_disposition::failed ||
            *lease_failure_cancel_calls != 1 ||
            lease_failure_gate->active() != 0) {
        std::fprintf(stderr, "inference lease registration failure did not cancel its task exactly once\n");
        return 1;
    }
    std::printf("inference_registration_failure_cancel=exactly_once\n");

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
    if (!manager_config.host_config.tooling_resolver) {
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
    if (sessions[0].lane_state != "idle") {
        std::fprintf(stderr, "session manager did not report idle lane state after failed turn: %s\n",
            sessions[0].lane_state.c_str());
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

    common_agent_runtime_session_manager_build_config reset_all_build_config = {
        memory_store,
        plan_store,
    };
    reset_all_build_config.resident_request = {
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
    reset_all_build_config.tooling_resolver =
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
    common_agent_runtime_session_manager reset_all_manager(
        make_agent_runtime_session_manager_config(std::move(reset_all_build_config)));
    common_agent_runtime_session_manager_turn_result reset_all_result;
    if (reset_all_manager.run_turn({
            "request-reset-all",
            {
                common_agent_runtime_host_mode::chat,
                "hello again",
                "session-reset-all",
                "namespace-reset-all",
                "",
                "turn-reset-all",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, reset_all_result, error)) {
        std::fprintf(stderr, "reset_all manager unexpectedly succeeded on resolver failure\n");
        return 1;
    }
    reset_all_manager.reset_all();
    if (!reset_all_manager.list_sessions().empty()) {
        std::fprintf(stderr, "session manager reset_all did not route through lane cleanup\n");
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
    if (!active_manager_config.host_config.tooling_resolver) {
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
            (active_turn->disposition != "continue_immediately" &&
             active_turn->disposition != "wait_for_inference")) {
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
            active_sessions[0].lane_state != "idle" ||
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
            queued_sessions[0].lane_state != "running_with_waiters" ||
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

    auto lifecycle_entered = std::make_shared<std::promise<void>>();
    auto lifecycle_entered_future = lifecycle_entered->get_future();
    auto lifecycle_release = std::make_shared<std::promise<void>>();
    auto lifecycle_release_future = lifecycle_release->get_future().share();
    auto lifecycle_call_count = std::make_shared<int>(0);

    common_agent_runtime_session_manager_build_config lifecycle_manager_build_config = {
        memory_store,
        plan_store,
    };
    lifecycle_manager_build_config.resident_request = {
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
    lifecycle_manager_build_config.tooling_resolver =
        [lifecycle_entered, lifecycle_release_future, lifecycle_call_count](
                const common_agent_runtime_resident_runtime *,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            ++(*lifecycle_call_count);
            if (*lifecycle_call_count == 1) {
                lifecycle_entered->set_value();
                lifecycle_release_future.wait();
            }
            tooling = {};
            error = "lifecycle-resolver turn=" + request.turn_id;
            return false;
        };
    auto lifecycle_manager_config = make_agent_runtime_session_manager_config(std::move(lifecycle_manager_build_config));
    common_agent_runtime_session_manager lifecycle_manager(std::move(lifecycle_manager_config));

    common_agent_runtime_session_manager_turn_result reset_first_result;
    std::string reset_first_error;
    auto reset_first_future = std::async(std::launch::async, [&]() {
        return lifecycle_manager.run_turn({
            "request-6",
            {
                common_agent_runtime_host_mode::chat,
                "reset-first",
                "session-d",
                "namespace-d",
                "",
                "turn-6",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, reset_first_result, reset_first_error);
    });

    if (lifecycle_entered_future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        std::fprintf(stderr, "lifecycle reset smoke did not enter the first resolver in time\n");
        return 1;
    }

    common_agent_runtime_session_manager_turn_result reset_second_result;
    std::string reset_second_error;
    auto reset_second_future = std::async(std::launch::async, [&]() {
        return lifecycle_manager.run_turn({
            "request-7",
            {
                common_agent_runtime_host_mode::chat,
                "reset-second",
                "session-d",
                "namespace-d",
                "",
                "turn-7",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, reset_second_result, reset_second_error);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::string reset_error;
    auto reset_future = std::async(std::launch::async, [&]() {
        return lifecycle_manager.reset_session({"namespace-d", "session-d"}, reset_error);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const auto resetting_sessions = lifecycle_manager.list_sessions();
    if (resetting_sessions.size() != 1 ||
            resetting_sessions[0].lane_state != "resetting" ||
            !resetting_sessions[0].has_active_turn ||
            resetting_sessions[0].queued_turn_count != 0) {
        std::fprintf(stderr, "session manager did not expose resetting lane state while reset was pending\n");
        return 1;
    }

    if (reset_second_future.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
            reset_second_future.get() ||
            reset_second_error != "session lane reset before turn execution") {
        std::fprintf(stderr, "session manager reset did not fail queued turn through lane transition\n");
        return 1;
    }

    lifecycle_release->set_value();

    if (reset_first_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready ||
            reset_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        std::fprintf(stderr, "session manager reset smoke did not finish after release\n");
        return 1;
    }
    if (reset_first_future.get()) {
        std::fprintf(stderr, "session manager reset smoke unexpectedly succeeded first turn\n");
        return 1;
    }
    if (!reset_future.get() || !reset_error.empty()) {
        std::fprintf(stderr, "session manager reset did not complete after active turn drained: %s\n", reset_error.c_str());
        return 1;
    }
    const auto reset_sessions = lifecycle_manager.list_sessions();
    if (reset_sessions.size() != 1 ||
            reset_sessions[0].lane_state != "idle" ||
            !reset_sessions[0].last_turn_id.empty() ||
            reset_sessions[0].has_active_turn) {
        std::fprintf(stderr, "session manager reset did not clear lane state after reset lifecycle\n");
        return 1;
    }

    auto close_entered = std::make_shared<std::promise<void>>();
    auto close_entered_future = close_entered->get_future();
    auto close_release = std::make_shared<std::promise<void>>();
    auto close_release_future = close_release->get_future().share();
    auto close_call_count = std::make_shared<int>(0);

    common_agent_runtime_session_manager_build_config close_manager_build_config = {
        memory_store,
        plan_store,
    };
    close_manager_build_config.resident_request = {
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
    close_manager_build_config.tooling_resolver =
        [close_entered, close_release_future, close_call_count](
                const common_agent_runtime_resident_runtime *,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            ++(*close_call_count);
            if (*close_call_count == 1) {
                close_entered->set_value();
                close_release_future.wait();
            }
            tooling = {};
            error = "close-resolver turn=" + request.turn_id;
            return false;
        };
    auto close_manager_config = make_agent_runtime_session_manager_config(std::move(close_manager_build_config));
    common_agent_runtime_session_manager close_manager(std::move(close_manager_config));

    common_agent_runtime_session_manager_turn_result close_first_result;
    std::string close_first_error;
    auto close_first_future = std::async(std::launch::async, [&]() {
        return close_manager.run_turn({
            "request-8",
            {
                common_agent_runtime_host_mode::chat,
                "close-first",
                "session-e",
                "namespace-e",
                "",
                "turn-8",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, close_first_result, close_first_error);
    });

    if (close_entered_future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        std::fprintf(stderr, "lifecycle close smoke did not enter the first resolver in time\n");
        return 1;
    }

    common_agent_runtime_session_manager_turn_result close_second_result;
    std::string close_second_error;
    auto close_second_future = std::async(std::launch::async, [&]() {
        return close_manager.run_turn({
            "request-9",
            {
                common_agent_runtime_host_mode::chat,
                "close-second",
                "session-e",
                "namespace-e",
                "",
                "turn-9",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, close_second_result, close_second_error);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::string close_error;
    auto close_future = std::async(std::launch::async, [&]() {
        return close_manager.close_session({"namespace-e", "session-e"}, close_error);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const auto closing_sessions = close_manager.list_sessions();
    if (closing_sessions.size() != 1 ||
            closing_sessions[0].lane_state != "closing" ||
            !closing_sessions[0].has_active_turn ||
            closing_sessions[0].queued_turn_count != 0) {
        std::fprintf(stderr, "session manager did not expose closing lane state while close was pending\n");
        return 1;
    }

    if (close_second_future.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
            close_second_future.get() ||
            close_second_error != "session lane closed before turn execution") {
        std::fprintf(stderr, "session manager close did not fail queued turn through lane transition\n");
        return 1;
    }

    close_release->set_value();

    if (close_first_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready ||
            close_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        std::fprintf(stderr, "session manager close smoke did not finish after release\n");
        return 1;
    }
    if (close_first_future.get()) {
        std::fprintf(stderr, "session manager close smoke unexpectedly succeeded first turn\n");
        return 1;
    }
    if (!close_future.get() || !close_error.empty()) {
        std::fprintf(stderr, "session manager close did not complete after active turn drained: %s\n", close_error.c_str());
        return 1;
    }
    if (!close_manager.list_sessions().empty()) {
        std::fprintf(stderr, "session manager close did not remove the lane after lifecycle transition\n");
        return 1;
    }

    auto waiting_poll_count = std::make_shared<int>(0);
    auto waiting_cancel_called = std::make_shared<bool>(false);
    auto waiting_tool_started = std::make_shared<std::promise<void>>();
    auto waiting_tool_started_future = waiting_tool_started->get_future();

    common_agent_runtime_session_manager_build_config waiting_manager_build_config = {
        memory_store,
        plan_store,
    };
    waiting_manager_build_config.resident_request = {
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
    waiting_manager_build_config.tooling_resolver =
        [](const common_agent_runtime_resident_runtime *,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            tooling = {};
            error = "tool-wait-resolver turn=" + request.turn_id;
            return false;
        };
    waiting_manager_build_config.pending_operation_resolver =
        [waiting_poll_count, waiting_tool_started, waiting_cancel_called](
                const common_agent_runtime_session_host_turn_request & request,
                std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
                std::string & error) {
            pending_operation = common_agent_runtime_session_manager_pending_operation{};
            pending_operation->pending_operation.operation_id = "tool:" + request.turn_id;
            pending_operation->pending_operation.kind = common_agent_runtime_pending_operation_kind::tool;
            pending_operation->pending_operation.detail = "session manager smoke pending tool";
            pending_operation->poll =
                [waiting_poll_count, waiting_tool_started](bool & ready, std::string & error) mutable {
                    ++(*waiting_poll_count);
                    if (*waiting_poll_count == 1) {
                        waiting_tool_started->set_value();
                    }
                    ready = false;
                    error.clear();
                    return true;
                };
            pending_operation->cancel = [waiting_cancel_called](std::string & error) {
                *waiting_cancel_called = true;
                error.clear();
                return true;
            };
            error.clear();
            return true;
        };
    auto waiting_manager_config = make_agent_runtime_session_manager_config(std::move(waiting_manager_build_config));
    common_agent_runtime_session_manager waiting_manager(std::move(waiting_manager_config));
    auto waiting_events = std::make_shared<std::vector<common_agent_daemon_event>>();
    auto waiting_events_mutex = std::make_shared<std::mutex>();
    waiting_manager.set_event_sink(
        [waiting_events, waiting_events_mutex](common_agent_daemon_event event) {
            std::lock_guard<std::mutex> lock(*waiting_events_mutex);
            waiting_events->push_back(std::move(event));
        });

    auto waiting_control = make_common_agent_runtime_execution_control({});
    common_agent_runtime_session_manager_turn_result waiting_result;
    std::string waiting_error;
    auto waiting_future = std::async(std::launch::async, [&]() {
        return waiting_manager.run_turn({
            "request-10",
            {
                common_agent_runtime_host_mode::chat,
                "tool-wait",
                "session-f",
                "namespace-f",
                "",
                "turn-10",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                waiting_control,
            },
        }, waiting_result, waiting_error);
    });

    if (waiting_tool_started_future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        std::fprintf(stderr, "session manager tool-wait smoke did not enter the pending tool state in time\n");
        return 1;
    }

    const auto waiting_active_turn = waiting_manager.describe_active_turn();
    if (!waiting_active_turn.has_value() ||
            waiting_active_turn->request_id != "request-10" ||
            waiting_active_turn->turn_id != "turn-10" ||
            waiting_active_turn->phase != "awaiting_tool" ||
            waiting_active_turn->disposition != "wait_for_tool" ||
            waiting_active_turn->pending_operation_kind != "tool" ||
            waiting_active_turn->pending_operation_detail != "session manager smoke pending tool") {
        std::fprintf(
            stderr,
            "session manager did not expose the waiting tool state: request='%s' turn='%s' phase='%s' disposition='%s' pending_kind='%s' pending_detail='%s'\n",
            waiting_active_turn.has_value() ? waiting_active_turn->request_id.c_str() : "",
            waiting_active_turn.has_value() ? waiting_active_turn->turn_id.c_str() : "",
            waiting_active_turn.has_value() ? waiting_active_turn->phase.c_str() : "",
            waiting_active_turn.has_value() ? waiting_active_turn->disposition.c_str() : "",
            waiting_active_turn.has_value() ? waiting_active_turn->pending_operation_kind.c_str() : "",
            waiting_active_turn.has_value() ? waiting_active_turn->pending_operation_detail.c_str() : "");
        return 1;
    }

    const auto waiting_sessions = waiting_manager.list_sessions();
    if (waiting_sessions.size() != 1 ||
            waiting_sessions[0].lane_state != "running" ||
            !waiting_sessions[0].has_active_turn ||
            waiting_sessions[0].active_turn_phase != "awaiting_tool" ||
            waiting_sessions[0].active_turn_disposition != "wait_for_tool" ||
            waiting_sessions[0].pending_operation_kind != "tool" ||
            waiting_sessions[0].pending_operation_detail != "session manager smoke pending tool") {
        std::fprintf(stderr, "session manager did not retain awaiting_tool lane diagnostics\n");
        return 1;
    }

    common_agent_runtime_active_turn_descriptor cancelled_waiting_turn;
    error.clear();
    if (!waiting_manager.request_cancel_active_turn("request-10", "", cancelled_waiting_turn, error) ||
            cancelled_waiting_turn.request_id != "request-10" ||
            cancelled_waiting_turn.turn_id != "turn-10" ||
            !cancelled_waiting_turn.cancellation_requested ||
            cancelled_waiting_turn.pending_operation_kind != "tool" ||
            cancelled_waiting_turn.pending_operation_detail != "session manager smoke pending tool") {
        std::fprintf(stderr, "session manager failed to cancel the waiting tool turn: %s\n", error.c_str());
        return 1;
    }

    if (waiting_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        std::fprintf(stderr, "session manager tool-wait smoke did not finish after cancellation\n");
        return 1;
    }
    if (waiting_future.get()) {
        std::fprintf(stderr, "session manager tool-wait smoke unexpectedly succeeded\n");
        return 1;
    }
    if (!waiting_result.cancelled || waiting_result.error != "turn cancelled by host") {
        std::fprintf(stderr, "session manager tool-wait smoke did not preserve cancellation result\n");
        return 1;
    }
    if (!*waiting_cancel_called) {
        std::fprintf(stderr, "session manager tool-wait smoke did not invoke the manager-owned cancel callback\n");
        return 1;
    }
    {
        std::lock_guard<std::mutex> lock(*waiting_events_mutex);
        if (!contains_event_type(*waiting_events, "turn.waiting_for_tool")) {
            std::fprintf(stderr, "session manager tool-wait smoke did not emit turn.waiting_for_tool\n");
            return 1;
        }
    }

    common_agent_runtime_session_manager_build_config failed_pending_build_config = {
        memory_store,
        plan_store,
    };
    failed_pending_build_config.resident_request = {
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
    failed_pending_build_config.pending_operation_resolver =
        [](const common_agent_runtime_session_host_turn_request & request,
                std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
                std::string & error) {
            pending_operation = common_agent_runtime_session_manager_pending_operation{};
            pending_operation->pending_operation.operation_id = "failed-tool:" + request.turn_id;
            pending_operation->pending_operation.kind = common_agent_runtime_pending_operation_kind::tool;
            pending_operation->pending_operation.detail = "session manager smoke failed pending tool";
            if (request.turn_id == "turn-timeout-pending") {
                pending_operation->pending_operation.deadline =
                    std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
            }
            pending_operation->poll = [](bool & ready, std::string & error) {
                ready = false;
                error = "synthetic pending operation failure";
                return false;
            };
            error.clear();
            return true;
        };
    auto failed_pending_manager_config =
        make_agent_runtime_session_manager_config(std::move(failed_pending_build_config));
    common_agent_runtime_session_manager failed_pending_manager(
        std::move(failed_pending_manager_config));
    auto failed_pending_events = std::make_shared<std::vector<common_agent_daemon_event>>();
    auto failed_pending_events_mutex = std::make_shared<std::mutex>();
    failed_pending_manager.set_event_sink(
        [failed_pending_events, failed_pending_events_mutex](common_agent_daemon_event event) {
            std::lock_guard<std::mutex> lock(*failed_pending_events_mutex);
            failed_pending_events->push_back(std::move(event));
        });
    common_agent_runtime_session_manager_turn_result failed_pending_result;
    std::string failed_pending_error;
    if (failed_pending_manager.run_turn({
            "request-failed-pending",
            {
                common_agent_runtime_host_mode::chat,
                "failed-pending",
                "session-failed-pending",
                "namespace-failed-pending",
                "",
                "turn-failed-pending",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, failed_pending_result, failed_pending_error)) {
        std::fprintf(stderr, "failed pending operation smoke unexpectedly succeeded\n");
        return 1;
    }
    if (failed_pending_error != "synthetic pending operation failure" ||
            failed_pending_result.error != failed_pending_error ||
            !contains_event_type(*failed_pending_events, "tool.failed") ||
            !contains_event_type(*failed_pending_events, "turn.failed")) {
        std::fprintf(stderr, "failed pending operation smoke did not preserve failure diagnostics\n");
        return 1;
    }

    common_agent_runtime_session_manager_turn_result timeout_pending_result;
    std::string timeout_pending_error;
    if (failed_pending_manager.run_turn({
            "request-timeout-pending",
            {
                common_agent_runtime_host_mode::chat,
                "timeout-pending",
                "session-failed-pending",
                "namespace-failed-pending",
                "",
                "turn-timeout-pending",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, timeout_pending_result, timeout_pending_error)) {
        std::fprintf(stderr, "timed-out pending operation smoke unexpectedly succeeded\n");
        return 1;
    }
    if (timeout_pending_error != "operation deadline exceeded" ||
            timeout_pending_result.error != timeout_pending_error ||
            !contains_event_type(*failed_pending_events, "tool.timed_out") ||
            !contains_event_type(*failed_pending_events, "turn.failed")) {
        std::fprintf(stderr, "timed-out pending operation smoke did not preserve timeout diagnostics\n");
        return 1;
    }

    auto inference_wait_release = std::make_shared<std::promise<void>>();
    auto inference_wait_release_future = inference_wait_release->get_future().share();
    auto inference_wait_poll_count = std::make_shared<int>(0);
    auto inference_wait_started = std::make_shared<std::promise<void>>();
    auto inference_wait_started_future = inference_wait_started->get_future();
    auto inference_wait_cancel_called = std::make_shared<bool>(false);

    common_agent_runtime_session_manager_build_config inference_wait_manager_build_config = {
        memory_store,
        plan_store,
    };
    inference_wait_manager_build_config.resident_request = {
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
    inference_wait_manager_build_config.tooling_resolver =
        [](const common_agent_runtime_resident_runtime *,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            tooling = {};
            error = "inference-wait-resolver turn=" + request.turn_id;
            return false;
        };
    inference_wait_manager_build_config.pending_operation_resolver =
        [inference_wait_release_future, inference_wait_poll_count, inference_wait_started, inference_wait_cancel_called](
                const common_agent_runtime_session_host_turn_request & request,
                std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
                std::string & error) {
            pending_operation = common_agent_runtime_session_manager_pending_operation{};
            pending_operation->pending_operation.operation_id = "inference-wait:" + request.turn_id;
            pending_operation->pending_operation.kind = common_agent_runtime_pending_operation_kind::inference;
            pending_operation->pending_operation.detail = "session manager smoke pending inference";
            pending_operation->waiting_phase = common_agent_runtime_turn_phase::awaiting_inference;
            pending_operation->waiting_disposition = common_agent_runtime_turn_disposition::wait_for_inference;
            pending_operation->poll =
                [inference_wait_release_future, inference_wait_poll_count, inference_wait_started](
                        bool & ready,
                        std::string & error) mutable {
                    ++(*inference_wait_poll_count);
                    if (*inference_wait_poll_count == 1) {
                        inference_wait_started->set_value();
                    }
                    ready =
                        inference_wait_release_future.wait_for(std::chrono::milliseconds(0)) ==
                        std::future_status::ready;
                    error.clear();
                    return true;
                };
            pending_operation->cancel = [inference_wait_cancel_called](std::string & error) {
                *inference_wait_cancel_called = true;
                error.clear();
                return true;
            };
            error.clear();
            return true;
        };
    common_agent_runtime_session_manager inference_wait_manager(
        make_agent_runtime_session_manager_config(std::move(inference_wait_manager_build_config)));
    auto inference_wait_events = std::make_shared<std::vector<common_agent_daemon_event>>();
    auto inference_wait_events_mutex = std::make_shared<std::mutex>();
    inference_wait_manager.set_event_sink(
        [inference_wait_events, inference_wait_events_mutex](common_agent_daemon_event event) {
            std::lock_guard<std::mutex> lock(*inference_wait_events_mutex);
            inference_wait_events->push_back(std::move(event));
        });

    common_agent_runtime_session_manager_turn_result inference_wait_result;
    std::string inference_wait_error;
    auto inference_wait_future = std::async(std::launch::async, [&]() {
        return inference_wait_manager.run_turn({
            "request-11",
            {
                common_agent_runtime_host_mode::chat,
                "inference-wait",
                "session-g",
                "namespace-g",
                "",
                "turn-11",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
                std::nullopt,
                make_common_agent_runtime_execution_control({}),
            },
        }, inference_wait_result, inference_wait_error);
    });

    if (inference_wait_started_future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        std::fprintf(stderr, "session manager inference-wait smoke did not enter the pending inference state in time\n");
        return 1;
    }

    const auto inference_wait_active_turn = inference_wait_manager.describe_active_turn();
    if (!inference_wait_active_turn.has_value() ||
            inference_wait_active_turn->request_id != "request-11" ||
            inference_wait_active_turn->turn_id != "turn-11" ||
            inference_wait_active_turn->phase != "awaiting_inference" ||
            inference_wait_active_turn->disposition != "wait_for_inference" ||
            inference_wait_active_turn->pending_operation_kind != "inference" ||
            inference_wait_active_turn->pending_operation_detail != "session manager smoke pending inference") {
        std::fprintf(
            stderr,
            "session manager did not expose the waiting inference state: request='%s' turn='%s' phase='%s' disposition='%s' pending_kind='%s' pending_detail='%s'\n",
            inference_wait_active_turn.has_value() ? inference_wait_active_turn->request_id.c_str() : "",
            inference_wait_active_turn.has_value() ? inference_wait_active_turn->turn_id.c_str() : "",
            inference_wait_active_turn.has_value() ? inference_wait_active_turn->phase.c_str() : "",
            inference_wait_active_turn.has_value() ? inference_wait_active_turn->disposition.c_str() : "",
            inference_wait_active_turn.has_value() ? inference_wait_active_turn->pending_operation_kind.c_str() : "",
            inference_wait_active_turn.has_value() ? inference_wait_active_turn->pending_operation_detail.c_str() : "");
        return 1;
    }

    const auto inference_wait_sessions = inference_wait_manager.list_sessions();
    if (inference_wait_sessions.size() != 1 ||
            inference_wait_sessions[0].lane_state != "running" ||
            !inference_wait_sessions[0].has_active_turn ||
            inference_wait_sessions[0].active_turn_phase != "awaiting_inference" ||
            inference_wait_sessions[0].active_turn_disposition != "wait_for_inference" ||
            inference_wait_sessions[0].pending_operation_kind != "inference" ||
            inference_wait_sessions[0].pending_operation_detail != "session manager smoke pending inference") {
        std::fprintf(stderr, "session manager did not retain awaiting_inference lane diagnostics for pending inference\n");
        return 1;
    }

    common_agent_runtime_active_turn_descriptor cancelled_inference_turn;
    error.clear();
    if (!inference_wait_manager.request_cancel_active_turn(
            "request-11", "", cancelled_inference_turn, error) ||
            !cancelled_inference_turn.cancellation_requested) {
        std::fprintf(stderr, "session manager failed to cancel inference before host execution: %s\n", error.c_str());
        return 1;
    }
    if (inference_wait_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        std::fprintf(stderr, "session manager pre-host cancellation smoke did not finish\n");
        return 1;
    }
    if (inference_wait_future.get() || !inference_wait_result.cancelled ||
            inference_wait_result.error != "turn cancelled by host" ||
            !*inference_wait_cancel_called) {
        std::fprintf(stderr, "session manager pre-host cancellation did not cancel the pending operation\n");
        return 1;
    }
    {
        std::lock_guard<std::mutex> lock(*inference_wait_events_mutex);
        if (!contains_event_type(*inference_wait_events, "turn.waiting_for_inference") ||
                !contains_event_type(*inference_wait_events, "turn.cancelled")) {
            std::fprintf(stderr, "session manager pre-host cancellation did not preserve inference and cancellation events\n");
            return 1;
        }
    }

    std::printf("session_manager_tooling_calls=%d\n", *call_count);
    std::printf("session_manager_queued_calls=%d\n", *queued_call_count);
    std::printf("session_manager_lifecycle_calls=%d\n", *lifecycle_call_count + *close_call_count);
    std::printf("session_manager_waiting_tool_polls=%d\n", *waiting_poll_count);
    std::printf("session_manager_waiting_inference_polls=%d\n", *inference_wait_poll_count);
    std::printf("inference_capacity_gate=single_active\n");
    return 0;
}
