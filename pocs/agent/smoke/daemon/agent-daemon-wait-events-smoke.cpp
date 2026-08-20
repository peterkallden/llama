#include "tools/agent/daemon/agent-daemon-dispatcher.h"

#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cstdio>
#include <memory>
#include <optional>
#include <string>

namespace {

common_agent_daemon_command make_turn_command(
        std::string request_id,
        std::string prompt,
        std::string turn_id) {
    common_agent_daemon_command command;
    command.request_id = std::move(request_id);
    command.type = common_agent_daemon_command_type::run_turn;
    command.turn = common_agent_daemon_turn_payload{};
    command.turn->request.request_id = command.request_id;
    command.turn->request.turn.mode = common_agent_runtime_host_mode::chat;
    command.turn->request.turn.prompt = std::move(prompt);
    command.turn->request.turn.session_id = "wait-smoke-session";
    command.turn->request.turn.namespace_id = "wait-smoke";
    command.turn->request.turn.project_id = "wait-smoke-project";
    command.turn->request.turn.turn_id = std::move(turn_id);
    command.turn->request.turn.memory_scope = common_memory_scope::session;
    command.turn->request.turn.plan_scope = common_plan_scope::turn;
    command.turn->request.turn.n_predict = 8;
    return command;
}

bool has_event_type(
        const common_agent_daemon_command_result & result,
        const char * event_type) {
    for (const auto & event : result.events) {
        if (event.type == event_type) {
            return true;
        }
    }
    return false;
}

bool has_typed_event(
        const common_agent_daemon_command_result & result,
        common_agent_daemon_event_type event_type) {
    for (const auto & event : result.events) {
        if (event.event_type == event_type) {
            return true;
        }
    }
    return false;
}

common_agent_daemon_runtime make_waiting_runtime(
        common_agent_runtime_pending_operation_kind kind,
        const std::string & pending_detail,
        const std::string & resolver_error) {
    auto memory_store = std::make_unique<common_memory_in_memory_store>();
    auto plan_store = std::make_unique<common_plan_in_memory_store>();

    common_agent_runtime_session_manager_build_config build_config = {
        *memory_store,
        *plan_store,
    };
    build_config.resident_request = {
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
    build_config.tooling_resolver =
        [resolver_error](
                const common_agent_runtime_resident_runtime *,
                const common_agent_runtime_session_host_turn_request &,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            tooling = {};
            error = resolver_error;
            return false;
        };
    auto poll_count = std::make_shared<int>(0);
    build_config.pending_operation_resolver =
        [kind, pending_detail, poll_count](
                const common_agent_runtime_session_host_turn_request & request,
                std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
                std::string & error) {
            pending_operation = common_agent_runtime_session_manager_pending_operation{};
            pending_operation->pending_operation.operation_id =
                std::string(kind == common_agent_runtime_pending_operation_kind::tool ? "tool:" : "inference:") +
                request.turn_id;
            pending_operation->pending_operation.kind = kind;
            pending_operation->pending_operation.detail = pending_detail;
            if (kind == common_agent_runtime_pending_operation_kind::inference) {
                pending_operation->waiting_phase = common_agent_runtime_turn_phase::awaiting_inference;
                pending_operation->waiting_disposition = common_agent_runtime_turn_disposition::wait_for_inference;
            }
            pending_operation->poll =
                [poll_count](bool & ready, std::string & error) mutable {
                    ++(*poll_count);
                    ready = *poll_count > 1;
                    error.clear();
                    return true;
                };
            error.clear();
            return true;
        };

    common_agent_daemon_runtime runtime;
    runtime.memory_store = std::move(memory_store);
    runtime.plan_store = std::move(plan_store);
    runtime.default_mode = common_agent_runtime_host_mode::chat;
    runtime.host = std::make_unique<common_agent_runtime_session_manager>(
        make_agent_runtime_session_manager_config(std::move(build_config)));
    return runtime;
}

} // namespace

int main() {
    common_agent_daemon_dispatcher tool_wait_dispatcher(
        make_waiting_runtime(
            common_agent_runtime_pending_operation_kind::tool,
            "wait-events smoke pending tool",
            "wait-events tool resolver"),
        8);

    common_agent_daemon_command_result tool_wait_result;
    std::string tool_wait_error;
    const bool tool_wait_ok = tool_wait_dispatcher.execute(
        make_turn_command(
            "turn-wait-tool",
            "wait for tool",
            "wait-tool"),
        tool_wait_result,
        tool_wait_error);

    if (tool_wait_ok ||
            !has_typed_event(tool_wait_result, common_agent_daemon_event_type::command_queued) ||
            !has_typed_event(tool_wait_result, common_agent_daemon_event_type::command_started) ||
            !has_event_type(tool_wait_result, "turn.waiting_for_tool") ||
            !has_typed_event(tool_wait_result, common_agent_daemon_event_type::turn_waiting_for_tool) ||
            tool_wait_result.turn_result.error.find("wait-events tool resolver") == std::string::npos) {
        std::fprintf(stderr, "tool wait scenario did not project the waiting event correctly\n");
        return 1;
    }

    common_agent_daemon_dispatcher inference_wait_dispatcher(
        make_waiting_runtime(
            common_agent_runtime_pending_operation_kind::inference,
            "wait-events smoke pending inference",
            "wait-events inference resolver"),
        8);

    common_agent_daemon_command_result inference_wait_result;
    std::string inference_wait_error;
    const bool inference_wait_ok = inference_wait_dispatcher.execute(
        make_turn_command(
            "turn-wait-inference",
            "wait for inference",
            "wait-inference"),
        inference_wait_result,
        inference_wait_error);

    if (inference_wait_ok ||
            !has_typed_event(inference_wait_result, common_agent_daemon_event_type::command_queued) ||
            !has_typed_event(inference_wait_result, common_agent_daemon_event_type::command_started) ||
            !has_event_type(inference_wait_result, "turn.waiting_for_inference") ||
            !has_typed_event(inference_wait_result, common_agent_daemon_event_type::turn_waiting_for_inference) ||
            inference_wait_result.turn_result.error.find("wait-events inference resolver") == std::string::npos) {
        std::fprintf(stderr, "inference wait scenario did not project the waiting event correctly\n");
        return 1;
    }

    std::printf("daemon_wait_tool_event=%s\n",
        has_event_type(tool_wait_result, "turn.waiting_for_tool") ? "yes" : "no");
    std::printf("daemon_wait_inference_event=%s\n",
        has_event_type(inference_wait_result, "turn.waiting_for_inference") ? "yes" : "no");
    return 0;
}
