#include "tools/agent/mcp/agent-mcp-http-server.h"
#include "tools/agent/daemon/agent-daemon-event-collector.h"
#include "agent/agent-runtime.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cpp-httplib/httplib.h>
#include <cstdio>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

using json = nlohmann::ordered_json;

class vertical_tool_runtime final : public common_agent_tool_runtime {
public:
    mutable int resource_reads = 0;
    mutable int memory_gets = 0;

    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }

    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        if (call.name != "resource_read" && call.name != "memory_get") {
            error = "vertical smoke only exposes resource_read and memory_get";
            return false;
        }
        const auto arguments = json::parse(call.arguments_json, nullptr, false);
        if (arguments.is_discarded() || !arguments.is_object()) {
            error = "vertical smoke received invalid tool arguments";
            return false;
        }
        const char * field = call.name == "resource_read" ? "uri" : "id";
        if (!arguments.contains(field) || !arguments[field].is_string() || arguments[field].get<std::string>().empty()) {
            error = std::string("vertical smoke requires ") + field;
            return false;
        }
        error.clear();
        return true;
    }

    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        const auto arguments = json::parse(call.arguments_json, nullptr, false);
        if (call.name == "resource_read") {
            ++resource_reads;
            const std::string uri = arguments.value("uri", "");
            return common_tool_execution_result::success(
                "User supplied resource confirms the bounded research criterion.",
                "user resource evidence",
                {common_runtime_resource_ref{uri, "user-reference", "vertical smoke input", "text/plain", 64,
                    common_runtime_resource_scope::turn, {}}});
        }
        ++memory_gets;
        return common_tool_execution_result::success(
            "Memory record confirms the bounded research criterion.",
            "memory evidence");
    }
};

class vertical_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "vertical-delegated-plan";
        proposal.plan.goal = "Produce a researched delegated answer";
        proposal.plan.success_criteria = "Research evidence is assessed before the draft is verified.";
        proposal.plan.status = common_plan_status::active;

        common_plan_step reason{"inspect", "Inspect", "Record the researched observations"};
        reason.status = common_plan_step_status::active;
        reason.mode = common_plan_step_mode::reasoning;
        common_plan_step answer{"answer", "Answer", "Return the verified delegated answer"};
        answer.mode = common_plan_step_mode::final_response;
        answer.depends_on = {"inspect"};
        proposal.plan.steps = {reason, answer};
        proposal.plan.active_step_id = reason.id;
        return proposal;
    }
};

class vertical_executor final : public common_action_executor {
public:
    std::string generate_reasoning(const common_agent_request &, const common_plan_state &, const common_plan_step &, std::string & error) override {
        error.clear();
        return R"({"summary":"research evidence assessed"})";
    }

    std::string generate_draft(const common_agent_request &, const common_plan_state &, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "Verified answer from memory evidence and user resource evidence.";
    }
};

class vertical_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        result.confidence = 0.95f;
        return result;
    }
};

static void append_runtime_events(
        common_agent_daemon_event_collector & collector,
        const common_agent_result & result,
        const std::string & request_id,
        const std::string & turn_id) {
    const common_agent_event_context context{
        "namespace-a", "project-a", "vertical-session", request_id, turn_id, ""};
    for (const auto & event : result.events) {
        collector.append(make_common_agent_daemon_event(
            common_agent_daemon_event_type::agent_runtime_event,
            request_id,
            turn_id,
            std::string(common_agent_event_type_name(event.type)) + ": " + event.detail,
            0,
            context));
    }
    collector.append(make_common_agent_daemon_event(
        common_agent_daemon_event_type::turn_completed,
        request_id,
        turn_id,
        "delegated terminal response ready",
        0,
        context));
}

int main() {
    std::string error;
    agent_mcp_server_tool_registry registry;
    auto authenticator = std::make_shared<agent_mcp_opaque_token_authenticator>();
    if (!authenticator->register_token("vertical-smoke-token", {
            "caller-a", "llama-agent", "namespace-a", "project-a", "agent", {"delegate_task"}, false,
        }, error)) {
        std::fprintf(stderr, "vertical smoke auth setup failed: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_event_collector event_collector;
    bool close_stream_after_terminal = false;
    agent_mcp_http_server_options options;
    options.listen_address = "127.0.0.1";
    options.port = 0;
    options.authenticator = authenticator;
    options.server_name = "agent-vertical-smoke";
    options.protocol_version = "2025-11-25";
    options.agent_tools_enabled = true;
    options.max_delegation_depth = 1;
    options.subscribe_events = [&event_collector](common_agent_event_stream_subscription subscription) {
        return event_collector.subscribe(std::move(subscription));
    };
    options.unsubscribe_events = [&event_collector](const std::string & subscription_id) {
        event_collector.unsubscribe(subscription_id);
    };
    options.wait_for_event = [&event_collector, &close_stream_after_terminal](
            const std::string & subscription_id,
            common_agent_event_stream_delivery & delivery,
            std::chrono::milliseconds timeout) {
        if (close_stream_after_terminal) return common_agent_event_stream_wait_status::closed;
        const auto status = event_collector.wait_next(subscription_id, delivery, timeout);
        if (status == common_agent_event_stream_wait_status::delivered &&
                delivery.kind == common_agent_event_stream_delivery_kind::event &&
                delivery.event.event_type == common_agent_daemon_event_type::turn_completed) {
            close_stream_after_terminal = true;
        }
        return status;
    };
    options.execute_agent_tool = [&event_collector](
            const agent_mcp_caller_policy & policy,
            const std::string & operation,
            const agent_mcp_json & arguments,
            agent_mcp_server_tool_result & result,
            std::string & callback_error) {
        if (operation != "delegate_task") {
            callback_error = "vertical smoke only accepts delegate_task";
            return false;
        }

        common_plan_in_memory_store plan_store;
        if (!plan_store.open("", callback_error)) return false;
        vertical_planner planner;
        vertical_executor executor;
        vertical_reflector reflector;
        vertical_tool_runtime tools;
        common_agent_runtime runtime(plan_store, planner, executor, reflector, &tools);

        common_agent_request request;
        request.prompt = arguments.value("task", "");
        request.namespace_id = policy.namespace_id;
        request.project_id = policy.project_id;
        request.session_id = "vertical-session";
        request.turn_id = "vertical-turn";
        request.max_iterations = 8;
        request.max_reflection_rounds = 2;
        request.max_tool_batches = 16;
        request.deliberation_policy = make_common_agent_deliberation_policy(common_agent_thinking_mode::deliberate);
        request.deliberation_policy.requested_mode = common_agent_thinking_request::deliberate;
        request.deliberation_policy.allow_escalation = true;
        request.deliberation_policy.max_research_iterations = 16;
        request.deliberation_policy.max_tool_rounds = 16;
        request.objective = common_agent_objective{
            "Validate the vertical delegated agent path",
            "Return a verified evidence-backed answer",
            {"memory and user resource evidence are collected"},
            {"preserve event/result separation", "keep the task bounded"}};
        request.input_resources.push_back({
            {"resource://user/vertical-reference", "vertical-reference", "User supplied reference", "text/plain", 64,
                common_runtime_resource_scope::turn, {}},
            "primary_source", true});
        common_memory_hit memory;
        memory.memory.id = "vertical-memory-1";
        memory.memory.summary = "A bounded memory reference for the vertical smoke.";
        memory.memory.confidence = 0.95f;
        memory.final_score = 0.95f;
        request.memories.push_back(memory);

        const auto runtime_result = runtime.run(request);
        append_runtime_events(event_collector, runtime_result, "vertical-request", request.turn_id);
        if (!runtime_result.error.empty() || runtime_result.response.empty() || !runtime_result.research_result ||
                !runtime_result.research_result->complete || !runtime_result.research_verification ||
                runtime_result.research_verification->decision != common_agent_research_verification_decision::accept ||
                tools.resource_reads == 0 || tools.memory_gets == 0) {
            callback_error = runtime_result.error.empty()
                ? "vertical delegated runtime did not complete its research contract"
                : runtime_result.error;
            result.failure_code = "agent.vertical_runtime_failed";
            result.failure_class = "execution";
            result.safe_summary = callback_error;
            return false;
        }
        result.structured_content = {
            {"operation", operation},
            {"thinking_mode", "research"},
            {"response", runtime_result.response},
            {"plan_id", runtime_result.plan_id.value_or("")},
            {"research_gaps", runtime_result.research_result->coverage.answered_gaps},
            {"evidence", runtime_result.research_result->evidence.size()},
            {"answer_verified", true},
        };
        result.content = {{{"type", "text"}, {"text", runtime_result.response}}};
        result.safe_summary = "vertical delegated answer ready";
        callback_error.clear();
        return true;
    };

    agent_mcp_http_server server(std::move(registry), std::move(options));
    if (!server.bind(error)) {
        std::fprintf(stderr, "vertical smoke bind failed: %s\n", error.c_str());
        return 1;
    }
    std::thread server_thread([&server, &error]() { server.listen(error); });
    httplib::Client client("127.0.0.1", server.port());
    httplib::Headers headers = {{"Authorization", "Bearer vertical-smoke-token"}};
    const auto initialize = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})", "application/json");
    if (!initialize || initialize->status != 200) {
        std::fprintf(stderr, "vertical smoke initialize failed\n");
        server.stop(); server_thread.join(); return 1;
    }
    headers.emplace("Mcp-Session-Id", initialize->get_header_value("Mcp-Session-Id"));
    headers.emplace("MCP-Protocol-Version", initialize->get_header_value("MCP-Protocol-Version"));

    std::string stream_body;
    std::mutex stream_mutex;
    bool stream_headers_seen = false;
    std::thread stream_thread([&]() {
        httplib::Client stream_client("127.0.0.1", server.port());
        stream_client.Get("/mcp", httplib::Headers{
                {"Authorization", "Bearer vertical-smoke-token"},
                {"Mcp-Session-Id", headers.find("Mcp-Session-Id")->second},
                {"MCP-Protocol-Version", headers.find("MCP-Protocol-Version")->second},
                {"Accept", "text/event-stream"},
                {"Last-Event-ID", "0"},
            },
            [&stream_headers_seen](const httplib::Response &) { stream_headers_seen = true; return true; },
            [&stream_body, &stream_mutex](const char * data, size_t length) {
                std::lock_guard<std::mutex> lock(stream_mutex);
                stream_body.append(data, length);
                return true;
            });
    });
    for (int attempt = 0; attempt < 100 && !stream_headers_seen; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    const auto delegated = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"delegate_task","arguments":{"task":"Compare the user reference and memory, verify the current answer, and return the evidence."}}})",
        "application/json");
    const auto delegated_json = delegated ? json::parse(delegated->body, nullptr, false) : json();
    const auto structured = delegated_json.value("result", json::object()).value("structuredContent", json::object());

    for (int attempt = 0; attempt < 200; ++attempt) {
        bool terminal_seen = false;
        {
            std::lock_guard<std::mutex> lock(stream_mutex);
            terminal_seen = stream_body.find("turn.completed") != std::string::npos;
        }
        if (terminal_seen) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    server.stop();
    stream_thread.join();
    server_thread.join();
    const std::string captured_stream = [&]() {
        std::lock_guard<std::mutex> lock(stream_mutex);
        return stream_body;
    }();
    const auto plan_position = captured_stream.find("plan_created");
    const auto research_position = captured_stream.find("research_started");
    const auto evidence_position = captured_stream.find("research_evidence_recorded");
    const auto verification_position = captured_stream.find("answer_reviewed");
    const auto terminal_position = captured_stream.find("turn.completed");
    const bool stream_ok = captured_stream.find("event: message") != std::string::npos &&
        captured_stream.find("thinking_mode_resolved") != std::string::npos &&
        captured_stream.find("research_completed") != std::string::npos &&
        plan_position != std::string::npos && research_position != std::string::npos &&
        evidence_position != std::string::npos && verification_position != std::string::npos &&
        terminal_position != std::string::npos && plan_position < research_position &&
        research_position < evidence_position && evidence_position < verification_position &&
        verification_position < terminal_position &&
        captured_stream.find("plan_id=vertical-delegated-plan") != std::string::npos;
    const bool terminal_ok = delegated && delegated->status == 200 &&
        delegated_json.value("result", json::object()).value("isError", true) == false &&
        structured.value("operation", "") == "delegate_task" &&
        structured.value("answer_verified", false) &&
        structured.value("plan_id", "") == "vertical-delegated-plan" &&
        structured.value("research_gaps", 0) == 1 && structured.value("evidence", 0) >= 2;
    if (!terminal_ok || !stream_ok) {
        std::fprintf(stderr, "vertical smoke failed terminal=%d stream=%d\n", terminal_ok ? 1 : 0, stream_ok ? 1 : 0);
        if (delegated) std::fprintf(stderr, "terminal=%s\n", delegated->body.c_str());
        std::fprintf(stderr, "stream=%s\n", captured_stream.c_str());
        return 1;
    }
    std::printf("agent_mcp_vertical=ok\nagent_mcp_vertical_terminal=ok\nagent_mcp_vertical_event_stream=ok\n");
    return 0;
}
