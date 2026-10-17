#include "tools/agent/daemon/agent-daemon-scope.h"
#include "tools/agent/daemon/agent-daemon-adapter.h"

#include <cassert>

int main() {
    agent_mcp_caller_policy policy;
    policy.caller_id = "web-client";
    policy.namespace_id = "local";
    policy.project_id = "default";
    policy.tool_profile = "all-configured";

    nlohmann::ordered_json upload = {{"command", "put_resource"}};
    common_agent_daemon_bind_caller_scope(upload, policy);
    assert(upload.value("namespace_id", "") == "local");
    assert(upload.value("project_id", "") == "default");

    nlohmann::ordered_json read = {
        {"command", "read_resource"},
        {"namespace_id", "attacker-namespace"},
        {"project_id", "attacker-project"},
    };
    common_agent_daemon_bind_caller_scope(read, policy);
    assert(read.value("namespace_id", "") == "local");
    assert(read.value("project_id", "") == "default");

    nlohmann::ordered_json turn = {
        {"command", "run_turn"},
        {"namespace_id", "attacker-namespace"},
        {"project_id", "attacker-project"},
    };
    common_agent_daemon_bind_caller_scope(turn, policy);
    assert(turn.value("namespace_id", "") == "local");
    assert(turn.value("project_id", "") == "default");
    assert(turn.value("session_id", "") == "web-client-session");

    nlohmann::ordered_json tool = {{"command", "execute_tool"}};
    common_agent_daemon_bind_caller_scope(tool, policy);
    assert(tool.value("namespace_id", "") == "local");
    assert(tool.value("project_id", "") == "default");
    assert(tool.value("tool_profile", "") == "all-configured");

    daemon_options options;
    common_agent_daemon_command parsed_app_call;
    std::string error;
    const nlohmann::ordered_json app_call = {
        {"command", "execute_tool"},
        {"tool_name", "document.open"},
        {"mcp_app_call", true},
        {"mcp_app_tool_name", "document.open"},
        {"arguments", nlohmann::ordered_json::object()},
    };
    assert(parse_agent_daemon_command(app_call, options,
        common_agent_runtime_host_mode::chat, parsed_app_call, error));
    assert(parsed_app_call.tool.has_value());
    assert(parsed_app_call.tool->allow_policy_gated_writes);

    common_agent_daemon_command parsed_direct_call;
    const nlohmann::ordered_json direct_call = {
        {"command", "execute_tool"},
        {"tool_name", "document.save"},
        {"arguments", nlohmann::ordered_json::object()},
    };
    assert(parse_agent_daemon_command(direct_call, options,
        common_agent_runtime_host_mode::chat, parsed_direct_call, error));
    assert(parsed_direct_call.tool.has_value());
    assert(!parsed_direct_call.tool->allow_policy_gated_writes);

    common_agent_daemon_command orientation;
    const nlohmann::ordered_json orientation_request = {
        {"command", "run_turn"}, {"mode", "agent"}, {"prompt", "client supplied text is ignored"},
        {"session_id", "s1"}, {"namespace_id", "local"}, {"turn_id", "orient-1"},
        {"attachment_orientation", true}, {"resource_refs", nlohmann::ordered_json::array({"resource://upload/1"})},
    };
    assert(parse_agent_daemon_command(orientation_request, options,
        common_agent_runtime_host_mode::agent, orientation, error));
    assert(orientation.turn.has_value());
    assert(orientation.turn->request.turn.attachment_orientation);
    assert(orientation.turn->request.turn.resource_read_binding.has_value());
    assert(orientation.turn->request.turn.resource_read_binding->intent == common_agent_resource_read_intent::orientation);
    assert(orientation.turn->request.turn.resource_read_binding->require_full_coverage);
    assert(orientation.turn->request.turn.require_tool_execution);
    assert(orientation.turn->request.turn.prompt.find("initial orientation") != std::string::npos);
    assert(orientation.turn->request.turn.allowed_exposed_tool_names.size() == 7);
    assert(orientation.turn->request.turn.input_resources.size() == 1);
    common_agent_daemon_command attachment_question;
    auto attachment_question_request = orientation_request;
    attachment_question_request.erase("attachment_orientation");
    attachment_question_request["attachment_context_required"] = true;
    attachment_question_request["prompt"] = "Who is this document about?";
    assert(parse_agent_daemon_command(attachment_question_request, options,
        common_agent_runtime_host_mode::agent, attachment_question, error));
    assert(attachment_question.turn.has_value());
    assert(attachment_question.turn->request.turn.prompt == "Who is this document about?");
    assert(attachment_question.turn->request.turn.require_tool_execution);
    assert(attachment_question.turn->request.turn.resource_read_binding.has_value());
    assert(attachment_question.turn->request.turn.resource_read_binding->intent == common_agent_resource_read_intent::question);
    assert(!attachment_question.turn->request.turn.resource_read_binding->require_full_coverage);
    assert(attachment_question.turn->request.turn.input_resources.front().required);
    auto chat_orientation = orientation_request;
    chat_orientation["mode"] = "chat";
    common_agent_daemon_command rejected_orientation;
    assert(!parse_agent_daemon_command(chat_orientation, options,
        common_agent_runtime_host_mode::agent, rejected_orientation, error));
    auto multi_resource_orientation = orientation_request;
    multi_resource_orientation["resource_refs"].push_back("resource://upload/2");
    assert(!parse_agent_daemon_command(multi_resource_orientation, options,
        common_agent_runtime_host_mode::agent, rejected_orientation, error));

    return 0;
}
