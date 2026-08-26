#include "tools/agent/daemon/agent-daemon-scope.h"

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

    return 0;
}
