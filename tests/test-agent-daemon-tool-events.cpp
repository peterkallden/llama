#include "tools/agent/daemon/agent-daemon-service.h"
#include <iostream>
#include <utility>

#define CHECK(condition) do { if (!(condition)) { std::cerr << __LINE__ << ": " << error << '\n'; return __LINE__; } } while (false)

int main() {
    std::string error;
    common_agent_daemon_runtime runtime;
    runtime.tool_executor = [](const common_agent_daemon_tool_payload & payload,
                              agent_tool_result & result, std::string &) {
        result.ok = true;
        common_runtime_resource_ref resource;
        resource.uri = "agent-resource://resource/" + payload.session.session_id;
        resource.name = "report.json";
        resource.mime_type = "application/vnd.llama-agent.document+json";
        result.resource_refs.push_back(resource);
        return true;
    };
    common_agent_daemon_service service(std::move(runtime));
    for (const auto * scope : {"a", "b"}) {
        common_agent_daemon_tool_payload payload;
        payload.session.namespace_id = std::string("namespace-") + scope;
        payload.session.session_id = std::string("session-") + scope;
        payload.project_id = std::string("project-") + scope;
        payload.tool_name = "investigation.report";
        payload.mcp_app_call = true;
        common_agent_daemon_command command;
        command.type = common_agent_daemon_command_type::execute_tool;
        command.request_id = std::string("request-") + scope;
        command.tool = payload;
        common_agent_daemon_command_result result;
        CHECK(service.execute(command, result, error));
        bool found = false;
        for (const auto & event : service.take_internal_events()) {
            if (event.event_type != common_agent_daemon_event_type::tool_artifact_created) continue;
            found = true;
            CHECK(event.request_id == command.request_id);
            CHECK(event.namespace_id == payload.session.namespace_id);
            CHECK(event.project_id == payload.project_id);
            CHECK(event.session_id == payload.session.session_id);
            CHECK(event.turn_id.empty());
            CHECK(event.resource_uri == "agent-resource://resource/" + payload.session.session_id);
            CHECK(event.resource_mime_type == "application/vnd.llama-agent.document+json");
        }
        CHECK(found);
    }
    return 0;
}
