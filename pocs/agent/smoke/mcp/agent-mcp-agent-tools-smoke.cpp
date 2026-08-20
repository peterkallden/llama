#include "tools/agent/mcp/agent-mcp-agent-tools.h"
#include "tools/agent/mcp/agent-mcp-http-server.h"

#include <cpp-httplib/httplib.h>
#include <cstdio>
#include <memory>
#include <thread>

int main() {
    agent_mcp_server_tool_registry registry;
    auto auth = std::make_shared<agent_mcp_opaque_token_authenticator>();
    std::string error;
    if (!auth->register_token("agent-tools-smoke", {
            "caller-a", "llama-agent", "namespace-a", "project-a", "agent", {"delegate_task", "summarize", "review_plan"}, false,
        }, error)) return 1;

    agent_mcp_http_server_options options;
    options.listen_address = "127.0.0.1";
    options.port = 0;
    options.authenticator = auth;
    options.server_name = "agent-tools-smoke";
    options.agent_tools_enabled = true;
    options.max_delegation_depth = 1;
    options.execute_agent_tool = [](
            const agent_mcp_caller_policy & policy,
            const std::string & operation,
            const agent_mcp_json & arguments,
            agent_mcp_server_tool_result & result,
            std::string & error) {
        if (policy.caller_id != "caller-a") { error = "wrong caller"; return false; }
        result.structured_content = {{"operation", operation}, {"response", arguments.value("task", arguments.value("text", "ok"))}};
        result.content = {{{"type", "text"}, {"text", "delegated"}}};
        result.safe_summary = "delegated";
        return true;
    };
    agent_mcp_http_server server(std::move(registry), std::move(options));
    if (!server.bind(error)) { std::fprintf(stderr, "bind: %s\n", error.c_str()); return 1; }
    std::thread thread([&server, &error]() { server.listen(error); });
    httplib::Client client("127.0.0.1", server.port());
    httplib::Headers headers = {{"Authorization", "Bearer agent-tools-smoke"}};
    auto init = client.Post("/mcp", headers, R"({"jsonrpc":"2.0","id":1,"method":"initialize"})", "application/json");
    if (!init || init->status != 200) { std::fprintf(stderr, "init failed\n"); server.stop(); thread.join(); return 1; }
    headers.emplace("Mcp-Session-Id", init->get_header_value("Mcp-Session-Id"));
    headers.emplace("MCP-Protocol-Version", init->get_header_value("MCP-Protocol-Version"));
    auto listed = client.Post("/mcp", headers, R"({"jsonrpc":"2.0","id":2,"method":"tools/list"})", "application/json");
    auto called = client.Post("/mcp", headers, R"({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"delegate_task","arguments":{"task":"hello"}}})", "application/json");
    auto depth = client.Post("/mcp", headers, R"({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"delegate_task","arguments":{"task":"hello","delegation_depth":1}}})", "application/json");
    auto deliberate = client.Post("/mcp", headers, R"({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"delegate_task","arguments":{"task":"hello","thinking_mode":"deliberate","max_plan_revisions":2}}})", "application/json");
    auto research = client.Post("/mcp", headers, R"({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"delegate_task","arguments":{"task":"hello","thinking_mode":"research","max_research_iterations":4,"resource_refs":["resource://input/1"]}}})", "application/json");
    auto invalid_mode = client.Post("/mcp", headers, R"({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"delegate_task","arguments":{"task":"hello","thinking_mode":"direct"}}})", "application/json");
    const auto list_json = listed ? agent_mcp_json::parse(listed->body, nullptr, false) : agent_mcp_json();
    const auto call_json = called ? agent_mcp_json::parse(called->body, nullptr, false) : agent_mcp_json();
    const auto depth_json = depth ? agent_mcp_json::parse(depth->body, nullptr, false) : agent_mcp_json();
    const auto deliberate_json = deliberate ? agent_mcp_json::parse(deliberate->body, nullptr, false) : agent_mcp_json();
    const auto research_json = research ? agent_mcp_json::parse(research->body, nullptr, false) : agent_mcp_json();
    const auto invalid_mode_json = invalid_mode ? agent_mcp_json::parse(invalid_mode->body, nullptr, false) : agent_mcp_json();
    bool schema_advertises_thinking_modes = false;
    if (list_json.contains("result") && list_json["result"].contains("tools")) {
        for (const auto & tool : list_json["result"]["tools"]) {
            if (tool.value("name", "") != "delegate_task") continue;
            const auto & schema = tool["inputSchema"];
            const auto modes = schema["properties"]["thinking_mode"]["enum"];
            schema_advertises_thinking_modes = modes.is_array() && modes.size() == 3 &&
                modes[0] == "reflective" && modes[1] == "deliberate" && modes[2] == "research" &&
                schema["properties"].contains("max_plan_revisions") &&
                schema["properties"].contains("max_research_iterations");
        }
    }
    const bool ok = listed && listed->status == 200 && list_json["result"]["tools"].size() == 3 &&
        schema_advertises_thinking_modes &&
        called && called->status == 200 && call_json["result"]["structuredContent"]["operation"] == "delegate_task" &&
        depth && depth->status == 200 && depth_json["result"]["isError"] == true &&
        deliberate && deliberate->status == 200 && deliberate_json["result"]["isError"] == false &&
        research && research->status == 200 && research_json["result"]["isError"] == false &&
        invalid_mode && invalid_mode->status == 200 && invalid_mode_json["result"]["isError"] == true;
    server.stop(); thread.join();
    if (!ok) {
        std::fprintf(stderr, "agent smoke failed statuses list=%d call=%d depth=%d deliberate=%d research=%d invalid_mode=%d\n",
            listed ? listed->status : 0, called ? called->status : 0, depth ? depth->status : 0,
            deliberate ? deliberate->status : 0, research ? research->status : 0,
            invalid_mode ? invalid_mode->status : 0);
        if (listed) std::fprintf(stderr, "list=%s\n", listed->body.c_str());
        if (called) std::fprintf(stderr, "call=%s\n", called->body.c_str());
        if (depth) std::fprintf(stderr, "depth=%s\n", depth->body.c_str());
        return 1;
    }
    std::printf("agent_mcp_tools=3\nagent_mcp_delegation=ok\nagent_mcp_depth_limit=ok\n");
    std::printf("agent_mcp_thinking_modes=reflective,deliberate,research\nagent_mcp_inbound_contract=ok\n");
    return 0;
}
