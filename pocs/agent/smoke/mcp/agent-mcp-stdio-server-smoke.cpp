#include "tools/agent/tooling/agent-tool-provider.h"
#include "tools/agent/mcp/agent-mcp-server-tool-registry.h"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

using json = nlohmann::ordered_json;

namespace {

std::filesystem::path get_server_path(const char * argv0) {
    std::filesystem::path argv_path = argv0 != nullptr ? std::filesystem::path(argv0) : std::filesystem::path();
    if (argv_path.has_parent_path()) {
        argv_path = std::filesystem::absolute(argv_path);
    } else {
        argv_path = std::filesystem::current_path() / argv_path;
    }
#ifdef _WIN32
    return argv_path.parent_path() / "llama-agent-mcp-stdio-server.exe";
#else
    return argv_path.parent_path() / "llama-agent-mcp-stdio-server";
#endif
}

std::filesystem::path get_fake_server_path(const char * argv0) {
    std::filesystem::path argv_path = argv0 != nullptr ? std::filesystem::path(argv0) : std::filesystem::path();
    if (argv_path.has_parent_path()) {
        argv_path = std::filesystem::absolute(argv_path);
    } else {
        argv_path = std::filesystem::current_path() / argv_path;
    }
#ifdef _WIN32
    return argv_path.parent_path() / "llama-agent-mcp-stdio-fake-server.exe";
#else
    return argv_path.parent_path() / "llama-agent-mcp-stdio-fake-server";
#endif
}

bool has_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}

std::string join_tool_names(const std::vector<common_chat_tool> & tools) {
    std::string joined;
    for (const auto & tool : tools) {
        if (!joined.empty()) {
            joined += ", ";
        }
        joined += tool.name;
    }
    return joined;
}

} // namespace

int main(int argc, char ** argv) {
    agent_mcp_server_tool_registry schema_registry;
    std::string schema_error;
    if (!schema_registry.register_tool({
            "schema_smoke",
            "Schema validation smoke",
            R"({"type":"object","additionalProperties":false,"required":["title"],"properties":{"title":{"type":"string","minLength":1}}})",
            true,
            false,
            false,
            false,
            false,
            [](const agent_mcp_json &, agent_mcp_server_tool_result & result, std::string & error) {
                result.ok = true;
                error.clear();
                return true;
            },
        }, schema_error)) {
        std::fprintf(stderr, "failed to register schema smoke tool: %s\n", schema_error.c_str());
        return 1;
    }
    agent_mcp_server_tool_result schema_result;
    if (schema_registry.call_tool("schema_smoke", agent_mcp_json::object(), schema_result, schema_error) ||
            schema_error.empty()) {
        std::fprintf(stderr, "MCP server accepted missing required argument\n");
        return 1;
    }
    if (schema_registry.call_tool(
            "schema_smoke",
            agent_mcp_json{{"title", 42}},
            schema_result,
            schema_error) || schema_error.empty()) {
        std::fprintf(stderr, "MCP server accepted wrong argument type\n");
        return 1;
    }
    if (!schema_registry.call_tool(
            "schema_smoke",
            agent_mcp_json{{"title", "valid"}},
            schema_result,
            schema_error) || !schema_result.ok) {
        std::fprintf(stderr, "MCP server rejected valid schema arguments: %s\n", schema_error.c_str());
        return 1;
    }

    const auto server_path = get_server_path(argc > 0 ? argv[0] : nullptr);
    if (!std::filesystem::exists(server_path)) {
        std::fprintf(stderr, "MCP stdio server not found: %s\n", server_path.string().c_str());
        return 1;
    }
    const auto fake_server_path = get_fake_server_path(argc > 0 ? argv[0] : nullptr);
    if (!std::filesystem::exists(fake_server_path)) {
        std::fprintf(stderr, "MCP stdio fake server not found: %s\n", fake_server_path.string().c_str());
        return 1;
    }

    std::string error;
    const auto config_root = std::filesystem::temp_directory_path() / "llama-agent-mcp-stdio-server-smoke";
    std::filesystem::create_directories(config_root);
    const auto minimal_config = config_root / "minimal.json";
    {
        std::ofstream out(minimal_config);
        out << json{
            {"model", {{"backend", "server-context"}}},
            {"tools", {{"profile", "minimal"}}},
            {"limits", {{"max_tool_rounds", 8}}},
        }.dump(2);
    }

    agent_mcp_stdio_client minimal_client({
        "local",
        {server_path.string(), "--config", minimal_config.string()},
        {},
    });
    mcp_agent_tool_provider minimal_provider("local", minimal_client);

    agent_tool_context minimal_context;
    minimal_context.request_id = "mcp-server-smoke";
    minimal_context.turn_id = "turn-1";

    std::unique_ptr<agent_tool_view> minimal_view = minimal_provider.resolve_tools(minimal_context, error);
    if (!minimal_view) {
        std::fprintf(stderr, "failed to resolve minimal MCP stdio server tool view: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(minimal_view->chat_tools(), "local_time_now") ||
            !has_tool(minimal_view->chat_tools(), "local_calculator") ||
            has_tool(minimal_view->chat_tools(), "local_echo")) {
        std::fprintf(stderr, "minimal MCP profile did not expose the expected native tools\n");
        return 1;
    }

    const auto calculator_result = minimal_view->call({
        "call-1",
        "local_calculator",
        R"({"expression":"17 * 23"})",
    }, error);
    if (!calculator_result.ok || calculator_result.content_json.find("391") == std::string::npos) {
        std::fprintf(stderr, "calculator tool did not return the expected result: %s\n", calculator_result.content_json.c_str());
        return 1;
    }

    const auto invalid_result = minimal_view->call({
        "call-2",
        "local_calculator",
        R"({"expression":"1 / 0"})",
    }, error);
    if (invalid_result.ok ||
            invalid_result.failure_class != common_tool_failure_class::validation ||
            invalid_result.failure_code != "tool.calculator.invalid_expression") {
        std::fprintf(stderr, "invalid calculator arguments were not rejected correctly\n");
        return 1;
    }

    const auto memory_config = config_root / "memory.json";
    {
        std::ofstream out(memory_config);
        out << json{
            {"model", {{"backend", "server-context"}}},
            {"tools", {{"profile", "memory"}}},
            {"limits", {{"max_tool_rounds", 8}}},
        }.dump(2);
    }
    agent_mcp_stdio_client memory_client({
        "local",
        {server_path.string(), "--config", memory_config.string()},
        {},
    });
    mcp_agent_tool_provider memory_provider("local", memory_client);

    agent_tool_context memory_context;
    memory_context.request_id = "mcp-server-smoke";
    memory_context.turn_id = "turn-1";
    memory_context.allow_policy_gated_writes = true;
    memory_context.allow_memory_proposals = true;

    std::unique_ptr<agent_tool_view> memory_view = memory_provider.resolve_tools(memory_context, error);
    if (!memory_view) {
        std::fprintf(stderr, "failed to resolve memory MCP stdio server tool view: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(memory_view->chat_tools(), "local_memory_search") ||
            !has_tool(memory_view->chat_tools(), "local_memory_remember")) {
        std::fprintf(stderr, "memory MCP profile did not expose expected memory tools: %s\n", join_tool_names(memory_view->chat_tools()).c_str());
        return 1;
    }

    const auto remember_result = memory_view->call({
        "call-memory-1",
        "local_memory_remember",
        R"({"kind":"procedure","content":"Verify scope before reading stored runtime evidence.","importance":0.8,"confidence":0.75,"rationale":"Useful MCP smoke memory."})",
    }, error);
    if (!remember_result.ok || remember_result.content_json.find("\"ok\":true") == std::string::npos) {
        std::fprintf(stderr, "memory_remember did not return the expected payload: %s\n", remember_result.content_json.c_str());
        return 1;
    }

    const auto memory_search_result = memory_view->call({
        "call-memory-2",
        "local_memory_search",
        R"({"query":"scope runtime evidence","limit":4})",
    }, error);
    if (!memory_search_result.ok || memory_search_result.content_json.find("Verify scope before reading stored runtime evidence.") == std::string::npos) {
        std::fprintf(stderr, "memory_search did not return the remembered content: %s\n", memory_search_result.content_json.c_str());
        return 1;
    }

    std::vector<mcp_agent_resource_definition> listed_resources;
    if (!memory_client.list_resources(listed_resources, error)) {
        std::fprintf(stderr, "resources/list failed against real MCP stdio server: %s\n", error.c_str());
        return 1;
    }

    const std::string repository_root = std::filesystem::weakly_canonical(std::filesystem::current_path()).string();
    const auto research_config = config_root / "research.json";
    {
        std::ofstream out(research_config);
        out << json{
            {"model", {{"backend", "server-context"}}},
            {"tools", {
                {"profile", "research"},
                {"repository_root", repository_root},
                {"providers", json::array({
                    {
                        {"type", "mcp"},
                        {"id", "github"},
                        {"enabled", true},
                        {"transport", "stdio"},
                        {"prefix", "github"},
                        {"server_name", "github"},
                        {"command", json::array({
                            fake_server_path.string(),
                        })},
                    },
                })},
            }},
            {"limits", {{"max_tool_rounds", 8}}},
        }.dump(2);
    }
    agent_mcp_stdio_client research_client({
        "local",
        {
            server_path.string(),
            "--config", research_config.string(),
        },
        {},
    });
    mcp_agent_tool_provider research_provider("local", research_client);

    agent_tool_context research_context;
    research_context.request_id = "mcp-server-smoke";
    research_context.turn_id = "turn-2";
    research_context.allow_network = true;

    std::unique_ptr<agent_tool_view> research_view = research_provider.resolve_tools(research_context, error);
    if (!research_view) {
        std::fprintf(stderr, "failed to resolve research MCP stdio server tool view: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(research_view->chat_tools(), "local_repository.list") ||
            !has_tool(research_view->chat_tools(), "local_github_search_issues")) {
        std::fprintf(
            stderr,
            "research MCP profile did not expose expected native+MCP tools: %s\n",
            join_tool_names(research_view->chat_tools()).c_str());
        return 1;
    }

    const auto repository_result = research_view->call({
        "call-3",
        "local_repository.list",
        R"({"path":"tools/agent/mcp","depth":0})",
    }, error);
    if (!repository_result.ok || repository_result.content_json.find("agent-mcp-stdio-server-main.cpp") == std::string::npos) {
        std::fprintf(stderr, "repository.list did not return the expected repository payload: %s\n", repository_result.content_json.c_str());
        return 1;
    }

    const auto github_result = research_view->call({
        "call-4",
        "local_github_search_issues",
        R"({"query":"runtime host"})",
    }, error);
    if (!github_result.ok || github_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "github_search_issues did not return the expected MCP payload: %s\n", github_result.content_json.c_str());
        return 1;
    }

    std::printf("mcp_server_minimal_tools=%zu\n", minimal_view->chat_tools().size());
    std::printf("mcp_server_calculator=%s\n", calculator_result.content_json.c_str());
    std::printf("mcp_server_memory_tools=%zu\n", memory_view->chat_tools().size());
    std::printf("mcp_server_memory_search=%s\n", memory_search_result.content_json.c_str());
    std::printf("mcp_server_listed_resources=%zu\n", listed_resources.size());
    std::printf("mcp_server_research_tools=%zu\n", research_view->chat_tools().size());
    std::printf("mcp_server_repository.list=%s\n", repository_result.content_json.c_str());
    std::printf("mcp_server_github_search=%s\n", github_result.content_json.c_str());
    std::printf("mcp_server_invalid_code=%s\n", invalid_result.failure_code.c_str());
    return 0;
}
