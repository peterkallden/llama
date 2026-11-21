#include "tools/agent/tooling/agent-tool-provider.h"
#include "tools/agent/mcp/agent-mcp-server-tool-registry.h"
#include "tools/agent/mcp/agent-mcp-stdio-server.h"
#include "tools/agent/mcp/agent-mcp-protocol.h"
#include "agent/tooling/contracts/schema-contract.h"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

namespace {

// Deliberately use literal wire messages and independent JSON-line reads:
// testing only our own client against our own server hid the framing bug.
bool check_stdio_wire_contract() {
    FILE * input = std::tmpfile();
    FILE * output = std::tmpfile();
    FILE * diagnostics = std::tmpfile();
    if (!input || !output || !diagnostics) {
        if (input) std::fclose(input);
        if (output) std::fclose(output);
        if (diagnostics) std::fclose(diagnostics);
        return false;
    }
    const std::string requests =
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"wire-test\",\"version\":\"1\"}}}\n"
        "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n"
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n"
        "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/list\"}\n"
        "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"echo\",\"arguments\":{\"text\":\"first\\nsecond\"}}}\n"
        "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"missing\",\"arguments\":{}}}\n"
        "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"tools/call\",\"params\":{\"name\":\"echo\",\"arguments\":{}}}\n"
        "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"resources/list\"}\n"
        "{\"jsonrpc\":\"2.0\",\"id\":8,\"method\":\"shutdown\"}\n"
        "{\"jsonrpc\":\"2.0\",\"method\":\"exit\"}\n";
    std::fwrite(requests.data(), 1, requests.size(), input);
    std::rewind(input);
    agent_mcp_server_tool_registry registry;
    std::string error;
    bool ok = registry.register_tool({
        "echo", "Echo text", R"({"type":"object","required":["text"],"properties":{"text":{"type":"string"}}})",
        true, false, false, false, false,
        [](const agent_mcp_json & args, agent_mcp_server_tool_result & result, std::string &) {
            result.ok = true;
            result.structured_content = args;
            result.safe_summary = "echoed";
            return true;
        },
    }, error);
    agent_mcp_stdio_server server(std::move(registry));
    ok = ok && server.run(input, output, diagnostics) == 0;
    std::rewind(output);
    std::vector<json> responses;
    char line[8192];
    while (std::fgets(line, sizeof(line), output)) {
        const auto response = json::parse(line, nullptr, false);
        ok = ok && response.is_object();
        responses.push_back(response);
    }
    ok = ok && responses.size() == 8;
    if (ok) {
        for (size_t i = 0; i < responses.size(); ++i) {
            ok = ok && responses[i].value("id", 0) == static_cast<int>(i + 1);
        }
        ok = ok && responses[0]["result"]["protocolVersion"] == "2025-11-25" &&
            responses[0]["result"]["capabilities"]["tools"].is_object() &&
            !responses[0]["result"]["capabilities"].contains("resources") &&
            responses[1]["result"].is_object() &&
            responses[2]["result"]["tools"].size() == 1 &&
            responses[3]["result"]["structuredContent"]["text"] == "first\nsecond" &&
            responses[4]["result"]["isError"] == true &&
            responses[5]["result"]["isError"] == true &&
            responses[6]["error"]["code"] == -32601;
    }
    std::fclose(input);
    std::fclose(output);
    std::fclose(diagnostics);
    return ok;
}

bool check_buffered_stdio_messages() {
    std::string buffer = R"({"jsonrpc":"2.0","id":1,"result":{"text":"a\nb"}})";
    json message;
    std::string error;
    if (parse_mcp_stdio_message(buffer, message, error) || !error.empty()) return false;
    buffer += "\r\n{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{}}\n";
    if (!parse_mcp_stdio_message(buffer, message, error) || !error.empty() ||
            message["result"]["text"] != "a\nb") return false;
    if (!parse_mcp_stdio_message(buffer, message, error) || !error.empty() ||
            message["id"] != 2 || !buffer.empty()) return false;
    buffer = "{broken\n";
    if (!parse_mcp_stdio_message(buffer, message, error) || error.empty()) return false;
    buffer = "[]\n";
    if (!parse_mcp_stdio_message(buffer, message, error) || error.empty()) return false;
    buffer.assign(common_mcp_max_stdio_message_bytes + 1, 'x');
    return parse_mcp_stdio_message(buffer, message, error) && !error.empty();
}

bool matches_advertised_result(const agent_tool_view & view, const agent_tool_result & result) {
    const auto provider_payload = json::parse(result.content_json, nullptr, false);
    if (!provider_payload.is_object() || !provider_payload.contains("result")) return false;
    for (const auto & tool : view.chat_tools()) {
        if (tool.name != result.tool_name) continue;
        std::string normalized;
        std::string error;
        if (common_schema_normalize_and_validate_object(
                provider_payload["result"].dump(), tool.output_schema, normalized, error)) return true;
        std::fprintf(stderr, "%s result does not match outputSchema: %s\n", tool.name.c_str(), error.c_str());
        return false;
    }
    return false;
}

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
    if (!check_stdio_wire_contract() || !check_buffered_stdio_messages()) {
        std::fprintf(stderr, "standard newline-delimited MCP stdio contract failed\n");
        return 1;
    }
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
    if (!schema_registry.register_tool({
            "throw_smoke",
            "Throws an exception for transport-boundary coverage",
            R"({"type":"object"})",
            true,
            false,
            false,
            false,
            false,
            [](const agent_mcp_json &, agent_mcp_server_tool_result &, std::string &) -> bool {
                throw std::runtime_error("private diagnostic detail");
            },
        }, schema_error)) {
        std::fprintf(stderr, "failed to register exception smoke tool: %s\n", schema_error.c_str());
        return 1;
    }
    if (schema_registry.call_tool("throw_smoke", agent_mcp_json::object(), schema_result, schema_error) ||
            schema_result.ok || schema_result.failure_code != "tool.executor_exception" ||
            schema_result.raw_diagnostic.find("private diagnostic detail") == std::string::npos ||
            agent_mcp_render_tool_call_result(schema_result).dump().find("private diagnostic detail") != std::string::npos ||
            !schema_registry.call_tool("schema_smoke", agent_mcp_json{{"title", "after exception"}}, schema_result, schema_error) ||
            !schema_result.ok) {
        std::fprintf(stderr, "MCP registry exception boundary leaked diagnostics or poisoned later calls\n");
        return 1;
    }
    if (!schema_registry.register_tool({
            "throw_unknown_smoke",
            "Throws a non-standard exception for boundary coverage",
            R"({"type":"object"})",
            true,
            false,
            false,
            false,
            false,
            [](const agent_mcp_json &, agent_mcp_server_tool_result &, std::string &) -> bool {
                throw 7;
            },
        }, schema_error) ||
            schema_registry.call_tool("throw_unknown_smoke", agent_mcp_json::object(), schema_result, schema_error) ||
            schema_result.failure_code != "tool.executor_exception" ||
            schema_result.raw_diagnostic.find("non-standard exception") == std::string::npos) {
        std::fprintf(stderr, "MCP registry did not contain a non-standard handler exception\n");
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
    if (!has_tool(minimal_view->chat_tools(), "time.now") ||
            !has_tool(minimal_view->chat_tools(), "math.calculate") ||
            has_tool(minimal_view->chat_tools(), "local_math.calculate") ||
            has_tool(minimal_view->chat_tools(), "local_echo")) {
        std::fprintf(stderr, "minimal MCP profile did not expose the expected native tools: %s\n",
            join_tool_names(minimal_view->chat_tools()).c_str());
        return 1;
    }
    for (const auto & tool : minimal_view->chat_tools()) {
        const auto schema = json::parse(tool.output_schema, nullptr, false);
        if (!schema.is_object() || !schema.contains("properties") ||
                !schema["properties"].contains("ok") || !schema["properties"].contains("result")) {
            std::fprintf(stderr, "real MCP stdio registration lost outputSchema for %s\n", tool.name.c_str());
            return 1;
        }
    }

    const auto calculator_result = minimal_view->call({
        "call-1",
        "math.calculate",
        R"({"expression":"17 * 23"})",
    }, error);
    if (!calculator_result.ok || calculator_result.content_json.find("391") == std::string::npos ||
            !matches_advertised_result(*minimal_view, calculator_result)) {
        std::fprintf(stderr, "calculator tool did not return the expected result: %s\n", calculator_result.content_json.c_str());
        return 1;
    }

    const auto invalid_result = minimal_view->call({
        "call-2",
        "math.calculate",
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
    if (!has_tool(memory_view->chat_tools(), "memory.search") ||
        has_tool(memory_view->chat_tools(), "local_memory.search") ||
        !has_tool(memory_view->chat_tools(), "memory.remember") ||
        has_tool(memory_view->chat_tools(), "local_memory.remember")) {
        std::fprintf(stderr, "memory MCP profile did not expose expected memory tools: %s\n", join_tool_names(memory_view->chat_tools()).c_str());
        return 1;
    }

    const auto remember_result = memory_view->call({
        "call-memory-1",
        "memory.remember",
        R"({"kind":"procedure","content":"Verify scope before reading stored runtime evidence.","importance":0.8,"confidence":0.75,"rationale":"Useful MCP smoke memory."})",
    }, error);
    if (!remember_result.ok || remember_result.content_json.find("\"ok\":true") == std::string::npos) {
        std::fprintf(stderr, "memory_remember did not return the expected payload: %s\n", remember_result.content_json.c_str());
        return 1;
    }

    const auto memory_search_result = memory_view->call({
        "call-memory-2",
        "memory.search",
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
                            "--mode", "delayed-tool-call",
                        })},
                    },
                })},
            }},
            {"limits", {{"max_tool_rounds", 8}, {"tool_timeout_ms", 5000}}},
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
    research_context.default_timeout_ms = 5000;

    std::unique_ptr<agent_tool_view> research_view = research_provider.resolve_tools(research_context, error);
    if (!research_view) {
        std::fprintf(stderr, "failed to resolve research MCP stdio server tool view: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(research_view->chat_tools(), "repository.list") ||
            has_tool(research_view->chat_tools(), "local_repository.list") ||
            !has_tool(research_view->chat_tools(), "local_github_search_issues")) {
        std::fprintf(
            stderr,
            "research MCP profile did not expose expected native+MCP tools: %s\n",
            join_tool_names(research_view->chat_tools()).c_str());
        return 1;
    }
    for (const auto & tool : research_view->chat_tools()) {
        if (tool.name != "resource.read" && tool.name != "resource.inspect") continue;
        const auto schema = json::parse(tool.parameters, nullptr, false);
        if (!schema.is_object() || !schema["properties"].contains("uri") ||
                schema["properties"].contains("id")) {
            std::fprintf(stderr, "MCP must advertise canonical resource URI arguments, not model handles\n");
            return 1;
        }
    }
    const auto missing_resource = research_view->call({
        "call-resource", "resource.inspect", R"({"uri":"agent-resource://resource/missing"})",
    }, error);
    if (missing_resource.ok || missing_resource.failure_class != common_tool_failure_class::not_found) {
        std::fprintf(stderr, "canonical resource URI was rejected before resource lookup\n");
        return 1;
    }

    const auto repository_result = research_view->call({
        "call-3",
        "repository.list",
        R"({"path":"tools/agent/mcp","depth":0})",
    }, error);
    if (!repository_result.ok || repository_result.content_json.find("agent-mcp-stdio-server-main.cpp") == std::string::npos ||
            !matches_advertised_result(*research_view, repository_result)) {
        std::fprintf(stderr, "repository.list did not return the expected repository payload: %s\n", repository_result.content_json.c_str());
        return 1;
    }

    const auto github_result = research_view->call({
        "call-4",
        "local_github_search_issues",
        R"({"query":"runtime host"})",
    }, error);
    if (!github_result.ok || github_result.content_json.find("stub issue") == std::string::npos ||
            !matches_advertised_result(*research_view, github_result)) {
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
