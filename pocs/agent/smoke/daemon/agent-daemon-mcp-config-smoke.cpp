#include "tools/agent/daemon/agent-daemon-adapter.h"
#include "tools/agent/daemon/agent-daemon-service.h"
#include "tools/agent/host/agent-host-config.h"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using json = nlohmann::ordered_json;

namespace {

bool has_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
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

} // namespace

int main(int argc, char ** argv) {
    const auto server_path = get_fake_server_path(argc > 0 ? argv[0] : nullptr);
    if (!std::filesystem::exists(server_path)) {
        std::fprintf(stderr, "fake MCP stdio server not found: %s\n", server_path.string().c_str());
        return 1;
    }

    const auto config_root = std::filesystem::temp_directory_path() / "llama-agent-daemon-mcp-config-smoke";
    std::filesystem::create_directories(config_root);
    const auto config_path = config_root / "daemon-config.json";
    {
        std::ofstream out(config_path);
        out << json{
            {"model", {
                {"backend", "server-context"},
                {"path", "fake.gguf"},
            }},
            {"tools", {
                {"profile", "minimal"},
                {"providers", json::array({
                    json{
                        {"type", "mcp"},
                        {"id", "github"},
                        {"enabled", true},
                        {"transport", "stdio"},
                        {"command", json::array({server_path.string()})},
                        {"prefix", "github"},
                        {"server_name", "github"},
                    },
                    json{
                        {"type", "mcp"},
                        {"id", "github_alt"},
                        {"enabled", true},
                        {"transport", "stdio"},
                        {"command", json::array({server_path.string()})},
                        {"prefix", "github_alt"},
                        {"server_name", "github-alt"},
                    },
                })},
            }},
            {"limits", {
                {"queue_capacity", 5},
                {"max_tool_rounds", 2},
            }},
        }.dump(2);
    }

    daemon_options options;
    char program[] = "llama-agent-daemon";
    char config_flag[] = "--config";
    std::string config_path_string = config_path.string();
    std::string error;
    std::vector<char> config_path_buffer(config_path_string.begin(), config_path_string.end());
    config_path_buffer.push_back('\0');
    char * parse_argv[] = {program, config_flag, config_path_buffer.data()};
    if (!parse_agent_daemon_args(3, parse_argv, options)) {
        std::fprintf(stderr, "daemon config parse failed\n");
        return 1;
    }

    agent_host_config loaded_config;
    if (!load_agent_host_config(config_path.string(), loaded_config, error)) {
        std::fprintf(stderr, "explicit host config load failed: %s\n", error.c_str());
        return 1;
    }
    if (loaded_config.schema_version != 1) {
        std::fprintf(stderr, "host config schema_version mismatch\n");
        return 1;
    }
    if (!validate_agent_host_config(loaded_config, error)) {
        std::fprintf(stderr, "host config validation failed: %s\n", error.c_str());
        return 1;
    }
    const json roundtrip = agent_host_config_to_json(loaded_config);
    if (!roundtrip.is_object() ||
            roundtrip.value("schema_version", 0) != 1 ||
            !roundtrip.contains("tools") ||
            !roundtrip["tools"].is_object()) {
        std::fprintf(stderr, "host config roundtrip serialization mismatch\n");
        return 1;
    }

    agent_host_config remote_config;
    const json remote_config_json = {
        {"schema_version", 1},
        {"tools", { {"providers", json::array({ json{
            {"type", "mcp"},
            {"id", "remote-github"},
            {"enabled", true},
            {"transport", "streamable_http"},
            {"url", "https://mcp.example.test/mcp"},
            {"token_env", "REMOTE_GITHUB_MCP_TOKEN"},
            {"allowed_tools", json::array({"search_issues"})},
            {"connect_timeout_ms", 4000},
            {"request_timeout_ms", 15000},
            {"shutdown_timeout_ms", 2000},
            {"max_result_bytes", 1048576},
        }})}}},
    };
    if (!parse_agent_host_config_json(remote_config_json, remote_config, error) ||
            remote_config.mcp_providers.size() != 1 ||
            remote_config.mcp_providers[0].url != "https://mcp.example.test/mcp" ||
            remote_config.mcp_providers[0].token_env != "REMOTE_GITHUB_MCP_TOKEN" ||
            remote_config.mcp_providers[0].allowed_tools.size() != 1 ||
            remote_config.mcp_providers[0].max_result_bytes != 1048576) {
        std::fprintf(stderr, "remote MCP provider config contract failed: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_runtime runtime;
    if (!initialize_agent_daemon_environment(options, runtime, error)) {
        std::fprintf(stderr, "daemon MCP environment init failed: %s\n", error.c_str());
        return 1;
    }
    if (!runtime.host) {
        std::fprintf(stderr, "daemon MCP environment did not create a session manager\n");
        return 1;
    }

    common_agent_runtime_tooling tooling;
    if (!resolve_agent_daemon_tooling(
            options,
            nullptr,
            {
                common_agent_runtime_host_mode::chat,
                "find tooling",
                "session-a",
                "namespace-a",
                "",
                "turn-a",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
            },
            *runtime.memory_store,
            *runtime.plan_store,
            runtime.resource_store.get(),
            tooling,
            error)) {
        std::fprintf(stderr, "daemon tooling resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!tooling.tool_view) {
        std::fprintf(stderr, "daemon tooling resolve did not return a tool view\n");
        return 1;
    }
    if (!has_tool(tooling.tools, "calculator") ||
            !has_tool(tooling.tools, "github_search_issues") ||
            !has_tool(tooling.tools, "github_alt_search_issues")) {
        std::fprintf(stderr, "daemon tooling resolve did not expose expected native+MCP tools\n");
        return 1;
    }

    auto calculator_result = tooling.tool_view->call({
        "call-1",
        "calculator",
        R"({"expression":"6 * 7"})",
    }, error);
    if (!calculator_result.ok || calculator_result.content_json.find("42") == std::string::npos) {
        std::fprintf(stderr, "daemon native tool call failed: %s\n", calculator_result.content_json.c_str());
        return 1;
    }

    auto github_result = tooling.tool_view->call({
        "call-2",
        "github_search_issues",
        R"({"query":"runtime host"})",
    }, error);
    if (!github_result.ok || github_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "daemon MCP tool call failed: %s\n", github_result.content_json.c_str());
        return 1;
    }
    auto github_alt_result = tooling.tool_view->call({
        "call-3",
        "github_alt_search_issues",
        R"({"query":"runtime host"})",
    }, error);
    if (!github_alt_result.ok || github_alt_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "daemon secondary MCP tool call failed: %s\n", github_alt_result.content_json.c_str());
        return 1;
    }
    const auto resolved_tool_count = tooling.tools.size();
    tooling.tool_view = nullptr;
    tooling.owned_resources.clear();
    tooling.tools.clear();
    tooling.profile_tools_active = false;

    common_agent_daemon_service service(std::move(runtime));
    common_agent_daemon_command_result status;
    if (!service.populate_status(status, error) || !status.status.ready || !status.status.live) {
        std::fprintf(stderr, "daemon MCP status was not ready after init: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command shutdown_command;
    shutdown_command.request_id = "shutdown-1";
    shutdown_command.type = common_agent_daemon_command_type::shutdown;
    common_agent_daemon_command_result shutdown_result;
    error.clear();
    if (!service.execute(shutdown_command, shutdown_result, error) ||
            !shutdown_result.ok ||
            shutdown_result.event != "shutdown") {
        std::fprintf(stderr, "daemon shutdown lifecycle contract failed: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command rejected_turn;
    rejected_turn.request_id = "turn-after-shutdown";
    rejected_turn.type = common_agent_daemon_command_type::run_turn;
    rejected_turn.turn = common_agent_daemon_turn_payload{};
    rejected_turn.turn->request.request_id = rejected_turn.request_id;
    rejected_turn.turn->request.turn.mode = common_agent_runtime_host_mode::chat;
    rejected_turn.turn->request.turn.prompt = "hello";
    rejected_turn.turn->request.turn.session_id = "session-a";
    rejected_turn.turn->request.turn.namespace_id = "namespace-a";
    rejected_turn.turn->request.turn.project_id = "project-a";
    rejected_turn.turn->request.turn.turn_id = "turn-a";
    rejected_turn.turn->request.turn.memory_scope = common_memory_scope::session;
    rejected_turn.turn->request.turn.plan_scope = common_plan_scope::turn;
    common_agent_daemon_command_result rejected_turn_result;
    error.clear();
    if (service.execute(rejected_turn, rejected_turn_result, error) ||
            rejected_turn_result.events.empty() ||
            rejected_turn_result.events.back().type != "turn.rejected" ||
            error.find("not accepting new turns") == std::string::npos) {
        std::fprintf(stderr, "daemon post-shutdown turn rejection contract failed: %s\n", error.c_str());
        return 1;
    }

    daemon_options bad_options = options;
    if (bad_options.mcp_providers.size() < 2) {
        std::fprintf(stderr, "daemon MCP bad tooling test requires at least two MCP providers\n");
        return 1;
    }
    bad_options.mcp_providers[1].prefix = bad_options.mcp_providers[0].prefix;
    common_agent_runtime_tooling bad_tooling;
    error.clear();
    if (resolve_agent_daemon_tooling(
            bad_options,
            nullptr,
            {
                common_agent_runtime_host_mode::chat,
                "find tooling",
                "session-a",
                "namespace-a",
                "",
                "turn-a",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
            },
            *runtime.memory_store,
            *runtime.plan_store,
            runtime.resource_store.get(),
            bad_tooling,
            error)) {
        std::fprintf(stderr, "daemon MCP bad tooling unexpectedly succeeded\n");
        return 1;
    }
    if (error.find("duplicate tool exposed in composite provider") == std::string::npos) {
        std::fprintf(stderr, "daemon MCP bad tooling did not preserve expected diagnostics: %s\n", error.c_str());
        return 1;
    }

    std::printf("daemon_mcp_ready=%s\n", status.status.ready ? "true" : "false");
    std::printf("daemon_mcp_status=%s\n", common_agent_daemon_state_name(status.status.state));
    std::printf("daemon_tooling_tools=%zu\n", resolved_tool_count);
    return 0;
}
