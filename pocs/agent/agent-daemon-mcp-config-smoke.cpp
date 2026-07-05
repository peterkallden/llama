#include "agent-daemon-adapter.h"
#include "agent-daemon-service.h"

#include <cstdio>
#include <filesystem>
#include <string>

namespace {

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

    daemon_options options;
    options.model = "fake.gguf";
    options.default_mode = "chat";
    options.mcp_tool_command = server_path.string();
    options.mcp_tool_server_name = "github";
    options.mcp_tool_prefix = "github";
    options.max_tool_rounds = 1;

    common_agent_daemon_runtime runtime;
    std::string error;
    if (!initialize_agent_daemon_environment(options, runtime, error)) {
        std::fprintf(stderr, "daemon MCP environment init failed: %s\n", error.c_str());
        return 1;
    }
    if (!runtime.host) {
        std::fprintf(stderr, "daemon MCP environment did not create a session manager\n");
        return 1;
    }

    common_agent_daemon_service service(std::move(runtime));
    common_agent_daemon_command_result status;
    if (!service.populate_status(status, error) || !status.ready || !status.live) {
        std::fprintf(stderr, "daemon MCP status was not ready after init: %s\n", error.c_str());
        return 1;
    }

    daemon_options bad_options = options;
    bad_options.mcp_tool_args = {"--mode", "bad-tools-list"};
    common_agent_daemon_runtime bad_runtime;
    if (!initialize_agent_daemon_environment(bad_options, bad_runtime, error)) {
        std::fprintf(stderr, "daemon MCP bad environment failed too early: %s\n", error.c_str());
        return 1;
    }
    common_agent_daemon_service bad_service(std::move(bad_runtime));
    common_agent_daemon_command_result bad_result;
    if (bad_service.execute({
            "bad-request",
            common_agent_daemon_command_type::run_turn,
            common_agent_runtime_session_manager_turn_request{
                common_agent_runtime_host_mode::chat,
                "hello",
                "session-a",
                "namespace-a",
                "",
                "turn-a",
                common_memory_scope::session,
                common_plan_scope::turn,
                0,
            },
            std::nullopt,
            {},
            {},
        }, bad_result, error)) {
        std::fprintf(stderr, "daemon MCP bad turn unexpectedly succeeded\n");
        return 1;
    }
    if (bad_result.error.find("invalid JSON-RPC payload") == std::string::npos) {
        std::fprintf(stderr, "daemon MCP bad turn did not preserve expected diagnostics: %s\n", bad_result.error.c_str());
        return 1;
    }

    std::printf("daemon_mcp_ready=%s\n", status.ready ? "true" : "false");
    std::printf("daemon_mcp_status=%s\n", status.state.c_str());
    return 0;
}
