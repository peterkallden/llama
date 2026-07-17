#include "agent-daemon-adapter.h"
#include "agent-daemon-dispatcher.h"
#include "../mcp/agent-mcp-http-server.h"

#include "log.h"

#include <memory>
#include <thread>
#include <nlohmann/json.hpp>

int main(int argc, char ** argv) {
    daemon_options options;
    if (!parse_agent_daemon_args(argc, argv, options)) {
        print_agent_daemon_usage(argv[0]);
        return 2;
    }
    common_log_set_verbosity_thold(LOG_LEVEL_WARN);

    common_agent_daemon_runtime runtime;
    std::string error;
    if (!initialize_agent_daemon_environment(options, runtime, error)) {
        std::fprintf(stderr, "failed to initialize daemon environment: %s\n", error.c_str());
        return 2;
    }
    common_agent_daemon_dispatcher dispatcher(std::move(runtime), options.queue_capacity, options.worker_count);
    std::unique_ptr<agent_mcp_http_server> http_server;
    std::thread http_thread;
    if (options.http_enabled) {
        agent_mcp_server_tool_registry registry;
        agent_mcp_http_server_options http_options;
        http_options.listen_address = options.http_listen_address;
        http_options.port = options.http_port;
        http_options.path = options.http_path;
        http_options.allowed_origin = options.http_allowed_origin;
        http_options.bearer_token = options.http_bearer_token;
        http_options.max_body_bytes = options.http_max_body_bytes;
        http_options.max_result_bytes = options.http_max_result_bytes;
        http_options.server_name = "llama-agent-daemon-mcp";
        http_options.default_policy = {
            "daemon-http",
            "llama-agent",
            "local",
            "",
            options.tool_profile,
            {},
            options.tool_profile == "memory" || options.tool_profile == "research",
        };
        http_options.execute_tool = [&dispatcher](
                const agent_mcp_caller_policy & policy,
                const std::string & tool_name,
                const agent_mcp_json & arguments,
                agent_mcp_server_tool_result & result,
                std::string & callback_error) {
            common_agent_daemon_command command;
            command.request_id = "mcp-http-tool";
            command.type = common_agent_daemon_command_type::execute_tool;
            command.tool = common_agent_daemon_tool_payload{
                {policy.namespace_id, policy.caller_id + "-session"},
                policy.project_id,
                policy.tool_profile,
                tool_name,
                arguments.dump(),
            };
            common_agent_daemon_command_result daemon_result;
            if (!dispatcher.execute(command, daemon_result, callback_error)) {
                result.ok = false;
                result.safe_summary = callback_error;
                result.content = {{{"type", "text"}, {"text", callback_error}}};
                return false;
            }
            result.ok = daemon_result.tool_result.ok;
            result.structured_content = nlohmann::ordered_json::parse(
                daemon_result.tool_result.content_json, nullptr, false);
            if (result.structured_content.is_discarded()) {
                result.structured_content = agent_mcp_json::object();
            }
            result.safe_summary = daemon_result.tool_result.content_summary;
            result.content = {{{"type", "text"}, {"text", result.safe_summary}}};
            callback_error = daemon_result.error;
            return result.ok;
        };
        http_server = std::make_unique<agent_mcp_http_server>(std::move(registry), std::move(http_options));
        if (!http_server->bind(error)) {
            std::fprintf(stderr, "failed to bind daemon MCP HTTP server: %s\n", error.c_str());
            return 2;
        }
        http_thread = std::thread([&http_server, &error]() {
            http_server->listen(error);
        });
    }

    const bool jsonl_ok = run_agent_daemon_jsonl_adapter(stdin, stdout, options, dispatcher, error);
    if (http_server) {
        http_server->stop();
        if (http_thread.joinable()) http_thread.join();
    }
    if (!jsonl_ok) {
        if (!error.empty()) {
            std::fprintf(stderr, "daemon adapter failed: %s\n", error.c_str());
        }
        return 2;
    }

    return 0;
}
