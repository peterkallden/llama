#include "agent-cli-command.h"

#include "agent-cli-config.h"
#include "agent-cli-run.h"
#include "../daemon/agent-daemon-client.h"
#include "../host/agent-host-config.h"

#include <cstdio>
#include <cstring>

namespace {

const char * find_cli_config_path(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0) {
            return i + 1 < argc ? argv[i + 1] : nullptr;
        }
    }
    return nullptr;
}

} // namespace

int run_memory_chat_command(const char * argv0, common_memory_store & store, args a) {
    if (a.model.empty() || a.prompt.empty()) {
        print_agent_usage(argv0, "chat");
        return 1;
    }

    std::string error;
    if (!validate_agent_memory_scope(a, error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    a.command = "chat";
    return run_agent_cli(store, a);
}

int run_agent_command_main(const char * argv0, int argc, char ** argv) {
    args a;
    const char * explicit_config_path = find_cli_config_path(argc, argv);
    std::string resolved_config_path;
    std::string config_resolution_error;
    if (resolve_agent_host_config_path(
            explicit_config_path != nullptr ? explicit_config_path : "",
            resolved_config_path,
            config_resolution_error)) {
        agent_host_config config;
        std::string config_error;
        if (!load_agent_host_config(resolved_config_path, config, config_error)) {
            std::fprintf(stderr, "%s\n", config_error.c_str());
            return 1;
        }
        apply_agent_host_config_to_args(config, a);
    } else if (!config_resolution_error.empty()) {
        std::fprintf(stderr, "%s\n", config_resolution_error.c_str());
        return 1;
    }
    if (!parse_agent_run_args(argc, argv, a)) {
        print_agent_usage(argv0);
        return 1;
    }

    if (a.command == "daemon-chat") {
        return run_daemon_chat_command(argv0, a);
    }
    if (a.command == "daemon-session") {
        return run_daemon_session_command(argv0, a);
    }

    if (a.model.empty() || a.prompt.empty()) {
        print_agent_usage(argv0);
        return 1;
    }

    std::string error;
    if (!validate_agent_memory_scope(a, error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    auto store = make_memory_store(a, error);
    if (!store) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!open_memory_store(*store, a, error)) {
        fprintf(stderr, "failed to open memory store: %s\n", error.c_str());
        return 1;
    }

    a.command = "chat";
    return run_agent_cli(*store, a);
}
