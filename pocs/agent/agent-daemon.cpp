#include "agent-daemon-adapter.h"
#include "agent-daemon-dispatcher.h"

#include "log.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

using json = nlohmann::ordered_json;

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
    common_agent_daemon_dispatcher dispatcher(std::move(runtime));
    std::cout << make_agent_daemon_ready_response(options).dump() << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }

        const auto parsed = json::parse(line, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            std::cout << make_agent_daemon_error_response("invalid JSON request").dump() << std::endl;
            continue;
        }

        common_agent_daemon_command command;
        if (!parse_agent_daemon_command(parsed, options, dispatcher.default_mode(), command, error)) {
            std::cout << make_agent_daemon_error_response(error).dump() << std::endl;
            continue;
        }

        common_agent_daemon_command_result result;
        error.clear();
        dispatcher.execute(command, result, error);
        if (!error.empty() && result.error.empty()) {
            result.error = error;
        }
        std::cout << make_agent_daemon_command_response(result).dump() << std::endl;
        if (dispatcher.shutdown_requested()) {
            break;
        }
    }

    return 0;
}
