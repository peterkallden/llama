#include "agent-daemon-adapter.h"

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

    common_agent_daemon_environment environment;
    std::string error;
    if (!initialize_agent_daemon_environment(options, environment, error)) {
        std::fprintf(stderr, "failed to initialize daemon environment: %s\n", error.c_str());
        return 2;
    }
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

        if (parsed.value("command", "") == "shutdown") {
            std::cout << make_agent_daemon_shutdown_response().dump() << std::endl;
            break;
        }

        common_agent_runtime_daemon_turn_request request;
        if (!parse_agent_daemon_turn_request(parsed, options, environment.default_mode, request, error)) {
            std::cout << make_agent_daemon_error_response(error).dump() << std::endl;
            continue;
        }

        common_agent_runtime_daemon_turn_result result;
        error.clear();
        environment.host->run_turn(request, result, error);
        if (!error.empty() && result.error.empty()) {
            result.error = error;
        }
        std::cout << make_agent_daemon_turn_response(result).dump() << std::endl;
    }

    return 0;
}
