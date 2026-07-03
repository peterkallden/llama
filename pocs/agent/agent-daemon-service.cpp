#include "agent-daemon-service.h"

common_agent_daemon_service::common_agent_daemon_service(common_agent_daemon_runtime runtime)
    : runtime(std::move(runtime)) {}

bool common_agent_daemon_service::execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error) {
    result = {};
    result.request_id = command.request_id;

    switch (command.type) {
        case common_agent_daemon_command_type::shutdown:
            shutdown_requested_flag = true;
            result.ok = true;
            result.event = "shutdown";
            error.clear();
            return true;

        case common_agent_daemon_command_type::run_turn:
            if (!command.turn.has_value()) {
                error = "run_turn command missing turn payload";
                result.error = error;
                return false;
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                result.error = error;
                return false;
            }

            error.clear();
            runtime.host->run_turn(*command.turn, result.turn_result, error);
            result.ok = result.turn_result.ok;
            if (!error.empty() && result.turn_result.error.empty()) {
                result.turn_result.error = error;
            }
            if (!result.turn_result.error.empty()) {
                result.error = result.turn_result.error;
            }
            return result.turn_result.ok;
    }

    error = "unsupported daemon command";
    result.error = error;
    return false;
}
