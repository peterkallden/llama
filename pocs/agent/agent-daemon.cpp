#include "agent-daemon-adapter.h"
#include "agent-daemon-dispatcher.h"
#include "agent-daemon-jsonl-protocol.h"

#include "log.h"

#include <iostream>
#include <string>

namespace {

bool emit_agent_daemon_jsonl_message(
        const nlohmann::ordered_json & message) {
    std::string error;
    return write_agent_daemon_jsonl_message(stdout, message, error);
}

} // namespace

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
    common_agent_daemon_dispatcher dispatcher(std::move(runtime), options.queue_capacity);
    if (!emit_agent_daemon_jsonl_message(make_agent_daemon_ready_response(options))) {
        return 2;
    }

    std::string protocol_error;
    nlohmann::ordered_json parsed;
    while (read_agent_daemon_jsonl_message(stdin, parsed, protocol_error)) {
        if (!parsed.is_object()) {
            emit_agent_daemon_jsonl_message(make_agent_daemon_error_response("invalid JSON request"));
            continue;
        }

        common_agent_daemon_command command;
        if (!parse_agent_daemon_command(parsed, options, dispatcher.default_mode(), command, error)) {
            emit_agent_daemon_jsonl_message(make_agent_daemon_error_response(error));
            continue;
        }

        common_agent_daemon_command_result result;
        error.clear();
        dispatcher.execute(command, result, error);
        if (!error.empty() && result.error.empty()) {
            result.error = error;
        }
        emit_agent_daemon_jsonl_message(make_agent_daemon_command_response(result));
        if (dispatcher.shutdown_requested()) {
            break;
        }
    }

    if (!protocol_error.empty() && !std::feof(stdin)) {
        emit_agent_daemon_jsonl_message(make_agent_daemon_error_response(protocol_error));
    }

    return 0;
}
