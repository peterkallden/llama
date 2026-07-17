#include "agent-daemon-adapter.h"
#include "agent-daemon-dispatcher.h"

#include "log.h"

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
    if (!run_agent_daemon_jsonl_adapter(stdin, stdout, options, dispatcher, error)) {
        if (!error.empty()) {
            std::fprintf(stderr, "daemon adapter failed: %s\n", error.c_str());
        }
        return 2;
    }

    return 0;
}
