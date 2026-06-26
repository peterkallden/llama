#include "agent-cli-config.h"
#include "agent-cli-run.h"

int main(int argc, char ** argv) {
    args a;
    if (!parse_agent_run_args(argc, argv, a)) {
        print_agent_usage(argv[0]);
        return 1;
    }

    if (a.model.empty() || a.prompt.empty()) {
        print_agent_usage(argv[0]);
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
