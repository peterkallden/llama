#include "agent-runtime-session-host.h"

#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cstdio>
#include <memory>
#include <string>

int main() {
    common_memory_in_memory_store memory_store;
    common_plan_in_memory_store plan_store;

    auto call_count = std::make_shared<int>(0);
    common_agent_runtime_session_host host(make_agent_runtime_session_host_config({
        memory_store,
        plan_store,
        {
            "",
            "",
            "",
            "",
            "fake.gguf",
            32,
            0,
            false,
            "server-context",
            common_memory_scope::session,
            common_plan_scope::turn,
        },
        {},
        {},
        {},
        common_memory_scope::session,
        false,
        {},
        {},
        [call_count](
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            tooling = {};
            ++(*call_count);
            error = "resolver-call=" + std::to_string(*call_count) + " turn=" + request.turn_id;
            return false;
        },
    }));

    common_agent_runtime_session_host_turn_result first_result;
    std::string error;
    if (host.run_turn({
            common_agent_runtime_host_mode::chat,
            "hello",
            "session-a",
            "namespace-a",
            "",
            "turn-1",
            common_memory_scope::session,
            common_plan_scope::turn,
            0,
        }, first_result, error)) {
        std::fprintf(stderr, "session host tooling smoke unexpectedly succeeded on first resolver failure\n");
        return 1;
    }
    if (error.find("resolver-call=1 turn=turn-1") == std::string::npos) {
        std::fprintf(stderr, "first resolver failure did not preserve turn-specific diagnostics: %s\n", error.c_str());
        return 1;
    }

    common_agent_runtime_session_host_turn_result second_result;
    if (host.run_turn({
            common_agent_runtime_host_mode::chat,
            "world",
            "session-a",
            "namespace-a",
            "",
            "turn-2",
            common_memory_scope::session,
            common_plan_scope::turn,
            0,
        }, second_result, error)) {
        std::fprintf(stderr, "session host tooling smoke unexpectedly succeeded on second resolver failure\n");
        return 1;
    }
    if (error.find("resolver-call=2 turn=turn-2") == std::string::npos) {
        std::fprintf(stderr, "second resolver failure did not preserve turn-specific diagnostics: %s\n", error.c_str());
        return 1;
    }

    std::printf("session_host_tooling_calls=%d\n", *call_count);
    return 0;
}
