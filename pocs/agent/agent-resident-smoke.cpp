#include "agent-runtime-host.h"

#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct resident_smoke_options {
    std::string model;
    std::string first_prompt = "Reply with OK only.";
    std::string second_prompt = "Reply with DONE only.";
    int n_predict = 32;
    int n_gpu_layers = 0;
};

bool parse_args(int argc, char ** argv, resident_smoke_options & options) {
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "--model") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.model = value;
        } else if (std::strcmp(argv[i], "--first-prompt") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.first_prompt = value;
        } else if (std::strcmp(argv[i], "--second-prompt") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.second_prompt = value;
        } else if (std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n-predict") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.n_predict = std::stoi(value);
        } else if (std::strcmp(argv[i], "-ngl") == 0 || std::strcmp(argv[i], "--n-gpu-layers") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.n_gpu_layers = std::stoi(value);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    if (options.model.empty()) {
        std::fprintf(stderr, "--model is required\n");
        return false;
    }

    return true;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model MODEL [--first-prompt TEXT] [--second-prompt TEXT] [--n-predict N] [-ngl N]\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    resident_smoke_options options;
    if (!parse_args(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }

    common_memory_in_memory_store memory_store;
    common_plan_in_memory_store plan_store;
    std::string error;
    if (!memory_store.open("", error)) {
        std::fprintf(stderr, "failed to open in-memory store: %s\n", error.c_str());
        return 1;
    }
    if (!plan_store.open("", error)) {
        std::fprintf(stderr, "failed to open in-memory plan store: %s\n", error.c_str());
        return 1;
    }

    common_agent_runtime_session_host runtime(
        make_agent_runtime_session_host_config({
            memory_store,
            plan_store,
            {
                options.first_prompt,
                "resident-smoke-session",
                "resident-smoke",
                {},
                options.model,
                options.n_predict,
                options.n_gpu_layers,
                false,
                "server-context",
                common_memory_scope::session,
                common_plan_scope::turn,
            },
            {},
            common_memory_scope::session,
            false,
            {},
            {},
            false,
            nullptr,
        }));
    common_agent_result first_result;
    common_agent_result second_result;
    void * first_keepalive = nullptr;
    void * second_keepalive = nullptr;
    common_agent_runtime_session_host_turn_result runtime_result;

    if (!runtime.run_turn({
            common_agent_runtime_host_mode::chat,
            options.first_prompt,
            "resident-smoke-session",
            "resident-smoke",
            {},
            "turn-1",
            common_memory_scope::session,
            common_plan_scope::turn,
            options.n_predict,
        }, runtime_result, error)) {
        std::fprintf(stderr, "first resident turn failed: %s\n", error.c_str());
        return 1;
    }
    first_result.response = runtime_result.response;
    first_result.total_decoded_tokens = runtime_result.total_decoded_tokens;
    const auto * first_session = runtime.session();
    if (first_session == nullptr) {
        std::fprintf(stderr, "resident runtime session was not initialized after first turn\n");
        return 1;
    }
    first_keepalive = first_session->inference_session.keepalive.get();
    if (!runtime.run_turn({
            common_agent_runtime_host_mode::chat,
            options.second_prompt,
            "resident-smoke-session",
            "resident-smoke",
            {},
            "turn-2",
            common_memory_scope::session,
            common_plan_scope::turn,
            options.n_predict,
        }, runtime_result, error)) {
        std::fprintf(stderr, "second resident turn failed: %s\n", error.c_str());
        return 1;
    }
    second_result.response = runtime_result.response;
    second_result.total_decoded_tokens = runtime_result.total_decoded_tokens;
    const auto * second_session = runtime.session();
    if (second_session == nullptr) {
        std::fprintf(stderr, "resident runtime session was not initialized after second turn\n");
        return 1;
    }
    second_keepalive = second_session->inference_session.keepalive.get();

    if (first_keepalive == nullptr || second_keepalive == nullptr || first_keepalive != second_keepalive) {
        std::fprintf(stderr, "resident host did not reuse the same server_context keepalive across turns\n");
        return 1;
    }

    std::printf("resident_keepalive_reused=yes\n");
    std::printf("turn1_response=%s\n", first_result.response.c_str());
    std::printf("turn2_response=%s\n", second_result.response.c_str());
    std::printf("turn1_tokens=%d\n", first_result.total_decoded_tokens);
    std::printf("turn2_tokens=%d\n", second_result.total_decoded_tokens);
    return 0;
}
