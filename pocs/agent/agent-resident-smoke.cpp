#include "agent-runtime-host.h"

#include "memory/memory-in-memory.h"

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

common_agent_runtime_turn_request make_turn_request(
        const resident_smoke_options & options,
        const std::string & prompt,
        const std::string & turn_id) {
    common_agent_runtime_turn_request turn_request;
    turn_request.request.prompt = prompt;
    turn_request.request.messages = {{"user", prompt}};
    turn_request.request.turn_id = turn_id;
    turn_request.request.session_id = "resident-smoke-session";
    turn_request.request.namespace_id = "resident-smoke";
    turn_request.scope.namespace_id = "resident-smoke";
    turn_request.scope.session_id = "resident-smoke-session";
    turn_request.scope.turn_id = turn_id;
    turn_request.scope.memory_scope = common_memory_scope::session;
    turn_request.scope.plan_scope = common_plan_scope::turn;
    turn_request.inference_options.model = options.model;
    turn_request.inference_options.n_predict = options.n_predict;
    turn_request.inference_options.n_gpu_layers = options.n_gpu_layers;
    turn_request.inference_options.fit_params = false;
    turn_request.policy.agent_inference_backend = "server-context";
    turn_request.orchestration_config.prompt = prompt;
    turn_request.generation_options.n_predict = options.n_predict;
    return turn_request;
}

bool run_turn(
        common_agent_runtime_resident_host & host,
        common_memory_store & memory_store,
        const resident_smoke_options & options,
        const std::string & prompt,
        const std::string & turn_id,
        common_agent_result & result,
        void *& keepalive_ptr,
        std::string & error) {
    const std::vector<common_memory_hit> memories;
    const std::vector<common_chat_tool> tools;
    common_agent_runtime_host_build_context build_context{
        memory_store,
        nullptr,
        make_turn_request(options, prompt, turn_id),
        nullptr,
        nullptr,
        memories,
        tools,
        false,
        nullptr,
        {},
    };
    auto inputs = make_agent_runtime_host_chat_inputs(build_context);
    if (!host.run_turn(inputs, result, error)) {
        return false;
    }
    keepalive_ptr = host.session().inference_session.keepalive.get();
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    resident_smoke_options options;
    if (!parse_args(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }

    common_memory_in_memory_store memory_store;
    std::string error;
    if (!memory_store.open("", error)) {
        std::fprintf(stderr, "failed to open in-memory store: %s\n", error.c_str());
        return 1;
    }

    common_agent_runtime_resident_host host;
    common_agent_result first_result;
    common_agent_result second_result;
    void * first_keepalive = nullptr;
    void * second_keepalive = nullptr;

    if (!run_turn(host, memory_store, options, options.first_prompt, "turn-1", first_result, first_keepalive, error)) {
        std::fprintf(stderr, "first resident turn failed: %s\n", error.c_str());
        return 1;
    }
    if (!run_turn(host, memory_store, options, options.second_prompt, "turn-2", second_result, second_keepalive, error)) {
        std::fprintf(stderr, "second resident turn failed: %s\n", error.c_str());
        return 1;
    }

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
