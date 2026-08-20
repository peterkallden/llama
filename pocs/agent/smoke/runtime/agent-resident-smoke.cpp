#include "tools/agent/runtime/agent-runtime-host.h"
#include "tools/agent/runtime/agent-runtime-session-host.h"

#include "log.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define fileno _fileno
inline int close_fd(int fd) { return _close(fd); }
#else
#include <unistd.h>
inline int close_fd(int fd) { return ::close(fd); }
#endif

namespace {

struct resident_smoke_options {
    std::string model;
    std::string first_prompt = "Reply with OK only.";
    std::string second_prompt = "Reply with DONE only.";
    int n_predict = 32;
    int second_n_predict = 0;
    int n_gpu_layers = 0;
    bool verbose_logs = false;
};

struct resident_smoke_log_guard {
    ggml_log_callback callback = nullptr;
    void * user_data = nullptr;

    resident_smoke_log_guard() {
        llama_log_get(&callback, &user_data);
        llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
            auto * guard = static_cast<resident_smoke_log_guard *>(user_data);
            const ggml_log_level effective_level =
                level >= GGML_LOG_LEVEL_WARN ? level : GGML_LOG_LEVEL_DEBUG;
            if (guard->callback != nullptr) {
                guard->callback(effective_level, text, guard->user_data);
            }
        }, this);
    }

    ~resident_smoke_log_guard() {
        llama_log_set(callback, user_data);
    }
};

struct resident_smoke_stdio_guard {
    explicit resident_smoke_stdio_guard(bool enabled)
        : enabled(enabled) {
        if (!enabled) {
            return;
        }

        capture = std::tmpfile();
        if (capture == nullptr) {
            enabled = false;
            return;
        }

        stdout_fd = dup(fileno(stdout));
        stderr_fd = dup(fileno(stderr));
        if (stdout_fd < 0 || stderr_fd < 0) {
            restore();
            enabled = false;
            return;
        }

        std::fflush(stdout);
        std::fflush(stderr);
        dup2(fileno(capture), fileno(stdout));
        dup2(fileno(capture), fileno(stderr));
    }

    ~resident_smoke_stdio_guard() {
        restore();
    }

    void dump_to(FILE * stream) {
        if (!capture || !stream) {
            return;
        }
        std::fflush(capture);
        std::fseek(capture, 0, SEEK_SET);
        char buffer[4096];
        while (std::fgets(buffer, sizeof(buffer), capture) != nullptr) {
            std::fputs(buffer, stream);
        }
        std::fflush(stream);
        std::fseek(capture, 0, SEEK_END);
    }

private:
    void restore() {
        if (!enabled) {
            return;
        }
        std::fflush(stdout);
        std::fflush(stderr);
        if (stdout_fd >= 0) {
            dup2(stdout_fd, fileno(stdout));
            close_fd(stdout_fd);
            stdout_fd = -1;
        }
        if (stderr_fd >= 0) {
            dup2(stderr_fd, fileno(stderr));
            close_fd(stderr_fd);
            stderr_fd = -1;
        }
        if (capture != nullptr) {
            std::fclose(capture);
            capture = nullptr;
        }
        enabled = false;
    }

    bool enabled = false;
    FILE * capture = nullptr;
    int stdout_fd = -1;
    int stderr_fd = -1;
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
        } else if (std::strcmp(argv[i], "--second-n-predict") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.second_n_predict = std::stoi(value);
        } else if (std::strcmp(argv[i], "-ngl") == 0 || std::strcmp(argv[i], "--n-gpu-layers") == 0) {
            const char * value = need_value(argv[i]); if (!value) return false; options.n_gpu_layers = std::stoi(value);
        } else if (std::strcmp(argv[i], "--verbose-logs") == 0) {
            options.verbose_logs = true;
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
        "usage: %s --model MODEL [--first-prompt TEXT] [--second-prompt TEXT] [--verbose-logs] [--n-predict N] [--second-n-predict N] [-ngl N]\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    resident_smoke_options options;
    if (!parse_args(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }
    common_log_set_verbosity_thold(LOG_LEVEL_WARN);
    resident_smoke_log_guard log_guard;

    common_memory_in_memory_store memory_store;
    common_plan_in_memory_store plan_store;
    std::string error;
    common_agent_result first_result;
    common_agent_result second_result;
    void * first_keepalive = nullptr;
    void * second_keepalive = nullptr;

    const auto run_smoke = [&]() -> bool {
        if (!memory_store.open("", error)) {
            error = "failed to open in-memory store: " + error;
            return false;
        }
        if (!plan_store.open("", error)) {
            error = "failed to open in-memory plan store: " + error;
            return false;
        }

        common_agent_runtime_policy policy;
        policy.agent_inference_backend = "server-context";
        common_agent_runtime_config runtime_config;
        runtime_config.generation_config.n_predict = options.n_predict;
        common_agent_orchestration_config orchestration_config;
        orchestration_config.prompt = options.first_prompt;

        common_agent_runtime_session_host runtime(
            make_agent_runtime_session_host_config({
                memory_store,
                plan_store,
                {
                    options.first_prompt,
                    "resident-smoke-session",
                    "resident-smoke",
                    {},
                    std::nullopt,
                    options.model,
                    options.n_predict,
                    options.n_gpu_layers,
                    false,
                    "server-context",
                    common_memory_scope::session,
                    common_plan_scope::turn,
                },
                std::move(policy),
                std::move(runtime_config),
                std::move(orchestration_config),
                common_memory_scope::session,
                false,
                {},
                {},
            }));
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
            error = "first resident turn failed: " + error;
            return false;
        }
        first_result.response = runtime_result.response;
        first_result.total_decoded_tokens = runtime_result.total_decoded_tokens;
        const auto * first_session = runtime.session();
        if (first_session == nullptr) {
            error = "resident runtime session was not initialized after first turn";
            return false;
        }
        const auto * inference_session = first_session->active_inference_session();
        if (inference_session == nullptr) {
            std::fprintf(stderr, "first resident session did not retain an active inference context\n");
            return 1;
        }
        first_keepalive = inference_session->keepalive.get();

        if (!runtime.run_turn({
                common_agent_runtime_host_mode::chat,
                options.second_prompt,
                "resident-smoke-session",
                "resident-smoke",
                {},
                "turn-2",
                common_memory_scope::session,
                common_plan_scope::turn,
                options.second_n_predict > 0 ? options.second_n_predict : options.n_predict,
            }, runtime_result, error)) {
            error = "second resident turn failed: " + error;
            return false;
        }
        second_result.response = runtime_result.response;
        second_result.total_decoded_tokens = runtime_result.total_decoded_tokens;
        const auto * second_session = runtime.session();
        if (second_session == nullptr) {
            error = "resident runtime session was not initialized after second turn";
            return false;
        }
        const auto * second_inference_session = second_session->active_inference_session();
        if (second_inference_session == nullptr) {
            std::fprintf(stderr, "second resident session did not retain an active inference context\n");
            return 1;
        }
        second_keepalive = second_inference_session->keepalive.get();

        if (first_keepalive == nullptr || second_keepalive == nullptr || first_keepalive != second_keepalive) {
            error = "resident host did not reuse the same server_context keepalive across turns";
            return false;
        }

        return true;
    };

    bool ok = false;
    {
        resident_smoke_stdio_guard output_guard(!options.verbose_logs);
        ok = run_smoke();
        if (!ok) {
            output_guard.dump_to(stderr);
        }
    }
    if (!ok) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    std::printf("resident_keepalive_reused=yes\n");
    std::printf("turn1_response=%s\n", first_result.response.c_str());
    std::printf("turn2_response=%s\n", second_result.response.c_str());
    std::printf("turn1_tokens=%d\n", first_result.total_decoded_tokens);
    std::printf("turn2_tokens=%d\n", second_result.total_decoded_tokens);
    std::printf("turn2_n_predict=%d\n", options.second_n_predict > 0 ? options.second_n_predict : options.n_predict);
    return 0;
}
