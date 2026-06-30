#include "agent-cli-command.h"

#include "agent-cli-config.h"
#include "agent-cli-run.h"

#include <nlohmann/json.hpp>
#include <sheredom/subprocess.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

std::filesystem::path get_daemon_executable_path(const char * argv0) {
    std::filesystem::path argv_path = argv0 != nullptr ? std::filesystem::path(argv0) : std::filesystem::path();
    if (argv_path.has_parent_path()) {
        argv_path = std::filesystem::absolute(argv_path);
    } else {
        argv_path = std::filesystem::current_path() / argv_path;
    }
    const auto parent = argv_path.parent_path();
#ifdef _WIN32
    return parent / "llama-agent-daemon.exe";
#else
    return parent / "llama-agent-daemon";
#endif
}

std::vector<char *> to_cstr_vec(const std::vector<std::string> & values) {
    std::vector<char *> result;
    result.reserve(values.size() + 1);
    for (const auto & value : values) {
        result.push_back(const_cast<char *>(value.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

bool read_protocol_message(FILE * stream, json & out, std::string & error) {
    out = json();
    error.clear();

    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), stream) != nullptr) {
        std::string line(buffer);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        const auto parsed = json::parse(line, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            std::fprintf(stderr, "%s\n", line.c_str());
            continue;
        }

        out = parsed;
        return true;
    }

    error = "daemon closed before returning a protocol response";
    return false;
}

bool write_protocol_message(FILE * stream, const json & message, std::string & error) {
    error.clear();
    const std::string line = message.dump() + "\n";
    if (std::fwrite(line.data(), 1, line.size(), stream) != line.size()) {
        error = "failed to write daemon request";
        return false;
    }
    if (std::fflush(stream) != 0) {
        error = "failed to flush daemon request";
        return false;
    }
    return true;
}

int run_daemon_chat_command(const char * argv0, args a) {
    if (a.model.empty() || a.prompt.empty()) {
        print_agent_usage(argv0, "daemon-chat");
        return 1;
    }

    if (a.agent_inference_backend != "server-context") {
        std::fprintf(stderr, "daemon-chat currently requires --agent-inference-backend server-context\n");
        return 1;
    }

    std::string error;
    if (!validate_agent_memory_scope(a, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (a.memory_scope != "session") {
        std::fprintf(stderr, "daemon-chat currently supports only --memory-scope session\n");
        return 1;
    }
    if (a.planning_mode != "off") {
        std::fprintf(stderr, "daemon-chat currently supports only --planning-mode off\n");
        return 1;
    }

    const auto daemon_path = get_daemon_executable_path(argv0);
    if (!std::filesystem::exists(daemon_path)) {
        std::fprintf(stderr, "agent daemon executable not found: %s\n", daemon_path.string().c_str());
        return 1;
    }

    std::vector<std::string> command_line = {
        daemon_path.string(),
        "--model", a.model,
        "--default-mode", a.planning_mode == "mini" ? "mini" : "chat",
        "--n-predict", std::to_string(a.n_predict),
        "--n-gpu-layers", std::to_string(a.n_gpu_layers),
    };

    subprocess_s proc;
    const int options =
        subprocess_option_no_window |
        subprocess_option_enable_async |
        subprocess_option_combined_stdout_stderr |
        subprocess_option_inherit_environment;
    auto argv = to_cstr_vec(command_line);

    if (subprocess_create(argv.data(), options, &proc) != 0) {
        std::fprintf(stderr, "failed to spawn agent daemon\n");
        return 1;
    }

    int exit_code = 1;
    int result_code = 1;
    FILE * daemon_in = subprocess_stdin(&proc);
    FILE * daemon_out = subprocess_stdout(&proc);
    if (daemon_in == nullptr || daemon_out == nullptr) {
        std::fprintf(stderr, "failed to acquire daemon pipes\n");
        subprocess_terminate(&proc);
        subprocess_join(&proc, &exit_code);
        subprocess_destroy(&proc);
        return 1;
    }

    json response;
    json shutdown_response;
    bool joined = false;
    bool destroy_needed = true;

    do {
        if (!read_protocol_message(daemon_out, response, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            break;
        }
        if (!response.value("ok", false) || response.value("event", "") != "ready") {
            std::fprintf(stderr, "unexpected daemon ready response: %s\n", response.dump().c_str());
            break;
        }

        json request = {
            {"prompt", a.prompt},
            {"session_id", a.memory_session},
            {"namespace_id", a.memory_namespace},
            {"turn_id", a.memory_turn},
            {"n_predict", a.n_predict},
            {"mode", "chat"},
        };

        if (!write_protocol_message(daemon_in, request, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            break;
        }
        if (!read_protocol_message(daemon_out, response, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            break;
        }

        if (!write_protocol_message(daemon_in, json({{"command", "shutdown"}}), error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            break;
        }
        if (!read_protocol_message(daemon_out, shutdown_response, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            break;
        }

        subprocess_join(&proc, &exit_code);
        joined = true;

        if (!response.value("ok", false)) {
            std::fprintf(stderr, "%s\n", response.value("error", "daemon turn failed").c_str());
            result_code = 1;
            break;
        }
        if (!shutdown_response.value("ok", false) || shutdown_response.value("event", "") != "shutdown") {
            std::fprintf(stderr, "unexpected daemon shutdown response: %s\n", shutdown_response.dump().c_str());
            break;
        }
        if (exit_code != 0) {
            std::fprintf(stderr, "agent daemon exited with code %d\n", exit_code);
            break;
        }

        std::printf("%s\n", response.value("response", "").c_str());
        result_code = 0;
    } while (false);

    if (!joined) {
        subprocess_terminate(&proc);
        subprocess_join(&proc, &exit_code);
    }
    if (destroy_needed) {
        subprocess_destroy(&proc);
    }
    return result_code;
}

} // namespace

int run_memory_chat_command(const char * argv0, common_memory_store & store, args a) {
    if (a.model.empty() || a.prompt.empty()) {
        print_agent_usage(argv0, "chat");
        return 1;
    }

    std::string error;
    if (!validate_agent_memory_scope(a, error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    a.command = "chat";
    return run_agent_cli(store, a);
}

int run_agent_command_main(const char * argv0, int argc, char ** argv) {
    args a;
    if (!parse_agent_run_args(argc, argv, a)) {
        print_agent_usage(argv0);
        return 1;
    }

    if (a.command == "daemon-chat") {
        return run_daemon_chat_command(argv0, a);
    }

    if (a.model.empty() || a.prompt.empty()) {
        print_agent_usage(argv0);
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
