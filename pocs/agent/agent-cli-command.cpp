#include "agent-cli-command.h"

#include "agent-cli-selection.h"
#include "agent-cli-config.h"
#include "agent-cli-run.h"

#include <nlohmann/json.hpp>
#include <sheredom/subprocess.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
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
            error = "daemon emitted a non-JSON protocol line: " + line;
            return false;
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

struct daemon_client_request {
    std::string prompt;
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    std::string turn_id;
    std::string memory_scope;
    std::string plan_scope;
    int n_predict = 0;
    std::string mode = "chat";
};

void forward_daemon_diagnostics(FILE * stream) {
    if (stream == nullptr) {
        return;
    }

    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), stream) != nullptr) {
        std::fwrite(buffer, 1, std::strlen(buffer), stderr);
        std::fflush(stderr);
    }
}

class agent_daemon_client_session {
public:
    bool start(const char * argv0, const args & a, std::string & error) {
        error.clear();
        if (running) {
            error = "daemon session already running";
            return false;
        }

        daemon_path = get_daemon_executable_path(argv0);
        if (!std::filesystem::exists(daemon_path)) {
            error = "agent daemon executable not found: " + daemon_path.string();
            return false;
        }

        command_line = {
            daemon_path.string(),
            "--model", a.model,
            "--default-mode", a.planning_mode == "mini" ? "mini" : "chat",
            "--planning-mode", a.planning_mode,
            "--reflection-mode", a.reflection_mode,
            "--memory-learn", a.memory_learn,
            "--agent-plan", a.agent_plan,
            "--n-predict", std::to_string(a.n_predict),
            "--n-gpu-layers", std::to_string(a.n_gpu_layers),
        };
        if (!a.embedding_model.empty()) {
            command_line.push_back("--embedding-model");
            command_line.push_back(a.embedding_model);
        }
        if (a.memory_learn_show_candidate) {
            command_line.push_back("--memory-learn-show-candidate");
        }
        if (a.memory_learn_min_confidence != 0.75f) {
            command_line.push_back("--memory-learn-min-confidence");
            command_line.push_back(std::to_string(a.memory_learn_min_confidence));
        }
        if (a.memory_learn_min_reuse != 0.65f) {
            command_line.push_back("--memory-learn-min-reuse");
            command_line.push_back(std::to_string(a.memory_learn_min_reuse));
        }
        if (a.plan_show_summary) {
            command_line.push_back("--plan-show-summary");
        }
        if (a.agent_trace) {
            command_line.push_back("--agent-trace");
        }
        auto argv = to_cstr_vec(command_line);
        const int options =
            subprocess_option_no_window |
            subprocess_option_enable_async |
            subprocess_option_inherit_environment;

        if (subprocess_create(argv.data(), options, &proc) != 0) {
            error = "failed to spawn agent daemon";
            return false;
        }

        daemon_in = subprocess_stdin(&proc);
        daemon_out = subprocess_stdout(&proc);
        daemon_err = subprocess_stderr(&proc);
        if (daemon_in == nullptr || daemon_out == nullptr || daemon_err == nullptr) {
            error = "failed to acquire daemon pipes";
            subprocess_terminate(&proc);
            subprocess_join(&proc, &exit_code);
            subprocess_destroy(&proc);
            daemon_in = nullptr;
            daemon_out = nullptr;
            daemon_err = nullptr;
            return false;
        }

        daemon_err_thread = std::thread(forward_daemon_diagnostics, daemon_err);

        json ready;
        if (!read_protocol_message(daemon_out, ready, error)) {
            terminate_if_running();
            return false;
        }
        if (!ready.value("ok", false) || ready.value("event", "") != "ready") {
            error = "unexpected daemon ready response: " + ready.dump();
            terminate_if_running();
            return false;
        }

        running = true;
        return true;
    }

    bool run_turn(
            const daemon_client_request & request,
            json & response,
            std::string & error) {
        response = json();
        if (!running || daemon_in == nullptr || daemon_out == nullptr) {
            error = "daemon session is not running";
            return false;
        }

        json protocol_request = {
            {"prompt", request.prompt},
            {"session_id", request.session_id},
            {"namespace_id", request.namespace_id},
            {"project_id", request.project_id},
            {"turn_id", request.turn_id},
            {"memory_scope", request.memory_scope},
            {"plan_scope", request.plan_scope},
            {"n_predict", request.n_predict},
            {"mode", request.mode},
        };

        if (!write_protocol_message(daemon_in, protocol_request, error)) {
            return false;
        }
        return read_protocol_message(daemon_out, response, error);
    }

    bool shutdown(std::string & error) {
        error.clear();
        if (!running) {
            return true;
        }

        json response;
        bool ok = write_protocol_message(daemon_in, json({{"command", "shutdown"}}), error) &&
                  read_protocol_message(daemon_out, response, error);
        if (ok && (!response.value("ok", false) || response.value("event", "") != "shutdown")) {
            error = "unexpected daemon shutdown response: " + response.dump();
            ok = false;
        }

        subprocess_join(&proc, &exit_code);
        subprocess_destroy(&proc);
        running = false;
        daemon_in = nullptr;
        daemon_out = nullptr;
        daemon_err = nullptr;
        if (daemon_err_thread.joinable()) {
            daemon_err_thread.join();
        }
        return ok && exit_code == 0;
    }

    ~agent_daemon_client_session() {
        std::string ignored;
        if (running) {
            terminate_if_running();
        }
    }

private:
    void terminate_if_running() {
        if (!daemon_in && !daemon_out && !running) {
            return;
        }
        subprocess_terminate(&proc);
        subprocess_join(&proc, &exit_code);
        subprocess_destroy(&proc);
        running = false;
        daemon_in = nullptr;
        daemon_out = nullptr;
        daemon_err = nullptr;
        if (daemon_err_thread.joinable()) {
            daemon_err_thread.join();
        }
    }

    std::filesystem::path daemon_path;
    std::vector<std::string> command_line;
    subprocess_s proc{};
    FILE * daemon_in = nullptr;
    FILE * daemon_out = nullptr;
    FILE * daemon_err = nullptr;
    std::thread daemon_err_thread;
    int exit_code = 1;
    bool running = false;
};

bool validate_daemon_command_args(const char * argv0, const args & a, bool require_prompt) {
    if (a.model.empty() || (require_prompt && a.prompt.empty())) {
        print_agent_usage(argv0, require_prompt ? "daemon-chat" : "daemon-session");
        return false;
    }
    if (a.agent_inference_backend != "server-context") {
        std::fprintf(stderr, "daemon-chat currently requires --agent-inference-backend server-context\n");
        return false;
    }

    std::string error;
    if (!validate_agent_memory_scope(a, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    common_plan_scope parsed_plan_scope = common_plan_scope::turn;
    if (!parse_plan_scope(a.plan_scope, parsed_plan_scope)) {
        std::fprintf(stderr, "unsupported plan scope: %s\n", a.plan_scope.c_str());
        return false;
    }
    if (a.planning_mode == "off" && a.agent_plan != "off") {
        std::fprintf(stderr, "daemon daemon commands require --planning-mode mini when --agent-plan is enabled\n");
        return false;
    }
    if (a.memory_learn == "post-turn" && a.planning_mode != "mini") {
        std::fprintf(stderr, "daemon daemon commands require --planning-mode mini when --memory-learn post-turn is enabled\n");
        return false;
    }
    if (a.memory_learn_min_confidence < 0.0f || a.memory_learn_min_confidence > 1.0f ||
            a.memory_learn_min_reuse < 0.0f || a.memory_learn_min_reuse > 1.0f) {
        std::fprintf(stderr, "memory learning thresholds must be between 0 and 1\n");
        return false;
    }
    return true;
}

daemon_client_request make_daemon_client_request(const args & a, const std::string & prompt, const std::string & turn_id = {}) {
    return {
        prompt,
        a.memory_session,
        a.memory_namespace,
        a.memory_project,
        turn_id.empty() ? a.memory_turn : turn_id,
        a.memory_scope,
        a.plan_scope,
        a.n_predict,
        a.planning_mode == "mini" ? "mini" : "chat",
    };
}

int run_daemon_chat_command(const char * argv0, const args & a) {
    if (!validate_daemon_command_args(argv0, a, true)) {
        return 1;
    }

    std::string error;
    agent_daemon_client_session session;
    if (!session.start(argv0, a, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    json response;
    if (!session.run_turn(make_daemon_client_request(a, a.prompt), response, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        session.shutdown(error);
        return 1;
    }
    if (!session.shutdown(error)) {
        if (!error.empty()) {
            std::fprintf(stderr, "%s\n", error.c_str());
        }
        return 1;
    }
    if (!response.value("ok", false)) {
        std::fprintf(stderr, "%s\n", response.value("error", "daemon turn failed").c_str());
        return 1;
    }
    std::printf("%s\n", response.value("response", "").c_str());
    return 0;
}

int run_daemon_session_command(const char * argv0, const args & a) {
    if (!validate_daemon_command_args(argv0, a, false)) {
        return 1;
    }

    std::string error;
    agent_daemon_client_session session;
    if (!session.start(argv0, a, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    int prompts_sent = 0;
    auto run_one = [&](const std::string & prompt, const std::string & turn_id = std::string()) -> bool {
        json response;
        if (!session.run_turn(make_daemon_client_request(a, prompt, turn_id), response, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return false;
        }
        if (!response.value("ok", false)) {
            std::fprintf(stderr, "%s\n", response.value("error", "daemon turn failed").c_str());
            return false;
        }
        std::printf("%s\n", response.value("response", "").c_str());
        ++prompts_sent;
        return true;
    };

    if (!a.prompt.empty() && !run_one(a.prompt)) {
        session.shutdown(error);
        return 1;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        if (line == "/quit" || line == "/exit") {
            break;
        }
        if (!run_one(line)) {
            session.shutdown(error);
            return 1;
        }
    }

    if (!session.shutdown(error)) {
        if (!error.empty()) {
            std::fprintf(stderr, "%s\n", error.c_str());
        }
        return 1;
    }

    if (prompts_sent == 0) {
        std::fprintf(stderr, "daemon-session requires at least one prompt via --prompt or stdin\n");
        return 1;
    }
    return 0;
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
    if (a.command == "daemon-session") {
        return run_daemon_session_command(argv0, a);
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
