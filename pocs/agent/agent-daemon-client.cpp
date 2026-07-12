#include "agent-daemon-client.h"

#include "agent-cli-config.h"
#include "agent-cli-selection.h"
#include "agent-daemon-jsonl-protocol.h"
#include "agent-resource-store.h"

#include <nlohmann/json.hpp>
#include <sheredom/subprocess.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string_view>
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

std::string default_plan_scope_for_memory_scope(const std::string & memory_scope) {
    if (memory_scope == "turn" || memory_scope == "session" ||
            memory_scope == "project" || memory_scope == "global") {
        return memory_scope;
    }
    return "session";
}

std::string effective_plan_scope_for_daemon_request(const args & a) {
    if (a.plan_scope != "turn") {
        return a.plan_scope;
    }
    if (a.planning_mode == "off") {
        return default_plan_scope_for_memory_scope(a.memory_scope);
    }
    return a.plan_scope;
}

std::string normalize_daemon_session_line(std::string line) {
    if (line.size() >= 3 &&
            (unsigned char) line[0] == 0xEF &&
            (unsigned char) line[1] == 0xBB &&
            (unsigned char) line[2] == 0xBF) {
        line.erase(0, 3);
    }
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.pop_back();
    }
    return line;
}

bool daemon_diagnostic_has_prefix(
        std::string_view line,
        std::string_view prefix) {
    return line.find(prefix) != std::string_view::npos;
}

bool is_routine_daemon_diagnostic(std::string_view line) {
    static const std::string_view noisy_prefixes[] = {
        "llama_model_loader:",
        "print_info:",
        "init_tokenizer:",
        "load:",
        "load_tensors:",
        "create_tensor:",
        "done_getting_tensors:",
        "repack:",
        "llama_context:",
        "llama_kv_cache:",
        "sched_reserve:",
        "graph_reserve:",
        "set_adapters_lora:",
        "adapters_lora_are_same:",
        "set_embeddings:",
        "set_abort_callback:",
        "common_speculative_init:",
        "srv          init:",
        "~llama_context:",
        ".",
    };

    for (const auto & prefix : noisy_prefixes) {
        if (daemon_diagnostic_has_prefix(line, prefix)) {
            return true;
        }
    }
    return false;
}

bool should_forward_daemon_diagnostic(
        std::string_view line,
        bool verbose) {
    if (verbose) {
        return true;
    }

    if (is_routine_daemon_diagnostic(line)) {
        return false;
    }

    return true;
}

void forward_daemon_diagnostics(FILE * stream, bool verbose) {
    if (stream == nullptr) {
        return;
    }

    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), stream) != nullptr) {
        std::string line(buffer);
        if (!should_forward_daemon_diagnostic(line, verbose)) {
            continue;
        }
        std::fprintf(stderr, "[daemon-stderr] %s", line.c_str());
        if (line.empty() || line.back() != '\n') {
            std::fputc('\n', stderr);
        }
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
            "--backend", a.backend,
            "--plan-backend", a.plan_backend,
            "--planning-mode", a.planning_mode,
            "--reflection-mode", a.reflection_mode,
            "--memory-learn", a.memory_learn,
            "--agent-plan", a.agent_plan,
            "--n-predict", std::to_string(a.n_predict),
            "--n-gpu-layers", std::to_string(a.n_gpu_layers),
        };
        if (!a.memory_db.empty()) {
            command_line.push_back("--memory-db");
            command_line.push_back(a.memory_db);
        }
        if (!a.plan_db.empty()) {
            command_line.push_back("--plan-db");
            command_line.push_back(a.plan_db);
        }
        if (!a.embedding_model.empty()) {
            command_line.push_back("--embedding-model");
            command_line.push_back(a.embedding_model);
        }
        if (!a.tool_profile.empty()) {
            command_line.push_back("--tool-profile");
            command_line.push_back(a.tool_profile);
        }
        if (!a.repository_root.empty()) {
            command_line.push_back("--repository-root");
            command_line.push_back(a.repository_root);
        }
        if (a.resource_blob_backend != "auto") {
            command_line.push_back("--resource-blob-backend");
            command_line.push_back(a.resource_blob_backend);
        }
        if (!a.resource_blob_root.empty()) {
            command_line.push_back("--resource-blob-root");
            command_line.push_back(a.resource_blob_root);
        }
        if (a.resource_metadata_backend != "auto") {
            command_line.push_back("--resource-metadata-backend");
            command_line.push_back(a.resource_metadata_backend);
        }
        if (!a.resource_metadata_db.empty()) {
            command_line.push_back("--resource-metadata-db");
            command_line.push_back(a.resource_metadata_db);
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
        if (a.max_tool_rounds > 0) {
            command_line.push_back("--max-tool-rounds");
            command_line.push_back(std::to_string(a.max_tool_rounds));
        }
        if (!a.mcp_tool_command.empty()) {
            command_line.push_back("--mcp-tool-command");
            command_line.push_back(a.mcp_tool_command);
        }
        for (const auto & mcp_arg : a.mcp_tool_args) {
            command_line.push_back("--mcp-tool-arg");
            command_line.push_back(mcp_arg);
        }
        if (!a.mcp_tool_server_name.empty() && a.mcp_tool_server_name != "mcp") {
            command_line.push_back("--mcp-tool-server-name");
            command_line.push_back(a.mcp_tool_server_name);
        }
        if (!a.mcp_tool_prefix.empty()) {
            command_line.push_back("--mcp-tool-prefix");
            command_line.push_back(a.mcp_tool_prefix);
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
        transport = jsonl_transport(daemon_in, daemon_out);

        daemon_err_thread = std::thread(forward_daemon_diagnostics, daemon_err, a.agent_trace);

        json ready;
        if (!transport.read(ready, error)) {
            terminate_if_running();
            return false;
        }
        agent_daemon_jsonl_ready_response ready_response;
        if (!parse_agent_daemon_jsonl_ready_response(ready, ready_response, error)) {
            error += ": " + ready.dump();
            terminate_if_running();
            return false;
        }

        running = true;
        return true;
    }

    bool run_turn(
            const agent_daemon_jsonl_turn_request & request,
            agent_daemon_jsonl_turn_response & response,
            std::string & error) {
        json message;
        if (!send_request(make_agent_daemon_jsonl_turn_request(request), message, error)) {
            return false;
        }
        return parse_agent_daemon_jsonl_turn_response(message, response, error);
    }

    bool status(agent_daemon_jsonl_status_response & response, std::string & error) {
        json message;
        if (!send_request(make_agent_daemon_jsonl_status_request({}), message, error)) {
            return false;
        }
        return parse_agent_daemon_jsonl_status_response(message, response, error);
    }

    bool reset_session(
            const std::string & session_id,
            const std::string & namespace_id,
            agent_daemon_jsonl_event_response & response,
            std::string & error) {
        json message;
        const bool ok = send_request(
            make_agent_daemon_jsonl_session_request({
                "reset_session",
                session_id,
                namespace_id,
            }),
            message,
            error);
        if (ok && !parse_agent_daemon_jsonl_event_response(message, response, error)) {
            error += ": " + message.dump();
            return false;
        }
        if (ok && response.event != "session_reset") {
            error = "unexpected daemon session_reset response: " + message.dump();
            return false;
        }
        return ok;
    }

    bool close_session(
            const std::string & session_id,
            const std::string & namespace_id,
            agent_daemon_jsonl_event_response & response,
            std::string & error) {
        json message;
        const bool ok = send_request(
            make_agent_daemon_jsonl_session_request({
                "close_session",
                session_id,
                namespace_id,
            }),
            message,
            error);
        if (ok && !parse_agent_daemon_jsonl_event_response(message, response, error)) {
            error += ": " + message.dump();
            return false;
        }
        if (ok && response.event != "session_closed") {
            error = "unexpected daemon session_closed response: " + message.dump();
            return false;
        }
        return ok;
    }

    bool shutdown(std::string & error) {
        error.clear();
        if (!running) {
            return true;
        }

        json response;
        bool ok = send_request(
                      make_agent_daemon_jsonl_shutdown_request({}),
                      response,
                      error);
        if (ok && !parse_agent_daemon_jsonl_event_response(response, "shutdown", error)) {
            error += ": " + response.dump();
            ok = false;
        }

        subprocess_join(&proc, &exit_code);
        subprocess_destroy(&proc);
        running = false;
        daemon_in = nullptr;
        daemon_out = nullptr;
        daemon_err = nullptr;
        transport = jsonl_transport();
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
    class jsonl_transport {
    public:
        jsonl_transport() = default;

        jsonl_transport(FILE * input, FILE * output) :
                input(input),
                output(output) {
        }

        bool write(const json & message, std::string & error) const {
            if (input == nullptr) {
                error = "daemon input pipe is not available";
                return false;
            }
            return write_agent_daemon_jsonl_message(input, message, error);
        }

        bool read(json & message, std::string & error) const {
            if (output == nullptr) {
                error = "daemon output pipe is not available";
                return false;
            }
            return read_agent_daemon_jsonl_message(output, message, error);
        }

    private:
        FILE * input = nullptr;
        FILE * output = nullptr;
    };

    bool send_request(
            const json & request,
            json & response,
            std::string & error) {
        response = json();
        if (!running) {
            error = "daemon session is not running";
            return false;
        }

        if (!transport.write(request, error)) {
            return false;
        }
        return transport.read(response, error);
    }
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
        transport = jsonl_transport();
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
    jsonl_transport transport;
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
        std::fprintf(stderr, "daemon commands require --planning-mode mini when --agent-plan is enabled\n");
        return false;
    }
    if (a.memory_learn == "post-turn" && a.planning_mode != "mini") {
        std::fprintf(stderr, "daemon commands require --planning-mode mini when --memory-learn post-turn is enabled\n");
        return false;
    }
    if (a.memory_learn_min_confidence < 0.0f || a.memory_learn_min_confidence > 1.0f ||
            a.memory_learn_min_reuse < 0.0f || a.memory_learn_min_reuse > 1.0f) {
        std::fprintf(stderr, "memory learning thresholds must be between 0 and 1\n");
        return false;
    }
    if (!validate_agent_resource_store_config({
            a.resource_blob_backend,
            a.resource_blob_root,
            a.resource_metadata_backend,
            a.resource_metadata_db,
        }, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    return true;
}

agent_daemon_jsonl_turn_request make_daemon_client_request(
        const args & a,
        const std::string & prompt,
        const std::string & turn_id = {}) {
    return {
        prompt,
        a.memory_session,
        a.memory_namespace,
        a.memory_project,
        turn_id.empty() ? a.memory_turn : turn_id,
        a.memory_scope,
        effective_plan_scope_for_daemon_request(a),
        a.n_predict,
        a.planning_mode == "mini" ? "mini" : "chat",
    };
}

} // namespace

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

    agent_daemon_jsonl_turn_response response;
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
    std::printf("%s\n", response.response.c_str());
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
        agent_daemon_jsonl_turn_response response;
        if (!session.run_turn(make_daemon_client_request(a, prompt, turn_id), response, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return false;
        }
        std::printf("%s\n", response.response.c_str());
        ++prompts_sent;
        return true;
    };

    if (!a.prompt.empty() && !run_one(a.prompt)) {
        session.shutdown(error);
        return 1;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        line = normalize_daemon_session_line(std::move(line));
        if (line.empty()) {
            continue;
        }
        if (line == "/quit" || line == "/exit") {
            break;
        }
        if (line == "/status") {
            agent_daemon_jsonl_status_response response;
            if (!session.status(response, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            std::printf("[daemon-status] %s\n", response.payload.dump().c_str());
            continue;
        }
        if (line == "/reset") {
            agent_daemon_jsonl_event_response response;
            if (!session.reset_session(a.memory_session, a.memory_namespace, response, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            std::printf("[daemon-reset] %s\n", response.event.c_str());
            continue;
        }
        if (line == "/close") {
            agent_daemon_jsonl_event_response response;
            if (!session.close_session(a.memory_session, a.memory_namespace, response, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            std::printf("[daemon-close] %s\n", response.event.c_str());
            continue;
        }
        if (line == "/help") {
            std::printf("[daemon-help] /status /reset /close /quit\n");
            continue;
        }
        if (!line.empty() && line.front() == '/') {
            std::fprintf(stderr, "unknown daemon-session command: %s\n", line.c_str());
            std::printf("[daemon-help] /status /reset /close /quit\n");
            continue;
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
