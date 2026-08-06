#include "agent-daemon-client.h"

#include "agent-daemon-client-admin.h"
#include "../cli/agent-cli-config.h"
#include "../cli/agent-cli-selection.h"
#include "../resource/agent-resource-store.h"
#include "agent-daemon-client-status.h"
#include "agent-daemon-jsonl-protocol.h"

#include <nlohmann/json.hpp>
#include <sheredom/subprocess.h>

#include <cstdio>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    if (!a.agent_runtime) {
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

void print_daemon_turn_summary(const agent_daemon_jsonl_turn_response & response) {
    if (!response.turn_summary.has_value()) return;
    const auto & summary = *response.turn_summary;
    auto join = [](const std::vector<std::string> & values) {
        std::string result;
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) result += ',';
            result += values[i];
        }
        return result;
    };
    std::fprintf(stderr,
        "turn-summary: mode=%s status=%s verified=%s phases=%s tools=%s unresolved=%zu\n",
        summary.mode.c_str(), summary.status.c_str(), summary.verified ? "yes" : "no",
        join(summary.phases).c_str(), join(summary.tools_used).c_str(), summary.unresolved_items);
}

bool split_daemon_session_command_argument(
        const std::string & line,
        const std::string & command,
        std::string & argument) {
    if (line == command) {
        argument.clear();
        return true;
    }
    const std::string prefix = command + " ";
    if (line.rfind(prefix, 0) != 0) {
        return false;
    }
    argument = line.substr(prefix.size());
    return true;
}

size_t count_listing_items(
        const agent_daemon_jsonl_listing_response & response,
        const char * field) {
    if (!response.payload.contains(field) || !response.payload[field].is_array()) {
        return 0;
    }
    return response.payload[field].size();
}

void print_daemon_resource_listing(
        const agent_daemon_jsonl_listing_response & response) {
    const auto count = count_listing_items(response, "resources");
    std::printf("[daemon-resources] count=%zu\n", count);

    if (!response.payload.contains("resources") ||
            !response.payload["resources"].is_array()) {
        return;
    }

    for (const auto & item : response.payload["resources"]) {
        if (!item.is_object()) {
            continue;
        }
        std::printf(
            "[daemon-resource-item] id=%s uri=%s name=%s mime=%s bytes=%zu scope=%s\n",
            item.value("resource_id", "").c_str(),
            item.value("uri", "").c_str(),
            item.value("name", "").c_str(),
            item.value("mime_type", "").c_str(),
            item.value("size_bytes", size_t{0}),
            item.value("scope", "").c_str());
    }
}

bool read_daemon_resource_file(
        const std::filesystem::path & path,
        std::string & text,
        std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "unable to open resource file: " + path.string();
        return false;
    }
    text.assign(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
    if (text.size() > 1024 * 1024) {
        error = "resource file exceeds the 1 MiB limit: " + path.string();
        return false;
    }
    if (text.find('\0') != std::string::npos) {
        error = "resource file contains NUL bytes: " + path.string();
        return false;
    }
    error.clear();
    return true;
}

std::string daemon_resource_mime_type(const std::filesystem::path & path) {
    auto extension = path.extension().string();
    for (auto & character : extension) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    if (extension == ".md" || extension == ".markdown") return "text/markdown";
    if (extension == ".json") return "application/json";
    if (extension == ".csv") return "text/csv";
    if (extension == ".html" || extension == ".htm") return "text/html";
    if (extension == ".xml") return "application/xml";
    return "text/plain";
}

void print_daemon_turn_failure(
        const agent_daemon_jsonl_turn_response & response,
        const std::string & fallback_error) {
    const auto summary =
        make_agent_daemon_client_turn_failure_summary(response, fallback_error);
    std::fprintf(
        stderr,
        "%s\n",
        render_agent_daemon_client_turn_failure_summary(summary).c_str());
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
            "--default-mode", a.agent_runtime ? "agent" : "chat",
            "--thinking-mode", a.thinking_mode,
            "--max-reflection-rounds", std::to_string(a.max_reflection_rounds),
            "--max-plan-revisions", std::to_string(a.max_plan_revisions),
            "--max-research-iterations", std::to_string(a.max_research_iterations),
            "--backend", a.backend,
            "--plan-backend", a.plan_backend,
            "--memory-learn", a.memory_learn,
            "--agent-plan", a.agent_plan,
            "--n-predict", std::to_string(a.n_predict),
            "--context-size", std::to_string(a.context_size),
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
        return make_admin_client().status(response, error);
    }

    bool list_sessions(agent_daemon_jsonl_status_response & response, std::string & error) {
        return make_admin_client().list_sessions(response, error);
    }

    bool get_session(
            const std::string & session_id,
            const std::string & namespace_id,
            agent_daemon_jsonl_status_response & response,
            std::string & error) {
        return make_admin_client().get_session(session_id, namespace_id, response, error);
    }

    bool list_resources(
            const agent_daemon_jsonl_list_resources_request & request,
            agent_daemon_jsonl_listing_response & response,
            std::string & error) {
        return make_admin_client().list_resources(request, response, error);
    }

    bool list_memories(
            const agent_daemon_jsonl_list_memories_request & request,
            agent_daemon_jsonl_listing_response & response,
            std::string & error) {
        return make_admin_client().list_memories(request, response, error);
    }

    bool list_plans(
            const agent_daemon_jsonl_list_plans_request & request,
            agent_daemon_jsonl_listing_response & response,
            std::string & error) {
        return make_admin_client().list_plans(request, response, error);
    }

    bool read_resource(
            const agent_daemon_jsonl_read_resource_request & request,
            agent_daemon_jsonl_resource_response & response,
            std::string & error) {
        return make_admin_client().read_resource(request, response, error);
    }

    bool put_resource(
            const agent_daemon_jsonl_put_resource_request & request,
            agent_daemon_jsonl_resource_response & response,
            std::string & error) {
        return make_admin_client().put_resource(request, response, error);
    }

    bool drain(agent_daemon_jsonl_lifecycle_response & response, std::string & error) {
        return make_admin_client().drain(response, error);
    }

    bool reset_session(
            const std::string & session_id,
            const std::string & namespace_id,
            agent_daemon_jsonl_lifecycle_response & response,
            std::string & error) {
        return make_admin_client().reset_session(session_id, namespace_id, response, error);
    }

    bool close_session(
            const std::string & session_id,
            const std::string & namespace_id,
            agent_daemon_jsonl_lifecycle_response & response,
            std::string & error) {
        return make_admin_client().close_session(session_id, namespace_id, response, error);
    }

    bool shutdown(std::string & error) {
        error.clear();
        if (!running) {
            return true;
        }

        agent_daemon_jsonl_lifecycle_response response;
        bool ok = make_admin_client().shutdown(response, error);
        if (ok &&
                response.status.state != "draining" &&
                response.status.state != "stopping" &&
                response.status.state != "stopped") {
            error = "unexpected daemon shutdown state: " + response.status.state;
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
            do {
                if (!read_agent_daemon_jsonl_message(output, message, error)) {
                    return false;
                }
            } while (message.value("message_type", std::string()) == "event");
            return true;
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

    agent_daemon_client_admin make_admin_client() {
        return agent_daemon_client_admin(
            [this](const json & request, json & response, std::string & error) {
                return send_request(request, response, error);
            });
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
    if (!a.agent_runtime && a.agent_plan != "off") {
        std::fprintf(stderr, "daemon commands require the agent runtime when --agent-plan is enabled\n");
        return false;
    }
    if (a.memory_learn == "post-turn" && !a.agent_runtime) {
        std::fprintf(stderr, "daemon commands require the agent runtime when --memory-learn post-turn is enabled\n");
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
        const std::string & turn_id = {},
        const std::vector<std::string> & resource_refs = {}) {
    agent_daemon_jsonl_turn_request request;
    request.prompt = prompt;
    request.session_id = a.memory_session;
    request.namespace_id = a.memory_namespace;
    request.project_id = a.memory_project;
    request.turn_id = turn_id.empty() ? a.memory_turn : turn_id;
    request.memory_scope = a.memory_scope;
    request.plan_scope = effective_plan_scope_for_daemon_request(a);
    request.n_predict = a.n_predict;
    request.mode = a.agent_runtime ? "agent" : "chat";
    request.resource_refs = resource_refs;
    request.include_summary = a.include_summary;
    request.turn_timeout_ms = a.turn_timeout_ms;
    request.inference_step_timeout_ms = a.inference_step_timeout_ms;
    request.tool_timeout_ms = a.tool_timeout_ms;
    request.mcp_connect_timeout_ms = a.mcp_connect_timeout_ms;
    request.mcp_request_timeout_ms = a.mcp_request_timeout_ms;
    request.mcp_shutdown_timeout_ms = a.mcp_shutdown_timeout_ms;
    return request;
}

bool import_daemon_cli_resources(
        agent_daemon_client_session & session,
        const args & a,
        const std::string & scope,
        std::vector<std::string> & resource_refs,
        std::string & error) {
    struct prepared_resource {
        std::filesystem::path path;
        std::string text;
        std::string mime_type;
    };
    std::vector<prepared_resource> prepared;
    prepared.reserve(a.resource_paths.size());
    for (const auto & resource_path : a.resource_paths) {
        const std::filesystem::path path(resource_path);
        std::string text;
        if (!read_daemon_resource_file(path, text, error)) {
            return false;
        }
        prepared.push_back({path, std::move(text), daemon_resource_mime_type(path)});
    }
    for (auto & item : prepared) {
        agent_daemon_jsonl_resource_response response;
        if (!session.put_resource({
                    item.path.filename().string(),
                    "Imported through daemon CLI --resource",
                    std::move(item.mime_type),
                    std::move(item.text),
                    scope,
                    a.memory_namespace,
                    a.memory_session,
                    a.memory_project,
                    a.memory_turn,
                },
                response,
                error)) {
            return false;
        }
        resource_refs.push_back(response.resource.uri);
    }
    return true;
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

    std::vector<std::string> resource_refs;
    if (!import_daemon_cli_resources(session, a, "turn", resource_refs, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        session.shutdown(error);
        return 1;
    }

    agent_daemon_jsonl_turn_response response;
    if (!session.run_turn(make_daemon_client_request(a, a.prompt, {}, resource_refs), response, error)) {
        print_daemon_turn_failure(response, error);
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
    if (a.include_summary) print_daemon_turn_summary(response);
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

    std::vector<std::string> resource_refs;
    if (!import_daemon_cli_resources(session, a, "session", resource_refs, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        session.shutdown(error);
        return 1;
    }

    int prompts_sent = 0;
    auto run_one = [&](const std::string & prompt, const std::string & turn_id = std::string()) -> bool {
        agent_daemon_jsonl_turn_response response;
        if (!session.run_turn(make_daemon_client_request(a, prompt, turn_id, resource_refs), response, error)) {
            print_daemon_turn_failure(response, error);
            return false;
        }
        std::printf("%s\n", response.response.c_str());
        if (a.include_summary) print_daemon_turn_summary(response);
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
        if (line == "/status" || line == "/status --verbose") {
            agent_daemon_jsonl_status_response response;
            if (!session.status(response, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            if (line == "/status --verbose") {
                std::printf("[daemon-status]\n%s\n",
                    render_agent_daemon_client_status_verbose(response).c_str());
            } else {
                const auto summary = make_agent_daemon_client_status_summary(response);
                std::printf("[daemon-status] %s\n", render_agent_daemon_client_status_summary(summary).c_str());
            }
            continue;
        }
        if (line == "/sessions") {
            agent_daemon_jsonl_status_response response;
            if (!session.list_sessions(response, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            const auto summary = make_agent_daemon_client_status_summary(response);
            std::printf("[daemon-sessions] %s\n", render_agent_daemon_client_status_summary(summary).c_str());
            continue;
        }
        if (line == "/session") {
            agent_daemon_jsonl_status_response response;
            if (!session.get_session(a.memory_session, a.memory_namespace, response, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            const auto summary = make_agent_daemon_client_status_summary(response);
            std::printf("[daemon-session] %s\n", render_agent_daemon_client_status_summary(summary).c_str());
            continue;
        }
        if (line == "/resources") {
            agent_daemon_jsonl_listing_response response;
            if (!session.list_resources({
                        a.memory_session,
                        a.memory_namespace,
                        a.memory_project,
                        a.memory_turn,
                    },
                    response,
                    error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            print_daemon_resource_listing(response);
            continue;
        }
        if (line == "/memories") {
            agent_daemon_jsonl_listing_response response;
            if (!session.list_memories({
                        a.memory_session,
                        a.memory_namespace,
                        a.memory_project,
                        a.memory_turn,
                    },
                    response,
                    error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            std::printf("[daemon-memories] count=%zu\n", count_listing_items(response, "memories"));
            continue;
        }
        if (line == "/plans") {
            agent_daemon_jsonl_listing_response response;
            if (!session.list_plans({
                        a.memory_session,
                        a.memory_namespace,
                        a.memory_project,
                        a.memory_turn,
                    },
                    response,
                    error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            std::printf("[daemon-plans] count=%zu\n", count_listing_items(response, "plans"));
            continue;
        }
        if (line == "/reset") {
            agent_daemon_jsonl_lifecycle_response response;
            if (!session.reset_session(a.memory_session, a.memory_namespace, response, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            const auto summary = make_agent_daemon_client_lifecycle_summary(response);
            std::printf("[daemon-reset] %s\n", render_agent_daemon_client_lifecycle_summary(summary).c_str());
            continue;
        }
        if (line == "/close") {
            agent_daemon_jsonl_lifecycle_response response;
            if (!session.close_session(a.memory_session, a.memory_namespace, response, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            const auto summary = make_agent_daemon_client_lifecycle_summary(response);
            std::printf("[daemon-close] %s\n", render_agent_daemon_client_lifecycle_summary(summary).c_str());
            continue;
        }
        if (line == "/drain") {
            agent_daemon_jsonl_lifecycle_response response;
            if (!session.drain(response, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            const auto summary = make_agent_daemon_client_lifecycle_summary(response);
            std::printf("[daemon-drain] %s\n", render_agent_daemon_client_lifecycle_summary(summary).c_str());
            continue;
        }
        std::string resource_uri;
        if (split_daemon_session_command_argument(line, "/resource-put", resource_uri)) {
            if (resource_uri.empty()) {
                std::fprintf(stderr, "daemon-session /resource-put requires a file path\n");
                continue;
            }
            const std::filesystem::path path(resource_uri);
            std::string text;
            if (!read_daemon_resource_file(path, text, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                continue;
            }
            agent_daemon_jsonl_resource_response response;
            if (!session.put_resource({
                        path.filename().string(),
                        "Imported through daemon-session JSONL admin",
                        daemon_resource_mime_type(path),
                        std::move(text),
                        "session",
                        a.memory_namespace,
                        a.memory_session,
                        a.memory_project,
                        a.memory_turn,
                    },
                    response,
                    error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            std::printf(
                "[daemon-resource-created] uri=%s mime=%s bytes=%zu scope=%s\n",
                response.resource.uri.c_str(),
                response.resource.mime_type.c_str(),
                response.resource.size_bytes,
                common_runtime_resource_scope_name(response.resource.scope));
            resource_refs.push_back(response.resource.uri);
            continue;
        }
        if (split_daemon_session_command_argument(line, "/resource", resource_uri)) {
            if (resource_uri.empty()) {
                std::fprintf(stderr, "daemon-session /resource requires a URI\n");
                continue;
            }
            agent_daemon_jsonl_resource_response response;
            if (!session.read_resource({
                        resource_uri,
                        a.memory_session,
                        a.memory_namespace,
                        a.memory_project,
                        a.memory_turn,
                        8192,
                    },
                    response,
                    error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                session.shutdown(error);
                return 1;
            }
            std::printf(
                "[daemon-resource] uri=%s mime=%s bytes=%zu content=%s\n",
                response.resource.uri.c_str(),
                response.resource.mime_type.c_str(),
                response.resource.size_bytes,
                response.content.c_str());
            continue;
        }
        if (line == "/help") {
            std::printf("[daemon-help] /status [/status --verbose] /sessions /session /resources /memories /plans /resource-put <path> /resource <uri> /reset /close /drain /quit\n");
            continue;
        }
        if (!line.empty() && line.front() == '/') {
            std::fprintf(stderr, "unknown daemon-session command: %s\n", line.c_str());
        std::printf("[daemon-help] /status [/status --verbose] /sessions /session /resources /memories /plans /resource-put <path> /resource <uri> /reset /close /drain /quit\n");
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
