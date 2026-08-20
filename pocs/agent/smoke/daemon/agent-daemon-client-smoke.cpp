#include "tools/agent/daemon/agent-daemon-client.h"
#include "tools/agent/daemon/agent-daemon-client-admin.h"

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define DUP _dup
#define DUP2 _dup2
#define FILENO _fileno
#define CLOSE _close
#else
#include <unistd.h>
#define DUP dup
#define DUP2 dup2
#define FILENO fileno
#define CLOSE close
#endif

namespace {

std::filesystem::path get_fake_daemon_source_path(const char * argv0) {
    std::filesystem::path argv_path = argv0 != nullptr ? std::filesystem::path(argv0) : std::filesystem::path();
    if (argv_path.has_parent_path()) {
        argv_path = std::filesystem::absolute(argv_path);
    } else {
        argv_path = std::filesystem::current_path() / argv_path;
    }
#ifdef _WIN32
    return argv_path.parent_path() / "llama-agent-daemon-client-fake-daemon.exe";
#else
    return argv_path.parent_path() / "llama-agent-daemon-client-fake-daemon";
#endif
}

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

class scoped_stdio_redirect {
public:
    scoped_stdio_redirect(const std::filesystem::path & stdin_path, const std::filesystem::path & stdout_path) {
        stdin_dup = DUP(FILENO(stdin));
        stdout_dup = DUP(FILENO(stdout));
        stdin_file = std::fopen(stdin_path.string().c_str(), "rb");
        stdout_file = std::fopen(stdout_path.string().c_str(), "wb");
        if (stdin_dup < 0 || stdout_dup < 0 || stdin_file == nullptr || stdout_file == nullptr) {
            restore();
            return;
        }
        if (DUP2(FILENO(stdin_file), FILENO(stdin)) < 0 ||
                DUP2(FILENO(stdout_file), FILENO(stdout)) < 0) {
            restore();
            return;
        }
        active = true;
    }

    bool ok() const {
        return active;
    }

    ~scoped_stdio_redirect() {
        restore();
    }

private:
    void restore() {
        if (stdout_file != nullptr) {
            std::fflush(stdout);
        }
        if (stdin_dup >= 0) {
            DUP2(stdin_dup, FILENO(stdin));
        }
        if (stdout_dup >= 0) {
            DUP2(stdout_dup, FILENO(stdout));
        }
        if (stdin_dup >= 0) {
            CLOSE(stdin_dup);
            stdin_dup = -1;
        }
        if (stdout_dup >= 0) {
            CLOSE(stdout_dup);
            stdout_dup = -1;
        }
        if (stdin_file != nullptr) {
            std::fclose(stdin_file);
            stdin_file = nullptr;
        }
        if (stdout_file != nullptr) {
            std::fclose(stdout_file);
            stdout_file = nullptr;
        }
        active = false;
    }

    int stdin_dup = -1;
    int stdout_dup = -1;
    FILE * stdin_file = nullptr;
    FILE * stdout_file = nullptr;
    bool active = false;
};

} // namespace

int main(int argc, char ** argv) {
    const auto fake_daemon_source = get_fake_daemon_source_path(argc > 0 ? argv[0] : nullptr);
    if (!std::filesystem::exists(fake_daemon_source)) {
        std::fprintf(stderr, "fake daemon executable not found: %s\n", fake_daemon_source.string().c_str());
        return 1;
    }

    const auto unique_suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto temp_root =
        std::filesystem::temp_directory_path() /
        std::filesystem::path("llama-agent-daemon-client-smoke-" + unique_suffix);
    std::error_code ec;
    std::filesystem::create_directories(temp_root, ec);
    if (ec) {
        std::fprintf(stderr, "failed to prepare temp smoke directory: %s\n", ec.message().c_str());
        return 1;
    }

#ifdef _WIN32
    const auto fake_daemon_target = temp_root / "llama-agent-daemon.exe";
    const auto fake_argv0 = temp_root / "llama-agent-smoke.exe";
#else
    const auto fake_daemon_target = temp_root / "llama-agent-daemon";
    const auto fake_argv0 = temp_root / "llama-agent-smoke";
#endif
    const auto stdin_path = temp_root / "stdin.txt";
    const auto stdout_path = temp_root / "stdout.txt";
    const auto resource_path = temp_root / "cli-input.md";

    std::filesystem::copy_file(
        fake_daemon_source,
        fake_daemon_target,
        std::filesystem::copy_options::overwrite_existing,
        ec);
    if (ec) {
        std::fprintf(stderr, "failed to stage fake daemon executable: %s\n", ec.message().c_str());
        return 1;
    }

    {
        std::ofstream resource_stream(resource_path, std::ios::binary);
        resource_stream << "# CLI input\n";
        if (!resource_stream.good()) {
            std::fprintf(stderr, "failed to write CLI resource smoke input\n");
            return 1;
        }
    }

    {
        std::ofstream stdin_stream(stdin_path, std::ios::binary);
        stdin_stream << "/sessions\n";
        stdin_stream << "/session\n";
        stdin_stream << "/resources\n";
        stdin_stream << "/memories\n";
        stdin_stream << "/plans\n";
        stdin_stream << "/resource agent-resource://resource/r-1\n";
        stdin_stream << "/reset\n";
        stdin_stream << "/close\n";
        stdin_stream << "/drain\n";
        stdin_stream << "/quit\n";
    }

    args options;
    options.model = "fake.gguf";
    options.prompt = "hello";
    options.agent_inference_backend = "server-context";
    options.memory_namespace = "namespace-a";
    options.memory_session = "session-a";
    options.memory_project = "project-a";
    options.memory_scope = "session";
    options.plan_scope = "session";
    options.backend = "server-context";
    options.plan_backend = "in-memory";
    options.memory_learn = "off";
    options.agent_plan = "off";
    options.n_predict = 32;
    options.n_gpu_layers = 0;
    options.resource_paths = {resource_path.string()};

    int rc = 1;
    {
        scoped_stdio_redirect redirect(stdin_path, stdout_path);
        if (!redirect.ok()) {
            std::fprintf(stderr, "failed to redirect smoke stdio\n");
            return 1;
        }
        rc = run_daemon_session_command(fake_argv0.string().c_str(), options);
        std::fflush(stdout);
    }

    if (rc != 0) {
        std::fprintf(stderr, "run_daemon_session_command returned %d\n", rc);
        return 1;
    }

    std::ifstream stdout_stream(stdout_path, std::ios::binary);
    const std::string output(
        (std::istreambuf_iterator<char>(stdout_stream)),
        std::istreambuf_iterator<char>());

    if (!contains(output, "stub turn response resources=1")) {
        std::fprintf(stderr, "daemon client smoke missing turn response: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "[daemon-sessions]")) {
        std::fprintf(stderr, "daemon client smoke missing /sessions output: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "[daemon-session]")) {
        std::fprintf(stderr, "daemon client smoke missing /session output: %s\n", output.c_str());
        return 1;
    }
    if (!contains(
                output,
                "active_turn=turn-active/awaiting_inference:continue_immediately pending=inference(session host turn execution)")) {
        std::fprintf(stderr, "daemon client smoke missing pending active-turn summary: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "[daemon-resources] count=1")) {
        std::fprintf(stderr, "daemon client smoke missing /resources output: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "[daemon-memories] count=1")) {
        std::fprintf(stderr, "daemon client smoke missing /memories output: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "[daemon-plans] count=1")) {
        std::fprintf(stderr, "daemon client smoke missing /plans output: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "[daemon-drain] drain")) {
        std::fprintf(stderr, "daemon client smoke missing /drain output: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "[daemon-resource] uri=agent-resource://resource/r-1")) {
        std::fprintf(stderr, "daemon client smoke missing /resource output: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "content={\"results\":[\"stub\"]}")) {
        std::fprintf(stderr, "daemon client smoke missing resource content: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "[daemon-reset] session_reset")) {
        std::fprintf(stderr, "daemon client smoke missing /reset output: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "[daemon-close] session_closed")) {
        std::fprintf(stderr, "daemon client smoke missing /close output: %s\n", output.c_str());
        return 1;
    }
    if (!contains(output, "namespace-a/session-a@project-a#pack-a")) {
        std::fprintf(stderr, "daemon client smoke missing session binding summary: %s\n", output.c_str());
        return 1;
    }
    if (!contains(
                output,
                "namespace-a/session-a@project-a#pack-a{state=running}[active=turn-active/awaiting_inference:continue_immediately pending=inference(session host turn execution)]")) {
        std::fprintf(stderr, "daemon client smoke missing pending session binding summary: %s\n", output.c_str());
        return 1;
    }

    {
        agent_daemon_client_admin admin(
            [](const nlohmann::ordered_json & request,
               nlohmann::ordered_json & response,
               std::string & error) {
                error.clear();
                const auto command = request.value("command", "");
                if (command == "read_resource") {
                    response = {
                        {"ok", false},
                        {"event", "resource_not_found"},
                        {"error", "resource is not available in this scope"},
                        {"state", "ready"},
                        {"live", true},
                        {"ready", true},
                        {"worker_running", true},
                        {"accepting_commands", true},
                        {"shutdown_requested", false},
                        {"sessions", 1},
                        {"queued_commands", 0},
                        {"max_queue_size", 8},
                        {"queue_capacity_remaining", 8},
                        {"resource", {
                            {"resource_id", "missing"},
                            {"uri", request.value("uri", "agent-resource://resource/missing")},
                            {"name", "missing.json"},
                            {"description", "Missing resource"},
                            {"mime_type", "application/json"},
                            {"size_bytes", 0},
                            {"metadata", nlohmann::ordered_json::object()},
                        }},
                        {"content", ""},
                    };
                    return true;
                }
                if (command == "put_resource") {
                    response = {
                        {"ok", true},
                        {"event", "resource_created"},
                        {"state", "ready"},
                        {"live", true},
                        {"ready", true},
                        {"worker_running", true},
                        {"accepting_commands", true},
                        {"shutdown_requested", false},
                        {"sessions", 1},
                        {"queued_commands", 0},
                        {"max_queue_size", 8},
                        {"queue_capacity_remaining", 8},
                        {"resource", {
                            {"resource_id", "r-uploaded"},
                            {"uri", "agent-resource://resource/r-uploaded"},
                            {"name", request.value("name", "notes.md")},
                            {"description", request.value("description", "Uploaded notes")},
                            {"mime_type", request.value("mime_type", "text/markdown")},
                            {"size_bytes", request.value("text", "").size()},
                            {"scope", request.value("scope", "session")},
                            {"metadata", nlohmann::ordered_json::object()},
                        }},
                        {"content", ""},
                    };
                    return true;
                }
                if (command == "drain") {
                    response = {
                        {"ok", true},
                        {"event", "status"},
                        {"state", "ready"},
                        {"live", true},
                        {"ready", true},
                        {"worker_running", true},
                        {"accepting_commands", true},
                        {"shutdown_requested", false},
                        {"sessions", 1},
                        {"queued_commands", 0},
                        {"max_queue_size", 8},
                        {"queue_capacity_remaining", 8},
                    };
                    return true;
                }
                if (command == "list_resources") {
                    response = {
                        {"ok", true},
                        {"event", "status"},
                        {"state", "ready"},
                        {"live", true},
                        {"ready", true},
                        {"worker_running", true},
                        {"accepting_commands", true},
                        {"shutdown_requested", false},
                        {"sessions", 1},
                        {"queued_commands", 0},
                        {"max_queue_size", 8},
                        {"queue_capacity_remaining", 8},
                        {"resources", nlohmann::ordered_json::array()},
                    };
                    return true;
                }
                error = "synthetic status transport failure";
                response = nlohmann::ordered_json();
                return false;
            });

        std::string negative_error;
        agent_daemon_jsonl_resource_response resource_response;
        if (admin.read_resource(
                    {
                        "agent-resource://resource/missing",
                        "session-a",
                        "namespace-a",
                        "project-a",
                        "turn-a",
                        1024,
                    },
                    resource_response,
                    negative_error) ||
                !contains(negative_error, "resource is not available in this scope") ||
                !contains(negative_error, "resource_not_found")) {
            std::fprintf(
                stderr,
                "daemon client smoke did not preserve resource error context: %s\n",
                negative_error.c_str());
            return 1;
        }

        negative_error.clear();
        agent_daemon_jsonl_lifecycle_response drain_response;
        if (admin.drain(drain_response, negative_error) ||
                !contains(negative_error, "unexpected daemon drain response") ||
                !contains(negative_error, "\"event\":\"status\"")) {
            std::fprintf(
                stderr,
                "daemon client smoke did not reject mismatched drain response: %s\n",
                negative_error.c_str());
            return 1;
        }

        negative_error.clear();
        agent_daemon_jsonl_listing_response listing_response;
        if (admin.list_resources(
                    {
                        "session-a",
                        "namespace-a",
                        "project-a",
                        "turn-a",
                    },
                    listing_response,
                    negative_error) ||
                !contains(negative_error, "unexpected daemon list_resources response") ||
                !contains(negative_error, "\"event\":\"status\"")) {
            std::fprintf(
                stderr,
                "daemon client smoke did not reject mismatched list_resources response: %s\n",
                negative_error.c_str());
            return 1;
        }

        negative_error.clear();
        if (!admin.put_resource(
                    {
                        "notes.md",
                        "Uploaded notes",
                        "text/markdown",
                        "# Notes\n",
                        "session",
                        "namespace-a",
                        "session-a",
                        "project-a",
                        "turn-a",
                    },
                    resource_response,
                    negative_error) ||
                resource_response.event != "resource_created" ||
                resource_response.resource.name != "notes.md" ||
                resource_response.resource.mime_type != "text/markdown" ||
                resource_response.resource.scope != common_runtime_resource_scope::session) {
            std::fprintf(
                stderr,
                "daemon client smoke did not put resource through JSONL: %s\n",
                negative_error.c_str());
            return 1;
        }

        negative_error.clear();
        agent_daemon_jsonl_status_response status_response;
        if (admin.status(status_response, negative_error) ||
                !contains(negative_error, "synthetic status transport failure")) {
            std::fprintf(
                stderr,
                "daemon client smoke did not preserve status transport failure: %s\n",
                negative_error.c_str());
            return 1;
        }
    }

    std::printf("daemon_client_smoke_output=%s\n", output.c_str());
    return 0;
}
