#include "agent-clangd-session.h"

#include <sheredom/subprocess.h>

#include <cstdio>
#include <exception>
#include <future>
#include <map>
#include <thread>
#include <utility>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
std::vector<char *> argv_for(const std::vector<std::string> & values) {
    std::vector<char *> result;
    for (const auto & value : values) result.push_back(const_cast<char *>(value.c_str()));
    result.push_back(nullptr);
    return result;
}
}

struct agent_clangd_session::impl {
    subprocess_s process{};
    FILE * input = nullptr;
    FILE * output = nullptr;
    bool running = false;
    bool initialized = false;
    bool joined = false;
    int exit_code = 1;
    int64_t next_id = 1;
    agent_clangd_message_decoder decoder;
    struct document { int64_t version; std::string text; };
    std::map<std::string, document> open_documents;
};

agent_clangd_session::agent_clangd_session(agent_clangd_session_config config)
    : config_(std::move(config)), state_(std::make_unique<impl>()) {}

agent_clangd_session::~agent_clangd_session() { shutdown(); }

bool agent_clangd_session::initialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_->initialized;
}

bool agent_clangd_session::ensure_started(std::string & error) {
    if (state_->initialized) return true;
    if (!state_->running) {
        if (config_.executable.empty()) { error = "clangd executable is empty"; return false; }
        // Keep clangd's workspace index enabled: workspace/symbol depends on
        // it, while the location-based queries are opened explicitly below.
        std::vector<std::string> command = {config_.executable, "--offset-encoding=utf-16"};
        if (!config_.compile_commands_dir.empty() && config_.compile_commands_dir != "auto") command.push_back("--compile-commands-dir=" + config_.compile_commands_dir);
        auto argv = argv_for(command);
        if (subprocess_create(argv.data(), subprocess_option_no_window | subprocess_option_enable_async |
                subprocess_option_inherit_environment | subprocess_option_search_user_path,
                &state_->process) != 0) {
            error = "failed to start clangd";
            return false;
        }
        state_->running = true;
        state_->joined = false;
        state_->input = subprocess_stdin(&state_->process);
        state_->output = subprocess_stdout(&state_->process);
        if (!state_->input || !state_->output) { error = "failed to acquire clangd pipes"; return false; }
#ifdef _WIN32
        _setmode(_fileno(state_->input), _O_BINARY);
        _setmode(_fileno(state_->output), _O_BINARY);
#endif
    }
    agent_clangd_json response;
    const auto root_uri = config_.repository_root.empty() ? agent_clangd_json(nullptr) : agent_clangd_json(agent_clangd_file_uri(std::filesystem::absolute(config_.repository_root)));
    const auto params = agent_clangd_json{{"processId", nullptr}, {"rootUri", root_uri}, {"capabilities", agent_clangd_json::object()}, {"workspaceFolders", agent_clangd_json::array()}};
    if (!send_request("initialize", params, response, error)) return false;
    if (!response.contains("result")) { error = "clangd initialize response did not contain result"; return false; }
    if (!send_notification("initialized", agent_clangd_json::object(), error)) return false;
    state_->initialized = true;
    return true;
}

bool agent_clangd_session::send_notification(const std::string & method, const agent_clangd_json & params, std::string & error) {
    std::string framed;
    try { framed = agent_clangd_encode_message(agent_clangd_notification(method, params)); }
    catch (const std::exception &) { error = "clangd notification contains invalid UTF-8 or JSON data"; return false; }
    if (!state_->input || std::fwrite(framed.data(), 1, framed.size(), state_->input) != framed.size() || std::fflush(state_->input) != 0) { error = "failed to write clangd notification"; return false; }
    return true;
}

bool agent_clangd_session::read_message(agent_clangd_json & message, std::string & error) {
    for (;;) {
        if (state_->decoder.pop(message)) return true;
        const int value = std::fgetc(state_->output);
        if (value == EOF) { error = "clangd closed its output"; return false; }
        const char byte = static_cast<char>(value);
        if (!state_->decoder.feed(&byte, 1, error)) return false;
    }
}

bool agent_clangd_session::send_request(const std::string & method, const agent_clangd_json & params, agent_clangd_json & response, std::string & error) {
    const auto id = state_->next_id++;
    std::string framed;
    try { framed = agent_clangd_encode_message(agent_clangd_request(id, method, params)); }
    catch (const std::exception &) { error = "clangd request contains invalid UTF-8 or JSON data"; return false; }
    if (!state_->input || std::fwrite(framed.data(), 1, framed.size(), state_->input) != framed.size() || std::fflush(state_->input) != 0) { error = "failed to write clangd request"; return false; }
    auto read = [this, id, &response, &error]() {
        for (;;) {
            agent_clangd_json message;
            if (!read_message(message, error)) return false;
            if (!message.contains("id") || message["id"] != id) continue;
            if (message.contains("error")) { error = "clangd returned JSON-RPC error: " + message["error"].dump(); return false; }
            response = std::move(message);
            return true;
        }
    };
    if (config_.request_timeout_ms == 0) return read();
    std::promise<bool> promise;
    auto future = promise.get_future();
    std::thread reader([&promise, &error, read = std::move(read)]() mutable {
        try { promise.set_value(read()); }
        catch (const std::exception &) { error = "clangd response could not be decoded"; promise.set_value(false); }
    });
    if (future.wait_for(std::chrono::milliseconds(config_.request_timeout_ms)) != std::future_status::ready) {
        subprocess_terminate(&state_->process);
        reader.join();
        error = "clangd request timed out";
        return false;
    }
    const bool ok = future.get();
    reader.join();
    return ok;
}

bool agent_clangd_session::request(const std::string & method, const agent_clangd_json & params, agent_clangd_json & response, std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ensure_started(error) && send_request(method, params, response, error)) return true;
    state_->initialized = false;
    shutdown_locked();
    return false;
}

bool agent_clangd_session::open_document(const std::string & uri, const std::string & language_id,
        const std::string & text, std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (text.size() > 4 * 1024 * 1024) { error = "clangd document exceeds the 4 MiB limit"; return false; }
    if (!ensure_started(error)) {
        state_->initialized = false;
        shutdown_locked();
        return false;
    }
    auto found = state_->open_documents.find(uri);
    if (found != state_->open_documents.end()) {
        if (found->second.text == text) return true;
        const int64_t version = found->second.version + 1;
        if (!send_notification("textDocument/didChange", {
                {"textDocument", {{"uri", uri}, {"version", version}}},
                {"contentChanges", agent_clangd_json::array({{{"text", text}}})}}, error)) {
            state_->initialized = false; shutdown_locked(); return false;
        }
        found->second = {version, text};
        return true;
    }
    if (state_->open_documents.size() >= 8) {
        const auto evicted = state_->open_documents.begin();
        if (!send_notification("textDocument/didClose", {{"textDocument", {{"uri", evicted->first}}}}, error)) {
            state_->initialized = false; shutdown_locked(); return false;
        }
        state_->open_documents.erase(evicted);
    }
    if (!send_notification("textDocument/didOpen", {
            {"textDocument", {{"uri", uri}, {"languageId", language_id}, {"version", 1}, {"text", text}}}
        }, error)) { state_->initialized = false; shutdown_locked(); return false; }
    state_->open_documents.emplace(uri, impl::document{1, text});
    return true;
}

void agent_clangd_session::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_locked();
}

void agent_clangd_session::shutdown_locked() {
    if (!state_ || !state_->running) return;
    if (state_->initialized) {
        agent_clangd_json ignored;
        std::string error;
        const auto timeout = config_.request_timeout_ms;
        config_.request_timeout_ms = config_.shutdown_timeout_ms;
        const bool stopped = send_request("shutdown", agent_clangd_json::object(), ignored, error);
        config_.request_timeout_ms = timeout;
        if (stopped) send_notification("exit", agent_clangd_json::object(), error);
    }
    if (!state_->joined) {
        if (subprocess_alive(&state_->process)) subprocess_terminate(&state_->process);
        subprocess_join(&state_->process, &state_->exit_code);
        state_->joined = true;
    }
    // subprocess_join owns and closes the child's stdin FILE*. state_->input
    // aliases that handle, so closing it here as well would double-free it.
    state_->input = nullptr;
    subprocess_destroy(&state_->process);
    state_->output = nullptr;
    state_->running = false;
    state_->initialized = false;
    state_->open_documents.clear();
    state_->decoder = agent_clangd_message_decoder();
}
