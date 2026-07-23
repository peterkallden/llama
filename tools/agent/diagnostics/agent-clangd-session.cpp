#include "agent-clangd-session.h"

#include <sheredom/subprocess.h>

#include <cstdio>
#include <future>
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
        std::vector<std::string> command = {config_.executable, "--background-index=false", "--offset-encoding=utf-16"};
        if (!config_.compile_commands_dir.empty() && config_.compile_commands_dir != "auto") command.push_back("--compile-commands-dir=" + config_.compile_commands_dir);
        auto argv = argv_for(command);
        if (subprocess_create(argv.data(), subprocess_option_no_window | subprocess_option_enable_async | subprocess_option_inherit_environment, &state_->process) != 0) {
            error = "failed to start clangd";
            return false;
        }
        state_->input = subprocess_stdin(&state_->process);
        state_->output = subprocess_stdout(&state_->process);
        if (!state_->input || !state_->output) { error = "failed to acquire clangd pipes"; shutdown(); return false; }
#ifdef _WIN32
        _setmode(_fileno(state_->input), _O_BINARY);
        _setmode(_fileno(state_->output), _O_BINARY);
#endif
        state_->running = true;
        state_->joined = false;
    }
    agent_clangd_json response;
    const auto root_uri = config_.repository_root.empty() ? agent_clangd_json(nullptr) : agent_clangd_json("file://" + config_.repository_root);
    const auto params = agent_clangd_json{{"processId", nullptr}, {"rootUri", root_uri}, {"capabilities", agent_clangd_json::object()}, {"workspaceFolders", agent_clangd_json::array()}};
    if (!send_request("initialize", params, response, error)) return false;
    if (!response.contains("result")) { error = "clangd initialize response did not contain result"; return false; }
    if (!send_notification("initialized", agent_clangd_json::object(), error)) return false;
    state_->initialized = true;
    return true;
}

bool agent_clangd_session::send_notification(const std::string & method, const agent_clangd_json & params, std::string & error) {
    const auto framed = agent_clangd_encode_message(agent_clangd_notification(method, params));
    if (!state_->input || std::fwrite(framed.data(), 1, framed.size(), state_->input) != framed.size() || std::fflush(state_->input) != 0) { error = "failed to write clangd notification"; return false; }
    return true;
}

bool agent_clangd_session::read_message(agent_clangd_json & message, std::string & error) {
    for (;;) {
        if (state_->decoder.pop(message)) return true;
        char buffer[4096];
        const auto count = std::fread(buffer, 1, sizeof(buffer), state_->output);
        if (count == 0) { error = "clangd closed its output"; return false; }
        if (!state_->decoder.feed(buffer, count, error)) return false;
    }
}

bool agent_clangd_session::send_request(const std::string & method, const agent_clangd_json & params, agent_clangd_json & response, std::string & error) {
    const auto id = state_->next_id++;
    const auto framed = agent_clangd_encode_message(agent_clangd_request(id, method, params));
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
    std::thread reader([&promise, read = std::move(read)]() mutable { promise.set_value(read()); });
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
    if (!ensure_started(error)) return false;
    return send_request(method, params, response, error);
}

void agent_clangd_session::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!state_ || !state_->running) return;
    if (state_->initialized) {
        agent_clangd_json ignored;
        std::string error;
        send_request("shutdown", agent_clangd_json::object(), ignored, error);
        send_notification("exit", agent_clangd_json::object(), error);
    }
    if (state_->input) { std::fclose(state_->input); state_->input = nullptr; }
    if (!state_->joined) {
        if (subprocess_alive(&state_->process)) subprocess_terminate(&state_->process);
        subprocess_join(&state_->process, &state_->exit_code);
        state_->joined = true;
    }
    subprocess_destroy(&state_->process);
    state_->output = nullptr;
    state_->running = false;
    state_->initialized = false;
}
