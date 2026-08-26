#include "agent-daemon-unix.h"

#include "agent-daemon-dispatcher.h"
#include "agent-daemon-scope.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sys/stat.h>
#include <thread>
#include <vector>

#ifdef _WIN32

bool run_agent_daemon_unix_socket_adapter(
        const daemon_options &,
        const std::shared_ptr<common_agent_daemon_config_store> &,
        common_agent_daemon_dispatcher &,
        const std::shared_ptr<const agent_mcp_authenticator> &,
        std::string & error) {
    error = "Unix domain sockets are not available on Windows; use TCP or named pipes";
    return false;
}

#else

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

class unix_jsonl_stream final : public agent_daemon_jsonl_stream {
public:
    unix_jsonl_stream(int socket, size_t max_line_bytes) : socket(socket), max_line_bytes(max_line_bytes) {}
    ~unix_jsonl_stream() override { if (socket >= 0) close(socket); }

    bool read(nlohmann::ordered_json & message, std::string & error) override {
        std::string line;
        char byte = 0;
        for (;;) {
            const ssize_t count = recv(socket, &byte, 1, 0);
            if (count == 0) { closed = true; error = "Unix socket client closed"; return false; }
            if (count < 0) { closed = true; error = "failed to read daemon Unix socket request"; return false; }
            if (byte == '\n') {
                while (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                const auto parsed = nlohmann::ordered_json::parse(line, nullptr, false);
                if (parsed.is_discarded() || !parsed.is_object()) {
                    error = "daemon received a non-JSON protocol line";
                    return false;
                }
                message = parsed;
                return true;
            }
            if (line.size() >= max_line_bytes) {
                error = "daemon Unix socket request exceeds configured max line bytes";
                return false;
            }
            line.push_back(byte);
        }
    }

    bool write(const nlohmann::ordered_json & message, std::string & error) override {
        const std::string line = message.dump() + "\n";
        size_t offset = 0;
        while (offset < line.size()) {
            const ssize_t count = send(socket, line.data() + offset, line.size() - offset, MSG_NOSIGNAL);
            if (count <= 0) { closed = true; error = "failed to write daemon Unix socket response"; return false; }
            offset += static_cast<size_t>(count);
        }
        return true;
    }

    bool eof() const override { return closed; }

private:
    int socket = -1;
    size_t max_line_bytes;
    bool closed = false;
};

bool prepare_unix_request(
        nlohmann::ordered_json & request,
        const std::shared_ptr<const agent_mcp_authenticator> & authenticator,
        bool & authenticated,
        agent_mcp_caller_policy & policy,
        std::string & error) {
    if (!authenticated) {
        const auto authorization = request.value("authorization", std::string());
        if (authorization.empty() || !authenticator) {
            error = "Unix socket client must authenticate before commands";
            return false;
        }
        if (!authenticator->authenticate({authorization}, policy, error)) return false;
        authenticated = true;
        request = {{"command", "status"}};
        return true;
    }
    const auto command = request.value("command", std::string());
    if ((command == "shutdown" || command == "drain" || command == "reload_config") && !policy.allow_admin) {
        error = "Unix socket caller policy does not allow daemon administration";
        return false;
    }
    common_agent_daemon_bind_caller_scope(request, policy);
    if (command == "run_turn") {
        request["_caller_allow_policy_gated_writes"] = policy.allow_writes;
        if (!policy.allowed_tools.empty()) request["_caller_allowed_tools"] = policy.allowed_tools;
    }
    return true;
}

void serve_unix_client(
        int client,
        const daemon_options & options,
        const std::shared_ptr<common_agent_daemon_config_store> & config_store,
        common_agent_daemon_dispatcher & dispatcher,
        const std::shared_ptr<const agent_mcp_authenticator> & authenticator) {
    unix_jsonl_stream stream(client, options.tcp_max_line_bytes);
    bool authenticated = false;
    agent_mcp_caller_policy policy;
    std::string error;
    run_agent_daemon_jsonl_stream(
        stream, options, config_store, dispatcher,
        [&](nlohmann::ordered_json & request, std::string & callback_error) {
            return prepare_unix_request(request, authenticator, authenticated, policy, callback_error);
        }, error);
}

} // namespace

bool run_agent_daemon_unix_socket_adapter(
        const daemon_options & options,
        const std::shared_ptr<common_agent_daemon_config_store> & config_store,
        common_agent_daemon_dispatcher & dispatcher,
        const std::shared_ptr<const agent_mcp_authenticator> & authenticator,
        std::string & error) {
    ::unlink(options.unix_socket_path.c_str());
    const int listener = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listener < 0) { error = "failed to create daemon Unix socket"; return false; }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (options.unix_socket_path.size() >= sizeof(address.sun_path)) {
        close(listener); error = "daemon Unix socket path is too long"; return false;
    }
    std::strncpy(address.sun_path, options.unix_socket_path.c_str(), sizeof(address.sun_path) - 1);
    if (bind(listener, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0 ||
            chmod(options.unix_socket_path.c_str(), static_cast<mode_t>(options.unix_socket_mode)) != 0 ||
            listen(listener, 16) != 0) {
        close(listener); ::unlink(options.unix_socket_path.c_str());
        error = "failed to bind daemon Unix socket";
        return false;
    }

    std::vector<std::thread> clients;
    while (!dispatcher.shutdown_requested()) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(listener, &read_set);
        timeval timeout{0, 250000};
        const int ready = select(listener + 1, &read_set, nullptr, nullptr, &timeout);
        if (ready == 0) continue;
        if (ready < 0) { error = "daemon Unix socket select failed"; break; }
        const int client = accept(listener, nullptr, nullptr);
        if (client < 0) { if (errno == EINTR) continue; error = "daemon Unix socket accept failed"; break; }
        clients.emplace_back(serve_unix_client, client, std::cref(options), config_store, std::ref(dispatcher), authenticator);
    }
    close(listener);
    for (auto & client : clients) if (client.joinable()) client.join();
    ::unlink(options.unix_socket_path.c_str());
    error.clear();
    return true;
}

#endif
