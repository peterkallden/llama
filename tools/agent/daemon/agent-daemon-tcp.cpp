#include "agent-daemon-tcp.h"

#include "agent-daemon-dispatcher.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using agent_socket_t = SOCKET;
static constexpr agent_socket_t invalid_agent_socket = INVALID_SOCKET;
static void close_agent_socket(agent_socket_t socket) { closesocket(socket); }
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using agent_socket_t = int;
static constexpr agent_socket_t invalid_agent_socket = -1;
static void close_agent_socket(agent_socket_t socket) { close(socket); }
#endif

namespace {

class tcp_jsonl_stream final : public agent_daemon_jsonl_stream {
public:
    tcp_jsonl_stream(agent_socket_t socket, size_t max_line_bytes)
        : socket(socket), max_line_bytes(max_line_bytes) {}

    ~tcp_jsonl_stream() override {
        if (socket != invalid_agent_socket) {
            close_agent_socket(socket);
        }
    }

    bool read(nlohmann::ordered_json & message, std::string & error) override {
        std::string line;
        char byte = 0;
        for (;;) {
#ifdef _WIN32
            const int count = recv(socket, &byte, 1, 0);
#else
            const ssize_t count = recv(socket, &byte, 1, 0);
#endif
            if (count == 0) {
                closed = true;
                error = "TCP client closed before returning a protocol response";
                return false;
            }
            if (count < 0) {
                closed = true;
                error = "failed to read daemon TCP request";
                return false;
            }
            if (byte == '\n') {
                while (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                return common_agent_jsonl_parse_line(line, message, error);
            }
            if (line.size() >= max_line_bytes) {
                error = "daemon TCP request exceeds configured max line bytes";
                return false;
            }
            line.push_back(byte);
        }
    }

    bool write(const nlohmann::ordered_json & message, std::string & error) override {
        const std::string line = common_agent_jsonl_make_line(message);
        size_t sent = 0;
        while (sent < line.size()) {
#ifdef _WIN32
            const int count = send(socket, line.data() + sent, static_cast<int>(line.size() - sent), 0);
#else
            const ssize_t count = send(socket, line.data() + sent, line.size() - sent, 0);
#endif
            if (count <= 0) {
                closed = true;
                error = "failed to write daemon TCP response";
                return false;
            }
            sent += static_cast<size_t>(count);
        }
        return true;
    }

    bool eof() const override { return closed; }

private:
    agent_socket_t socket;
    size_t max_line_bytes;
    bool closed = false;
};

bool bind_listener(const daemon_options & options, agent_socket_t & listener, std::string & error) {
    listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener == invalid_agent_socket) {
        error = "failed to create daemon TCP listener";
        return false;
    }
    int reuse = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&reuse), sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(options.tcp_port));
    if (inet_pton(AF_INET, options.tcp_listen_address.c_str(), &address.sin_addr) != 1) {
        close_agent_socket(listener);
        listener = invalid_agent_socket;
        error = "daemon TCP listen address must be an IPv4 address";
        return false;
    }
    if (bind(listener, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0 || listen(listener, 16) != 0) {
        close_agent_socket(listener);
        listener = invalid_agent_socket;
        error = "failed to bind daemon TCP listener";
        return false;
    }
    return true;
}

bool bind_tcp_policy(
        nlohmann::ordered_json & request,
        const std::shared_ptr<const agent_mcp_authenticator> & authenticator,
        bool & authenticated,
        agent_mcp_caller_policy & policy,
        std::string & error) {
    if (!authenticated) {
        const auto authorization = request.value("authorization", std::string());
        if (authorization.empty() || !authenticator) {
            error = "TCP client must authenticate with an authorization field before commands";
            return false;
        }
        if (!authenticator->authenticate({authorization}, policy, error)) return false;
        authenticated = true;
        request = { {"command", "status"} };
        return true;
    }

    if (request.contains("namespace_id")) request["namespace_id"] = policy.namespace_id;
    if (request.contains("project_id")) request["project_id"] = policy.project_id;
    const auto command = request.value("command", std::string());
    if ((command == "shutdown" || command == "drain" || command == "reload_config") &&
            !policy.allow_admin) {
        error = "TCP caller policy does not allow daemon administration";
        return false;
    }
    if (!policy.allowed_tools.empty() && command == "run_turn") {
        request["_caller_allowed_tools"] = policy.allowed_tools;
    }
    if (command == "run_turn") {
        request["_caller_allow_policy_gated_writes"] = policy.allow_writes;
    }
    if (command == "run_turn") {
        request["namespace_id"] = policy.namespace_id;
        request["project_id"] = policy.project_id;
        if (request.value("session_id", std::string()).empty()) {
            request["session_id"] = policy.caller_id + "-session";
        }
    }
    if (command == "execute_tool") {
        request["namespace_id"] = policy.namespace_id;
        request["project_id"] = policy.project_id;
        if (!policy.tool_profile.empty()) request["tool_profile"] = policy.tool_profile;
    }
    return true;
}

void serve_tcp_client(
        agent_socket_t client,
        const daemon_options & options,
        const std::shared_ptr<common_agent_daemon_config_store> & config_store,
        common_agent_daemon_dispatcher & dispatcher,
        const std::shared_ptr<const agent_mcp_authenticator> & authenticator) {
    tcp_jsonl_stream stream(client, options.tcp_max_line_bytes);
    bool authenticated = false;
    agent_mcp_caller_policy policy;
    std::string error;
    run_agent_daemon_jsonl_stream(
        stream,
        options,
        config_store,
        dispatcher,
        [&](nlohmann::ordered_json & request, std::string & callback_error) {
            return bind_tcp_policy(request, authenticator, authenticated, policy, callback_error);
        },
        error);
}

} // namespace

bool run_agent_daemon_tcp_adapter(
        const daemon_options & options,
        const std::shared_ptr<common_agent_daemon_config_store> & config_store,
        common_agent_daemon_dispatcher & dispatcher,
        const std::shared_ptr<const agent_mcp_authenticator> & authenticator,
        std::string & error) {
#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        error = "failed to initialize Windows sockets";
        return false;
    }
#endif
    agent_socket_t listener = invalid_agent_socket;
    if (!bind_listener(options, listener, error)) {
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    std::vector<std::thread> clients;
    while (!dispatcher.shutdown_requested()) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(listener, &read_set);
        timeval timeout{};
        timeout.tv_sec = 0;
        timeout.tv_usec = 250000;
        const int ready = select(static_cast<int>(listener + 1), &read_set, nullptr, nullptr, &timeout);
        if (ready == 0) continue;
        if (ready < 0) {
            error = "daemon TCP listener select failed";
            break;
        }
        sockaddr_storage address{};
#ifdef _WIN32
        int address_size = sizeof(address);
#else
        socklen_t address_size = sizeof(address);
#endif
        const agent_socket_t client = accept(listener, reinterpret_cast<sockaddr *>(&address), &address_size);
        if (client == invalid_agent_socket) {
            if (dispatcher.shutdown_requested()) break;
            error = "daemon TCP accept failed";
            close_agent_socket(listener);
#ifdef _WIN32
            WSACleanup();
#endif
            return false;
        }
        clients.emplace_back(serve_tcp_client, client, std::cref(options), config_store, std::ref(dispatcher), authenticator);
    }
    close_agent_socket(listener);
    for (auto & client : clients) if (client.joinable()) client.join();
#ifdef _WIN32
    WSACleanup();
#endif
    error.clear();
    return true;
}
