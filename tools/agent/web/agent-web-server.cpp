#include "agent-web-server.h"

#include "../../../common/agent/protocol/agent-jsonl.h"
#include "../../../common/agent/protocol/agent-sse.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>
#include <string_view>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using agent_web_socket_t = SOCKET;
static constexpr agent_web_socket_t invalid_agent_web_socket = INVALID_SOCKET;
static void close_agent_web_socket(agent_web_socket_t socket) { closesocket(socket); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using agent_web_socket_t = int;
static constexpr agent_web_socket_t invalid_agent_web_socket = -1;
static void close_agent_web_socket(agent_web_socket_t socket) { close(socket); }
#endif

using json = nlohmann::ordered_json;

namespace {

std::string web_request_id(const json & command) {
    const auto request_id = command.value("request_id", std::string());
    if (!request_id.empty()) return request_id;
    const auto turn_id = command.value("turn_id", std::string());
    if (!turn_id.empty()) return turn_id;

    static std::atomic<uint64_t> next_id{1};
    return "web-turn-" + std::to_string(next_id.fetch_add(1));
}

class daemon_connection {
public:
    ~daemon_connection() {
        if (socket_ != invalid_agent_web_socket) close_agent_web_socket(socket_);
    }

    bool connect_to(const agent_web_server_options & options, std::string & error) {
        socket_ = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_ == invalid_agent_web_socket) {
            error = "web adapter could not create daemon TCP socket";
            return false;
        }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<uint16_t>(options.daemon_port));
        if (inet_pton(AF_INET, options.daemon_address.c_str(), &address.sin_addr) != 1 ||
                connect(socket_, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0) {
            error = "web adapter could not connect to daemon TCP endpoint";
            close_agent_web_socket(socket_);
            socket_ = invalid_agent_web_socket;
            return false;
        }
        return true;
    }

    bool write(const json & message, std::string & error) {
        const auto line = common_agent_jsonl_make_line(message);
        size_t sent = 0;
        while (sent < line.size()) {
#ifdef _WIN32
            const int count = send(socket_, line.data() + sent,
                static_cast<int>(line.size() - sent), 0);
#else
            const ssize_t count = send(socket_, line.data() + sent, line.size() - sent, 0);
#endif
            if (count <= 0) {
                error = "web adapter failed to write daemon JSONL message";
                return false;
            }
            sent += static_cast<size_t>(count);
        }
        return true;
    }

    bool read(json & message, std::string & error, int timeout_ms = -1) {
        std::string line;
        for (;;) {
            if (!wait_readable(timeout_ms, error)) return false;
            char byte = 0;
#ifdef _WIN32
            const int count = recv(socket_, &byte, 1, 0);
#else
            const ssize_t count = recv(socket_, &byte, 1, 0);
#endif
            if (count == 0) {
                error = "daemon JSONL connection closed";
                return false;
            }
            if (count < 0) {
                error = "web adapter failed to read daemon JSONL message";
                return false;
            }
            if (byte == '\n') {
                while (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                return common_agent_jsonl_parse_line(line, message, error);
            }
            line.push_back(byte);
            if (line.size() > 1024 * 1024) {
                error = "daemon JSONL message exceeds web adapter limit";
                return false;
            }
            timeout_ms = -1;
        }
    }

private:
    bool wait_readable(int timeout_ms, std::string & error) {
        if (timeout_ms < 0) return true;
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(socket_, &read_set);
        timeval timeout{};
        timeout.tv_sec = timeout_ms / 1000;
        timeout.tv_usec = (timeout_ms % 1000) * 1000;
        const int result = select(static_cast<int>(socket_) + 1, &read_set, nullptr, nullptr, &timeout);
        if (result == 0) {
            error.clear();
            return false;
        }
        if (result < 0) {
            error = "web adapter select on daemon socket failed";
            return false;
        }
        return true;
    }

    agent_web_socket_t socket_ = invalid_agent_web_socket;
};

bool is_event(const json & message) {
    return message.value("message_type", std::string()) == "event";
}

bool authorize_web_request(
        const httplib::Request & request,
        httplib::Response & response,
        const agent_web_server_options & options) {
    if (!options.allowed_origin.empty()) {
        const auto origin = request.get_header_value("Origin");
        if (!origin.empty() && origin != options.allowed_origin) {
            response.status = 403;
            response.set_content(R"({"error":"origin is not allowed"})", "application/json");
            return false;
        }
    }
    if (!options.web_bearer_token.empty() &&
            request.get_header_value("Authorization") != "Bearer " + options.web_bearer_token) {
        response.status = 401;
        response.set_header("WWW-Authenticate", "Bearer");
        response.set_content(R"({"error":"unauthorized"})", "application/json");
        return false;
    }
    return true;
}

std::string daemon_authorization(const agent_web_server_options & options) {
    if (!options.daemon_authorization.empty()) return options.daemon_authorization;
    return options.web_bearer_token.empty()
        ? std::string()
        : "Bearer " + options.web_bearer_token;
}

bool authenticate_daemon(
        daemon_connection & connection,
        const agent_web_server_options & options,
        std::string & error) {
    json ready;
    if (!connection.read(ready, error)) return false;
    if (!ready.value("event", std::string()).empty() &&
            ready.value("event", std::string()) != "ready") {
        error = "unexpected daemon readiness message";
        return false;
    }
    if (!connection.write({
        {"authorization", daemon_authorization(options)},
    }, error)) {
        return false;
    }
    // The daemon turns the first authenticated message into a status probe.
    // Consume that response before the adapter sends the actual command.
    json authentication_response;
    do {
        if (!connection.read(authentication_response, error)) return false;
    } while (is_event(authentication_response));
    if (!authentication_response.value("ok", false)) {
        error = authentication_response.value(
            "error", std::string("daemon authentication failed"));
        return false;
    }
    return true;
}

bool read_daemon_response(
        daemon_connection & connection,
        json & response,
        std::string & error) {
    for (;;) {
        if (!connection.read(response, error)) return false;
        if (!is_event(response)) return true;
    }
}

bool send_daemon_command(
        const agent_web_server_options & options,
        const json & command,
        json & response,
        std::string & error) {
    daemon_connection connection;
    if (!connection.connect_to(options, error) ||
            !authenticate_daemon(connection, options, error) ||
            !connection.write(command, error) ||
            !read_daemon_response(connection, response, error)) {
        return false;
    }
    return true;
}

void set_json_error(httplib::Response & response, int status, const std::string & error) {
    response.status = status;
    response.set_content(json{{"error", error}}.dump(), "application/json");
}

std::string safe_download_filename(const json & resource, const std::string & uri) {
    auto name = resource.value("name", std::string());
    if (name.empty()) {
        const auto slash = uri.find_last_of('/');
        name = slash == std::string::npos ? "artifact" : uri.substr(slash + 1);
    }
    for (char & character : name) {
        if (character == '\r' || character == '\n' || character == '"' || character == '/' || character == '\\') character = '_';
    }
    return name.empty() ? "artifact" : name;
}

std::string request_param_or(
        const httplib::Request & request,
        const char * name,
        const char * fallback) {
    return request.has_param(name) ? request.get_param_value(name) : fallback;
}

} // namespace

bool run_agent_web_server(
        const agent_web_server_options & options,
        std::string & error) {
    if (options.port <= 0 || options.port > 65535 || options.daemon_port <= 0 || options.daemon_port > 65535) {
        error = "web and daemon ports must be between 1 and 65535";
        return false;
    }

#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        error = "web adapter failed to initialize Windows sockets";
        return false;
    }
#endif

    httplib::Server server;
    server.set_payload_max_length(options.max_body_bytes);
    server.set_pre_routing_handler([&options](
            const httplib::Request & request,
            httplib::Response & response) {
        if (options.allowed_origin.empty()) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        const auto origin = request.get_header_value("Origin");
        if (!origin.empty() && origin != options.allowed_origin) {
            response.status = 403;
            response.set_content(R"({"error":"origin is not allowed"})", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!origin.empty()) response.set_header("Access-Control-Allow-Origin", origin);
        response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        response.set_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Last-Event-ID");
        if (request.method == "OPTIONS") {
            response.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    auto authorize = [&options](const httplib::Request & request, httplib::Response & response) {
        return authorize_web_request(request, response, options);
    };

    server.Post("/api/v1/turns", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        auto command = json::parse(request.body, nullptr, false);
        if (command.is_discarded() || !command.is_object()) {
            set_json_error(response, 400, "request body must be a JSON object");
            return;
        }
        command["command"] = "run_turn";
        command["request_id"] = web_request_id(command);
        json result;
        std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            set_json_error(response, 502, command_error);
            return;
        }
        response.set_content(result.dump(), "application/json");
    });

    server.Post(R"(/api/v1/turns/([^/]+)/cancel)", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        json command = {
            {"command", "cancel_turn"},
            {"target_turn_id", request.matches[1].str()},
        };
        json result;
        std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            set_json_error(response, 502, command_error);
            return;
        }
        response.set_content(result.dump(), "application/json");
    });

    server.Post("/api/v1/resources", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        auto command = json::parse(request.body, nullptr, false);
        if (command.is_discarded() || !command.is_object()) {
            set_json_error(response, 400, "request body must be a JSON object");
            return;
        }
        command["command"] = "put_resource";
        json result;
        std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            set_json_error(response, 502, command_error);
            return;
        }
        response.set_content(result.dump(), "application/json");
    });

    server.Get("/api/v1/resources/download", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        if (!request.has_param("uri")) {
            set_json_error(response, 400, "resource download requires uri");
            return;
        }
        const auto uri = request.get_param_value("uri");
        const json command = {
            {"command", "read_resource"},
            {"uri", uri},
            {"namespace_id", request_param_or(request, "namespace_id", "default-namespace")},
            {"session_id", request_param_or(request, "session_id", "default-session")},
            {"project_id", request_param_or(request, "project_id", "")},
            {"turn_id", request_param_or(request, "turn_id", "")},
            {"max_bytes", options.max_download_bytes},
        };
        json result;
        std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            set_json_error(response, 502, command_error);
            return;
        }
        if (!result.value("ok", false) || !result.contains("resource") || !result.contains("content")) {
            set_json_error(response, 404, result.value("error", "resource could not be downloaded"));
            return;
        }
        const auto & resource = result["resource"];
        response.set_header("Content-Disposition", "attachment; filename=\"" + safe_download_filename(resource, uri) + "\"");
        response.set_content(result["content"].get<std::string>(), resource.value("mime_type", "application/octet-stream"));
    });

    server.Get("/api/v1/status", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        json result;
        std::string command_error;
        if (!send_daemon_command(options, {{"command", "status"}}, result, command_error)) {
            set_json_error(response, 502, command_error);
            return;
        }
        response.set_content(result.dump(), "application/json");
    });

    server.Get("/api/v1/events", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        auto connection = std::make_shared<daemon_connection>();
        std::string connection_error;
        if (!connection->connect_to(options, connection_error) ||
                !authenticate_daemon(*connection, options, connection_error)) {
            set_json_error(response, 502, connection_error);
            return;
        }
        response.set_header("Cache-Control", "no-cache");
        response.set_header("Connection", "keep-alive");
        response.set_header("X-Accel-Buffering", "no");
        response.set_chunked_content_provider(
            "text/event-stream",
            [connection](size_t, httplib::DataSink & sink) mutable {
                json message;
                std::string read_error;
                if (!connection->read(message, read_error, 15000)) {
                    if (read_error.empty()) return sink.write(": heartbeat\n\n", 13);
                    return false;
                }
                if (!is_event(message)) return true;
                const auto payload = common_agent_sse_format_message(message, "agent");
                return sink.write(payload.data(), payload.size());
            });
    });

    if (!server.listen(options.listen_address.c_str(), options.port)) {
        error = "web adapter failed to bind HTTP listener";
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
#ifdef _WIN32
    WSACleanup();
#endif
    error.clear();
    return true;
}
