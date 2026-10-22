#include "agent-web-server.h"
#include "../../../common/base64.hpp"

#include "../../../common/agent/protocol/agent-jsonl.h"
#include "../../../common/agent/protocol/agent-sse.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <memory>
#include <limits>
#include <string_view>
#include <ctime>

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

bool send_daemon_command(const agent_web_server_options & options,
        const json & command, json & result, std::string & error);

bool has_prefix(const std::string & value, const char * prefix) {
    const std::string_view expected(prefix);
    return value.size() >= expected.size() && value.compare(0, expected.size(), expected) == 0;
}

struct active_web_turn { std::string session_id; std::string project_id; };
std::mutex active_web_turn_mutex;
std::map<std::string, active_web_turn> active_web_turns;

std::string web_request_id(const json & command) {
    const auto request_id = command.contains("request_id") && command["request_id"].is_string()
        ? command["request_id"].get<std::string>() : std::string();
    if (!request_id.empty()) return request_id;
    const auto turn_id = command.contains("turn_id") && command["turn_id"].is_string()
        ? command["turn_id"].get<std::string>() : std::string();
    if (!turn_id.empty()) return turn_id;

    static std::atomic<uint64_t> next_id{1};
    return "web-turn-" + std::to_string(next_id.fetch_add(1));
}

constexpr const char * k_chat_not_found = "chat is not available in this namespace";
constexpr const char * k_project_not_found = "project is not available in this namespace";
constexpr const char * k_web_namespace = "web";
constexpr const char * k_web_catalog_project = "__llama_agent_web_catalog__";

bool daemon_command(const agent_web_server_options & options, const json & command,
        json & result, std::string & error) {
    return send_daemon_command(options, command, result, error);
}

bool list_catalog(const agent_web_server_options & options, json & resources, std::string & error) {
    json result;
    if (!daemon_command(options, {{"command", "list_resources"}, {"namespace_id", k_web_namespace},
            {"session_id", "web-catalog"}, {"project_id", k_web_catalog_project}}, result, error)) return false;
    if (!result.value("ok", false)) { error = result.value("error", "project catalog could not be read"); return false; }
    resources = result.value("resources", json::array());
    if (!resources.is_array()) { error = "project catalog response is invalid"; return false; }
    error.clear();
    return true;
}

bool read_catalog_record(const agent_web_server_options & options, const std::string & uri,
        json & value, std::string & error) {
    json result;
    if (!daemon_command(options, {{"command", "read_resource"}, {"uri", uri}, {"namespace_id", k_web_namespace},
            {"session_id", "web-catalog"}, {"project_id", k_web_catalog_project}, {"max_bytes", 64 * 1024}}, result, error)) return false;
    if (!result.value("ok", false) || !result.contains("content") || !result["content"].is_string()) {
        error = result.value("error", "project catalog record could not be read"); return false;
    }
    value = json::parse(result["content"].get<std::string>(), nullptr, false);
    if (!value.is_object()) { error = "project catalog record is invalid"; return false; }
    error.clear();
    return true;
}

bool put_catalog_record(const agent_web_server_options & options, const json & value,
        std::string & error) {
    const auto kind = value.value("kind", std::string());
    const auto id = value.value("id", std::string());
    if ((kind != "project" && kind != "chat") || id.empty()) { error = "invalid catalog record"; return false; }
    json result;
    auto stored = value;
    stored["catalog_revision"] = static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    const auto name = std::string("llama-agent-web-") + kind + "-" + id + ".json";
    if (!daemon_command(options, {{"command", "put_resource"}, {"name", name},
            {"description", "Host-owned web navigation metadata."}, {"mime_type", "application/json"},
            {"text", stored.dump()}, {"scope", "project"}, {"namespace_id", k_web_namespace},
            {"session_id", "web-catalog"}, {"project_id", k_web_catalog_project}}, result, error)) return false;
    if (!result.value("ok", false)) { error = result.value("error", "catalog record could not be saved"); return false; }
    error.clear();
    return true;
}

bool find_project(const agent_web_server_options & options, const std::string & project_id,
        json & project, std::string & error) {
    json resources;
    if (!list_catalog(options, resources, error)) return false;
    bool found = false;
    for (const auto & descriptor : resources) {
        if (!descriptor.is_object() || !has_prefix(descriptor.value("name", std::string()), "llama-agent-web-project-")) continue;
        json record;
        if (!read_catalog_record(options, descriptor.value("uri", std::string()), record, error)) return false;
        if (record.value("kind", std::string()) == "project" && record.value("id", std::string()) == project_id) {
            if (!found || record.value("catalog_revision", uint64_t(0)) > project.value("catalog_revision", uint64_t(0))) project = std::move(record);
            found = true;
        }
    }
    if (found) { error.clear(); return true; }
    error = k_project_not_found;
    return false;
}

bool find_chat(const agent_web_server_options & options, const std::string & session_id,
        json & chat, std::string & error) {
    json resources;
    if (!list_catalog(options, resources, error)) return false;
    bool found = false;
    for (const auto & descriptor : resources) {
        if (!descriptor.is_object() || !has_prefix(descriptor.value("name", std::string()), "llama-agent-web-chat-")) continue;
        json record;
        if (!read_catalog_record(options, descriptor.value("uri", std::string()), record, error)) return false;
        if (record.value("kind", std::string()) == "chat" && record.value("id", std::string()) == session_id) {
            if (!found || record.value("catalog_revision", uint64_t(0)) > chat.value("catalog_revision", uint64_t(0))) chat = std::move(record);
            found = true;
        }
    }
    if (found) { error.clear(); return true; }
    error = k_chat_not_found;
    return false;
}

bool validate_chat_scope(const agent_web_server_options & options, const std::string & session_id,
        const std::string & project_id, json & chat, std::string & error) {
    if (!find_chat(options, session_id, chat, error)) return false;
    if (chat.value("project_id", std::string()) != project_id) {
        error = "chat belongs to a different project";
        return false;
    }
    if (!project_id.empty()) {
        json project;
        if (!find_project(options, project_id, project, error)) return false;
    }
    error.clear();
    return true;
}

bool parse_chat_scope(const json & value, std::string & session_id, std::string & project_id, std::string & error) {
    if (!value.is_object() || !value.contains("session_id") || !value["session_id"].is_string() ||
            (value.contains("project_id") && !value["project_id"].is_string())) {
        error = "request requires a string session_id and optional string project_id";
        return false;
    }
    session_id = value["session_id"].get<std::string>();
    project_id = value.value("project_id", std::string());
    if (session_id.empty() || session_id.size() > 128 || !has_prefix(session_id, "web-") ||
            session_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos) {
        error = "chat id is invalid";
        return false;
    }
    error.clear();
    return true;
}

class daemon_connection {
public:
    ~daemon_connection() {
        if (socket_ != invalid_agent_web_socket) close_agent_web_socket(socket_);
    }

    bool connect_to(const agent_web_server_options & options, std::string & error) {
        // JSON strings can expand each byte to a six-byte Unicode escape.
        // Include UI resources and the configured download size, not only uploads.
        const auto payload_limit = std::max({options.max_body_bytes, options.max_download_bytes,
            size_t{2 * 1024 * 1024}});
        constexpr size_t framing_bytes = 64 * 1024;
        if (payload_limit > (std::numeric_limits<size_t>::max() - framing_bytes) / 6) {
            error = "web adapter response limit is too large";
            return false;
        }
        max_line_bytes_ = payload_limit * 6 + framing_bytes;
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
            int flags = 0;
#ifdef MSG_NOSIGNAL
            flags = MSG_NOSIGNAL;
#endif
            const ssize_t count = send(socket_, line.data() + sent, line.size() - sent, flags);
#endif
            if (count <= 0) {
#ifndef _WIN32
                if (count < 0 && errno == EINTR) continue;
#endif
                error = "web adapter failed to write daemon JSONL message";
                return false;
            }
            sent += static_cast<size_t>(count);
        }
        return true;
    }

    bool read(json & message, std::string & error, int timeout_ms = -1) {
        error.clear();
        for (;;) {
            const auto end = read_buffer_.find('\n');
            if ((end == std::string::npos && read_buffer_.size() > max_line_bytes_) ||
                    (end != std::string::npos && end > max_line_bytes_)) {
                error = "daemon JSONL message exceeds web adapter response limit";
                return false;
            }
            if (end != std::string::npos) {
                auto line = read_buffer_.substr(0, end);
                read_buffer_.erase(0, end + 1);
                while (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                return common_agent_jsonl_parse_line(line, message, error);
            }
            if (!wait_readable(timeout_ms, error)) return false;
            char bytes[8192];
#ifdef _WIN32
            const int count = recv(socket_, bytes, sizeof(bytes), 0);
#else
            const ssize_t count = recv(socket_, bytes, sizeof(bytes), 0);
#endif
            if (count == 0) {
                error = "daemon JSONL connection closed";
                return false;
            }
            if (count < 0) {
#ifndef _WIN32
                if (errno == EINTR) continue;
#endif
                error = "web adapter failed to read daemon JSONL message";
                return false;
            }
            // Keep partial lines across heartbeat timeouts and retain any
            // following messages already received in the same socket read.
            read_buffer_.append(bytes, static_cast<size_t>(count));
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
    std::string read_buffer_;
    size_t max_line_bytes_ = 0;
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

    server.Get("/api/v1/projects", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        json resources; std::string error;
        if (!list_catalog(options, resources, error)) { set_json_error(response, 502, error); return; }
        std::map<std::string, json> latest;
        for (const auto & descriptor : resources) {
            if (!descriptor.is_object() || !has_prefix(descriptor.value("name", std::string()), "llama-agent-web-project-")) continue;
            json value;
            if (!read_catalog_record(options, descriptor.value("uri", std::string()), value, error)) { set_json_error(response, 502, error); return; }
            const auto id = value.value("id", std::string());
            if (value.value("kind", std::string()) != "project" || id.empty()) continue;
            const auto found = latest.find(id);
            if (found == latest.end() || value.value("catalog_revision", uint64_t(0)) > found->second.value("catalog_revision", uint64_t(0))) latest[id] = std::move(value);
        }
        json projects = json::array();
        std::vector<json> ordered;
        for (auto & entry : latest) { entry.second.erase("catalog_revision"); ordered.push_back(std::move(entry.second)); }
        std::sort(ordered.begin(), ordered.end(), [](const json & a, const json & b) { return a.value("last_active_at", int64_t(0)) > b.value("last_active_at", int64_t(0)); });
        for (auto & item : ordered) projects.push_back(std::move(item));
        response.set_content(json{{"ok", true}, {"projects", std::move(projects)}}.dump(), "application/json");
    });

    server.Post("/api/v1/projects", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        const auto input = json::parse(request.body, nullptr, false);
        if (!input.is_object() || !input.contains("name") || !input["name"].is_string()) { set_json_error(response, 400, "project name is required"); return; }
        auto name = input["name"].get<std::string>();
        while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) name.pop_back();
        const auto first = name.find_first_not_of(" \t\r\n");
        if (first != std::string::npos) name.erase(0, first);
        if (name.empty() || name.size() > 120) { set_json_error(response, 400, "project name must contain 1 to 120 characters"); return; }
        static std::atomic<uint64_t> next_project{1};
        const auto ticks = std::chrono::system_clock::now().time_since_epoch().count();
        const auto id = "project-" + std::to_string(static_cast<uint64_t>(ticks)) + "-" + std::to_string(next_project.fetch_add(1));
        const auto now = static_cast<int64_t>(std::time(nullptr));
        const json project{{"schema", "llama-agent-web-project.v1"}, {"kind", "project"}, {"id", id}, {"name", name}, {"created_at", now}, {"last_active_at", now}};
        std::string error;
        if (!put_catalog_record(options, project, error)) { set_json_error(response, 502, error); return; }
        response.set_content(json{{"ok", true}, {"project", project}}.dump(), "application/json");
    });

    server.Get("/api/v1/chats", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        const auto project_id = request_param_or(request, "project_id", "");
        if (!project_id.empty()) { json project; std::string error; if (!find_project(options, project_id, project, error)) { set_json_error(response, 403, error); return; } }
        json resources; std::string error;
        if (!list_catalog(options, resources, error)) { set_json_error(response, 502, error); return; }
        std::map<std::string, json> latest;
        for (const auto & descriptor : resources) {
            if (!descriptor.is_object() || !has_prefix(descriptor.value("name", std::string()), "llama-agent-web-chat-")) continue;
            json value;
            if (!read_catalog_record(options, descriptor.value("uri", std::string()), value, error)) { set_json_error(response, 502, error); return; }
            const auto id = value.value("id", std::string());
            if (value.value("kind", std::string()) != "chat" || id.empty() || value.value("project_id", std::string()) != project_id) continue;
            const auto found = latest.find(id);
            if (found == latest.end() || value.value("catalog_revision", uint64_t(0)) > found->second.value("catalog_revision", uint64_t(0))) latest[id] = std::move(value);
        }
        std::vector<json> ordered;
        for (auto & entry : latest) { entry.second.erase("catalog_revision"); ordered.push_back(std::move(entry.second)); }
        std::sort(ordered.begin(), ordered.end(), [](const json & a, const json & b) {
            return a.value("last_active_at", int64_t(0)) > b.value("last_active_at", int64_t(0));
        });
        std::map<std::string, std::string> daemon_active_turns;
        json session_snapshot; std::string session_snapshot_error;
        if (daemon_command(options, {{"command", "list_sessions"}}, session_snapshot, session_snapshot_error) &&
                session_snapshot.value("ok", false) && session_snapshot.contains("session_keys") && session_snapshot["session_keys"].is_array()) {
            for (const auto & entry : session_snapshot["session_keys"]) {
                if (entry.is_object() && entry.value("namespace_id", std::string()) == k_web_namespace &&
                        entry.contains("active_turn_id") && entry["active_turn_id"].is_string()) {
                    daemon_active_turns[entry.value("session_id", std::string())] = entry["active_turn_id"].get<std::string>();
                }
            }
        }
        size_t offset = 0;
        if (request.has_param("cursor")) {
            try { offset = static_cast<size_t>(std::stoull(request.get_param_value("cursor"))); }
            catch (...) { set_json_error(response, 400, "chat cursor is invalid"); return; }
        }
        if (offset > ordered.size()) { set_json_error(response, 400, "chat cursor is out of range"); return; }
        const size_t limit = std::clamp<size_t>(request.has_param("limit") ? static_cast<size_t>(std::max(1, std::atoi(request.get_param_value("limit").c_str()))) : 30, 1, 100);
        json chats = json::array();
        for (size_t i = offset; i < ordered.size() && i < offset + limit; ++i) {
            auto item = std::move(ordered[i]);
            const auto session_id = item.value("id", std::string());
            const auto daemon_active = daemon_active_turns.find(session_id);
            if (daemon_active != daemon_active_turns.end()) item["active_turn_id"] = daemon_active->second;
            {
                std::lock_guard<std::mutex> lock(active_web_turn_mutex);
                const auto active = active_web_turns.find(daemon_active == daemon_active_turns.end() ? std::string() : daemon_active->second);
                if (active == active_web_turns.end()) {
                    const auto web_active = std::find_if(active_web_turns.begin(), active_web_turns.end(), [&](const auto & candidate) {
                        return candidate.second.session_id == session_id;
                    });
                    if (web_active != active_web_turns.end()) item["active_turn_id"] = web_active->first;
                }
            }
            chats.push_back(std::move(item));
        }
        const auto next = offset + chats.size() < ordered.size() ? std::to_string(offset + chats.size()) : std::string();
        response.set_content(json{{"ok", true}, {"chats", std::move(chats)}, {"next_cursor", next}}.dump(), "application/json");
    });

    server.Post("/api/v1/chats", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        const auto input = json::parse(request.body, nullptr, false);
        if (!input.is_object() || !input.contains("session_id") || !input["session_id"].is_string() || input["session_id"].get<std::string>().empty() ||
                (input.contains("project_id") && !input["project_id"].is_string()) || (input.contains("title") && !input["title"].is_string())) {
            set_json_error(response, 400, "chat creation requires a session_id"); return;
        }
        const auto id = input["session_id"].get<std::string>();
        if (id.size() > 128 || !has_prefix(id, "web-") || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos) {
            set_json_error(response, 400, "chat id is invalid"); return;
        }
        if (input.contains("project_id") && !input["project_id"].is_string()) { set_json_error(response, 400, "project_id must be a string"); return; }
        const auto project_id = input.value("project_id", std::string());
        json existing; std::string error;
        if (find_chat(options, id, existing, error)) { set_json_error(response, 409, "chat id already exists"); return; }
        if (error != k_chat_not_found) { set_json_error(response, 502, error); return; }
        if (!project_id.empty()) { json project; if (!find_project(options, project_id, project, error)) { set_json_error(response, 403, error); return; } }
        const auto now = static_cast<int64_t>(std::time(nullptr));
        const json chat{{"schema", "llama-agent-web-chat.v1"}, {"kind", "chat"}, {"id", id},
            {"project_id", project_id}, {"title", input.value("title", std::string("New chat"))},
            {"created_at", now}, {"last_active_at", now}, {"turn_count", 0}, {"attachments", json::array()}};
        if (!put_catalog_record(options, chat, error)) { set_json_error(response, 502, error); return; }
        response.set_content(json{{"ok", true}, {"chat", chat}}.dump(), "application/json");
    });

    server.Post("/api/v1/chats/migrate-legacy", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        const auto input = json::parse(request.body, nullptr, false);
        if (!input.is_object() || !input.contains("session_id") || !input["session_id"].is_string()) { set_json_error(response, 400, "legacy migration requires a session_id"); return; }
        const auto session_id = input["session_id"].get<std::string>();
        if (session_id.size() > 128 || !has_prefix(session_id, "web-") || session_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos) {
            set_json_error(response, 400, "legacy chat id is invalid"); return;
        }
        json existing; std::string error;
        if (find_chat(options, session_id, existing, error)) { existing.erase("catalog_revision"); response.set_content(json{{"ok", true}, {"chat", existing}}.dump(), "application/json"); return; }
        if (error != k_chat_not_found) { set_json_error(response, 502, error); return; }
        constexpr const char * legacy_project_id = "web-example";
        json project;
        if (!find_project(options, legacy_project_id, project, error)) {
            if (error != k_project_not_found) { set_json_error(response, 502, error); return; }
            const auto now = static_cast<int64_t>(std::time(nullptr));
            project = json{{"schema", "llama-agent-web-project.v1"}, {"kind", "project"}, {"id", legacy_project_id},
                {"name", "Imported project"}, {"created_at", now}, {"last_active_at", now}};
            if (!put_catalog_record(options, project, error)) { set_json_error(response, 502, error); return; }
        }
        json listing;
        if (!daemon_command(options, {{"command", "list_resources"}, {"namespace_id", k_web_namespace},
                {"session_id", session_id}, {"project_id", legacy_project_id}}, listing, error) || !listing.value("ok", false)) {
            set_json_error(response, 502, error.empty() ? listing.value("error", "legacy chat resources could not be checked") : error); return;
        }
        if (!listing.contains("resources") || !listing["resources"].is_array()) { set_json_error(response, 502, "legacy resource listing is invalid"); return; }
        const auto now = static_cast<int64_t>(std::time(nullptr));
        size_t transcript_records = 0;
        json attachments = json::array();
        for (const auto & item : listing.value("resources", json::array())) {
            if (item.is_object() && item.contains("metadata") && item["metadata"].is_object() &&
                    item["metadata"].value("purpose", std::string()) == "agent.context.turn.v1") ++transcript_records;
            if (item.is_object() && item.value("scope", std::string()) == "session" &&
                    !(item.contains("metadata") && item["metadata"].is_object() && has_prefix(item["metadata"].value("purpose", std::string()), "agent.context."))) {
                attachments.push_back({{"uri", item.value("uri", std::string())}, {"name", item.value("name", std::string("Resource"))},
                    {"mime_type", item.value("mime_type", std::string("application/octet-stream"))}});
            }
        }
        const json chat{{"schema", "llama-agent-web-chat.v1"}, {"kind", "chat"}, {"id", session_id},
            {"project_id", legacy_project_id}, {"title", "Imported chat"}, {"created_at", now},
            {"last_active_at", now}, {"turn_count", transcript_records ? 1 : 0}, {"attachments", std::move(attachments)}};
        if (!put_catalog_record(options, chat, error)) { set_json_error(response, 502, error); return; }
        project.erase("catalog_revision");
        response.set_content(json{{"ok", true}, {"chat", chat}, {"migrated", true}}.dump(), "application/json");
    });

    server.Get(R"(/api/v1/chats/([^/]+)/messages)", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        const auto session_id = request.matches[1].str();
        const auto project_id = request_param_or(request, "project_id", "");
        json chat; std::string error;
        if (!validate_chat_scope(options, session_id, project_id, chat, error)) { set_json_error(response, error == k_chat_not_found ? 404 : 403, error); return; }
        json listing;
        if (!daemon_command(options, {{"command", "list_resources"}, {"namespace_id", k_web_namespace}, {"session_id", session_id}, {"project_id", project_id}}, listing, error) || !listing.value("ok", false)) {
            set_json_error(response, 502, error.empty() ? listing.value("error", "chat history could not be listed") : error); return;
        }
        if (!listing.contains("resources") || !listing["resources"].is_array()) { set_json_error(response, 502, "chat history listing is invalid"); return; }
        struct message_row { uint64_t sequence; json value; };
        std::vector<message_row> rows;
        for (const auto & descriptor : listing.value("resources", json::array())) {
            if (!descriptor.is_object() || !descriptor.contains("metadata") || !descriptor["metadata"].is_object() ||
                    descriptor["metadata"].value("purpose", std::string()) != "agent.context.turn.v1") continue;
            json stored;
            json read;
            if (!daemon_command(options, {{"command", "read_resource"}, {"uri", descriptor.value("uri", std::string())},
                    {"namespace_id", k_web_namespace}, {"session_id", session_id}, {"project_id", project_id}, {"max_bytes", 1024 * 1024}}, read, error) ||
                    !read.value("ok", false) || !read.contains("content") || !read["content"].is_string()) {
                set_json_error(response, 502, error.empty() ? read.value("error", "persisted chat history could not be read") : error);
                return;
            }
            stored = json::parse(read["content"].get<std::string>(), nullptr, false);
            if (!stored.is_object()) { set_json_error(response, 502, "persisted chat history contains an invalid record"); return; }
            const auto role = stored.value("role", std::string());
            if (role != "user" && role != "assistant") { set_json_error(response, 502, "persisted chat history contains an invalid role"); return; }
            rows.push_back({stored.value("sequence", uint64_t(0)), json{{"id", stored.value("request_id", std::string()) + "-" + role},
                {"turn_id", stored.value("turn_id", std::string())}, {"role", role}, {"content", stored.value("content", std::string())},
                {"recorded_at", stored.value("recorded_at", int64_t(0))}, {"evidence_refs", stored.value("evidence_refs", json::array())}}});
        }
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.sequence < b.sequence; });
        const auto total_messages = rows.size();
        const size_t limit = std::clamp<size_t>(request.has_param("limit")
            ? static_cast<size_t>(std::max(1, std::atoi(request.get_param_value("limit").c_str()))) : 50, 1, 100);
        size_t end = total_messages;
        if (request.has_param("cursor")) {
            try { end = static_cast<size_t>(std::stoull(request.get_param_value("cursor"))); }
            catch (...) { set_json_error(response, 400, "history cursor is invalid"); return; }
        }
        if (end > total_messages) { set_json_error(response, 409, "history cursor is stale; reload the chat"); return; }
        const size_t limit_begin = end > limit ? end - limit : 0;
        size_t page_begin = end;
        size_t page_content_bytes = 0;
        std::vector<json> reversed_page;
        for (size_t index = end; index > limit_begin; --index) {
            auto & message = rows[index - 1].value;
            const auto content_bytes = message.value("content", std::string()).size();
            if (!reversed_page.empty() && page_content_bytes + content_bytes > 512 * 1024) break;
            message["sequence"] = rows[index - 1].sequence;
            reversed_page.push_back(message);
            page_content_bytes += content_bytes;
            page_begin = index - 1;
        }
        json messages = json::array();
        for (auto it = reversed_page.rbegin(); it != reversed_page.rend(); ++it) messages.push_back(std::move(*it));
        const auto next = page_begin > 0 ? std::to_string(page_begin) : std::string();
        chat.erase("catalog_revision");
        const bool history_available = total_messages > 0 || chat.value("turn_count", size_t(0)) == 0;
        response.set_content(json{{"ok", true}, {"chat", chat}, {"messages", std::move(messages)},
            {"history_available", history_available}, {"next_cursor", next}}.dump(), "application/json");
    });

    server.Post(R"(/api/v1/chats/([^/]+)/resources)", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        const auto input = json::parse(request.body, nullptr, false);
        if (!input.is_object() || !input.contains("resource_uri") || !input["resource_uri"].is_string() ||
                !input.contains("scope") || !input["scope"].is_string() || !input.contains("action") || !input["action"].is_string() ||
                (input.contains("project_id") && !input["project_id"].is_string())) {
            set_json_error(response, 400, "resource association requires resource_uri, scope and action"); return;
        }
        const auto session_id = request.matches[1].str();
        const auto project_id = input.value("project_id", std::string());
        const auto uri = input["resource_uri"].get<std::string>();
        const auto resource_scope = input["scope"].get<std::string>();
        const auto action = input["action"].get<std::string>();
        if (uri.empty() || (resource_scope != "project" && resource_scope != "session") || (action != "attach" && action != "detach")) {
            set_json_error(response, 400, "resource association arguments are invalid"); return;
        }
        json chat; std::string error;
        if (!validate_chat_scope(options, session_id, project_id, chat, error)) { set_json_error(response, error == k_chat_not_found ? 404 : 403, error); return; }
        json listing;
        if (!daemon_command(options, {{"command", "list_resources"}, {"namespace_id", k_web_namespace},
                {"session_id", session_id}, {"project_id", project_id}}, listing, error) || !listing.value("ok", false)) {
            set_json_error(response, 502, error.empty() ? listing.value("error", "resource authorization failed") : error); return;
        }
        if (!listing.contains("resources") || !listing["resources"].is_array()) { set_json_error(response, 502, "resource authorization response is invalid"); return; }
        bool authorized = false;
        for (const auto & item : listing.value("resources", json::array()))
            if (item.is_object() && item.value("uri", std::string()) == uri && item.value("scope", std::string()) == resource_scope) authorized = true;
        if (!authorized) { set_json_error(response, 403, "resource is outside the chat's authorized scope"); return; }
        auto refs = chat.value("attachments", json::array());
        if (!refs.is_array()) refs = json::array();
        auto found = std::find_if(refs.begin(), refs.end(), [&](const json & item) { return item.is_object() && item.value("uri", std::string()) == uri; });
        if (action == "detach") {
            if (found != refs.end()) refs.erase(found);
        } else if (found == refs.end()) {
            const auto descriptor = std::find_if(listing["resources"].begin(), listing["resources"].end(), [&](const json & item) { return item.is_object() && item.value("uri", std::string()) == uri && item.value("scope", std::string()) == resource_scope; });
            refs.push_back({{"uri", uri}, {"name", descriptor->value("name", std::string("Resource"))},
                {"mime_type", descriptor->value("mime_type", std::string("application/octet-stream"))}});
        }
        chat["attachments"] = std::move(refs);
        chat["last_active_at"] = static_cast<int64_t>(std::time(nullptr));
        if (!put_catalog_record(options, chat, error)) { set_json_error(response, 502, error); return; }
        response.set_content(json{{"ok", true}, {"chat", chat}}.dump(), "application/json");
    });

    server.Post("/api/v1/turns", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        auto command = json::parse(request.body, nullptr, false);
        if (command.is_discarded() || !command.is_object()) {
            set_json_error(response, 400, "request body must be a JSON object");
            return;
        }
        std::string session_id, project_id, catalog_error;
        if (!parse_chat_scope(command, session_id, project_id, catalog_error)) { set_json_error(response, 400, catalog_error); return; }
        if (!command.contains("turn_id") || !command["turn_id"].is_string() || command["turn_id"].get<std::string>().empty()) {
            set_json_error(response, 400, "turn_id is required"); return;
        }
        if (!command.contains("prompt") || !command["prompt"].is_string()) { set_json_error(response, 400, "prompt must be a string"); return; }
        command["namespace_id"] = k_web_namespace;
        json chat;
        if (!validate_chat_scope(options, session_id, project_id, chat, catalog_error)) {
            set_json_error(response, 403, catalog_error);
            return;
        }
        const auto now = static_cast<int64_t>(std::time(nullptr));
        chat["last_active_at"] = now;
        chat["turn_count"] = chat.value("turn_count", size_t(0)) + 1;
        if (chat.value("title", std::string()) == "New chat") {
            auto title = command.value("prompt", std::string());
            const auto newline = title.find_first_of("\r\n");
            if (newline != std::string::npos) title.resize(newline);
            if (title.size() > 80) title.resize(77), title += "...";
            if (!title.empty()) chat["title"] = title;
        }
        if (!put_catalog_record(options, chat, catalog_error)) { set_json_error(response, 502, catalog_error); return; }
        if (!project_id.empty()) {
            json project;
            if (!find_project(options, project_id, project, catalog_error)) { set_json_error(response, 403, catalog_error); return; }
            project["last_active_at"] = now;
            if (!put_catalog_record(options, project, catalog_error)) { set_json_error(response, 502, catalog_error); return; }
        }
        command["command"] = "run_turn";
        command["request_id"] = web_request_id(command);
        const auto turn_id = command["turn_id"].get<std::string>();
        {
            std::lock_guard<std::mutex> lock(active_web_turn_mutex);
            if (active_web_turns.count(turn_id)) { set_json_error(response, 409, "turn id is already active"); return; }
            active_web_turns.emplace(turn_id, active_web_turn{session_id, project_id});
        }
        json result;
        std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            { std::lock_guard<std::mutex> lock(active_web_turn_mutex); active_web_turns.erase(turn_id); }
            set_json_error(response, 502, command_error);
            return;
        }
        { std::lock_guard<std::mutex> lock(active_web_turn_mutex); active_web_turns.erase(turn_id); }
        response.set_content(result.dump(), "application/json");
    });

    server.Post(R"(/api/v1/chats/([^/]+)/turns/([^/]+)/cancel)", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        const auto session_id = request.matches[1].str();
        const auto turn_id = request.matches[2].str();
        const auto input = json::parse(request.body, nullptr, false);
        if (!input.is_object() || (input.contains("project_id") && !input["project_id"].is_string())) { set_json_error(response, 400, "cancel request is invalid"); return; }
        const auto project_id = input.value("project_id", std::string());
        json chat; std::string scope_error;
        if (!validate_chat_scope(options, session_id, project_id, chat, scope_error)) { set_json_error(response, 403, scope_error); return; }
        {
            std::lock_guard<std::mutex> lock(active_web_turn_mutex);
            const auto active = active_web_turns.find(turn_id);
            if (active == active_web_turns.end() || active->second.session_id != session_id || active->second.project_id != project_id) {
                set_json_error(response, 404, "active turn was not found in this chat"); return;
            }
        }
        json command = {
            {"command", "cancel_turn"},
            {"target_turn_id", turn_id},
        };
        json result;
        std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            set_json_error(response, 502, command_error);
            return;
        }
        response.set_content(result.dump(), "application/json");
    });

    server.Post("/api/v1/tools/call", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        auto input = json::parse(request.body, nullptr, false);
        if (input.is_discarded() || !input.is_object() ||
                !input.contains("tool_name") || !input["tool_name"].is_string() ||
                !input.contains("arguments") || !input["arguments"].is_object()) {
            set_json_error(response, 400, "tool call requires tool_name and object arguments");
            return;
        }
        std::string session_id, project_id, scope_error;
        if (!parse_chat_scope(input, session_id, project_id, scope_error)) { set_json_error(response, 400, scope_error); return; }
        json chat;
        if (!validate_chat_scope(options, session_id, project_id, chat, scope_error)) { set_json_error(response, 403, scope_error); return; }
        json command = {
            {"command", "execute_tool"},
            {"request_id", web_request_id(input)},
            {"tool_name", input["tool_name"]},
            {"arguments", input["arguments"]},
            {"read_only_only", true},
            {"namespace_id", k_web_namespace},
            {"session_id", session_id},
            {"project_id", project_id},
        };
        json result;
        std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            set_json_error(response, 502, command_error);
            return;
        }
        response.status = result.value("ok", false) ? 200 : 422;
        response.set_content(result.dump(), "application/json");
    });

    server.Post("/api/v1/mcp-apps/resource", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        auto input = json::parse(request.body, nullptr, false);
        if (input.is_discarded() || !input.is_object() ||
                !input.contains("tool_name") || !input["tool_name"].is_string() ||
                !input.contains("resource_uri") || !input["resource_uri"].is_string()) {
            set_json_error(response, 400, "MCP App resource read requires tool_name and resource_uri");
            return;
        }
        std::string session_id, project_id, scope_error;
        if (!parse_chat_scope(input, session_id, project_id, scope_error)) { set_json_error(response, 400, scope_error); return; }
        json chat;
        if (!validate_chat_scope(options, session_id, project_id, chat, scope_error)) { set_json_error(response, 403, scope_error); return; }
        json command = {
            {"command", "execute_tool"},
            {"request_id", web_request_id(input)},
            {"tool_name", input["tool_name"]},
            {"arguments", json::object()},
            {"mcp_app_resource_uri", input["resource_uri"]},
            {"namespace_id", k_web_namespace},
            {"session_id", session_id},
            {"project_id", project_id},
        };
        json result; std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            set_json_error(response, 502, command_error); return;
        }
        response.status = result.value("ok", false) ? 200 : 422;
        response.set_content(result.dump(), "application/json");
    });

    server.Post("/api/v1/mcp-apps/call", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        auto input = json::parse(request.body, nullptr, false);
        if (input.is_discarded() || !input.is_object() ||
                !input.contains("tool_name") || !input["tool_name"].is_string() ||
                !input.contains("app_tool_name") || !input["app_tool_name"].is_string() ||
                !input.contains("arguments") || !input["arguments"].is_object()) {
            set_json_error(response, 400, "MCP App tool call requires app_tool_name, tool_name and object arguments");
            return;
        }
        std::string session_id, project_id, scope_error;
        if (!parse_chat_scope(input, session_id, project_id, scope_error)) { set_json_error(response, 400, scope_error); return; }
        json chat;
        if (!validate_chat_scope(options, session_id, project_id, chat, scope_error)) { set_json_error(response, 403, scope_error); return; }
        json command = {
            {"command", "execute_tool"},
            {"request_id", web_request_id(input)},
            {"tool_name", input["tool_name"]},
            {"arguments", input["arguments"]},
            {"mcp_app_call", true},
            {"mcp_app_tool_name", input["app_tool_name"]},
            {"namespace_id", k_web_namespace},
            {"session_id", session_id},
            {"project_id", project_id},
        };
        json result; std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            set_json_error(response, 502, command_error); return;
        }
        response.status = result.value("ok", false) ? 200 : 422;
        response.set_content(result.dump(), "application/json");
    });

    server.Post("/api/v1/resources", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        auto command = json::parse(request.body, nullptr, false);
        if (command.is_discarded() || !command.is_object()) {
            set_json_error(response, 400, "request body must be a JSON object");
            return;
        }
        std::string session_id, project_id, scope_error;
        if (!parse_chat_scope(command, session_id, project_id, scope_error)) { set_json_error(response, 400, scope_error); return; }
        if (command.contains("scope") && (!command["scope"].is_string() || command["scope"] != "session")) {
            set_json_error(response, 400, "web attachments must use session scope"); return;
        }
        json chat;
        if (!validate_chat_scope(options, session_id, project_id, chat, scope_error)) { set_json_error(response, 403, scope_error); return; }
        command["namespace_id"] = k_web_namespace;
        command["command"] = "put_resource";
        json result;
        std::string command_error;
        if (!send_daemon_command(options, command, result, command_error)) {
            set_json_error(response, 502, command_error);
            return;
        }
        if (!result.value("ok", false)) { response.status = 422; response.set_content(result.dump(), "application/json"); return; }
        if (result.contains("resource") && result["resource"].is_object()) {
            const auto & resource = result["resource"];
            const auto uri = resource.value("uri", std::string());
            if (!uri.empty()) {
                auto refs = chat.value("attachments", json::array());
                if (!refs.is_array()) refs = json::array();
                const bool exists = std::any_of(refs.begin(), refs.end(), [&](const json & item) {
                    return item.is_object() && item.value("uri", std::string()) == uri;
                });
                if (!exists) refs.push_back({{"uri", uri}, {"name", resource.value("name", std::string("Resource"))},
                    {"mime_type", resource.value("mime_type", std::string("application/octet-stream"))}});
                chat["attachments"] = std::move(refs);
                chat["last_active_at"] = static_cast<int64_t>(std::time(nullptr));
                if (!put_catalog_record(options, chat, scope_error)) { set_json_error(response, 502, scope_error); return; }
            }
        }
        response.set_content(result.dump(), "application/json");
    });

    server.Get("/api/v1/resources", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        const auto scope = request_param_or(request, "scope", "project");
        const auto session_id = request_param_or(request, "session_id", "");
        const auto project_id = request_param_or(request, "project_id", "");
        if (scope != "project" && scope != "session") { set_json_error(response, 400, "resource scope must be project or session"); return; }
        if (scope == "project") {
            json project; std::string scope_error;
            if (project_id.empty() || !find_project(options, project_id, project, scope_error)) { set_json_error(response, 403, project_id.empty() ? "project_id is required" : scope_error); return; }
        } else {
            json chat; std::string scope_error;
            if (!validate_chat_scope(options, session_id, project_id, chat, scope_error)) { set_json_error(response, 403, scope_error); return; }
        }
        json result; std::string error;
        if (!daemon_command(options, {{"command", "list_resources"}, {"namespace_id", k_web_namespace},
                {"session_id", session_id}, {"project_id", project_id}}, result, error) || !result.value("ok", false)) {
            set_json_error(response, 502, error.empty() ? result.value("error", "resource list failed") : error); return;
        }
        const auto query = request_param_or(request, "q", "");
        const auto mime = request_param_or(request, "mime_type", "");
        std::vector<json> filtered;
        for (const auto & item : result.value("resources", json::array())) {
            if (!item.is_object() || has_prefix(item.value("name", std::string()), "llama-agent-web-")) continue;
            if (item.value("scope", std::string()) != scope) continue;
            if (item.contains("metadata") && item["metadata"].is_object() &&
                    has_prefix(item["metadata"].value("purpose", std::string()), "agent.context.")) continue;
            const auto name = item.value("name", std::string());
            const auto description = item.value("description", std::string());
            if (!query.empty() && name.find(query) == std::string::npos && description.find(query) == std::string::npos) continue;
            if (!mime.empty() && item.value("mime_type", std::string()).find(mime) == std::string::npos) continue;
            filtered.push_back(item);
        }
        size_t offset = 0;
        if (request.has_param("cursor")) { try { offset = static_cast<size_t>(std::stoull(request.get_param_value("cursor"))); } catch (...) { set_json_error(response, 400, "resource cursor is invalid"); return; } }
        if (offset > filtered.size()) { set_json_error(response, 400, "resource cursor is out of range"); return; }
        const size_t limit = std::clamp<size_t>(request.has_param("limit") ? static_cast<size_t>(std::max(1, std::atoi(request.get_param_value("limit").c_str()))) : 30, 1, 100);
        json resources = json::array();
        for (size_t i = offset; i < filtered.size() && i < offset + limit; ++i) resources.push_back(std::move(filtered[i]));
        const auto next = offset + resources.size() < filtered.size() ? std::to_string(offset + resources.size()) : std::string();
        response.set_content(json{{"ok", true}, {"resources", std::move(resources)}, {"next_cursor", next}}.dump(), "application/json");
    });

    server.Get("/api/v1/resources/download", [&options, authorize](const httplib::Request & request, httplib::Response & response) {
        if (!authorize(request, response)) return;
        if (!request.has_param("uri")) {
            set_json_error(response, 400, "resource download requires uri");
            return;
        }
        const auto uri = request.get_param_value("uri");
        const auto session_id = request_param_or(request, "session_id", "");
        const auto project_id = request_param_or(request, "project_id", "");
        if (session_id.empty()) { set_json_error(response, 400, "resource download requires an authorized chat session"); return; }
        if (!project_id.empty()) { json project; std::string scope_error; if (!find_project(options, project_id, project, scope_error)) { set_json_error(response, 403, scope_error); return; } }
        { json chat; std::string scope_error; if (!validate_chat_scope(options, session_id, project_id, chat, scope_error)) { set_json_error(response, 403, scope_error); return; } }
        const json command = {
            {"command", "read_resource"},
            {"encoding", "base64"},
            {"uri", uri},
            {"namespace_id", k_web_namespace},
            {"session_id", session_id},
            {"project_id", project_id},
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
        if (result.value("content_encoding", std::string()) != "base64") {
            set_json_error(response, 502, "resource download did not return byte-safe content");
            return;
        }
        try {
            const auto bytes = base64::decode(result["content"].get<std::string>());
            if (bytes.size() > options.max_download_bytes) {
                set_json_error(response, 502, "resource download exceeds its byte limit");
                return;
            }
            response.set_content(bytes, resource.value("mime_type", "application/octet-stream"));
        } catch (const std::exception &) {
            set_json_error(response, 502, "resource download returned invalid byte encoding");
        }
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
            [connection, started = false](size_t, httplib::DataSink & sink) mutable {
                if (!started) {
                    started = true;
                    constexpr char ready[] = ": connected\n\n";
                    return sink.write(ready, sizeof(ready) - 1);
                }
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
