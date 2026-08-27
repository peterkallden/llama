#include "agent-openapi-http.h"

#include <cpp-httplib/httplib.h>

#include <cstdlib>
#include <sstream>

namespace {
struct url_parts { std::string scheme, host, base_path; int port = 0; };

bool parse_url(const std::string & input, url_parts & url, std::string & error) {
    const auto scheme_end = input.find("://");
    if (scheme_end == std::string::npos) { error = "OpenAPI base_url must include a scheme"; return false; }
    url.scheme = input.substr(0, scheme_end);
    const auto authority_start = scheme_end + 3;
    const auto path_start = input.find('/', authority_start);
    const auto authority = input.substr(authority_start, path_start == std::string::npos ? std::string::npos : path_start - authority_start);
    const auto port_start = authority.rfind(':');
    if (port_start != std::string::npos) {
        url.host = authority.substr(0, port_start);
        url.port = std::atoi(authority.substr(port_start + 1).c_str());
    } else {
        url.host = authority;
        url.port = url.scheme == "https" ? 443 : 80;
    }
    url.base_path = path_start == std::string::npos ? "" : input.substr(path_start);
    if ((url.scheme != "http" && url.scheme != "https") || url.host.empty() || url.port <= 0 || url.port > 65535) {
        error = "OpenAPI base_url has an unsupported scheme or invalid host";
        return false;
    }
    return true;
}

template<typename Client>
void configure(Client & client, const agent_host_openapi_provider_config & config) {
    client.set_connection_timeout(config.connect_timeout_ms / 1000, static_cast<time_t>((config.connect_timeout_ms % 1000) * 1000));
    client.set_read_timeout(config.request_timeout_ms / 1000, static_cast<time_t>((config.request_timeout_ms % 1000) * 1000));
}

std::string join_path(const std::string & base, const std::string & path) {
    if (base.empty()) return path.empty() ? "/" : path;
    if (path.empty()) return base;
    return base.back() == '/' && path.front() == '/' ? base + path.substr(1) : base + path;
}
}

agent_openapi_executor make_agent_openapi_http_executor(agent_host_openapi_provider_config config) {
    return [config = std::move(config)](
            const agent_tool_context & context, const agent_openapi_operation & operation,
            const std::string & arguments_json, agent_openapi_execution_result & result,
            std::string & error) {
        if (!context.allow_network) { error = "OpenAPI tool requires network capability"; return false; }
        url_parts url;
        if (!parse_url(config.base_url, url, error)) return false;
        const auto arguments = nlohmann::json::parse(arguments_json, nullptr, false);
        if (!arguments.is_object()) { error = "OpenAPI tool arguments must be a JSON object"; return false; }
        std::string path = operation.path;
        for (auto it = arguments.begin(); it != arguments.end(); ++it) {
            const std::string marker = "{" + it.key() + "}";
            const auto position = path.find(marker);
            if (position != std::string::npos && it.value().is_string()) path.replace(position, marker.size(), it.value().get<std::string>());
        }
        httplib::Headers headers;
        if (config.auth_type == "bearer" && !config.token_env.empty()) {
            const char * token = std::getenv(config.token_env.c_str());
            if (token == nullptr || *token == '\0') { error = "OpenAPI bearer token environment variable is missing"; return false; }
            headers.emplace("Authorization", std::string("Bearer ") + token);
        }
        const std::string request_path = join_path(url.base_path, path);
        const std::string body = arguments.contains("body") ? arguments["body"].dump() : arguments_json;
        httplib::Result response;
        if (url.scheme == "http") {
            httplib::Client client(url.host, url.port); configure(client, config);
            if (operation.method == "get") response = client.Get(request_path, headers);
            else if (operation.method == "head") response = client.Head(request_path, headers);
            else if (operation.method == "post") response = client.Post(request_path, headers, body, "application/json");
            else if (operation.method == "put") response = client.Put(request_path, headers, body, "application/json");
            else if (operation.method == "patch") response = client.Patch(request_path, headers, body, "application/json");
            else if (operation.method == "delete") response = client.Delete(request_path, headers, body, "application/json");
            else { error = "unsupported OpenAPI HTTP method"; return false; }
        } else {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
            httplib::SSLClient client(url.host, url.port); configure(client, config);
            if (operation.method == "get") response = client.Get(request_path, headers);
            else if (operation.method == "post") response = client.Post(request_path, headers, body, "application/json");
            else if (operation.method == "put") response = client.Put(request_path, headers, body, "application/json");
            else if (operation.method == "patch") response = client.Patch(request_path, headers, body, "application/json");
            else if (operation.method == "delete") response = client.Delete(request_path, headers, body, "application/json");
            else { error = "unsupported OpenAPI HTTPS method"; return false; }
#else
            error = "HTTPS OpenAPI transport requires OpenSSL"; return false;
#endif
        }
        if (!response) { error = "OpenAPI HTTP request failed: " + httplib::to_string(response.error()); return false; }
        if (response->body.size() > config.max_result_bytes) { error = "OpenAPI response exceeded max_result_bytes"; return false; }
        result.ok = response->status >= 200 && response->status < 300;
        result.structured_content_json = response->body;
        result.text_content = response->body;
        if (!result.ok) { result.failure_code = "openapi.http_status"; result.safe_summary = "The OpenAPI request returned an error status."; error = "OpenAPI request returned HTTP " + std::to_string(response->status); }
        return true;
    };
}
