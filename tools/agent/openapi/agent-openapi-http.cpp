#include "agent-openapi-http.h"

#include <cpp-httplib/httplib.h>

#include <cstdlib>
#ifndef _WIN32
#include <cstring>
#include <netdb.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#endif
#include <sstream>
#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>

namespace {
struct url_parts { std::string scheme, host, base_path; int port = 0; };
struct oauth_token_cache {
    std::mutex mutex;
    std::string access_token;
    std::chrono::steady_clock::time_point expires_at{};
};

bool parse_url(const std::string & input, url_parts & url, std::string & error) {
    const auto scheme_end = input.find("://");
    if (scheme_end == std::string::npos) { error = "OpenAPI base_url must include a scheme"; return false; }
    url.scheme = input.substr(0, scheme_end);
    const auto authority_start = scheme_end + 3;
    const auto path_start = input.find('/', authority_start);
    const auto authority = input.substr(authority_start, path_start == std::string::npos ? std::string::npos : path_start - authority_start);
    if (authority.find('@') != std::string::npos) { error = "OpenAPI base_url must not contain userinfo"; return false; }
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

#ifndef _WIN32
bool private_address(const sockaddr * address) {
    if (address->sa_family == AF_INET) {
        const auto value = ntohl(reinterpret_cast<const sockaddr_in *>(address)->sin_addr.s_addr);
        return (value >> 24) == 10 || (value >> 20) == 0xAC1 || (value >> 16) == 0xC0A8 ||
            (value >> 24) == 127 || (value >> 28) == 0xE || (value >> 24) == 0;
    }
    if (address->sa_family == AF_INET6) {
        const auto * bytes = reinterpret_cast<const unsigned char *>(
            &reinterpret_cast<const sockaddr_in6 *>(address)->sin6_addr);
        return bytes[0] == 0 || bytes[0] == 0xFF || bytes[0] == 0xFC || bytes[0] == 0xFD ||
            (bytes[0] == 0xFE && (bytes[1] & 0xC0) == 0x80) ||
            (bytes[15] == 1 && std::all_of(bytes, bytes + 15, [](unsigned char byte) { return byte == 0; }));
    }
    return true;
}

bool resolve_validated_host(const std::string & host, std::string & address, bool & private_result) {
    addrinfo hints{}; hints.ai_socktype = SOCK_STREAM; hints.ai_family = AF_UNSPEC;
    addrinfo * results = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &results) != 0) return false;
    private_result = false;
    bool have_address = false;
    for (auto * item = results; item != nullptr; item = item->ai_next) {
        if (private_address(item->ai_addr)) private_result = true;
        char numeric[NI_MAXHOST] = {};
        if (!have_address && getnameinfo(item->ai_addr, item->ai_addrlen, numeric, sizeof(numeric), nullptr, 0, NI_NUMERICHOST) == 0) {
            address = numeric;
            have_address = true;
        }
    }
    freeaddrinfo(results);
    return have_address;
}
#else
bool resolve_validated_host(const std::string &, std::string &, bool &) {
    // Fail closed when the platform resolver policy is not implemented yet.
    return true;
}
#endif

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

template<typename Client>
bool configure_basic_auth(
        Client & client,
        const agent_host_openapi_provider_config & config,
        const char * username,
        const char * password,
        std::string & error) {
    if (config.auth.type == "basic") {
        if (username == nullptr || password == nullptr) {
            error = "OpenAPI basic auth environment variable is missing";
            return false;
        }
        client.set_basic_auth(username, password);
    }
    return true;
}

const agent_openapi_security_scheme * selected_security_scheme(
        const agent_openapi_operation & operation,
        const agent_host_openapi_provider_config & config,
        std::string & error) {
    if (!operation.auth_required) return nullptr;
    if (config.auth.type == "none") {
        error = "OpenAPI operation requires authentication but provider auth is none";
        return nullptr;
    }
    std::string selected = config.auth.scheme;
    if (selected.empty() && operation.security_schemes.size() == 1) {
        selected = operation.security_schemes.front();
    }
    if (selected.empty()) {
        error = "OpenAPI operation has multiple authentication alternatives; auth.scheme is required";
        return nullptr;
    }
    if (std::find(operation.security_schemes.begin(), operation.security_schemes.end(), selected) ==
            operation.security_schemes.end()) {
        error = "OpenAPI auth.scheme is not allowed by this operation: " + selected;
        return nullptr;
    }
    for (const auto & scheme : operation.security_definitions) {
        if (scheme.name == selected) return &scheme;
    }
    error = "OpenAPI auth.scheme was not found in the catalog: " + selected;
    return nullptr;
}

bool acquire_oauth_client_credentials_token(
        const agent_host_openapi_provider_config & config,
        const agent_openapi_security_scheme & scheme,
        const std::shared_ptr<oauth_token_cache> & cache,
        std::string & token,
        std::string & error) {
    std::lock_guard<std::mutex> lock(cache->mutex);
    const auto now = std::chrono::steady_clock::now();
    if (!cache->access_token.empty() && cache->expires_at > now + std::chrono::seconds(30)) {
        token = cache->access_token;
        return true;
    }
    const std::string token_url = config.auth.token_url.empty()
        ? scheme.token_url : config.auth.token_url;
    if (token_url.empty()) {
        error = "OpenAPI OAuth client credentials requires token_url";
        return false;
    }
    url_parts url;
    if (!parse_url(token_url, url, error)) {
        error = "OpenAPI OAuth token_url is invalid: " + error;
        return false;
    }
    if (url.scheme != "https" && !(url.scheme == "http" && config.allow_private_network)) {
        error = "OpenAPI OAuth token_url must use HTTPS";
        return false;
    }
    std::string validated_address;
    bool private_address_result = true;
#ifndef _WIN32
    if (!resolve_validated_host(url.host, validated_address, private_address_result)) {
        error = "OpenAPI OAuth token_url host could not be resolved";
        return false;
    }
#else
    resolve_validated_host(url.host, validated_address, private_address_result);
#endif
    if (!config.allow_private_network && private_address_result) {
        error = "OpenAPI OAuth token_url targets a private or local network address";
        return false;
    }
    const char * client_id = std::getenv(config.auth.client_id_env.c_str());
    const char * client_secret = std::getenv(config.auth.client_secret_env.c_str());
    if (client_id == nullptr || *client_id == '\0' || client_secret == nullptr || *client_secret == '\0') {
        error = "OpenAPI OAuth client credential environment variable is missing";
        return false;
    }
    httplib::Params form;
    form.emplace("grant_type", "client_credentials");
    if (!config.auth.scopes.empty()) {
        std::string scope;
        for (const auto & item : config.auth.scopes) {
            if (!scope.empty()) scope += ' ';
            scope += item;
        }
        form.emplace("scope", scope);
    }
    httplib::Headers headers{{"Accept", "application/json"}};
    httplib::Result response;
    if (url.scheme == "http") {
        httplib::Client client(url.host, url.port);
        client.set_hostname_addr_map({{url.host, validated_address}});
        client.set_follow_location(false);
        configure(client, config);
        client.set_basic_auth(client_id, client_secret);
        response = client.Post(url.base_path.empty() ? "/" : url.base_path, headers, form);
    } else {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient client(url.host, url.port);
        client.set_hostname_addr_map({{url.host, validated_address}});
        client.set_follow_location(false);
        configure(client, config);
        client.set_basic_auth(client_id, client_secret);
        response = client.Post(url.base_path.empty() ? "/" : url.base_path, headers, form);
#else
        error = "HTTPS OpenAPI OAuth token transport requires OpenSSL";
        return false;
#endif
    }
    if (!response) {
        error = "OpenAPI OAuth token request failed: " + httplib::to_string(response.error());
        return false;
    }
    if (response->status < 200 || response->status >= 300) {
        error = "OpenAPI OAuth token endpoint returned HTTP " + std::to_string(response->status);
        return false;
    }
    if (response->body.size() > config.max_result_bytes) {
        error = "OpenAPI OAuth token response exceeded max_result_bytes";
        return false;
    }
    const auto value = nlohmann::json::parse(response->body, nullptr, false);
    if (!value.is_object() || !value.value("access_token", "").size()) {
        error = "OpenAPI OAuth token response did not contain access_token";
        return false;
    }
    token = value.value("access_token", "");
    const auto expires_in = value.value("expires_in", int64_t(300));
    cache->access_token = token;
    cache->expires_at = now + std::chrono::seconds(std::max<int64_t>(60, expires_in - 30));
    return true;
}
}

agent_openapi_executor make_agent_openapi_http_executor(agent_host_openapi_provider_config config) {
    return [config = std::move(config), oauth_cache = std::make_shared<oauth_token_cache>()](
            const agent_tool_context & context, const agent_openapi_operation & operation,
            const std::string & arguments_json, agent_openapi_execution_result & result,
            std::string & error) {
        if (!context.allow_network) { error = "OpenAPI tool requires network capability"; return false; }
        url_parts url;
        if (!parse_url(config.base_url, url, error)) return false;
        std::string validated_address;
        bool private_address_result = true;
#ifndef _WIN32
        if (!resolve_validated_host(url.host, validated_address, private_address_result)) {
            error = "OpenAPI base_url host could not be resolved";
            return false;
        }
#else
        resolve_validated_host(url.host, validated_address, private_address_result);
#endif
        if (!config.allow_private_network && private_address_result) {
            error = "OpenAPI request targets a private or local network address";
            return false;
        }
        const auto arguments = nlohmann::json::parse(arguments_json, nullptr, false);
        if (!arguments.is_object()) { error = "OpenAPI tool arguments must be a JSON object"; return false; }
        const auto * security_scheme = selected_security_scheme(operation, config, error);
        if (operation.auth_required && security_scheme == nullptr) return false;
        std::string path = operation.path;
        for (const auto & name : operation.path_parameters) {
            const auto it = arguments.find(name);
            if (it == arguments.end() || !it->is_string()) { error = "OpenAPI path parameter is missing or not a string: " + name; return false; }
            const std::string marker = "{" + name + "}";
            const auto position = path.find(marker);
            if (position != std::string::npos) path.replace(position, marker.size(), httplib::encode_uri_component(it->get<std::string>()));
        }
        httplib::Params query;
        for (const auto & name : operation.query_parameters) {
            const auto it = arguments.find(name);
            if (it == arguments.end()) continue;
            query.emplace(name, it->is_string() ? it->get<std::string>() : it->dump());
        }
        httplib::Headers headers;
        std::string token;
        if (operation.auth_required && (config.auth.type == "bearer" || config.auth.type == "api_key")) {
            const char * value = std::getenv(config.auth.token_env.c_str());
            if (value == nullptr || *value == '\0') { error = "OpenAPI credential environment variable is missing"; return false; }
            token = value;
        } else if (operation.auth_required && config.auth.type == "oauth2_client_credentials") {
            if (security_scheme->type != "oauth2" || security_scheme->flow != "clientCredentials") {
                error = "OpenAPI OAuth auth does not match a clientCredentials security scheme"; return false;
            }
            if (!acquire_oauth_client_credentials_token(config, *security_scheme, oauth_cache, token, error)) return false;
        }
        if (security_scheme != nullptr && (config.auth.type == "bearer" ||
                config.auth.type == "oauth2_client_credentials")) {
            if (config.auth.type == "bearer" &&
                    (security_scheme->type != "http" || security_scheme->scheme != "bearer")) {
                error = "OpenAPI bearer auth does not match the selected security scheme"; return false;
            }
            if (config.auth.type == "oauth2_client_credentials" && security_scheme->type != "oauth2") {
                error = "OpenAPI OAuth auth does not match the selected security scheme"; return false;
            }
            headers.emplace("Authorization", std::string("Bearer ") + token);
        } else if (security_scheme != nullptr && config.auth.type == "api_key") {
            if (security_scheme->type != "apiKey" || security_scheme->parameter_name.empty()) {
                error = "OpenAPI api_key auth does not match the selected security scheme"; return false;
            }
            if (security_scheme->location == "header") headers.emplace(security_scheme->parameter_name, token);
            else if (security_scheme->location == "query") query.emplace(security_scheme->parameter_name, token);
            else if (security_scheme->location == "cookie") headers.emplace("Cookie", security_scheme->parameter_name + "=" + token);
            else { error = "OpenAPI api_key security scheme has an unsupported location"; return false; }
        }
        const char * username = nullptr;
        const char * password = nullptr;
        if (operation.auth_required && config.auth.type == "basic") {
            username = std::getenv(config.auth.username_env.c_str());
            password = std::getenv(config.auth.password_env.c_str());
            if (security_scheme == nullptr || security_scheme->type != "http" || security_scheme->scheme != "basic") {
                error = "OpenAPI basic auth does not match the selected security scheme"; return false;
            }
        }
        if (operation.auth_required && config.auth.type == "mutual_tls" &&
                (security_scheme == nullptr || security_scheme->type != "mutualTLS")) {
            error = "OpenAPI mutual_tls auth does not match the selected security scheme"; return false;
        }
        std::string request_path = join_path(url.base_path, path);
        request_path = httplib::append_query_params(request_path, query);
        // Public APIs and API gateways commonly use the User-Agent for
        // diagnostics and abuse/rate-limit handling. Keep it deterministic
        // and host-owned; callers still cannot provide arbitrary headers.
        headers.emplace("User-Agent", "llama-agent-openapi-http/1.0");
        const std::string body = arguments.contains("body") ? arguments["body"].dump() : "{}";
        httplib::Result response;
        if (url.scheme == "http") {
            if (config.auth.type == "mutual_tls") { error = "OpenAPI mutual_tls requires HTTPS"; return false; }
            httplib::Client client(url.host, url.port); client.set_hostname_addr_map({{url.host, validated_address}}); client.set_follow_location(false); configure(client, config);
            if (!configure_basic_auth(client, config, username, password, error)) return false;
            if (operation.method == "get") response = client.Get(request_path, headers);
            else if (operation.method == "head") response = client.Head(request_path, headers);
            else if (operation.method == "post") response = client.Post(request_path, headers, body, "application/json");
            else if (operation.method == "put") response = client.Put(request_path, headers, body, "application/json");
            else if (operation.method == "patch") response = client.Patch(request_path, headers, body, "application/json");
            else if (operation.method == "delete") response = client.Delete(request_path, headers, body, "application/json");
            else { error = "unsupported OpenAPI HTTP method"; return false; }
        } else {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
            std::unique_ptr<httplib::SSLClient> client;
            if (config.auth.type == "mutual_tls") {
                const char * cert = std::getenv(config.auth.client_cert_path_env.c_str());
                const char * key = std::getenv(config.auth.client_key_path_env.c_str());
                if (cert == nullptr || *cert == '\0' || key == nullptr || *key == '\0') { error = "OpenAPI client certificate path environment variable is missing"; return false; }
                client = std::make_unique<httplib::SSLClient>(url.host, url.port, cert, key);
            } else {
                client = std::make_unique<httplib::SSLClient>(url.host, url.port);
            }
            client->set_hostname_addr_map({{url.host, validated_address}}); client->set_follow_location(false); configure(*client, config);
            if (!configure_basic_auth(*client, config, username, password, error)) return false;
            if (!config.auth.ca_cert_path_env.empty()) {
                const char * ca_path = std::getenv(config.auth.ca_cert_path_env.c_str());
                if (ca_path == nullptr || *ca_path == '\0') { error = "OpenAPI CA certificate path environment variable is missing"; return false; }
                client->set_ca_cert_path(ca_path);
            }
            if (operation.method == "get") response = client->Get(request_path, headers);
            else if (operation.method == "post") response = client->Post(request_path, headers, body, "application/json");
            else if (operation.method == "put") response = client->Put(request_path, headers, body, "application/json");
            else if (operation.method == "patch") response = client->Patch(request_path, headers, body, "application/json");
            else if (operation.method == "delete") response = client->Delete(request_path, headers, body, "application/json");
            else { error = "unsupported OpenAPI HTTPS method"; return false; }
#else
            error = "HTTPS OpenAPI transport requires OpenSSL"; return false;
#endif
        }
        if (!response) { error = "OpenAPI HTTP request failed: " + httplib::to_string(response.error()); return false; }
        if (response->body.size() > config.max_result_bytes) { error = "OpenAPI response exceeded max_result_bytes"; return false; }
        result.ok = response->status >= 200 && response->status < 300;
        result.http_status = response->status;
        result.mime_type = response->get_header_value("Content-Type");
        if (result.mime_type.empty()) result.mime_type = "application/octet-stream";
        result.structured_content_json = response->body;
        result.text_content = response->body;
        if (!result.ok) { result.failure_code = "openapi.http_status"; result.safe_summary = "The OpenAPI request returned an error status."; error = "OpenAPI request returned HTTP " + std::to_string(response->status); }
        return true;
    };
}
