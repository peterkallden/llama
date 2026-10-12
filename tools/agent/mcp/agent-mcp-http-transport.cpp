#include "agent-mcp-http-transport.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <cstdlib>

namespace {
struct parsed_url { std::string scheme, host, path = "/"; int port = 0; };

bool parse_url(const std::string & value, parsed_url & result, std::string & error) {
    const size_t scheme_end = value.find("://");
    if (scheme_end == std::string::npos) { error = "MCP HTTP provider url must include a scheme"; return false; }
    result.scheme = value.substr(0, scheme_end);
    const size_t authority_start = scheme_end + 3;
    const size_t path_start = value.find('/', authority_start);
    const std::string authority = value.substr(authority_start,
        path_start == std::string::npos ? std::string::npos : path_start - authority_start);
    if (authority.empty()) { error = "MCP HTTP provider url is missing a host"; return false; }
    const size_t port_start = authority.rfind(':');
    if (port_start != std::string::npos && authority.find(']') == std::string::npos) {
        result.host = authority.substr(0, port_start);
        result.port = std::atoi(authority.substr(port_start + 1).c_str());
    } else { result.host = authority; result.port = result.scheme == "https" ? 443 : 80; }
    if (result.host.empty() || result.port <= 0 || result.port > 65535) {
        error = "MCP HTTP provider url has an invalid host or port"; return false;
    }
    result.path = path_start == std::string::npos ? "/" : value.substr(path_start);
    return true;
}

template<typename Client>
void configure_client(Client & client, uint32_t connect_ms, uint32_t request_ms) {
    client.set_connection_timeout(connect_ms / 1000, static_cast<time_t>((connect_ms % 1000) * 1000));
    client.set_read_timeout(request_ms / 1000, static_cast<time_t>((request_ms % 1000) * 1000));
}

class httplib_transport final : public agent_mcp_http_transport {
public:
    bool post(const agent_mcp_http_request & request,
            agent_mcp_http_response & response, std::string & error) override {
        parsed_url url;
        if (!parse_url(request.url, url, error)) return false;
        if (url.scheme != "http" && url.scheme != "https") {
            error = "unsupported MCP HTTP provider scheme: " + url.scheme; return false;
        }
        httplib::Headers headers;
        for (const auto & header : request.headers) headers.emplace(header.first, header.second);
        httplib::Result result;
        if (url.scheme == "http") {
            httplib::Client client(url.host, url.port);
            configure_client(client, request.connect_timeout_ms, request.request_timeout_ms);
            client.set_follow_location(false);
            result = client.Post(url.path, headers, request.body, "application/json");
        } else {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
            httplib::SSLClient client(url.host, url.port);
            configure_client(client, request.connect_timeout_ms, request.request_timeout_ms);
            client.set_follow_location(false);
            result = client.Post(url.path, headers, request.body, "application/json");
#else
            error = "HTTPS MCP transport requires an OpenSSL-enabled transport";
            return false;
#endif
        }
        if (!result) { error = "MCP HTTP request failed: " + httplib::to_string(result.error()); return false; }
        response.status = result->status;
        response.body = result->body;
        if (response.body.size() > request.max_result_bytes) { error = "MCP HTTP response exceeded max_result_bytes"; return false; }
        if (result->has_header("Mcp-Session-Id")) response.headers.emplace("Mcp-Session-Id", result->get_header_value("Mcp-Session-Id"));
        return true;
    }
};
}

std::shared_ptr<agent_mcp_http_transport> make_agent_mcp_httplib_transport() {
    return std::make_shared<httplib_transport>();
}
