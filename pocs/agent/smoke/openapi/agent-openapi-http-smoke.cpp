#include "tools/agent/openapi/agent-openapi-http.h"

#include <cpp-httplib/httplib.h>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <thread>

int main() {
    httplib::Server server;
    server.Get("/sales/42", [](const httplib::Request & request, httplib::Response & response) {
        assert(request.get_param_value("limit") == "10");
        response.set_content(R"({"id":42,"total":7})", "application/json");
    });
    server.Get("/basic", [](const httplib::Request & request, httplib::Response & response) {
        assert(request.get_header_value("Authorization") == "Basic dXNlcjpwYXNz");
        response.set_content(R"({"ok":true})", "application/json");
    });
    server.Get("/key", [](const httplib::Request & request, httplib::Response & response) {
        assert(request.get_param_value("api_key") == "key-123");
        response.set_content(R"({"ok":true})", "application/json");
    });
    int oauth_token_requests = 0;
    server.Post("/token", [&oauth_token_requests](const httplib::Request & request, httplib::Response & response) {
        ++oauth_token_requests;
        assert(request.get_header_value("Authorization") == "Basic b2F1dGgtaWQ6b2F1dGgtc2VjcmV0");
        assert(request.get_param_value("grant_type") == "client_credentials");
        response.set_content(R"({"access_token":"oauth-access","expires_in":300})", "application/json");
    });
    server.Get("/oauth", [](const httplib::Request & request, httplib::Response & response) {
        assert(request.get_header_value("Authorization") == "Bearer oauth-access");
        response.set_content(R"({"ok":true})", "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    assert(port > 0);
    std::thread server_thread([&server] { server.listen_after_bind(); });

    agent_host_openapi_provider_config config;
    config.id = "sales-api";
    config.base_url = "http://127.0.0.1:" + std::to_string(port);
    config.connect_timeout_ms = 1000;
    config.request_timeout_ms = 2000;
    config.max_result_bytes = 1024;
    config.allow_private_network = true;
    agent_openapi_catalog catalog;
    catalog.provider_id = config.id;
    catalog.prefix = "sales";
    catalog.operations.push_back({
        "getSale", "get", "/sales/{id}", "Get sale", "Get sale", R"({"type":"object"})",
        {"id"}, {"limit"}, agent_openapi_access::read, true, false});

    int observed_status = 0;
    std::string observed_mime;
    agent_openapi_tool_provider provider(std::move(catalog), make_agent_openapi_http_executor(config),
        [&](const agent_tool_context &, const agent_openapi_operation &,
            const std::string &,
            agent_openapi_execution_result & execution, std::string &) {
            observed_status = execution.http_status;
            observed_mime = execution.mime_type;
            return true;
        });
    agent_tool_context context;
    context.allow_network = true;
    std::string error;
    auto view = provider.resolve_tools(context, error);
    assert(view != nullptr);
    const auto result = view->call({"http-1", "sales.getSale", R"({"id":"42","limit":"10"})"}, error);
    assert(result.ok);
    assert(result.content_json.find("\"total\":7") != std::string::npos);
    assert(observed_status == 200);
    assert(observed_mime == "application/json");

    config.allow_private_network = false;
    agent_openapi_tool_provider private_denied_provider(
        agent_openapi_catalog{
            "sales-api", config.base_url, "sales", {
                {"getSale", "get", "/sales/{id}", "Get sale", "Get sale", R"({"type":"object"})",
                    {"id"}, {"limit"}, agent_openapi_access::read, true, false}}},
        make_agent_openapi_http_executor(config));
    auto private_denied_view = private_denied_provider.resolve_tools(context, error);
    assert(private_denied_view != nullptr);
    const auto private_denied = private_denied_view->call(
        {"http-private", "sales.getSale", R"({"id":"42"})"}, error);
    assert(!private_denied.ok);

    context.allow_network = false;
    agent_openapi_tool_provider denied_provider(std::move(agent_openapi_catalog{
        "sales-api", config.base_url, "sales", {
            {"getSale", "get", "/sales/{id}", "Get sale", "Get sale", R"({"type":"object"})",
                {"id"}, {"limit"}, agent_openapi_access::read, true, false}}}),
        make_agent_openapi_http_executor(config));
    auto denied_view = denied_provider.resolve_tools(context, error);
    assert(denied_view != nullptr);
    const auto denied = denied_view->call({"http-2", "sales.getSale", R"({"id":"42"})"}, error);
    assert(!denied.ok);

    setenv("OPENAPI_BASIC_USER", "user", 1);
    setenv("OPENAPI_BASIC_PASSWORD", "pass", 1);
    setenv("OPENAPI_API_KEY", "key-123", 1);
    context.allow_network = true;
    auto run_authenticated = [&](agent_host_openapi_provider_config authenticated_config,
            agent_openapi_operation authenticated_operation) {
        agent_openapi_catalog authenticated_catalog;
        authenticated_catalog.provider_id = authenticated_config.id;
        authenticated_catalog.base_url = authenticated_config.base_url;
        authenticated_catalog.prefix = authenticated_config.prefix;
        authenticated_catalog.operations.push_back(std::move(authenticated_operation));
        agent_openapi_tool_provider authenticated_provider(
            std::move(authenticated_catalog),
            make_agent_openapi_http_executor(authenticated_config));
        std::string authenticated_error;
        auto authenticated_view = authenticated_provider.resolve_tools(context, authenticated_error);
        assert(authenticated_view != nullptr);
        const auto authenticated_result = authenticated_view->call(
            {"auth-call", authenticated_config.prefix + ".auth", "{}"}, authenticated_error);
        assert(authenticated_result.ok);
        const auto cached_result = authenticated_view->call(
            {"auth-call-2", authenticated_config.prefix + ".auth", "{}"}, authenticated_error);
        assert(cached_result.ok);
    };
    agent_openapi_operation basic_operation;
    basic_operation.operation_id = "auth";
    basic_operation.method = "get";
    basic_operation.path = "/basic";
    basic_operation.access = agent_openapi_access::read;
    basic_operation.read_only = true;
    basic_operation.auth_required = true;
    basic_operation.security_schemes = {"basicAuth"};
    basic_operation.security_definitions = {{"basicAuth", "http", "basic", "", ""}};
    agent_host_openapi_provider_config basic_config = config;
    basic_config.prefix = "basic";
    basic_config.auth.type = "basic";
    basic_config.auth.scheme = "basicAuth";
    basic_config.auth.username_env = "OPENAPI_BASIC_USER";
    basic_config.auth.password_env = "OPENAPI_BASIC_PASSWORD";
    run_authenticated(basic_config, basic_operation);

    agent_openapi_operation api_key_operation;
    api_key_operation.operation_id = "auth";
    api_key_operation.method = "get";
    api_key_operation.path = "/key";
    api_key_operation.access = agent_openapi_access::read;
    api_key_operation.read_only = true;
    api_key_operation.auth_required = true;
    api_key_operation.security_schemes = {"apiKeyAuth"};
    api_key_operation.security_definitions = {{"apiKeyAuth", "apiKey", "", "api_key", "query"}};
    agent_host_openapi_provider_config api_key_config = config;
    api_key_config.prefix = "key";
    api_key_config.auth.type = "api_key";
    api_key_config.auth.scheme = "apiKeyAuth";
    api_key_config.auth.token_env = "OPENAPI_API_KEY";
    run_authenticated(api_key_config, api_key_operation);

    setenv("OPENAPI_OAUTH_ID", "oauth-id", 1);
    setenv("OPENAPI_OAUTH_SECRET", "oauth-secret", 1);
    agent_openapi_operation oauth_operation;
    oauth_operation.operation_id = "auth";
    oauth_operation.method = "get";
    oauth_operation.path = "/oauth";
    oauth_operation.access = agent_openapi_access::read;
    oauth_operation.read_only = true;
    oauth_operation.auth_required = true;
    oauth_operation.security_schemes = {"oauthAuth"};
    oauth_operation.security_definitions = {{"oauthAuth", "oauth2", "", "", "", "clientCredentials", ""}};
    agent_host_openapi_provider_config oauth_config = config;
    oauth_config.prefix = "oauth";
    oauth_config.auth.type = "oauth2_client_credentials";
    oauth_config.auth.scheme = "oauthAuth";
    oauth_config.auth.token_url = "http://127.0.0.1:" + std::to_string(port) + "/token";
    oauth_config.auth.client_id_env = "OPENAPI_OAUTH_ID";
    oauth_config.auth.client_secret_env = "OPENAPI_OAUTH_SECRET";
    run_authenticated(oauth_config, oauth_operation);
    assert(oauth_token_requests == 1);

    server.stop();
    server_thread.join();
    std::cout << "agent-openapi-http-smoke: ok\n";
}
