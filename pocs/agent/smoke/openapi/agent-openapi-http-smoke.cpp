#include "tools/agent/openapi/agent-openapi-http.h"

#include <cpp-httplib/httplib.h>

#include <cassert>
#include <iostream>
#include <thread>

int main() {
    httplib::Server server;
    server.Get("/sales/42", [](const httplib::Request & request, httplib::Response & response) {
        assert(request.get_param_value("limit") == "10");
        response.set_content(R"({"id":42,"total":7})", "application/json");
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
    const auto result = view->call({"http-1", "sales.getSale", R"({"id":"42"})"}, error);
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

    server.stop();
    server_thread.join();
    std::cout << "agent-openapi-http-smoke: ok\n";
}
