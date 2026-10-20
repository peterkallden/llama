#include "tools/agent/openapi/agent-openapi-http.h"
#include "tools/agent/data/agent-investigation-dataset-source.h"

#include <iostream>

int main() {
    using json = nlohmann::ordered_json;
    agent_host_investigation_source_config source;
    source.id = "logs";
    source.base_url = "https://search.example.invalid";
    source.allowed_indices = {"logs-2026.10"};
    source.allowed_namespaces = {"tenant-a"};
    source.max_page_size = 10;
    agent_tool_context context;
    context.scope.namespace_id = "tenant-a";
    std::string response, error;

    if (agent_investigation_http_search(context, source, "other-index",
            json{{"query", {{"match_all", json::object()}}}}, response, error) ||
            error.find("allowlist") == std::string::npos) return 1;
    if (agent_investigation_http_search(context, source, "logs-2026.10",
            json{{"query", {{"match_all", json::object()}}}, {"size", 11}}, response, error) ||
            error.find("page bound") == std::string::npos) return 2;
    if (agent_investigation_http_search(context, source, "logs-2026.10",
            json{{"query", json::object()}, {"script_fields", json::object()}}, response, error) ||
            error.find("does not accept scripts") == std::string::npos) return 3;
    if (agent_investigation_http_search(context, source, "logs-2026.10",
            json{{"query", {{"match_all", json::object()}}}, {"size", 10}}, response, error) ||
            error.find("network capability") == std::string::npos) {
        std::cerr << "investigation HTTP connector did not honor caller network policy: " << error << '\n';
        return 4;
    }
    agent_host_investigation_config config;
    config.enabled = true;
    config.sources.push_back(source);
    agent_investigation_virtual_dataset_source virtual_source(config);
    common_agent_dataset_descriptor descriptor;
    descriptor.ref.uri = "dataset://investigation/logs/logs-2026.10";
    descriptor.ref.source_provider = "logs";
    descriptor.ref.source_request_json = R"({"index":"logs-2026.10"})";
    if (virtual_source.execute(descriptor, "data.query",
            R"({"dataset":"dataset://investigation/logs/logs-2026.10"})", context.scope,
            false, response, error) || error.find("network permission") == std::string::npos) {
        std::cerr << "virtual dataset adapter did not enforce the resolved host permission: " << error << '\n';
        return 5;
    }
    context.scope.namespace_id = "tenant-b";
    if (virtual_source.execute(descriptor, "data.query",
            R"({"dataset":"dataset://investigation/logs/logs-2026.10"})", context.scope,
            true, response, error) || error.find("not authorized for this namespace") == std::string::npos) {
        std::cerr << "virtual dataset adapter did not enforce the configured namespace: " << error << '\n';
        return 6;
    }
    return 0;
}
