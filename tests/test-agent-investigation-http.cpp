#include "tools/agent/openapi/agent-openapi-http.h"
#include "tools/agent/data/agent-investigation-dataset-source.h"

#include <iostream>
#include <cstdlib>

int main() {
    using json = nlohmann::ordered_json;
    if (!agent_investigation_search_page_complete(true, false, true, 0, 0, 4, 4) ||
            agent_investigation_search_page_complete(true, false, true, 1, 0, 4, 4) ||
            agent_investigation_search_page_complete(true, true, true, 0, 0, 4, 4) ||
            agent_investigation_search_page_complete(true, false, false, 0, 0, 4, 4) ||
            agent_investigation_search_page_complete(false, false, true, 0, 0, 4, 4)) return 9;
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

    // Optional real-service smoke. The regular CTest contract remains offline;
    // setting this endpoint exercises the same virtual-dataset adapter against
    // a running Elasticsearch/OpenSearch instance populated by the smoke runner.
    const char * real_endpoint = std::getenv("LLAMA_AGENT_INVESTIGATION_REAL_ENDPOINT");
    if (real_endpoint && *real_endpoint) {
        agent_host_investigation_source_config real_source;
        real_source.id = "real";
        real_source.dialect = std::getenv("LLAMA_AGENT_INVESTIGATION_REAL_DIALECT")
            ? std::getenv("LLAMA_AGENT_INVESTIGATION_REAL_DIALECT") : "elasticsearch";
        real_source.base_url = real_endpoint;
        real_source.allowed_indices = {"llama-agent-investigation-smoke"};
        real_source.allowed_namespaces = {"tenant-a"};
        real_source.allow_private_network = true;
        real_source.max_page_size = 2;
        real_source.fields = {
            {"event.id", "event.id"}, {"event.kind", "event.kind"},
            {"host.name", "host.name"}, {"@timestamp", "@timestamp"},
        };
        agent_host_investigation_config real_config;
        real_config.enabled = true;
        real_config.max_materialized_rows = 8;
        real_config.sources.push_back(real_source);
        agent_investigation_virtual_dataset_source real_virtual_source(real_config);
        common_agent_dataset_descriptor real_descriptor;
        real_descriptor.ref.uri = "dataset://investigation/real/llama-agent-investigation-smoke";
        real_descriptor.ref.source_provider = "real";
        real_descriptor.ref.source_request_json = R"({"index":"llama-agent-investigation-smoke"})";
        context.scope.namespace_id = "tenant-a";
        const auto request = json{{"dataset", real_descriptor.ref.uri},
            {"select", {"event.id", "host.name", "@source.index", "@source.id"}}, {"where", json::array({
                {{"field", "event.kind"}, {"operator", "="}, {"value", "auth"}}})},
            {"order_by", json::array({{{"field", "event.id"}, {"direction", "asc"}}})},
            {"limit", 2}, {"offset", 0}}.dump();
        if (!real_virtual_source.execute(real_descriptor, "data.query", request,
                context.scope, true, response, error)) {
            std::cerr << "real search provider smoke failed: " << error << '\n';
            return 7;
        }
        const auto page = json::parse(response, nullptr, false);
        if (!page.is_object() || !page.contains("rows") || !page["rows"].is_array() ||
                page["rows"].size() != 2 || page.value("complete", true) ||
                !page.value("result_truncated", false) ||
                page["rows"][0] != json::array({"evt-001", "SRV-APP-07",
                    "llama-agent-investigation-smoke", "evt-001"})) {
            std::cerr << "real search provider returned an unexpected filtered/paged result: " << response << '\n';
            return 8;
        }
    }
    return 0;
}
