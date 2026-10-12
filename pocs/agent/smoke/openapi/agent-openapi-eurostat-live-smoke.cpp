#include "tools/agent/openapi/agent-openapi-catalog.h"
#include "tools/agent/openapi/agent-openapi-provider.h"
#include "tools/agent/openapi/agent-openapi-http.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>

using json = nlohmann::ordered_json;

int main() {
    // Eurostat publishes a WADL rather than an OpenAPI document. This small
    // local adapter describes only the stable Statistics API operation used
    // by this opt-in live smoke; it does not make WADL look like OpenAPI.
    const json document = {
        {"openapi", "3.0.0"},
        {"info", {{"title", "Eurostat Statistics API"}, {"version", "1.0"}}},
        {"paths", {
            {"/data/{datasetCode}", {
                {"get", {
                    {"operationId", "getData"},
                    {"summary", "Get a Eurostat JSON-stat dataset"},
                    {"parameters", json::array({
                        json{{"name", "datasetCode"}, {"in", "path"}, {"required", true},
                            {"schema", {{"type", "string"}}}},
                        json{{"name", "lang"}, {"in", "query"},
                            {"schema", {{"type", "string"}}}},
                        json{{"name", "geo"}, {"in", "query"},
                            {"schema", {{"type", "string"}}}},
                        json{{"name", "time"}, {"in", "query"},
                            {"schema", {{"type", "string"}}}},
                    })},
                    {"responses", {{"200", {
                        {"description", "JSON-stat dataset"},
                        {"content", {{"application/json", {
                            {"schema", {{"type", "object"}}},
                        }}}},
                    }}}},
                }},
            }},
        }},
    };

    agent_host_openapi_provider_config config;
    config.id = "eurostat-statistics";
    config.base_url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0";
    config.prefix = "eurostat";
    config.access = "read_only";
    config.allow_private_network = false;
    config.connect_timeout_ms = 5000;
    config.request_timeout_ms = 20000;
    config.max_result_bytes = 1024 * 1024;

    agent_openapi_catalog catalog;
    std::string error;
    if (!build_agent_openapi_catalog(document, config, catalog, error)) {
        std::fprintf(stderr, "Eurostat adapter catalog failed: %s\n", error.c_str());
        return 1;
    }

    agent_openapi_tool_provider provider(
        std::move(catalog), make_agent_openapi_http_executor(config));
    agent_tool_context context;
    context.allow_network = true;
    auto view = provider.resolve_tools(context, error);
    if (!view) {
        std::fprintf(stderr, "Eurostat provider view failed: %s\n", error.c_str());
        return 1;
    }
    const auto result = view->call({
        "eurostat-live-1", "eurostat.getData",
        R"({"datasetCode":"DEMO_R_D3DENS","lang":"EN","geo":"SE","time":"2020"})"
    }, error);
    if (!result.ok || result.content_json.find(R"("version":"2.0")") == std::string::npos ||
            result.content_json.find(R"("class":"dataset")") == std::string::npos ||
            result.content_json.find(R"("dimension")") == std::string::npos) {
        std::fprintf(stderr, "Eurostat live smoke failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("eurostat_jsonstat=2.0\n");
    std::printf("eurostat_dataset=DEMO_R_D3DENS\n");
    return 0;
}
