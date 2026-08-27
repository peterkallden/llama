#include "tools/agent/openapi/agent-openapi-catalog.h"
#include "tools/agent/openapi/agent-openapi-provider.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <cassert>
#include <iostream>

int main() {
    const nlohmann::json document = {
        {"openapi", "3.0.3"},
        {"components", {"schemas", {{"SaleId", {{"type", "string"}}}}}},
        {"paths", {
            {"/sales", {
                {"get", {{"operationId", "listSales"}, {"summary", "List sales"},
                    {"parameters", {{{"name", "id"}, {"in", "query"}, {"x-agent-inferable", true}, {"schema", {{"type", "string"}}}}}},
                    {"responses", {{"200", {{"content", {{"application/json", {{"schema", {{"type", "array"}, {"items", {{"type", "object"}}}}}}}}}}}}}}},
                {"post", {{"operationId", "createSale"}, {"summary", "Create sale"}}},
            }},
            {"/sales/{id}", {
                {"get", {{"operationId", "getSale"}, {"summary", "Get sale"},
                    {"parameters", {{{"name", "id"}, {"in", "path"}, {"required", true}, {"schema", {{"type", "string"}}}}}},
                    {"responses", {{"200", {{"content", {{"application/json", {{"schema", {{"type", "object"}}}}}}}}}}}}},
                {"delete", {{"operationId", "deleteSale"}}},
            }},
        }},
    };
    agent_host_openapi_provider_config config;
    config.id = "sales-api";
    config.prefix = "sales";
    config.access = "read_only";
    config.exposure = "auto";
    agent_openapi_catalog catalog;
    std::string error;
    if (!build_agent_openapi_catalog(document, config, catalog, error) ||
            catalog.operations.size() != 2 ||
            catalog.operations[0].operation_id != "listSales" ||
            !catalog.operations[0].read_only ||
            catalog.operations[0].input_schema_json.find("\"type\":\"string\"") == std::string::npos ||
            catalog.operations[0].input_schema_json.find("x-agent-autowire-fields") == std::string::npos ||
            catalog.operations[0].result_schema_json.find("\"type\":\"array\"") == std::string::npos) {
        std::cerr << "OpenAPI catalog inferability contract failed: " << error
                  << " operations=" << catalog.operations.size();
        if (!catalog.operations.empty()) {
            std::cerr << " id=" << catalog.operations[0].operation_id
                      << " schema=" << catalog.operations[0].input_schema_json;
        }
        std::cerr << "\n";
        return 1;
    }
    std::string compact_error;
    const auto compact = common_render_compact_tool_description(
        "sales.listSales", "List sales", catalog.operations[0].input_schema_json,
        R"({"type":"object"})", compact_error);
    if (!compact_error.empty() || compact.find("id?:string [may be inferred]") == std::string::npos ||
            agent_openapi_exposed_tool_name(catalog, catalog.operations[0]) != "sales.listSales") {
        std::cerr << "OpenAPI compact inferability rendering failed: " << compact_error
                  << " compact=" << compact << "\n";
        return 1;
    }

    config.access = "read_write";
    config.operations["createSale"].access = "write";
    assert(build_agent_openapi_catalog(document, config, catalog, error));
    assert(catalog.operations.size() == 3);
    assert(catalog.operations[2].requires_confirmation);

    config.exposure = "include";
    config.operations.clear();
    config.operations["listSales"].access = "read";
    assert(build_agent_openapi_catalog(document, config, catalog, error));
    assert(catalog.operations.size() == 1);

    config.exposure = "auto";
    config.access = "read_only";
    assert(build_agent_openapi_catalog(document, config, catalog, error));
    assert(catalog.relations.size() == 1);
    assert(catalog.relations[0].collection_operation_id == "listSales");
    assert(catalog.relations[0].item_operation_id == "getSale");
    assert(catalog.relations[0].item_parameter == "id");

    std::cout << "agent-openapi-catalog-smoke: ok\n";
    return 0;
}
