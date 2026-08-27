#include "tools/agent/openapi/agent-openapi-catalog.h"

#include <cassert>
#include <iostream>

int main() {
    const nlohmann::json document = {
        {"openapi", "3.0.3"},
        {"paths", {
            {"/sales", {
                {"get", {{"operationId", "listSales"}, {"summary", "List sales"}}},
                {"post", {{"operationId", "createSale"}, {"summary", "Create sale"}}},
            }},
            {"/sales/{id}", {
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
    assert(build_agent_openapi_catalog(document, config, catalog, error));
    assert(catalog.operations.size() == 1);
    assert(catalog.operations[0].operation_id == "listSales");
    assert(catalog.operations[0].read_only);
    assert(agent_openapi_exposed_tool_name(catalog, catalog.operations[0]) == "sales.listSales");

    config.access = "read_write";
    config.operations["createSale"].access = "write";
    assert(build_agent_openapi_catalog(document, config, catalog, error));
    assert(catalog.operations.size() == 2);
    assert(catalog.operations[1].requires_confirmation);

    config.exposure = "include";
    config.operations.clear();
    config.operations["listSales"].access = "read";
    assert(build_agent_openapi_catalog(document, config, catalog, error));
    assert(catalog.operations.size() == 1);

    std::cout << "agent-openapi-catalog-smoke: ok\n";
    return 0;
}
