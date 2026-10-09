#include "tools/agent/openapi/agent-openapi-catalog.h"
#include "tools/agent/openapi/agent-openapi-provider.h"
#include "tools/agent/openapi/dynamic/agent-openapi-dynamic-admission.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    const nlohmann::json document = {
        {"openapi", "3.0.3"},
        {"components", {
            {"schemas", {
                {"SaleId", {{"type", "string"}}},
                {"SaleInput", {
                    {"type", "object"},
                    {"properties", {{"quantity", {{"type", "integer"}}}}},
                    {"required", {"quantity"}},
                }},
                {"SaleResponse", {
                    {"type", "object"},
                    {"properties", {
                        {"id", {{"$ref", "#/components/schemas/SaleId"}}},
                        {"status", {{"type", "string"}}},
                    }},
                }},
            }},
            {"parameters", {
                {"search", {{"name", "search"}, {"in", "query"},
                    {"description", "Full-text search across titles and abstracts."},
                    {"schema", {{"type", "string"}}}}},
                {"perPage", {{"name", "per-page"}, {"in", "query"},
                    {"schema", {{"type", "integer"}}}}},
                {"saleId", {{"name", "id"}, {"in", "path"}, {"required", true},
                    {"schema", {{"type", "string"}}}}},
            }},
            {"requestBodies", {
                {"CreateSale", {
                    {"required", true},
                    {"content", {{"application/json", {
                        {"schema", {{"$ref", "#/components/schemas/SaleInput"}}},
                    }}}},
                }},
            }},
            {"responses", {
                {"SaleCreated", {
                    {"description", "Created sale"},
                    {"content", {{"application/json", {
                        {"schema", {{"$ref", "#/components/schemas/SaleResponse"}}},
                    }}}},
                }},
            }},
            {"securitySchemes", {
                {"bearerAuth", {{"type", "http"}, {"scheme", "bearer"}, {"bearerFormat", "JWT"}}},
            }},
        }},
        {"security", json::array({json({{"bearerAuth", json::array()}})})},
        {"paths", {
            {"/sales", {
                {"get", {{"operationId", "listSales"}, {"summary", "List sales"},
                    {"parameters", {
                        {{"name", "id"}, {"in", "query"}, {"x-agent-inferable", true},
                            {"schema", {{"type", "string"}}}},
                        {{"$ref", "#/components/parameters/search"}},
                        {{"name", "filter"}, {"in", "query"},
                            {"description", "Filter by publication_year:2024 or topics.id:T…"},
                            {"schema", {{"type", "string"}}}},
                        {{"$ref", "#/components/parameters/perPage"}},
                        {{"name", "cursor"}, {"in", "query"},
                            {"schema", {{"type", "string"}}}},
                        {{"name", "select"}, {"in", "query"},
                            {"schema", {{"type", "string"}}}},
                    }},
                    {"responses", {{"200", {{"content", {{"application/json", {{"schema", {{"type", "array"}, {"items", {{"type", "object"}}}}}}}}}}}}}}},
                {"post", {
                    {"operationId", "createSale"}, {"summary", "Create sale"},
                    {"requestBody", {{"$ref", "#/components/requestBodies/CreateSale"}}},
                    {"responses", {{"201", {{"$ref", "#/components/responses/SaleCreated"}}}}},
                }},
            }},
            {"/sales/{id}", {
                {"parameters", {{{"$ref", "#/components/parameters/saleId"}}}},
                {"get", {{"operationId", "getSale"}, {"summary", "Get sale"},
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
    config.operations["listSales"].required_parameters = {"search"};
    agent_openapi_catalog catalog;
    std::string error;
    if (!build_agent_openapi_catalog(document, config, catalog, error) ||
            catalog.operations.size() != 2 ||
            catalog.operations[0].operation_id != "listSales" ||
            !catalog.operations[0].read_only ||
            catalog.operations[0].input_schema_json.find("\"type\":\"string\"") == std::string::npos ||
            catalog.operations[0].input_schema_json.find(
                "Full-text search across titles and abstracts.") == std::string::npos ||
            catalog.operations[0].input_schema_json.find(
                "publication_year:2024 or topics.id:T") == std::string::npos ||
            std::find(catalog.operations[0].query_parameters.begin(),
                catalog.operations[0].query_parameters.end(), "search") ==
                catalog.operations[0].query_parameters.end() ||
            std::find(catalog.operations[0].query_parameters.begin(),
                catalog.operations[0].query_parameters.end(), "per-page") ==
                catalog.operations[0].query_parameters.end() ||
            catalog.operations[0].paging.kind != agent_openapi_paging_kind::cursor ||
            catalog.operations[0].paging.page_size_parameter != "per-page" ||
            catalog.operations[0].paging.cursor_parameter != "cursor" ||
            catalog.operations[0].paging.projection_parameter != "select" ||
            catalog.operations[0].input_schema_json.find("x-agent-autowire-fields") == std::string::npos ||
            catalog.operations[0].host_required_parameters != std::vector<std::string>{"search"} ||
            catalog.operations[0].input_schema_json.find("\"required\"") != std::string::npos ||
            catalog.operations[0].result_schema_json.find("\"type\":\"array\"") == std::string::npos ||
            !catalog.operations[0].auth_required ||
            catalog.operations[0].security_schemes.size() != 1 ||
            catalog.operations[0].security_schemes[0] != "bearerAuth" ||
            catalog.security_schemes.size() != 1 ||
            catalog.security_schemes[0].parameter_name != "") {
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
    if (!compact_error.empty() || compact.find("id (optional, string) [host may infer]") == std::string::npos ||
            agent_openapi_exposed_tool_name(catalog, catalog.operations[0]) != "sales.listSales") {
        std::cerr << "OpenAPI compact inferability rendering failed: " << compact_error
                  << " compact=" << compact << "\n";
        return 1;
    }

    config.access = "read_write";
    config.operations["createSale"].access = "write";
    assert(build_agent_openapi_catalog(document, config, catalog, error));
    assert(catalog.operations.size() == 3);
    const auto create_sale = std::find_if(catalog.operations.begin(), catalog.operations.end(),
        [](const agent_openapi_operation & operation) { return operation.operation_id == "createSale"; });
    assert(create_sale != catalog.operations.end() && create_sale->requires_confirmation);
    assert(create_sale->input_schema_json.find("\"quantity\"") != std::string::npos);
    assert(create_sale->input_schema_json.find("\"body\"") != std::string::npos);
    assert(create_sale->input_schema_json.find("\"required\":[\"body\"]") != std::string::npos);
    assert(create_sale->result_schema_json.find("\"status\"") != std::string::npos);

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
    const auto get_sale = std::find_if(catalog.operations.begin(), catalog.operations.end(),
        [](const agent_openapi_operation & operation) { return operation.operation_id == "getSale"; });
    assert(get_sale != catalog.operations.end());
    assert(std::find(get_sale->path_parameters.begin(), get_sale->path_parameters.end(), "id") !=
        get_sale->path_parameters.end());
    assert(get_sale->paging.kind == agent_openapi_paging_kind::none);

    agent_openapi_result_projection projection;
    assert(classify_agent_openapi_result_json(
        R"([{"id":"s1","region":"north"},{"id":"s2","region":"south"}])",
        agent_openapi_result_projection_limits{}, projection, error));
    assert(projection.kind == agent_openapi_result_projection_kind::dataset);
    assert(projection.row_count == 2);
    assert(projection.columns.size() == 2);
    assert(classify_agent_openapi_result_json(
        R"([{"id":"s1","details":{"amount":10}}])",
        agent_openapi_result_projection_limits{}, projection, error));
    assert(projection.kind == agent_openapi_result_projection_kind::json_resource);

    std::vector<agent_openapi_item_reference> references;
    assert(make_agent_openapi_item_references(
        catalog, "listSales", R"([{"id":"s1","region":"north"},{"id":"s2"}])",
        agent_openapi_result_projection_limits{}, references, error));
    assert(references.size() == 2);
    assert(references[0].candidate == "getSale#1");
    assert(references[0].item_value == "s1");
    assert(references[0].item_parameter == "id");
    assert(references[0].provider_id == "sales-api");
    assert(references[1].row_index == 1);

    agent_openapi_result_projection_limits small_limits;
    small_limits.max_rows = 1;
    assert(!make_agent_openapi_item_references(
        catalog, "listSales", R"([{"id":"s1"},{"id":"s2"}])",
        small_limits, references, error));

    auto external_ref_document = document;
    external_ref_document["paths"]["/sales"]["get"]["responses"]["200"] = {
        {"$ref", "https://example.test/openapi.json#/components/responses/Sales"}
    };
    assert(!build_agent_openapi_catalog(external_ref_document, config, catalog, error));
    assert(error.find("external") != std::string::npos);

    const nlohmann::json dynamic_document = {
        {"openapi", "3.0.3"},
        {"info", {{"title", "Public Data API"}, {"version", "1"}}},
        {"servers", {{{"url", "https://api.example.test/v1"}}}},
        {"paths", {
            {"/items", {{"get", {{"operationId", "listItems"}, {"summary", "List items"},
                {"responses", {{"200", {{"description", "ok"}}}}}}},
                {"post", {{"operationId", "createItem"}, {"summary", "Create item"},
                {"responses", {{"201", {{"description", "ok"}}}}}}}}},
            {"/private", {{"get", {{"operationId", "privateItems"},
                {"security", {{{"apiKey", json::array()}}}},
                {"responses", {{"200", {{"description", "ok"}}}}}}}}},
        }},
        {"components", {{"securitySchemes", {{"apiKey", {{"type", "apiKey"}, {"in", "header"}, {"name", "key"}}}}}}},
    };
    agent_openapi_dynamic_registration dynamic_registration;
    assert(admit_agent_dynamic_openapi_document(
        dynamic_document, "resource://session/spec", "0123456789abcdef",
        dynamic_registration, error));
    assert(dynamic_registration.config.access == "read_only");
    assert(dynamic_registration.config.auth.type == "none");
    assert(dynamic_registration.catalog.operations.size() == 1);
    assert(dynamic_registration.catalog.operations.front().operation_id == "listItems");
    agent_openapi_tool_provider dynamic_provider(
        dynamic_registration.catalog,
        [](const agent_tool_context &, const agent_openapi_operation &,
           const std::string &, agent_openapi_execution_result &, std::string &) {
            return false;
        });
    agent_tool_context dynamic_context;
    dynamic_context.allow_network = true;
    auto dynamic_view = dynamic_provider.resolve_tools(dynamic_context, error);
    assert(dynamic_view && dynamic_view->chat_tools().size() == 1);
    assert(dynamic_view->chat_tools().front().name ==
        dynamic_registration.config.prefix + ".listItems");
    agent_openapi_dynamic_registry dynamic_registry;
    assert(dynamic_registry.add(dynamic_registration, error));
    assert(dynamic_registry.contains(dynamic_registration.config.id));
    assert(dynamic_registry.snapshot().size() == 1);
    assert(dynamic_registry.add(dynamic_registration, error));
    assert(dynamic_registry.snapshot().size() == 1);
    dynamic_registry.clear();
    assert(dynamic_registry.snapshot().empty());

    auto http_dynamic_document = dynamic_document;
    http_dynamic_document["servers"][0]["url"] = "http://api.example.test/v1";
    assert(!admit_agent_dynamic_openapi_document(
        http_dynamic_document, "resource://session/spec", "0123456789abcdef",
        dynamic_registration, error));
    assert(error.find("HTTPS") != std::string::npos);

    std::cout << "agent-openapi-catalog-smoke: ok\n";
    return 0;
}
