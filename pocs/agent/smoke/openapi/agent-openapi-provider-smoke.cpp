#include "tools/agent/openapi/agent-openapi-provider.h"

#include <cassert>
#include <iostream>

int main() {
    agent_openapi_catalog catalog;
    catalog.provider_id = "sales-api";
    catalog.prefix = "sales";
    agent_openapi_operation list_sales;
    list_sales.operation_id = "listSales";
    list_sales.method = "get";
    list_sales.path = "/sales";
    list_sales.summary = "List sales";
    list_sales.description = "List sales";
    list_sales.input_schema_json = R"({"type":"object","properties":{"limit":{"type":"integer","minimum":1,"maximum":100}}})";
    list_sales.host_required_parameters = {"limit"};
    list_sales.result_schema_json = R"({
        "type":"object",
        "properties":{
            "results":{"type":"array","items":{"type":"object","properties":{
                "id":{"type":"string"},"display_name":{"type":"string"}
            }}}
        }
    })";
    list_sales.access = agent_openapi_access::read;
    list_sales.read_only = true;
    list_sales.paging.kind = agent_openapi_paging_kind::cursor;
    list_sales.paging.page_size_parameter = "per-page";
    list_sales.paging.cursor_parameter = "cursor";
    list_sales.paging.projection_parameter = "select";
    catalog.operations.push_back(std::move(list_sales));
    agent_openapi_operation get_sale;
    get_sale.operation_id = "getSale";
    get_sale.method = "get";
    get_sale.path = "/sales/{id}";
    get_sale.summary = "Get sale";
    get_sale.description = "Get sale";
    get_sale.input_schema_json = R"({"type":"object"})";
    get_sale.path_parameters = {"id"};
    get_sale.access = agent_openapi_access::read;
    get_sale.read_only = true;
    catalog.operations.push_back(std::move(get_sale));
    catalog.relations.push_back({"listSales", "getSale", "/sales", "id"});
    bool called = false;
    bool materialized = false;
    agent_openapi_tool_provider provider(std::move(catalog),
        [&](const agent_tool_context &, const agent_openapi_operation & operation,
            const std::string & arguments, agent_openapi_execution_result & result, std::string &) {
            called = operation.operation_id == "listSales" && arguments == R"({"limit":10})";
            result.ok = true;
            result.structured_content_json = R"({"count":1})";
            return true;
        },
        [&](const agent_tool_context &, const agent_openapi_operation &,
            const std::string &,
            agent_openapi_execution_result & result, std::string &) {
            materialized = true;
            common_runtime_resource_ref resource;
            resource.uri = "resource://openapi/sales/list-1";
            resource.name = "sales-list";
            resource.scope = common_runtime_resource_scope::turn;
            result.resource_refs.push_back(std::move(resource));
            common_agent_dataset_ref dataset;
            dataset.uri = "dataset://openapi/sales";
            dataset.name = "sales";
            dataset.row_count = 1;
            dataset.column_count = 2;
            dataset.source_resource_uri = "resource://openapi/sales/list-1";
            dataset.source_representation = "openapi:json";
            result.dataset_refs.push_back(std::move(dataset));
            return true;
        });
    agent_tool_context context;
    context.allow_network = true;
    std::string error;
    auto view = provider.resolve_tools(context, error);
    if (!view) {
        std::cerr << "OpenAPI tool view resolution failed: " << error << "\n";
        return 1;
    }
    if (view->chat_tools().size() != 2) {
        std::cerr << "OpenAPI tool view exposed " << view->chat_tools().size()
                  << " tools instead of 2: " << error << "\n";
        return 1;
    }
    if (view->chat_tools()[0].name != "sales.listSales") {
        std::cerr << "OpenAPI exposed tool name was not family-qualified with a dot\n";
        return 1;
    }
    const auto openapi_metadata = view->learning_metadata("sales.listSales");
    assert(openapi_metadata.tool_family == "sales");
    assert(openapi_metadata.provider_kind == "openapi");
    assert(view->chat_tools()[0].description.find("workflow: sales.listSales -> choose/bind id -> sales.getSale") != std::string::npos);
    assert(view->chat_tools()[0].description.find("pagination: host-classified cursor") != std::string::npos);
    assert(view->chat_tools()[0].description.find("per-page") != std::string::npos);
    assert(view->chat_tools()[0].description.find("cursor") != std::string::npos);
    assert(view->chat_tools()[0].description.find("select") != std::string::npos);
    assert(view->chat_tools()[0].description.find("returns: results:{") != std::string::npos);
    assert(view->chat_tools()[0].description.find("display_name?:string") != std::string::npos);
    assert(view->chat_tools()[0].result_schema.find("display_name") != std::string::npos);
    assert(view->chat_tools()[0].parameters.find(R"("required":["limit"])" ) != std::string::npos);
    common_plan_state invalid_plan;
    invalid_plan.steps.push_back({});
    invalid_plan.steps[0].tool_call = common_plan_tool_call{
        "sales.getSale", R"({"id":"$previous.id"})"};
    assert(!view->validate_plan(invalid_plan, error));
    common_plan_state valid_plan;
    common_plan_step collection_step;
    collection_step.tool_call = common_plan_tool_call{"sales.listSales", R"({})"};
    valid_plan.steps.push_back(collection_step);
    valid_plan.steps.push_back(invalid_plan.steps[0]);
    assert(view->validate_plan(valid_plan, error));
    assert(view->is_read_only("sales.listSales"));
    assert(!view->is_policy_gated("sales.listSales"));
    agent_tool_call call{"call-1", "sales.listSales", R"({"limit":10})"};
    assert(view->validate(call, error));
    // The host-required projection is model-facing only. Execution validation
    // still uses the OpenAPI input schema and therefore remains optional here.
    agent_tool_call model_omitted_call{"call-optional", "sales.listSales", R"({})"};
    assert(view->validate(model_omitted_call, error));
    const auto result = view->call(call, error);
    assert(result.ok && called && materialized);
    assert(result.content_json.find(R"("count":1)") != std::string::npos);
    assert(result.resource_refs.size() == 1);
    assert(result.resource_refs.front().uri == "resource://openapi/sales/list-1");
    assert(result.dataset_refs.size() == 1);
    assert(result.dataset_refs.front().uri == "dataset://openapi/sales");
    assert(result.content_json.find(R"("dataset_refs")") != std::string::npos);
    std::cout << "agent-openapi-provider-smoke: ok\n";
}
