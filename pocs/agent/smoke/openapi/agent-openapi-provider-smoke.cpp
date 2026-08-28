#include "tools/agent/openapi/agent-openapi-provider.h"

#include <cassert>
#include <iostream>

int main() {
    agent_openapi_catalog catalog;
    catalog.provider_id = "sales-api";
    catalog.prefix = "sales";
    catalog.operations.push_back({
        "listSales", "get", "/sales", "List sales", "List sales", R"({"type":"object"})",
        {}, {}, agent_openapi_access::read, true, false});
    catalog.operations.push_back({
        "getSale", "get", "/sales/{id}", "Get sale", "Get sale", R"({"type":"object"})",
        {"id"}, {}, agent_openapi_access::read, true, false});
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
            return true;
        });
    agent_tool_context context;
    std::string error;
    auto view = provider.resolve_tools(context, error);
    assert(view != nullptr);
    assert(view->chat_tools().size() == 2);
    assert(view->chat_tools()[0].name == "sales.listSales");
    assert(view->chat_tools()[0].description.find("workflow: sales.listSales -> choose/bind id -> sales.getSale") != std::string::npos);
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
    const auto result = view->call(call, error);
    assert(result.ok && called && materialized);
    assert(result.content_json == R"({"count":1})");
    assert(result.resource_refs.size() == 1);
    assert(result.resource_refs.front().uri == "resource://openapi/sales/list-1");
    std::cout << "agent-openapi-provider-smoke: ok\n";
}
