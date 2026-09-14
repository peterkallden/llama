#include "agent/tool-family-index.h"
#include "agent/tooling/catalog/model-projection.h"
#include "plan/plan-json.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>

static bool has_tool(const std::vector<common_chat_tool> & tools,
        const std::string & name) {
    return std::any_of(tools.begin(), tools.end(), [&](const auto & tool) {
        return tool.name == name;
    });
}

int main() {
    const std::vector<common_chat_tool> tools = {
        {"dataset.list", "List available datasets", "{}", "{}"},
        {"dataset.inspect", "Inspect one dataset", "{}", "{}"},
        {"data.aggregate", "Aggregate a dataset", "{}", "{}"},
        {"openapi.search", "Search the configured API", "{}", "{}"},
        {"openapi.get", "Get one API record", "{}", "{}"},
        {"memory.search", "Search memory", "{}", "{}"},
    };
    const auto families = common_generate_tool_family_index(tools, {
        {"openapi", "Search the configured read-only API"}});
    const auto rendered = common_render_tool_family_index(families);
    assert(rendered.find("dataset: Choose and inspect datasets for analysis") != std::string::npos);
    assert(rendered.find("openapi: Search the configured read-only API") != std::string::npos);
    // The first routing prompt is deliberately compact. Exact tool names and
    // schemas are supplied only after the host accepts the family selection.
    assert(rendered.find("dataset.list") == std::string::npos);
    assert(rendered.find("openapi.get") == std::string::npos);

    common_tool_family_selection selection;
    std::string error;
    assert(common_parse_tool_family_selection_text(
        "TOOLS: dataset, openapi", families, selection, error));
    assert(selection.needs_tools && selection.family_ids.size() == 2);
    const auto selected = common_filter_tools_by_families(
        tools, selection.family_ids);
    assert(selected.size() == 4);
    assert(has_tool(selected, "dataset.list") && has_tool(selected, "openapi.get"));
    assert(!has_tool(selected, "memory.search"));
    assert(!common_parse_tool_family_selection_text(
        "TOOLS: datasets", families, selection, error));

    common_tool_definition dataset_inspect;
    dataset_inspect.name = "dataset.inspect";
    dataset_inspect.input_schema_json = R"({
        "type":"object",
        "required":["dataset"],
        "properties":{
            "dataset":{"type":"string","x-agent-type":"dataset_ref"},
            "max_scan_rows":{"type":"integer"},
            "materialize":{"type":"boolean"},
            "backend":{"type":"string"}
        }
    })";
    const auto inspect_projection = nlohmann::json::parse(
        common_tool_default_model_input_projection(dataset_inspect));
    assert(inspect_projection["properties"].contains("dataset"));
    assert(!inspect_projection["properties"].contains("max_scan_rows"));
    assert(!inspect_projection["properties"].contains("materialize"));
    assert(!inspect_projection["properties"].contains("backend"));

    common_tool_definition api_search;
    api_search.name = "openapi.search";
    api_search.input_schema_json = R"({
        "type":"object",
        "required":["query"],
        "properties":{
            "query":{"type":"string"},
            "limit":{"type":"integer"},
            "timeout_ms":{"type":"integer"},
            "execution_class":{"type":"string"}
        }
    })";
    api_search.result_schema_json = R"({
        "type":"object",
        "properties":{
            "items":{"type":"array","x-agent-type":"resource_ref"},
            "next":{"type":"string","x-agent-type":"continuation_ref"},
            "debug":{"type":"string"}
        }
    })";
    const auto api_input = nlohmann::json::parse(
        common_tool_default_model_input_projection(api_search));
    assert(api_input["properties"].contains("query"));
    assert(api_input["properties"].contains("limit"));
    assert(!api_input["properties"].contains("timeout_ms"));
    assert(!api_input["properties"].contains("execution_class"));
    const auto api_result = nlohmann::json::parse(
        common_tool_default_model_result_projection(api_search));
    assert(api_result["properties"].contains("items"));
    assert(api_result["properties"].contains("next"));
    assert(!api_result["properties"].contains("debug"));

    // This is the compact chain a small model can emit. The host parser
    // turns the indexed model reference into a typed step binding.
    common_plan_state plan;
    std::vector<common_plan_operation> operations;
    const auto chain = R"({
        "goal":"inspect the first available dataset",
        "steps":[
            {"tool":"dataset.list","args":{},"as":"candidates"},
            {"tool":"dataset.inspect","args":{"dataset":"$candidates.datasets[0]"}}
        ]
    })";
    assert(common_plan_parse_proposal_json(chain, plan, operations, error));
    assert(operations.size() == 3); // native final synthesis is host-owned
    const auto bound = nlohmann::json::parse(
        operations[1].step->tool_call->arguments_json);
    assert(bound["dataset"].value("$from_step", "") == "step_1");
    assert(bound["dataset"].value("$json_pointer", "") == "/datasets/0");

    // Repair surfaces that have appeared with compact models: unknown family,
    // missing indexed reference and a direct tool-chain self-reference.
    assert(!common_plan_parse_proposal_json(
        R"({"goal":"inspect","steps":[{"tool":"dataset.inspect","args":{"dataset":"$missing.datasets[0]"}}]})",
        plan, operations, error));
    assert(error.find("plan.binding") != std::string::npos);
    assert(!common_plan_parse_proposal_json(
        R"({"goal":"inspect","steps":[{"tool":"dataset.list","args":{"dataset":"$step_1.datasets[0]"}}]})",
        plan, operations, error));
    assert(error.find("plan.binding") != std::string::npos);
    return 0;
}
