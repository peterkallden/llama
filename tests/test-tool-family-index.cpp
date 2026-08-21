#include "agent/tool-family-index.h"
#include "agent/tool-workflow-index.h"

#include <cassert>

int main() {
    const std::vector<common_chat_tool> tools = {
        {"data.join", "join", "{}", "{}"},
        {"data.aggregate", "aggregate", "{}", "{}"},
        {"dataset.list", "list", "{}", "{}"},
        {"web_search", "search", "{}", "{}"},
        {"calculator", "calculate", "{}", "{}"},
    };

    const auto families = common_generate_tool_family_index(tools);
    assert(families.size() == 4);
    assert(families[0].description == "Query and transform datasets");
    const auto rendered = common_render_tool_family_index(families);
    assert(rendered.find("data: Query and transform datasets") != std::string::npos);
    assert(rendered.find("dataset: Choose and inspect datasets for analysis") != std::string::npos);
    assert(rendered.find("web: Search and retrieve information from the web") != std::string::npos);
    assert(rendered.find("calculator: Operations provided by the calculator tool family") != std::string::npos);
    assert(rendered.find("aggregate") == std::string::npos);
    assert(rendered.find("dataset.list") == std::string::npos);

    const auto selected = common_filter_tools_by_families(tools, {"data", "dataset"});
    assert(selected.size() == 3);
    assert(selected[0].name == "data.join");
    assert(selected[2].name == "dataset.list");

    const auto bounded = common_render_tool_family_index(families, 32);
    assert(bounded.size() <= 32);

    common_tool_family_selection selection;
    std::string error;
    assert(common_parse_tool_family_selection(
        R"({"needs_tools":true,"families":["data","dataset"],"reason":"inspect data"})",
        selection,
        error));
    assert(selection.needs_tools && selection.family_ids.size() == 2);
    assert(!common_parse_tool_family_selection(
        R"({"needs_tools":true,"families":["data","data"]})",
        selection,
        error));

    const auto workflows = common_generate_tool_workflow_index();
    const auto workflow_view = common_render_tool_workflow_index(workflows, {"dataset", "data", "statistics"});
    assert(workflow_view.find("dataset.discover") != std::string::npos);
    assert(workflow_view.find("dataset.list() as=candidates") != std::string::npos);
    assert(workflow_view.find("dataset.select(name=$candidates.names[index])") != std::string::npos);
    assert(workflow_view.find("dataset.join") != std::string::npos);
    assert(workflow_view.find("dataset.select(name=<left_name>) as=left") != std::string::npos);
    assert(workflow_view.find("data.join(left=$left.dataset, right=$right.dataset") != std::string::npos);
    assert(workflow_view.find("dataset.inspect_named") != std::string::npos);
    const auto statistics_only = common_render_tool_workflow_index(workflows, {"statistics"});
    assert(statistics_only.find("dataset.summarize") != std::string::npos);
    assert(statistics_only.find("dataset.join") == std::string::npos);
    common_tool_workflow_selection workflow_selection;
    assert(common_parse_tool_workflow_selection(
        R"({"workflows":["dataset.discover","dataset.join"]})", workflow_selection, error));
    assert(workflow_selection.workflow_ids.size() == 2);
    assert(!common_parse_tool_workflow_selection(
        R"({"workflows":["dataset.join","dataset.join"]})", workflow_selection, error));

    const std::vector<common_tool_workflow_step_view> valid_join = {
        {"dataset.select", R"({"name":"orders"})"},
        {"dataset.select", R"({"name":"customers"})"},
        {"data.join", R"({"left":"$left.dataset","right":"$right.dataset"})"},
    };
    assert(common_validate_tool_workflow_plan(
        workflows, {"dataset.join"}, valid_join, error));
    const std::vector<common_tool_workflow_step_view> invalid_join = {
        {"data.join", R"({"left":"$left.dataset","right":"$right.dataset"})"},
    };
    assert(!common_validate_tool_workflow_plan(
        workflows, {"dataset.join"}, invalid_join, error));
    assert(error.find("required_producer_missing") != std::string::npos);
    const std::vector<common_tool_workflow_step_view> direct_join = {
        {"data.join", R"({"left":"dataset://orders","right":"dataset://customers"})"},
    };
    assert(common_validate_tool_workflow_plan(
        workflows, {"dataset.join"}, direct_join, error));

    const auto slots = common_expand_tool_workflow_slots(
        workflows, {"dataset.discover", "dataset.inspect_named", "dataset.join", "dataset.summarize"});
    assert(slots.size() == 8);
    assert(slots[0].tool_name == "dataset.list");
    assert(slots[1].tool_name == "dataset.select");
    assert(slots[1].alias == "left");
    assert(slots[2].tool_name == "dataset.select");
    assert(slots[3].tool_name == "dataset.inspect");
    assert(slots[3].fixed_arguments.at("dataset") == "$left.dataset");
    assert(slots[5].tool_name == "data.join");
    assert(slots[5].fixed_arguments.at("left") == "$left.dataset");
    assert(slots[6].tool_name == "data.aggregate");
    assert(slots[6].fixed_arguments.at("dataset") == "$joined.dataset");
    return 0;
}
