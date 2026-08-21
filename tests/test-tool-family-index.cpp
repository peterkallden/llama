#include "agent/tool-family-index.h"

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
    assert(rendered.find("data: Query and transform datasets; operations: aggregate, join") != std::string::npos);
    assert(rendered.find("dataset: Choose and inspect datasets for analysis; operations: list") != std::string::npos);
    assert(rendered.find("web: Search and retrieve information from the web; operations: search") != std::string::npos);
    assert(rendered.find("calculator: Operations provided by the calculator tool family; operations: calculator") != std::string::npos);

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
    return 0;
}
