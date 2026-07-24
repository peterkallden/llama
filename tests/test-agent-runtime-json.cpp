#include "agent/runtime-json-contracts.h"
#include "plan/plan-json.h"

#include <cassert>

int main() {
    const auto reasoning_object = common_agent_runtime_reasoning_observation_to_json(
        R"({"summary":"grounded","citations":["obs-1"]})");
    assert(reasoning_object.is_object());
    assert(reasoning_object.value("summary", "") == "grounded");
    assert(reasoning_object.contains("citations"));

    const auto reasoning_text = common_agent_runtime_reasoning_observation_to_json(
        "plain reasoning fallback");
    assert(reasoning_text.is_object());
    assert(reasoning_text.value("summary", "") == "plain reasoning fallback");
    assert(reasoning_text.value("format", "") == "unstructured");

    common_agent_request request;
    request.prompt = "Search planner runtime contract details";

    nlohmann::ordered_json normalized;
    bool changed = false;
    std::string error;
    assert(common_agent_runtime_apply_safe_tool_defaults_to_json(
        request,
        "web_search",
        nlohmann::ordered_json::object(),
        normalized,
        changed,
        error));
    assert(changed);
    assert(normalized.value("query", "") == "Search planner runtime contract details");
    assert(normalized.value("limit", 0) == 5);

    normalized = nlohmann::ordered_json();
    changed = false;
    assert(common_agent_runtime_apply_safe_tool_defaults_to_json(
        request,
        "resource_read",
        nlohmann::ordered_json::object({{"uri", "agent-resource://turn/t/r"}}),
        normalized,
        changed,
        error));
    assert(changed);
    assert(normalized.value("uri", "") == "agent-resource://turn/t/r");
    assert(normalized.value("max_bytes", 0) == 8192);

    normalized = nlohmann::ordered_json();
    changed = false;
    assert(!common_agent_runtime_apply_safe_tool_defaults_to_json(
        request,
        "web_search",
        nlohmann::ordered_json::array(),
        normalized,
        changed,
        error));
    assert(error == "tool arguments must be a JSON object");

    std::string normalized_tool_arguments;
    assert(common_plan_normalize_tool_arguments_json(
        "lookup",
        R"({"tool":{"name":"lookup","args":{"id":"first"}}})",
        normalized_tool_arguments,
        error));
    const auto normalized_tool = nlohmann::ordered_json::parse(normalized_tool_arguments);
    assert(normalized_tool.size() == 1 && normalized_tool.value("id", "") == "first");

    return 0;
}
