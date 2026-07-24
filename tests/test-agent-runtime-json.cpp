#include "agent/runtime-json-contracts.h"
#include "plan/plan-json.h"

#include <cassert>

#define TEST_ASSERT(expr) do { \
    const bool test_result = static_cast<bool>(expr); \
    assert(test_result); \
    if (!test_result) return 1; \
} while (false)

int main() {
    const auto reasoning_object = common_agent_runtime_reasoning_observation_to_json(
        R"({"summary":"grounded","citations":["obs-1"]})");
    TEST_ASSERT(reasoning_object.is_object());
    TEST_ASSERT(reasoning_object.value("summary", "") == "grounded");
    TEST_ASSERT(reasoning_object.contains("citations"));

    const auto reasoning_text = common_agent_runtime_reasoning_observation_to_json(
        "plain reasoning fallback");
    TEST_ASSERT(reasoning_text.is_object());
    TEST_ASSERT(reasoning_text.value("summary", "") == "plain reasoning fallback");
    TEST_ASSERT(reasoning_text.value("format", "") == "unstructured");

    common_agent_request request;
    request.prompt = "Search planner runtime contract details";

    nlohmann::ordered_json normalized;
    bool changed = false;
    std::string error;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        request,
        "web_search",
        nlohmann::ordered_json::object(),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("query", "") == "Search planner runtime contract details");
    TEST_ASSERT(normalized.value("limit", 0) == 5);

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        request,
        "resource_read",
        nlohmann::ordered_json::object({{"uri", "agent-resource://turn/t/r"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("uri", "") == "agent-resource://turn/t/r");
    TEST_ASSERT(normalized.value("max_bytes", 0) == 8192);

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(!common_agent_runtime_apply_safe_tool_defaults_to_json(
        request,
        "web_search",
        nlohmann::ordered_json::array(),
        normalized,
        changed,
        error));
    TEST_ASSERT(error == "tool arguments must be a JSON object");

    std::string normalized_tool_arguments;
    TEST_ASSERT(common_plan_normalize_tool_arguments_json(
        "lookup",
        R"({"tool":{"name":"lookup","args":{"id":"first"}}})",
        normalized_tool_arguments,
        error));

    const auto normalized_tool = nlohmann::ordered_json::parse(normalized_tool_arguments);
    TEST_ASSERT(normalized_tool.size() == 1 && normalized_tool.value("id", "") == "first");

    return 0;
}

#undef TEST_ASSERT
