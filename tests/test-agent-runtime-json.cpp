#include "agent/runtime-json-contracts.h"
#include "agent/input-resources.h"
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

    request.input_resources.push_back({
        common_runtime_resource_ref{"agent-resource://turn/t/document.json", "document", "", "application/json", 0},
        "reference",
        true});
    request.input_resources.push_back({
        common_runtime_resource_ref{"agent-resource://turn/t/second", "second.csv", "", "text/csv", 0},
        "reference",
        false});
    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        request,
        "resource_read",
        nlohmann::ordered_json::object({{"id", "r2"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("uri", "") == "agent-resource://turn/t/second");
    TEST_ASSERT(!normalized.contains("id"));

    common_agent_request csv_request = request;
    csv_request.input_resources.resize(1);
    csv_request.input_resources.front().resource.name = "data.csv";
    csv_request.input_resources.front().resource.mime_type = "text/csv";
    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        csv_request,
        "dataset.sample",
        nlohmann::ordered_json::object({{"rows", 20}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://turn/t/document.json");
    TEST_ASSERT(normalized.value("rows", 0) == 20);

    csv_request.available_resources.push_back({
        common_runtime_resource_ref{"agent-resource://session/old.csv", "old.csv", "", "text/csv", 0},
        "scoped_reference",
        false});
    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        csv_request,
        "dataset.inspect",
        nlohmann::ordered_json::object({{"resource", "s1"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://session/old.csv");

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        csv_request,
        "dataset.inspect",
        nlohmann::ordered_json::object({{"dataset", "$datasets.datasets[]"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://turn/t/document.json");

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        request,
        "dataset.sample",
        nlohmann::ordered_json::object(),
        normalized,
        changed,
        error));
    TEST_ASSERT(!changed);
    TEST_ASSERT(!normalized.contains("resource"));

    const auto rendered_resources = common_agent_render_input_resource_context(request.input_resources);
    TEST_ASSERT(rendered_resources.find("id=r1") != std::string::npos);
    TEST_ASSERT(rendered_resources.find("id=r2") != std::string::npos);
    TEST_ASSERT(rendered_resources.find("agent-resource://turn/t/document.json") == std::string::npos);
    TEST_ASSERT(rendered_resources.find("second.csv") != std::string::npos);

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(!common_agent_runtime_apply_safe_tool_defaults_to_json(
        request,
        "resource_read",
        nlohmann::ordered_json::object({{"id", "r3"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(error.find("Choose one of") != std::string::npos);

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

    common_agent_request document_request = request;
    document_request.input_resources.resize(1);
    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        document_request,
        "document.tables",
        nlohmann::ordered_json::object(),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://turn/t/document.json");

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        document_request,
        "document.tables",
        nlohmann::ordered_json::object({{"resource", "r1"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://turn/t/document.json");

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        document_request,
        "document.table",
        nlohmann::ordered_json::object({{"resource", "r1"}, {"table", "Budget summary"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://turn/t/document.json");

    normalized = nlohmann::ordered_json();
    changed = false;
    document_request.input_resources.front().resource.name = "document-table-model.json";
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        document_request,
        "document.tables",
        nlohmann::ordered_json::object({{"resource", "document-table-model.json"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://turn/t/document.json");

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        document_request,
        "document.tables",
        nlohmann::ordered_json::object({{
            "resource",
            "agent-resource://turn/t/document.json name=document-table-model.json role=reference mime_type=application/json"
        }}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://turn/t/document.json");

    normalized_tool_arguments.clear();
    TEST_ASSERT(common_plan_normalize_tool_arguments_json(
        "document.table",
        R"({"uri":"agent-resource://turn/t/document.json","table":"Budget summary"})",
        normalized_tool_arguments,
        error));
    const auto normalized_document_tool = nlohmann::ordered_json::parse(normalized_tool_arguments);
    TEST_ASSERT(normalized_document_tool.value("resource", "") == "agent-resource://turn/t/document.json");
    TEST_ASSERT(!normalized_document_tool.contains("uri"));

    return 0;
}

#undef TEST_ASSERT
