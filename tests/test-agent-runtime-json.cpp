#include "agent/runtime-json-contracts.h"
#include "agent/input-resources.h"
#include "agent/dataset-contracts.h"
#include "plan/plan-json.h"

#include <cassert>

#define TEST_ASSERT(expr) do { \
    const bool test_result = static_cast<bool>(expr); \
    assert(test_result); \
    if (!test_result) return 1; \
} while (false)

int main() {
    common_agent_dataset_ref provenance_ref;
    provenance_ref.uri = "dataset://local/orders";
    provenance_ref.name = "orders";
    provenance_ref.row_count = 2;
    provenance_ref.column_count = 3;
    provenance_ref.source_resource_uri = "resource://turn/t/orders.json";
    provenance_ref.source_representation = "openapi:json-array";
    provenance_ref.source_provider = "sales-api";
    provenance_ref.source_operation = "listOrders";
    provenance_ref.source_request_json = R"({"limit":2})";
    provenance_ref.retrieved_at = 123;
    provenance_ref.content_hash = "sha256:abc";
    std::string error;
    const auto compact_ref = common_agent_dataset_ref_to_json(
        provenance_ref, common_agent_dataset_ref_json_projection::compact);
    TEST_ASSERT(!compact_ref.contains("source_provider"));
    TEST_ASSERT(!compact_ref.contains("source_request_json"));
    const auto full_ref = common_agent_dataset_ref_to_json(
        provenance_ref, common_agent_dataset_ref_json_projection::full);
    common_agent_dataset_ref decoded_ref;
    TEST_ASSERT(common_agent_dataset_ref_from_json(full_ref, decoded_ref, error));
    TEST_ASSERT(decoded_ref.source_provider == provenance_ref.source_provider);
    TEST_ASSERT(decoded_ref.source_operation == provenance_ref.source_operation);
    TEST_ASSERT(decoded_ref.source_request_json == provenance_ref.source_request_json);
    TEST_ASSERT(decoded_ref.retrieved_at == provenance_ref.retrieved_at);
    TEST_ASSERT(decoded_ref.content_hash == provenance_ref.content_hash);
    TEST_ASSERT(common_agent_dataset_uri_scope_component("turn/a 1") == "turn-a-1");
    TEST_ASSERT(common_agent_dataset_uri_is_current_turn(
        "dataset://agent/turn/turn-a-1/step-2", "turn/a 1"));
    TEST_ASSERT(!common_agent_dataset_uri_is_current_turn(
        "dataset://agent/turn/other/step-2", "turn/a 1"));

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
        nlohmann::ordered_json::object({{"dataset", "s1"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://session/old.csv");

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        csv_request,
        "dataset.inspect",
        nlohmann::ordered_json::object({{"dataset", "old.csv"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://session/old.csv");

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        csv_request,
        "resource_read",
        nlohmann::ordered_json::object({{"uri", "old.csv"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("uri", "") == "agent-resource://session/old.csv");

    common_agent_request ambiguous_resources = csv_request;
    ambiguous_resources.available_resources.push_back({
        common_runtime_resource_ref{"agent-resource://project/old.csv", "old.csv", "", "text/csv", 0},
        "scoped_reference",
        false});
    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(!common_agent_runtime_apply_safe_tool_defaults_to_json(
        ambiguous_resources,
        "dataset.inspect",
        nlohmann::ordered_json::object({{"dataset", "old.csv"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(error.find("ambiguous resource 'old.csv'") != std::string::npos);
    TEST_ASSERT(error.find("s1") != std::string::npos);
    TEST_ASSERT(error.find("s2") != std::string::npos);

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(!common_agent_runtime_apply_safe_tool_defaults_to_json(
        csv_request,
        "resource_inspect",
        nlohmann::ordered_json::object({{"uri", "missing.csv"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(error.find("unknown resource 'missing.csv'") != std::string::npos);
    TEST_ASSERT(error.find("r1") != std::string::npos);
    TEST_ASSERT(error.find("s1") != std::string::npos);

    normalized = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_safe_tool_defaults_to_json(
        csv_request,
        "dataset.inspect",
        nlohmann::ordered_json::object({{"dataset", "TAB6623_sv_sample.csv"}}),
        normalized,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(normalized.value("resource", "") == "agent-resource://turn/t/document.json");

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

    common_agent_dataset_descriptor orders;
    orders.ref.name = "orders";
    orders.ref.uri = "dataset://local/orders";
    orders.ref.source_resource_uri = "agent-resource://turn/t/orders.csv";
    common_agent_dataset_descriptor customers;
    customers.ref.name = "customers";
    customers.ref.uri = "dataset://local/customers";
    request.available_datasets = {orders, customers};
    const auto rendered_datasets = common_agent_render_dataset_inventory(request.available_datasets);
    TEST_ASSERT(rendered_datasets.find("id=d1 name=orders uri=dataset://local/orders") != std::string::npos);
    TEST_ASSERT(rendered_datasets.find("id=d2 name=customers uri=dataset://local/customers") != std::string::npos);
    TEST_ASSERT(rendered_datasets.find("$datasets.datasets[index].dataset") != std::string::npos);
    TEST_ASSERT(rendered_datasets.find("agent-resource://turn/t/orders.csv") != std::string::npos);

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
