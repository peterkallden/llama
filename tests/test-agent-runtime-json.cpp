#include "agent/runtime-json-contracts.h"
#include "agent/resource-chunk-extraction.h"
#include "agent/input-resources.h"
#include "agent/agent-inference.h"
#include "agent/dataset-contracts.h"
#include "plan/plan-json.h"

#include <cassert>
#include <cstdio>

#define TEST_ASSERT(expr) do { \
    const bool test_result = static_cast<bool>(expr); \
    assert(test_result); \
    if (!test_result) { std::fprintf(stderr, "TEST_ASSERT failed at line %d: %s\n", __LINE__, #expr); return 1; } \
} while (false)

int main() {
    common_runtime_resource_lineage processor_lineage{
        "resource://source/pdf", 0, 1, 0, 5018540, 0,
        "resource.process:pdf-page-image-v1>tesseract-ocr-v1"};
    assert(!common_runtime_resource_lineage_is_chunk(processor_lineage));
    common_runtime_resource_lineage chunk_lineage{
        "resource://derived/ocr-text", 0, 6, 0, 4096, 0, "host.chunk.text.v1"};
    assert(common_runtime_resource_lineage_is_chunk(chunk_lineage));

    common_agent_resource_chunk_extraction chunk_extraction;
    std::string chunk_error;
    const std::string chunk_text = "Radiation flux increased.\nThe camera remained operational during the campaign.";
    TEST_ASSERT(common_agent_parse_resource_chunk_extraction(
        "status: findings\nfinding: Radiation flux increased.\n"
        "finding: The camera remained operational during the campaign.",
        chunk_text, chunk_extraction, chunk_error));
    TEST_ASSERT(chunk_extraction.findings.size() == 2);
    TEST_ASSERT(chunk_extraction.findings.front().finding == "Radiation flux increased.");
    TEST_ASSERT(chunk_extraction.findings.front().evidence == "Radiation flux increased.");
    TEST_ASSERT(!common_agent_parse_resource_chunk_extraction(
        "status: findings\nfinding: findings",
        chunk_text, chunk_extraction, chunk_error));
    TEST_ASSERT(chunk_error.find("does not occur") != std::string::npos);
    TEST_ASSERT(common_agent_parse_resource_chunk_extraction(
        "status: no_findings", chunk_text, chunk_extraction, chunk_error));
    TEST_ASSERT(chunk_extraction.no_findings && chunk_extraction.findings.empty());
    TEST_ASSERT(!common_agent_parse_resource_chunk_extraction(
        "I will read this slice now.", chunk_text, chunk_extraction, chunk_error));
    TEST_ASSERT(!common_agent_parse_resource_chunk_extraction(
        "status: findings\nfinding: not in this resource",
        chunk_text, chunk_extraction, chunk_error));
    TEST_ASSERT(chunk_error.find("does not occur") != std::string::npos);
    TEST_ASSERT(!common_agent_parse_resource_chunk_extraction(
        "status: findings\nfinding:", chunk_text, chunk_extraction, chunk_error));
    TEST_ASSERT(chunk_error.find("must not be empty") != std::string::npos);
    TEST_ASSERT(!common_agent_parse_resource_chunk_extraction(
        "status: findings\nfinding: This finding has more than twenty words and should be rejected by the host parser because it is too long to be a concise evidence excerpt.",
        chunk_text, chunk_extraction, chunk_error));
    TEST_ASSERT(chunk_error.find("at most 20 words") != std::string::npos);
    const std::string twenty_word_evidence =
        "status: findings\nfinding: The camera remained operational during the campaign despite the high neutron flux and elevated radiation exposure throughout the experiment period.";
    const std::string twenty_word_chunk =
        "The camera remained operational during the campaign despite the high neutron flux and elevated radiation exposure throughout the experiment period.";
    TEST_ASSERT(common_agent_parse_resource_chunk_extraction(
        twenty_word_evidence, twenty_word_chunk, chunk_extraction, chunk_error));
    TEST_ASSERT(common_agent_resource_chunk_word_count(chunk_extraction.findings[0].evidence) == 20);
    TEST_ASSERT(!common_agent_parse_resource_chunk_extraction(
        "status: findings\nfinding: The camera remained operational during the campaign despite the high neutron flux and elevated radiation exposure throughout the experiment across all operating shifts.",
        "The camera remained operational during the campaign despite the high neutron flux and elevated radiation exposure throughout the experiment across all operating shifts.",
        chunk_extraction, chunk_error));
    TEST_ASSERT(chunk_error.find("at most 20 words") != std::string::npos);
    const std::string quoted_repeated_status =
        "status: findings\n"
        "finding: \"The cameras are installed in lead shield boxes.\"\n"
        "status: findings\n"
        "finding: \"Radiation flux was modeled by MCNP-6.\"";
    TEST_ASSERT(common_agent_parse_resource_chunk_extraction(
        quoted_repeated_status,
        "The cameras are installed in lead shield boxes. Radiation flux was modeled by MCNP-6.",
        chunk_extraction, chunk_error));
    TEST_ASSERT(chunk_extraction.findings.size() == 2);
    TEST_ASSERT(chunk_extraction.findings[1].evidence == "Radiation flux was modeled by MCNP-6.");
    common_agent_input_resource processor_resource;
    processor_resource.resource.uri = "resource://derived/ocr";
    processor_resource.resource.lineage = processor_lineage;
    const auto processor_context = common_agent_render_input_resource_context({processor_resource});
    TEST_ASSERT(processor_context.find("chunk_index=") == std::string::npos);
    common_agent_input_resource chunk_resource;
    chunk_resource.resource.uri = "resource://derived/ocr";
    chunk_resource.resource.lineage = chunk_lineage;
    const auto chunk_context = common_agent_render_input_resource_context({chunk_resource});
    TEST_ASSERT(chunk_context.find("chunk_index=0 chunk_count=6") != std::string::npos);

    common_agent_generation_resource_chunk generation_chunk;
    generation_chunk.text = chunk_text;
    generation_chunk.source.uri = "resource://derived/ocr";
    generation_chunk.chunk_index = 2;
    generation_chunk.chunk_count = 6;
    generation_chunk.byte_offset = 8192;
    generation_chunk.byte_length = chunk_text.size();
    common_agent_generation_request generation_request;
    generation_request.resource_chunk = generation_chunk;
    TEST_ASSERT(generation_request.resource_chunk->text == chunk_text);
    TEST_ASSERT(generation_request.resource_chunk->chunk_index == 2);
    common_agent_generation_options extraction_options;
    extraction_options.n_predict = 128;
    auto extraction_request = common_agent_make_resource_chunk_extraction_request(
        "Summarize the camera result.", generation_chunk, common_agent_scope{},
        extraction_options, "chunk-test");
    TEST_ASSERT(extraction_request.purpose == common_agent_generation_purpose::resource_chunk_extraction);
    TEST_ASSERT(extraction_request.tools.empty());
    TEST_ASSERT(extraction_request.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE);
    TEST_ASSERT(extraction_request.messages.size() == 2);
    TEST_ASSERT(extraction_request.messages[1].content.find(chunk_text) != std::string::npos);
    TEST_ASSERT(extraction_request.messages[1].content.find("Select a quote only if it helps answer the task.") != std::string::npos);
    TEST_ASSERT(extraction_request.messages[1].content.find("resource.read") == std::string::npos);
    TEST_ASSERT(extraction_request.grammar.empty());
    TEST_ASSERT(extraction_request.messages[0].content.find("status: findings") != std::string::npos);
    TEST_ASSERT(extraction_request.messages[0].content.find("finding: <exact contiguous quote") != std::string::npos);
    TEST_ASSERT(extraction_request.messages[0].content.find("at most 20 words") != std::string::npos);
    auto repair_request = common_agent_make_resource_chunk_extraction_request(
        "Summarize the camera result.", generation_chunk, common_agent_scope{},
        extraction_options, "chunk-repair-test",
        "return one exact quote that appears in the resource text.");
    TEST_ASSERT(repair_request.messages[0].content.find("Correction:") != std::string::npos);
    TEST_ASSERT(extraction_request.resource_chunk->byte_offset == 8192);

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
    bool changed = false;

    request.tool_argument_bindings.push_back({
        "openalex.listWorks",
        R"({"search":"machine learning","per_page":1,"select":"id,display_name"})",
        "config://openapi/openalex/listWorks",
        true});
    nlohmann::ordered_json bound_arguments;
    changed = false;
    TEST_ASSERT(common_agent_runtime_apply_host_tool_arguments_to_json(
        request,
        "openalex.listWorks",
        nlohmann::ordered_json::object(),
        bound_arguments,
        changed,
        error));
    TEST_ASSERT(changed);
    TEST_ASSERT(bound_arguments.value("search", "") == "machine learning");
    TEST_ASSERT(bound_arguments.value("per_page", 0) == 1);
    TEST_ASSERT(bound_arguments.value("select", "") == "id,display_name");

    bound_arguments = nlohmann::ordered_json();
    changed = false;
    TEST_ASSERT(!common_agent_runtime_apply_host_tool_arguments_to_json(
        request,
        "openalex.listWorks",
        nlohmann::ordered_json::object({{"search", "conflicting model value"}}),
        bound_arguments,
        changed,
        error));
    TEST_ASSERT(error.find("search") != std::string::npos);

    nlohmann::ordered_json normalized;
    changed = false;
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
