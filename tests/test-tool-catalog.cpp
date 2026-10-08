#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "plan/plan-contract.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <string>

int main() {
    common_tool_catalog catalog;
    common_tool_bootstrap_result first;
    std::string error;
    if (!catalog.bootstrap("memory", first, error)) {
        std::fprintf(stderr, "memory catalog bootstrap failed: %s\n", error.c_str());
        return 1;
    }
    assert(!first.definitions_created.empty());
    assert(catalog.find_definition("memory_search"));
    const auto * memory_remember = catalog.find_definition("memory_remember");
    if (!memory_remember) {
        std::fprintf(stderr, "missing memory_remember tool definition\n");
        return 1;
    }
    assert(memory_remember->requires_confirmation);
    assert(memory_remember->risk_class == common_tool_risk_class::memory_proposal);
    for (const auto * name : {"memory_get", "memory_propose_update", "memory_propose_forget"}) {
        const auto * definition = catalog.find_definition(name);
        if (!definition) {
            std::fprintf(stderr, "missing memory tool definition: %s\n", name);
            return 1;
        }
        const auto schema = nlohmann::json::parse(definition->input_schema_json);
        const auto & id = schema["properties"]["id"];
        assert(id.value("type", "") == "string");
        assert(id.value("minLength", 0) == 1);
        assert(id.value("maxLength", 0) == 256);
    }
    const auto * link_definition = catalog.find_definition("memory_link");
    if (!link_definition) {
        std::fprintf(stderr, "missing memory_link tool definition\n");
        return 1;
    }
    const auto link_schema = nlohmann::json::parse(link_definition->input_schema_json);
    assert(link_schema["properties"]["from"].value("minLength", 0) == 1);
    assert(link_schema["properties"]["to"].value("maxLength", 0) == 256);
    assert(link_schema["properties"]["relation"].value("minLength", 0) == 1);
    const auto * compact_definition = catalog.find_definition("memory_compact_propose");
    if (!compact_definition) {
        std::fprintf(stderr, "missing memory_compact_propose tool definition\n");
        return 1;
    }
    const auto compact_schema = nlohmann::json::parse(compact_definition->input_schema_json);
    assert(compact_schema["properties"]["source_ids"]["items"].value("maxLength", 0) == 256);
    const auto * web_search = catalog.find_definition("web_search");
    const auto * web_fetch = catalog.find_definition("web_fetch");
    if (!web_search || !web_fetch) {
        std::fprintf(stderr, "missing web tool definitions\n");
        return 1;
    }
    assert(web_search->executor_id == "builtin.web_search");
    assert(web_fetch->executor_id == "builtin.web_fetch");

    const auto * calculator = catalog.find_definition("math.calculate");
    const auto * calculator_legacy = catalog.find_definition("calculator");
    const auto * data_query = catalog.find_definition("data.query");
    const auto * data_join = catalog.find_definition("data.join");
    assert(calculator && calculator_legacy && calculator_legacy->name == "math.calculate" && data_query && data_join);
    assert(std::find(data_join->capabilities.begin(), data_join->capabilities.end(), "data.join") !=
        data_join->capabilities.end());
    const auto calculator_model = nlohmann::json::parse(calculator->model_input_schema_json);
    const auto data_query_model = nlohmann::json::parse(data_query->model_input_schema_json);
    assert(calculator_model["properties"].contains("expression"));
    assert(data_query_model["properties"].contains("dataset"));
    assert(!data_query_model["properties"].contains("max_scan_rows"));
    assert(!data_query_model["properties"].contains("materialize"));
    for (const auto * operation : {"data.query", "data.filter", "data.aggregate",
            "data.join", "data.transform"}) {
        const auto * definition = catalog.find_definition(operation);
        assert(definition);
        assert(std::find(definition->capabilities.begin(), definition->capabilities.end(), operation) !=
            definition->capabilities.end());
    }

    const auto * document_tables = catalog.find_definition("document.tables");
    const auto * document_table = catalog.find_definition("document.table");
    const auto * data_aggregate = catalog.find_definition("data.aggregate");
    const auto * dataset_schema = catalog.find_definition("dataset.schema");
    const auto * dataset_sample = catalog.find_definition("dataset.sample");
    const auto * dataset_validate = catalog.find_definition("dataset.validate");
    const auto * artifact_export = catalog.find_definition("artifact.export");
    const auto * value_counts = catalog.find_definition("statistics.value_counts");
    const auto * native_crash = catalog.find_definition("diagnostics.native_crash");
    assert(document_tables && document_table && data_aggregate && dataset_schema && dataset_sample && dataset_validate && artifact_export && value_counts && native_crash);
    assert(artifact_export->risk_class == common_tool_risk_class::artifact_write);
    assert(artifact_export->requires_confirmation);
    const auto native_crash_input = nlohmann::json::parse(native_crash->input_schema_json);
    const auto native_crash_model_result = nlohmann::json::parse(native_crash->result_schema_json);
    assert(native_crash_input["required"].size() == 2);
    assert(native_crash_input["properties"]["executable"].value("x-agent-type", "") == "resource_ref");
    assert(native_crash_input["properties"]["dump"].value("x-agent-type", "") == "resource_ref");
    assert(!native_crash_model_result["properties"].contains("raw_output"));
    const auto document_tables_result = nlohmann::json::parse(document_tables->result_schema_json);
    const auto document_table_result = nlohmann::json::parse(document_table->result_schema_json);
    const auto document_tables_input = nlohmann::json::parse(document_tables->input_schema_json);
    const auto document_table_input = nlohmann::json::parse(document_table->input_schema_json);
    const auto aggregate_input = nlohmann::json::parse(data_aggregate->input_schema_json);
    const auto document_table_model_input = nlohmann::json::parse(document_table->model_input_schema_json);
    const auto aggregate_model_input = nlohmann::json::parse(data_aggregate->model_input_schema_json);
    const auto aggregate_result = nlohmann::json::parse(data_aggregate->result_schema_json);
    const auto aggregate_model_result = nlohmann::json::parse(data_aggregate->model_result_schema_json);
    const auto dataset_schema_result = nlohmann::json::parse(dataset_schema->result_schema_json);
    const auto dataset_sample_result = nlohmann::json::parse(dataset_sample->result_schema_json);
    const auto dataset_validate_result = nlohmann::json::parse(dataset_validate->result_schema_json);
    const auto validate_input = nlohmann::json::parse(dataset_validate->input_schema_json);
    const auto validate_model_input = nlohmann::json::parse(dataset_validate->model_input_schema_json);
    const auto export_model_input = nlohmann::json::parse(artifact_export->model_input_schema_json);
    assert(document_tables_result["required"].size() == 3);
    assert(document_tables_input["properties"]["resource"].value("x-agent-type", "") == "resource_ref");
    assert(document_table_input["properties"]["resource"].value("x-agent-type", "") == "resource_ref");
    assert(document_table_input["properties"]["node_id"].value("x-agent-type", "") == "table_ref");
    assert(aggregate_input["properties"]["dataset"].value("x-agent-type", "") == "dataset_ref");
    assert(aggregate_input["properties"]["dataset"].value("x-agent-inferable", false));
    assert(aggregate_model_input["properties"]["dataset"].value("x-agent-type", "") == "dataset_ref");
    assert(aggregate_model_input["properties"]["dataset"].value("x-agent-inferable", false));
    assert(aggregate_model_input["x-agent-autowire-fields"][0] == "dataset");
    const auto aggregate_contract = common_project_model_tool_contract(
        data_aggregate->name,
        data_aggregate->description,
        data_aggregate->model_input_schema_json,
        data_aggregate->model_result_schema_json,
        error);
    assert(error.empty());
    assert(aggregate_contract.inputs.size() == 3);
    assert(aggregate_contract.inputs[0].name == "dataset");
    assert(aggregate_contract.inputs[0].may_be_inferred);
    assert(aggregate_contract.outputs.size() == 2);
    assert(aggregate_contract.outputs[0].name == "rows");
    std::string role_error;
    const auto role_contract = common_project_model_tool_contract(
        "role-test", "role test",
        R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref","x-agent-inferable":true}}})",
        R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref","x-agent-role":"dataflow"},"rows":{"type":"array","x-agent-role":"evidence"},"elapsed_ms":{"type":"integer","x-agent-role":"diagnostic"}}})",
        role_error);
    assert(role_error.empty());
    assert(role_contract.outputs[0].role == "dataflow");
    assert(role_contract.outputs[1].role == "evidence");
    assert(role_contract.outputs[2].role == "diagnostic");
    assert(validate_input["properties"]["dataset"].value("x-agent-type", "") == "dataset_ref");
    assert(validate_model_input["properties"]["dataset"].value("x-agent-type", "") == "dataset_ref");
    assert(export_model_input["properties"]["source_dataset"].value("x-agent-type", "") == "dataset_ref");
    assert(!aggregate_model_input["properties"].contains("materialize"));
    assert(!aggregate_model_input["properties"].contains("result_dataset"));
    assert(validate_model_input["required"].size() == 1);
    assert(validate_model_input["required"][0] == "rules");
    assert(document_table_model_input["properties"].contains("table"));
    assert(document_table_model_input["required"].size() == 2);
    assert(!document_table_model_input["properties"].contains("node_id"));
    assert(!document_table_model_input["properties"].contains("table_index"));
    assert(document_table_result["properties"]["dataset"].value("x-agent-type", "") == "dataset_ref");
    assert(aggregate_result["properties"]["dataset"].value("x-agent-type", "") == "dataset_ref");
    assert(aggregate_result["properties"].contains("materialized"));
    assert(aggregate_model_result["properties"].contains("rows"));
    assert(aggregate_model_result["properties"].contains("dataset"));
    assert(aggregate_model_result["additionalProperties"] == false);
    assert(!aggregate_model_result["properties"].contains("materialized"));
    assert(!aggregate_model_result["properties"].contains("scan_truncated"));
    assert(dataset_schema_result["properties"].contains("columns"));
    assert(dataset_sample_result["properties"].contains("rows"));
    assert(dataset_validate_result["properties"]["valid"].value("type", "") == "boolean");
    const auto value_counts_schema = nlohmann::json::parse(value_counts->input_schema_json);
    assert(value_counts_schema["required"].size() == 2);
    assert(value_counts_schema["properties"]["limit"].value("maximum", 0) == 1000);

    const auto * build = catalog.find_definition("development.build");
    const auto * test = catalog.find_definition("development.test");
    if (!build || !test) {
        std::fprintf(stderr, "missing developer tool definitions\n");
        return 1;
    }
    assert(build && test);
    assert(build->executor_id == "sandbox.development.build");
    assert(test->executor_id == "sandbox.development.test");
    assert(build->risk_class == common_tool_risk_class::sandbox_execution);
    assert(test->risk_class == common_tool_risk_class::sandbox_execution);
    assert(build->requires_confirmation && test->requires_confirmation);
    assert(build->input_schema_json.find("required\":[\"target\"]") != std::string::npos);
    assert(test->input_schema_json.find("required\":[\"target\"]") != std::string::npos);
    for (const auto * name : {"resource.inspect", "resource.read", "web.search",
            "web.fetch", "diagnostics.compile", "diagnostics.test_failures"}) {
        const auto * definition = catalog.find_definition(name);
        assert(definition);
        assert(std::find(definition->capabilities.begin(), definition->capabilities.end(), name) !=
            definition->capabilities.end());
    }
    for (const auto * name : {"document.tables", "document.table"}) {
        const auto * definition = catalog.find_definition(name);
        assert(definition);
        assert(std::find(definition->capabilities.begin(), definition->capabilities.end(),
            "document.inspect") != definition->capabilities.end());
    }

    const auto * resource_read = catalog.find_definition("resource.read");
    assert(resource_read);
    const auto resource_schema = nlohmann::json::parse(resource_read->input_schema_json);
    const auto resource_model_schema = nlohmann::json::parse(resource_read->model_input_schema_json);
    assert(resource_model_schema["required"].size() == 1);
    assert(resource_model_schema["required"][0] == "id");
    assert(resource_model_schema["properties"].contains("id"));
    assert(!resource_model_schema["properties"].contains("uri"));
    assert(resource_schema["required"][0] == "uri");
    assert(resource_schema["properties"]["offset"].value("minimum", 1) == 0);
    assert(resource_schema["properties"]["offset"].value("maximum", 0) == 1073741824);
    assert(resource_schema["properties"]["max_bytes"].value("maximum", 0) == 32768);
    std::string compact_error;
    const auto compact_read = common_render_compact_tool_description(
        resource_read->name,
        resource_read->description,
        resource_read->model_input_schema_json,
        resource_read->result_schema_json,
        compact_error);
    assert(compact_error.empty());
    assert(compact_read.find("resource.read:") != std::string::npos);
    assert(compact_read.find("Returns: result.") != std::string::npos);
    assert(compact_read.find("id (required, string)") != std::string::npos);
    assert(compact_read.find("representation (optional, text|bytes)") != std::string::npos);
    assert(compact_read.find("max_bytes (optional, integer[1..32768])") != std::string::npos);
    const auto aggregate_compact = common_render_compact_tool_description(
        data_aggregate->name,
        data_aggregate->description,
        common_tool_model_input_schema(*data_aggregate),
        common_tool_model_result_schema(*data_aggregate),
        compact_error);
    assert(compact_error.empty());
    assert(aggregate_compact.find("data.aggregate:") != std::string::npos);
    assert(aggregate_compact.find("dataset (optional, dataset_ref) [host may infer]") != std::string::npos);
    assert(aggregate_compact.find("measure: <function> <column> <as>") != std::string::npos);
    assert(aggregate_compact.find("named form: measure: function=<value> column=<value> as=<value>") != std::string::npos);
    const auto aggregate_returns = aggregate_compact.find(" Returns:");
    assert(aggregate_returns != std::string::npos);
    assert(aggregate_compact.substr(0, aggregate_returns).find("materialize") == std::string::npos);
    assert(aggregate_compact.find("materialized") == std::string::npos);
    assert(aggregate_compact.find("scan_truncated") == std::string::npos);
    assert(aggregate_compact.find("Returns: rows, dataset") != std::string::npos);
    assert(aggregate_compact.find("example:\nopen! data.aggregate\ndataset: $orders.dataset\nmeasure: sum amount total_amount") != std::string::npos);

    const auto * data_filter = catalog.find_definition("data.filter");
    const auto * data_transform = catalog.find_definition("data.transform");
    assert(data_filter && data_transform);
    const auto filter_compact = common_render_compact_tool_description(
        data_filter->name,
        data_filter->description,
        data_filter->model_input_schema_json,
        common_tool_model_result_schema(*data_filter),
        compact_error);
    assert(compact_error.empty());
    assert(filter_compact.find("conditions (required, object list)") != std::string::npos);
    assert(filter_compact.find("conditions[0].field: string") != std::string::npos);
    assert(filter_compact.find("Comparison operator. Null checks do not need value.") != std::string::npos);
    assert(filter_compact.find("example:\nopen! data.filter\ndataset: $orders.dataset\nconditions[0].field: status") != std::string::npos);
    const auto join_compact = common_render_compact_tool_description(
        data_join->name,
        data_join->description,
        data_join->model_input_schema_json,
        common_tool_model_result_schema(*data_join),
        compact_error);
    assert(compact_error.empty());
    assert(join_compact.find("on (required, directive(on) list)") != std::string::npos);
    assert(join_compact.find("on: <left> <right>") != std::string::npos);
    assert(join_compact.find("example:\nopen! data.join\nleft: $orders.dataset\nright: $customers.dataset\ntype: left\non: customer_id id") != std::string::npos);
    const auto transform_compact = common_render_compact_tool_description(
        data_transform->name,
        data_transform->description,
        data_transform->model_input_schema_json,
        common_tool_model_result_schema(*data_transform),
        compact_error);
    assert(compact_error.empty());
    assert(transform_compact.find("operations[0].type: rename|drop") != std::string::npos);
    assert(transform_compact.find("rename requires from and to; drop requires column.") != std::string::npos);
    assert(transform_compact.find("example:\nopen! data.transform\ndataset: $orders.dataset\noperations[0].type: rename") != std::string::npos);
    const auto value_counts_compact = common_render_compact_tool_description(
        value_counts->name,
        value_counts->description,
        value_counts->model_input_schema_json,
        common_tool_model_result_schema(*value_counts),
        compact_error);
    assert(compact_error.empty());
    assert(value_counts_compact.find("Exact column name from the selected dataset schema.") != std::string::npos);
    assert(value_counts_compact.find("example:\nopen! statistics.value_counts\ndataset: $orders.dataset\ncolumn: status\nlimit: 20") != std::string::npos);

    const auto * statistics_describe = catalog.find_definition("statistics.describe");
    assert(statistics_describe);
    const auto describe_compact = common_render_compact_tool_description(
        statistics_describe->name,
        statistics_describe->description,
        statistics_describe->model_input_schema_json,
        common_tool_model_result_schema(*statistics_describe),
        compact_error);
    assert(compact_error.empty());
    assert(describe_compact.find("Describe selected or schema-declared numeric columns in a bounded dataset. Returns: ") != std::string::npos);
    assert(describe_compact.find("columns (per-column count, null_count, min, max, mean and stddev)") != std::string::npos);
    assert(describe_compact.find("columns (optional, string list): Columns to describe") != std::string::npos);
    assert(describe_compact.find("group_by (optional, string list): Return separate statistics for each group.") != std::string::npos);
    const auto artifact_compact = common_render_compact_tool_description(
        artifact_export->name,
        artifact_export->description,
        artifact_export->model_input_schema_json,
        common_tool_model_result_schema(*artifact_export),
        compact_error);
    assert(compact_error.empty());
    assert(artifact_compact.find("rule: Choose one export mode: provide name and content for text, or source_dataset for CSV.") != std::string::npos);
    assert(artifact_compact.find("example:\nopen! artifact.export\nname: summary.txt\ncontent: \"Total orders: 42\"") != std::string::npos);

    std::vector<common_plan_schema_field> extracted_fields;
    std::string extraction_error;
    assert(common_plan_extract_schema_fields(R"({
        "type":"object",
        "properties":{
            "dataset":{"type":"string","x-agent-type":"dataset_ref","x-agent-role":"dataflow","x-agent-inferable":true},
            "rows":{"type":"array","x-agent-role":"evidence"}
        },
        "required":["dataset"]
    })", extracted_fields, extraction_error));
    assert(extraction_error.empty());
    assert(extracted_fields.size() == 2);
    assert(extracted_fields[0].name == "dataset");
    assert(extracted_fields[0].semantic_type == "dataset_ref");
    assert(extracted_fields[0].role == "dataflow");
    assert(extracted_fields[0].required && extracted_fields[0].inferable);
    assert(extracted_fields[1].name == "rows");
    assert(extracted_fields[1].role == "evidence");
    assert(!extracted_fields[1].required && !extracted_fields[1].inferable);
    assert(!common_plan_extract_schema_fields("[]", extracted_fields, extraction_error));
    assert(!extraction_error.empty());

    assert(build->policy_json.find("execution_class\":\"developer-build\"") != std::string::npos);
    assert(test->policy_json.find("filesystem\":\"workspace-write\"") != std::string::npos);

    const auto read = catalog.load_profile("memory-read", error);
    assert(error.empty());
    assert(read.size() == 9);
    for (const auto & definition : read) assert(definition.risk_class == common_tool_risk_class::local_read);

    const auto analysis = catalog.load_profile("analysis", error);
    assert(error.empty());
    assert(analysis.size() > read.size());
    assert(catalog.find_profile("analysis"));
    bool analysis_has_dataset_schema = false;
    bool analysis_has_document_tables = false;
    bool analysis_has_web_fetch = false;
    bool analysis_has_artifact_export = false;
    for (const auto & definition : analysis) {
        analysis_has_dataset_schema = analysis_has_dataset_schema || definition.name == "dataset.schema";
        analysis_has_document_tables = analysis_has_document_tables || definition.name == "document.tables";
        analysis_has_web_fetch = analysis_has_web_fetch || definition.name == "web.fetch";
        analysis_has_artifact_export = analysis_has_artifact_export || definition.name == "artifact.export";
        if (definition.name == "data.join") {
            assert(std::find(definition.capabilities.begin(), definition.capabilities.end(), "data.join") !=
                definition.capabilities.end());
        }
        if (definition.name == "artifact.export") {
            assert(definition.risk_class == common_tool_risk_class::artifact_write);
            assert(definition.requires_confirmation);
            assert(std::find(definition.capabilities.begin(), definition.capabilities.end(),
                "artifact.export") != definition.capabilities.end());
        } else {
            assert(!definition.requires_confirmation);
        }
        assert(definition.name != "memory_remember");
        assert(definition.name != "development.build");
    }
    assert(analysis_has_dataset_schema && analysis_has_document_tables && analysis_has_web_fetch &&
        analysis_has_artifact_export);
    const auto * analysis_profile = catalog.find_profile("analysis");
    assert(analysis_profile && analysis_profile->allow_policy_gated_writes == true);
    const auto research = catalog.load_profile("research", error);
    assert(error.empty());
    assert(std::any_of(research.begin(), research.end(), [](const auto & definition) {
        return definition.name == "artifact.export";
    }));
    const auto developer_read = catalog.load_profile("developer-read", error);
    assert(error.empty());
    assert(std::none_of(developer_read.begin(), developer_read.end(), [](const auto & definition) {
        return definition.name == "artifact.export";
    }));

    common_tool_bootstrap_result second;
    assert(catalog.bootstrap("memory", second, error));
    assert(second.definitions_created.empty());
    assert(second.definitions_unchanged.size() >= first.definitions_created.size());
    assert(!catalog.bootstrap("not-a-profile", second, error));
    assert(error == "tool profile is unavailable: not-a-profile");
    return 0;
}
