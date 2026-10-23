#include "agent/tooling/schema/tool-output-codec.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/contracts/agent-request.h"
#include "plan/plan-json.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

namespace {

common_chat_tool list_works_tool() {
    return {
        "openalex.listWorks",
        "List OpenAlex works.",
        R"json({"type":"object","required":["search"],"properties":{"search":{"type":"string"},"per_page":{"type":"integer","minimum":1,"maximum":100},"select":{"type":"string"}}})json",
        R"json({"type":"object"})json",
    };
}

common_chat_tool aggregate_tool() {
    return {
        "data.aggregate",
        "Aggregate rows.",
        R"json({"type":"object","additionalProperties":false,"required":["dataset","measures"],"properties":{"dataset":{"type":"string"},"group_by":{"type":"array","items":{"type":"string"}},"measures":{"type":"array","minItems":1,"x-agent-flat":{"directive":"measure","positional":["function","column","as"]},"items":{"type":"object","additionalProperties":false,"required":["function"],"properties":{"function":{"type":"string","enum":["sum","avg"]},"column":{"type":"string"},"as":{"type":"string"}}}}}})json",
        R"json({"type":"object"})json",
    };
}

common_chat_tool describe_tool() {
    return {
        "statistics.describe",
        "Describe selected dataset columns.",
        R"json({"type":"object","additionalProperties":false,"required":["columns"],"properties":{"columns":{"type":"array","items":{"type":"string"}},"select":{"type":"string"}}})json",
        R"json({"type":"object"})json",
    };
}

common_chat_tool query_tool() {
    return {
        "data.query",
        "Query a dataset.",
        R"json({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string"},"where":{"type":"array","x-agent-flat":{"directive":"where","positional":["field","operator","value"],"two_slot_value_default_operator":"contains"},"items":{"type":"object","required":["field","operator"],"properties":{"field":{"type":"string"},"operator":{"type":"string","enum":["=","contains","in","not_null"]},"value":{}}}},"order_by":{"type":"array","x-agent-flat":{"directive":"order_by","positional":["field","direction"]},"items":{"type":"object","required":["field"],"properties":{"field":{"type":"string"},"direction":{"type":"string","enum":["asc","desc"]}}}}}})json",
        R"json({"type":"object"})json",
    };
}

void test_formats() {
    std::string error;
    common_agent_tool_output_format format;
    assert(common_parse_agent_tool_output_format("jsonl", format, error));
    assert(format == common_agent_tool_output_format::jsonl);
    assert(common_parse_agent_tool_output_format("compact_dsl", format, error));
    assert(format == common_agent_tool_output_format::compact_dsl);
    assert(common_parse_agent_tool_output_format("native", format, error));
    assert(format == common_agent_tool_output_format::native);
    assert(common_parse_agent_tool_output_format("json", format, error));
    assert(format == common_agent_tool_output_format::native);
    assert(common_agent_tool_output_format_name(format) == std::string("json"));
    assert(common_parse_agent_tool_output_format("dsl", format, error));
    assert(format == common_agent_tool_output_format::compact_dsl);
    assert(common_agent_tool_output_format_name(format) == std::string("dsl"));
    assert(common_parse_agent_tool_output_format("compact_dsl", format, error));
    assert(format == common_agent_tool_output_format::compact_dsl);
    assert(!common_parse_agent_tool_output_format("xml", format, error));
}

void test_request_default_is_dsl() {
    common_agent_request request;
    assert(request.tool_output_format == common_agent_tool_output_format::compact_dsl);
}

void test_compact_variants_normalize_identically() {
    const std::string command =
        R"(open! openalex.listWorks search="machine learning" per_page=1 select="id,display_name")";
    const std::string call =
        R"(openalex.listWorks(search="machine learning", per_page=1, select="id,display_name"))";
    common_agent_tool_call command_call;
    common_agent_tool_call call_call;
    std::string error;
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl, command, command_call, error));
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl, call, call_call, error));
    assert(command_call.name == "openalex.listWorks");
    assert(command_call.arguments_json == call_call.arguments_json);
    assert(command_call.arguments_json.find("\"per_page\":1") != std::string::npos);

    std::string normalized;
    assert(common_normalize_comma_separated_string(
        "id, display_name", normalized, error));
    assert(normalized == "id,display_name");
    assert(!common_normalize_comma_separated_string(
        "id,,display_name", normalized, error));
}

void test_jsonl_normalization() {
    common_agent_tool_call call;
    std::string error;
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::jsonl,
        R"({"tool":"openalex.listWorks","arguments":{"search":"machine learning","per_page":1}})",
        call, error));
    assert(call.name == "openalex.listWorks");
    assert(call.arguments_json.find("machine learning") != std::string::npos);
    assert(!common_parse_model_tool_call(
        common_agent_tool_output_format::jsonl,
        "{\"tool\":\"openalex.listWorks\"}", call, error));
}

void test_v1_schema_boundary_and_rendering() {
    std::string reason;
    assert(common_compact_dsl_schema_supported(list_works_tool().parameters, reason));
    assert(common_compact_dsl_schema_supported(
        R"json({"type":"object","properties":{"filters":{"type":"array","items":{"type":"object","properties":{"field":{"type":"string"},"op":{"type":"string"}}}}}})json",
        reason));

    std::vector<std::string> unsupported;
    std::string error;
    const auto instructions = common_render_model_tool_output_instructions(
        common_agent_tool_output_format::compact_dsl,
        {list_works_tool(), {"complex", "Complex", R"json({"type":"object","properties":{"x":{"type":"object","properties":{"a":{"type":"string"}}}}})json", "{}"}},
        &unsupported,
        error);
    assert(!instructions.empty());
    assert(unsupported.empty());
    assert(instructions.find("openalex.listWorks") != std::string::npos);
    assert(instructions.find("complex") != std::string::npos);
    assert(instructions.find("one `field: value` per line") != std::string::npos);
    assert(instructions.find("comma-separated values such as `columns: region, amount`") != std::string::npos);
    assert(instructions.find("bracketed list such as `columns: [region, amount]`") != std::string::npos);
    assert(instructions.find("Quote scalar strings containing commas") != std::string::npos);
}

void test_schema_aware_flat_dsl_and_named_slots() {
    const auto tool = aggregate_tool();
    const std::vector<common_chat_tool> tools{tool};
    common_agent_tool_call call;
    std::string error;
    const std::string positional =
        "open! data.aggregate\n"
        "dataset: $inspect.dataset\n"
        "group_by: region\n"
        "measure: sum amount total_amount\n"
        "measure: avg amount average_amount";
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl, positional, tools, call, error));
    const auto args = nlohmann::json::parse(call.arguments_json);
    assert(call.name == "data.aggregate");
    assert(args["dataset"] == "$inspect.dataset");
    assert(args["group_by"] == nlohmann::json::array({"region"}));
    assert(args["measures"].size() == 2);
    assert(args["measures"][0] == nlohmann::json({{"function", "sum"}, {"column", "amount"}, {"as", "total_amount"}}));

    const std::string bracketed_array =
        "open! data.aggregate\n"
        "dataset: \"sales, net\"\n"
        "group_by: [region, \"sales, net\"]\n"
        "measure: sum amount total_amount";
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl, bracketed_array, tools, call, error));
    const auto bracketed_args = nlohmann::json::parse(call.arguments_json);
    assert(bracketed_args["dataset"] == "sales, net");
    assert(bracketed_args["group_by"] == nlohmann::json::array({"region", "sales, net"}));

    const std::vector<common_chat_tool> describe_tools{describe_tool()};
    const std::string comma_separated_columns =
        "open! statistics.describe\n"
        "columns: amount, region";
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl, comma_separated_columns, describe_tools, call, error));
    const auto comma_separated_args = nlohmann::json::parse(call.arguments_json);
    assert(comma_separated_args["columns"] == nlohmann::json::array({"amount", "region"}));

    const std::string comma_containing_column_name =
        "open! statistics.describe\n"
        "columns: amount, \"sales, net\"";
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl, comma_containing_column_name, describe_tools, call, error));
    const auto quoted_comma_args = nlohmann::json::parse(call.arguments_json);
    assert(quoted_comma_args["columns"] == nlohmann::json::array({"amount", "sales, net"}));

    const std::string quoted_scalar_comma =
        "open! statistics.describe\n"
        "columns: amount\n"
        "select: \"id, display_name\"";
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl, quoted_scalar_comma, describe_tools, call, error));
    const auto quoted_scalar_args = nlohmann::json::parse(call.arguments_json);
    assert(quoted_scalar_args["select"] == "id, display_name");

    assert(!common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! data.aggregate\ndataset: $inspect.dataset\ngroup_by: [region, amount]\ngroup_by: channel\nmeasure: sum amount total_amount",
        tools, call, error));
    assert(!common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! data.aggregate\ndataset: [$inspect.dataset]\nmeasure: sum amount total_amount",
        tools, call, error));

    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! data.aggregate\ngroup_by: null\nmeasure: sum amount total_amount",
        tools, call, error));
    const auto omitted_optional = nlohmann::json::parse(call.arguments_json);
    assert(!omitted_optional.contains("group_by"));
    assert(omitted_optional["measures"].size() == 1);

    const std::string named =
        "open! data.aggregate\n"
        "dataset: $inspect.dataset\n"
        "measure: function=sum column=amount as=total_amount";
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl, named, tools, call, error));
    const auto named_args = nlohmann::json::parse(call.arguments_json);
    assert(named_args["measures"][0]["as"] == "total_amount");
    assert(!common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! data.aggregate\ndataset: \nmeasure: sum", tools, call, error));
    assert(error.find("empty") != std::string::npos);
    assert(!common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! data.aggregate\ndataset: x\nmeasure: sum {column=amount}", tools, call, error));
}

void test_predicate_directive_avoids_indexed_bracket_paths() {
    const auto tool = query_tool();
    std::string error;
    const auto description = common_render_compact_tool_description(
        tool.name, tool.description, tool.parameters, tool.result_schema, error);
    assert(error.empty());
    assert(description.find("where: <field> <operator> <value>") != std::string::npos);
    assert(description.find("where[0].field") == std::string::npos);
    assert(description.find("order_by: <field> <direction>") != std::string::npos);
    assert(description.find("order_by[0].field") == std::string::npos);

    common_agent_tool_call call;
    const std::vector<common_chat_tool> tools{tool};
    const bool parsed = common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! data.query\n"
        "where: title contains \"population density\"\n"
        "where: geo in [SE, NO]\n"
        "where: status not_null\n"
        "order_by: code asc\n"
        "order_by: title desc",
        tools, call, error);
    if (!parsed) std::fprintf(stderr, "predicate directive parse failed: %s\n", error.c_str());
    assert(parsed);
    const auto arguments = nlohmann::json::parse(call.arguments_json);
    assert(arguments["where"].size() == 3);
    assert(arguments["where"][0] == nlohmann::json({
        {"field", "title"}, {"operator", "contains"}, {"value", "population density"}}));
    assert(arguments["where"][1]["value"] == nlohmann::json({"SE", "NO"}));
    assert(arguments["where"][2] == nlohmann::json({
        {"field", "status"}, {"operator", "not_null"}}));
    assert(arguments["order_by"] == nlohmann::json::array({
        nlohmann::json{{"field", "code"}, {"direction", "asc"}},
        nlohmann::json{{"field", "title"}, {"direction", "desc"}},
    }));

    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! data.query\nwhere: title \"population density\"",
        tools, call, error));
    const auto short_form = nlohmann::json::parse(call.arguments_json);
    assert(short_form["where"][0] == nlohmann::json({
        {"field", "title"}, {"operator", "contains"}, {"value", "population density"}}));

    assert(!common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! data.query\nwhere[[0]].field: title", tools, call, error));
    assert(!common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! data.query\norder_by[[0]].field: code", tools, call, error));
}

void test_flat_paths_and_multiline_plan() {
    const auto tool = aggregate_tool();
    const std::vector<common_chat_tool> tools{tool};
    std::string proposal;
    std::string error;
    const std::string dsl =
        "plan goal=\"Aggregate rows\"\n"
        "step as=result | open! data.aggregate\n"
        "dataset: $inspect.dataset\n"
        "group_by: region\n"
        "measure: sum amount total";
    assert(common_parse_compact_dsl_plan(dsl, tools, proposal, error));
    const auto plan = nlohmann::json::parse(proposal);
    assert(plan["steps"][0]["args"]["measures"][0]["function"] == "sum");
    assert(plan["steps"][0]["args"]["measures"][0]["as"] == "total");

    const std::string implicit_id_dsl =
        "plan goal=\"Aggregate the selected dataset\"\n"
        "step | open! data.aggregate\n"
        "dataset: d1\n"
        "measure: sum amount total";
    assert(common_parse_compact_dsl_plan(implicit_id_dsl, tools, proposal, error));
    const auto implicit_id_plan = nlohmann::json::parse(proposal);
    assert(implicit_id_plan["steps"].size() == 1);
    assert(implicit_id_plan["steps"][0]["tool"] == "data.aggregate");

    const common_chat_tool nested{
        "nested.tool", "Nested fields.",
        R"json({"type":"object","properties":{"operations":{"type":"array","items":{"type":"object","properties":{"kind":{"type":"string"},"column":{"type":"string"}}}}}})json",
        R"json({"type":"object"})json"};
    common_agent_tool_call call;
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! nested.tool\noperations[0].kind: rename\noperations[0].column: amount",
        std::vector<common_chat_tool>{nested}, call, error));
    const auto nested_args = nlohmann::json::parse(call.arguments_json);
    assert(nested_args["operations"][0]["kind"] == "rename");
    assert(nested_args["operations"][0]["column"] == "amount");
}

void test_nested_scalar_lists() {
    const common_chat_tool tool{"nested.lists", "Nested lists.",
        R"json({"type":"object","properties":{"operations":{"type":"array","items":{"type":"object","properties":{"values":{"type":"array","items":{"type":"string"}},"value":{},"column":{"type":"string"}}}}}})json",
        "{}"};
    const std::vector<common_chat_tool> tools{tool};
    common_agent_tool_call call;
    std::string error;
    assert(common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl,
        "open! nested.lists\noperations[0].values: [north, east]\noperations[0].value: [10, 11]",
        tools, call, error));
    const auto arguments = nlohmann::json::parse(call.arguments_json);
    assert(arguments["operations"][0]["values"] == nlohmann::json::array({"north", "east"}));
    assert(arguments["operations"][0]["value"] == nlohmann::json::array({10, 11}));
    assert(!common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl,
        "open! nested.lists\noperations[0].column: [amount]", tools, call, error));
    assert(!common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl,
        "open! nested.lists\noperations[0].missing: [amount]", tools, call, error));
    assert(!common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl,
        "open! nested.lists\noperations[0].value: [10]\noperations[0].value: [11]", tools, call, error));
}

void test_nested_arguments_and_multistep_plan() {
    const std::string call_text =
        R"(open! data.aggregate dataset=$inspect.dataset measures=[{function=sum; field="amount"}, {function=count; field="region"}] filter={field=region; op=eq; value="north"})";
    common_agent_tool_call call;
    std::string error;
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl, call_text, call, error));
    assert(call.name == "data.aggregate");
    assert(call.arguments_json.find("\"function\":\"sum\"") != std::string::npos);
    assert(call.arguments_json.find("\"dataset\":\"$inspect.dataset\"") != std::string::npos);

    const std::string dsl =
        "plan goal=\"Aggregate selected rows\"\n"
        "step as=inspect | open! dataset.inspect resource=r1\n"
        "step as=result | open! data.aggregate dataset=$inspect.dataset measures=[{function=sum; field=amount}]";
    std::string proposal;
    assert(common_parse_compact_dsl_plan(dsl, proposal, error));
    assert(proposal.find("\"goal\":\"Aggregate selected rows\"") != std::string::npos);
    assert(proposal.find("\"as\":\"inspect\"") != std::string::npos);
    assert(proposal.find("\"$inspect.dataset\"") != std::string::npos);
    common_plan_state plan;
    std::vector<common_plan_operation> operations;
    assert(common_plan_parse_proposal_json(proposal, plan, operations, error, 6));
    // Proposals also contain dependency operations between executable steps.
    // Select the tool steps rather than treating every operation as one.
    std::vector<const common_plan_step *> tool_steps;
    for (const auto & operation : operations) {
        if (operation.step && operation.step->tool_call) tool_steps.push_back(&*operation.step);
    }
    assert(tool_steps.size() == 2);
    assert(tool_steps[0]->tool_call->name == "dataset.inspect");
    assert(tool_steps[1]->tool_call->name == "data.aggregate");
    assert(tool_steps[0]->semantic_alias && *tool_steps[0]->semantic_alias == "inspect");
    const auto aggregate_args = nlohmann::json::parse(
        tool_steps[1]->tool_call->arguments_json, nullptr, false);
    assert(aggregate_args["dataset"].value("$from_step", "") == "step_1");
    assert(aggregate_args["dataset"].value("$json_pointer", "") == "/dataset");
    assert(!common_parse_compact_dsl_plan(
        "plan goal=\"test\"\nstep | open! tool a=1\nstep | open! tool a=2\nstep | open! tool a=3\nstep | open! tool a=4\nstep | open! tool a=5\nstep | open! tool a=6\nstep | open! tool a=7\nstep | open! tool a=8\nstep | open! tool a=9",
        proposal, error));
}

void test_invalid_compact_output() {
    common_agent_tool_call call;
    std::string error;
    assert(!common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! openalex.listWorks search= per_page=1", call, error));
    assert(!common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "openalex.listWorks(search=\"x\" search=\"y\")", call, error));
}

void test_compact_single_workflow_action() {
    common_agent_tool_call call;
    std::string error;
    assert(common_parse_model_tool_call(
        common_agent_tool_output_format::compact_dsl,
        "open! statistics.describe column=amount",
        call, error));
    assert(call.name == "statistics.describe");
    assert(call.arguments_json == R"({"column":"amount"})");
}

void test_nested_tool_argument_schema_validation() {
    const std::string schema = R"json({
        "type":"object",
        "additionalProperties":false,
        "required":["operation"],
        "properties":{
            "operation":{"type":"string","enum":["aggregate","filter"]},
            "measures":{"type":"array","minItems":1,"items":{
                "type":"object","additionalProperties":false,
                "required":["function","column"],
                "properties":{
                    "function":{"type":"string","enum":["sum","count"]},
                    "column":{"type":"string","minLength":1}
                }
            }}
        }
    })json";
    std::string normalized;
    std::string error;
    assert(common_schema_normalize_and_validate_object(
        R"({"operation":"aggregate","measures":[{"function":"sum","column":"sales"}]})",
        schema, normalized, error));
    assert(!common_schema_normalize_and_validate_object(
        R"({"operation":"aggregate","measures":[{"function":"median","column":"sales"}]})",
        schema, normalized, error));
    assert(!common_schema_normalize_and_validate_object(
        R"({"operation":"aggregate","measures":[{"function":"sum"}]})",
        schema, normalized, error));
    assert(!common_schema_normalize_and_validate_object(
        R"({"operation":"unknown"})", schema, normalized, error));
    assert(!common_schema_normalize_and_validate_object(
        R"({"measures":[]})", schema, normalized, error));
    assert(error == "required contract field is missing: operation");
    const std::string nullable = R"({"type":"object","properties":{"exit_code":{"type":["integer","null"]},"source":{"type":["object","null"]}}})";
    assert(common_schema_normalize_and_validate_object(
        R"({"exit_code":0,"source":{}})", nullable, normalized, error));
    assert(common_schema_normalize_and_validate_object(
        R"({"exit_code":null,"source":null})", nullable, normalized, error));
    std::string in_place = R"({"exit_code":0,"source":{}})";
    assert(common_schema_normalize_and_validate_object(in_place, nullable, in_place, error));
    assert(in_place == R"({"exit_code":0,"source":{}})");
    assert(!common_schema_normalize_and_validate_object(
        R"({"exit_code":"0"})", nullable, normalized, error));
    assert(normalized.empty());
    assert(!common_schema_normalize_and_validate_object(
        R"({"source":[]})", nullable, normalized, error));
    assert(!common_schema_normalize_and_validate_object(
        R"({"value":null})", R"({"type":"object","properties":{"value":{"type":["null",2]}}})", normalized, error));
    assert(!common_schema_normalize_and_validate_object(
        R"({})", R"({"type":["object","null"]})", normalized, error));
    assert(!common_schema_normalize_and_validate_object(
        R"({"value":"x"})", R"({"type":"object","properties":{"value":{"minLength":"invalid"}}})", normalized, error));

}

} // namespace

int main() {
    test_formats();
    test_request_default_is_dsl();
    test_compact_variants_normalize_identically();
    test_jsonl_normalization();
    test_v1_schema_boundary_and_rendering();
    test_schema_aware_flat_dsl_and_named_slots();
    test_predicate_directive_avoids_indexed_bracket_paths();
    test_flat_paths_and_multiline_plan();
    test_nested_scalar_lists();
    test_nested_arguments_and_multistep_plan();
    test_invalid_compact_output();
    test_compact_single_workflow_action();
    test_nested_tool_argument_schema_validation();
    return 0;
}
