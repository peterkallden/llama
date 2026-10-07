#include "agent/tooling/schema/tool-output-codec.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/contracts/agent-request.h"
#include "plan/plan-json.h"

#include <cassert>
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
    assert(operations.size() >= 2);
    assert(operations[0].step->tool_call->name == "dataset.inspect");
    assert(operations[1].step->tool_call->name == "data.aggregate");
    assert(operations[0].step->semantic_alias && *operations[0].step->semantic_alias == "inspect");
    const auto aggregate_args = nlohmann::json::parse(
        operations[1].step->tool_call->arguments_json, nullptr, false);
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
}

} // namespace

int main() {
    test_formats();
    test_request_default_is_dsl();
    test_compact_variants_normalize_identically();
    test_jsonl_normalization();
    test_v1_schema_boundary_and_rendering();
    test_nested_arguments_and_multistep_plan();
    test_invalid_compact_output();
    test_compact_single_workflow_action();
    test_nested_tool_argument_schema_validation();
    return 0;
}
