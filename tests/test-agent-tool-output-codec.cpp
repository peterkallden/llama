#include "agent/tooling/schema/tool-output-codec.h"

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
    assert(!common_parse_agent_tool_output_format("xml", format, error));
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
    assert(!common_compact_dsl_schema_supported(
        R"json({"type":"object","properties":{"filters":{"type":"array","items":{"type":"object"}}}})json",
        reason));

    std::vector<std::string> unsupported;
    std::string error;
    const auto instructions = common_render_model_tool_output_instructions(
        common_agent_tool_output_format::compact_dsl,
        {list_works_tool(), {"complex", "Complex", R"json({"type":"object","properties":{"x":{"type":"object"}}})json", "{}"}},
        &unsupported,
        error);
    assert(!instructions.empty());
    assert(unsupported.size() == 1 && unsupported.front() == "complex");
    assert(instructions.find("openalex.listWorks") != std::string::npos);
    assert(instructions.find("complex") == std::string::npos);
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

} // namespace

int main() {
    test_formats();
    test_compact_variants_normalize_identically();
    test_jsonl_normalization();
    test_v1_schema_boundary_and_rendering();
    test_invalid_compact_output();
    return 0;
}
