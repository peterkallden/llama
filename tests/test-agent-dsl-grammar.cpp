#include "agent/tooling/schema/tool-dsl-grammar.h"
#include "agent/tooling/schema/tool-output-codec.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include <nlohmann/json.hpp>
#include "../src/llama-grammar.h"
#include "../src/unicode.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool condition, const std::string & message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

bool matches(const std::string & grammar, const std::string & text) {
    std::unique_ptr<llama_grammar, decltype(&llama_grammar_free_impl)> sampler(
        llama_grammar_init_impl(nullptr, grammar.c_str(), "root", false, nullptr, 0, nullptr, 0),
        llama_grammar_free_impl);
    require(sampler != nullptr, "generated GBNF failed to compile:\n" + grammar);
    try {
        for (const auto character : unicode_cpts_from_utf8(text)) {
            llama_grammar_accept(sampler.get(), character);
            if (llama_grammar_get_stacks(sampler.get()).empty()) return false;
        }
    } catch (const std::runtime_error &) { return false; }
    const auto & stacks = llama_grammar_get_stacks(sampler.get());
    return std::any_of(stacks.begin(), stacks.end(), [](const auto & stack) { return stack.empty(); });
}

void action_contracts() {
    const common_chat_tool inspect{"code.inspect", "Inspect", R"({"type":"object","additionalProperties":false})", "{}"};
    const common_chat_tool symbols{"code.symbols", "Symbols",
        R"({"type":"object","additionalProperties":false,"required":["symbol"],"properties":{"symbol":{"type":"string"}}})", "{}"};
    std::string error;
    const auto grammar = common_build_tool_dsl_grammar({inspect, symbols}, common_tool_dsl_grammar_form::action, error);
    require(!grammar.empty() && error.empty(), "mixed tool grammar was not produced");
    for (const auto & good : {"open! code.inspect", "open! code.inspect\n", "open! code.symbols\nsymbol: renderer"}) {
        require(matches(grammar, good), std::string("valid action rejected: ") + good);
        common_agent_tool_call call;
        require(common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl, good, {inspect, symbols}, call, error), error);
    }
    for (const auto & bad : {"open! code.inspect\nsymbol: renderer", "open! code.inspect\nfield: result",
            "open! code.symbols", "open! code.symbols\nsymbol: a\nsymbol: b", "open! code.symbols\nbackend: true", "open! code.inspect.", "open! private.inspect"}) {
        require(!matches(grammar, bad), std::string("cross-tool or invented field accepted: ") + bad);
    }
    const auto no_args = common_build_tool_dsl_grammar({inspect}, common_tool_dsl_grammar_form::action, error);
    require(!no_args.empty() && matches(no_args, "open! code.inspect"), "all-no-argument catalog disabled the grammar");
    const auto args = common_build_tool_dsl_grammar({symbols}, common_tool_dsl_grammar_form::arguments, error);
    require(matches(args, "symbol: renderer") && !matches(args, "backend: true") && !matches(args, "open! code.symbols"),
        "argument repair escaped its selected tool contract");
    const auto empty_args = common_build_tool_dsl_grammar({inspect}, common_tool_dsl_grammar_form::arguments, error);
    require(matches(empty_args, "") && !matches(empty_args, "field: result"), "empty argument grammar was not empty");
}

void plan_contracts() {
    const common_chat_tool inspect{"code.inspect", "Inspect", R"({"type":"object","additionalProperties":false})", "{}"};
    const common_chat_tool lookup{"lookup", "Lookup",
        R"({"type":"object","properties":{"symbol":{"type":"string"}}})", "{}"};
    std::string error;
    const auto grammar = common_build_tool_dsl_grammar({inspect, lookup}, common_tool_dsl_grammar_form::plan, error, 2, 32);
    const std::string one = "plan goal=\"Inspect\"\nstep | open! code.inspect";
    const std::string chain = one + "\nstep as=definition | open! lookup\nsymbol: $previous.symbol";
    require(matches(grammar, one) && matches(grammar, chain) && matches(grammar, chain + "\n"), "valid plan rejected");
    require(matches(grammar, "plan goal=\"Undersök kod\"\nstep | open! code.inspect"), "UTF-8 goal rejected");
    for (const auto & bad : {one + "\nfield: result", one + "\nsymbol: renderer",
            chain + "\nstep | open! code.inspect", "plan goal=\"" + std::string(33, 'x') + "\"\nstep | open! code.inspect",
            std::string("plan goal=\"Inspect\""), std::string("{\"goal\":\"Inspect\"}")}) {
        require(!matches(grammar, bad), "invalid or out-of-budget plan accepted by grammar");
    }
    std::string canonical;
    require(common_parse_compact_dsl_plan(chain, {inspect, lookup}, canonical, error), error);
    const auto single = common_build_tool_dsl_grammar({inspect}, common_tool_dsl_grammar_form::plan, error, 1);
    require(matches(single, one) && !matches(single, one + "\nstep | open! code.inspect"), "one-step host bound was lost");
}

void nested_contracts() {
    const common_chat_tool structured{"data.transform", "Transform", R"({"type":"object","properties":{
        "operations":{"type":"array","items":{"type":"object","additionalProperties":false,"properties":{"column":{"type":"string"}}}},
        "measures":{"type":"array","x-agent-flat":{"directive":"measure","positional":["function","column"]},"items":{"type":"object","properties":{"function":{"type":"string"},"column":{"type":"string"}}}},
        "metadata":{"type":"object","additionalProperties":true}}})", "{}"};
    std::string error;
    const auto grammar = common_build_tool_dsl_grammar({structured}, common_tool_dsl_grammar_form::action, error);
    for (const auto & good : {"open! data.transform\noperations[0].column: amount", "open! data.transform\nmeasure: sum amount",
            "open! data.transform\nmetadata.origin: fixture", "open! data.transform\nmeasures[0].function: sum\nmeasures[0].column: amount"}) {
        require(matches(grammar, good), std::string("declared nested path/directive rejected: ") + good);
        common_agent_tool_call call;
        require(common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl, good, {structured}, call, error), error);
    }
    require(!matches(grammar, "open! data.transform\noperations[[0]].column: amount"), "array path gained doubled brackets");
    require(!matches(grammar, "open! data.transform\noperations[0].missing: amount"), "undeclared closed-object path accepted");
    const common_chat_tool alternatives{"choose", "Choose", R"({"type":"object","oneOf":[
        {"properties":{"dataset":{"type":"string"}}},{"properties":{"resource":{"type":"string"}}}]})", "{}"};
    const auto union_grammar = common_build_tool_dsl_grammar({alternatives}, common_tool_dsl_grammar_form::action, error);
    require(matches(union_grammar, "open! choose\ndataset: d1") && matches(union_grammar, "open! choose\nresource: r1"),
        "schema alternatives lost declared fields");
}

void typed_contracts() {
    const common_chat_tool tool{"typed", "Typed", R"({"type":"object","additionalProperties":false,
        "required":["timezone","enabled","count","mode"],"properties":{
        "timezone":{"type":"string"},"enabled":{"type":"boolean"},"count":{"type":"integer"},
        "mode":{"type":"string","enum":["fast","safe"]},"ratio":{"type":"number"},
        "tags":{"type":"array","items":{"type":"string"}}}})", "{}"};
    std::string error;
    const auto grammar = common_build_tool_dsl_grammar({tool}, common_tool_dsl_grammar_form::action, error);
    require(!grammar.empty(), error);
    const std::string good = "open! typed\ntimezone: UTC\nenabled: true\ncount: 2\nmode: fast";
    for (const auto & text : {good, good + "\nratio: -0.25e+2", good + "\ntags: one\ntags: two",
            good + "\ntags: [one, two]", good + "\ntags: one, two"}) {
        require(matches(grammar, text), "typed valid call rejected: " + text);
        common_agent_tool_call call;
        require(common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl, text, {tool}, call, error), error);
        std::string normalized;
        require(common_schema_normalize_and_validate_object(call.arguments_json, tool.parameters, normalized, error), error);
    }
    for (const auto & text : {good + "\ntimezone: UTC", std::string("open! typed"),
            std::string("open! typed\ntimezone: UTC\nenabled: yes\ncount: 2\nmode: fast"),
            std::string("open! typed\ntimezone: UTC\nenabled: true\ncount: 2.5\nmode: fast"),
            std::string("open! typed\ntimezone: UTC\nenabled: true\ncount: 2\nmode: invented")}) {
        require(!matches(grammar, text), "invalid typed call accepted: " + text);
    }
    // The generation order is canonical, but existing stored DSL stays readable.
    common_agent_tool_call call;
    require(common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl,
        "open! typed\nmode: fast\ncount: 2\nenabled: true\ntimezone: UTC", {tool}, call, error), error);
    const auto arguments = common_build_tool_dsl_grammar({tool}, common_tool_dsl_grammar_form::arguments, error);
    require(matches(arguments, good.substr(12)) && !matches(arguments, "timezone: UTC"), "repair lost required inputs");
    const common_chat_tool optional{"optional", "Optional", R"({"type":"object","properties":{"note":{"type":"string"}}})", "{}"};
    const auto empty = common_build_tool_dsl_grammar({optional}, common_tool_dsl_grammar_form::arguments, error);
    require(matches(empty, "") && matches(empty, "note: \"åäö with spaces\""), "optional/Unicode arguments rejected");
    for (const auto & text : {"true", "false", "null", "42", "-2"}) {
        require(!matches(empty, std::string("note: ") + text), "string grammar admitted a non-string literal");
        require(matches(empty, std::string("note: \"") + text + "\""), "quoted string literal rejected");
    }
    const common_chat_tool annotated{"annotated", "Annotated", R"({"type":"object","properties":{
        "conditions":{"type":"string","items":{"type":"object"}}}})", "{}"};
    const auto annotated_grammar = common_build_tool_dsl_grammar({annotated}, common_tool_dsl_grammar_form::action, error);
    require(!annotated_grammar.empty() && matches(annotated_grammar, "open! annotated\nconditions: \"amount > 2\""),
        "non-applicable items annotation changed a string field into an array: " + error);
    const common_chat_tool alternatives{"choose", "Choose", R"({"type":"object","properties":{
        "kind":{"type":"string"}},"oneOf":[
        {"required":["kind"],"properties":{"kind":{"const":"left"}}},
        {"required":["kind"],"properties":{"kind":{"const":"right"}}}]})", "{}"};
    const auto choice = common_build_tool_dsl_grammar({alternatives}, common_tool_dsl_grammar_form::action, error);
    require(!choice.empty() && matches(choice, "open! choose\nkind: left") && !matches(choice, "open! choose\nkind: other"),
        "branch constraints were lost: " + error);
    const common_chat_tool malformed{"bad", "Bad", R"({"type":"object","properties":[]})", "{}"};
    require(common_build_tool_dsl_grammar({malformed}, common_tool_dsl_grammar_form::action, error).empty() && !error.empty(),
        "malformed schema did not fail explicitly");
    require(common_build_tool_dsl_grammar({tool}, common_tool_dsl_grammar_form::action, error) == grammar,
        "grammar is not deterministic");
}

void array_ambiguity_contracts() {
    const common_chat_tool checks{"checks", "Checks", R"({"type":"object","required":["rules"],"properties":{
        "rules":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object","required":["type","column"],
        "properties":{"type":{"const":"numeric_range"},"column":{"type":"string"},
        "min":{"type":"number"},"max":{"type":"number"}},"anyOf":[{"required":["min"]},{"required":["max"]}]}}}})", "{}"};
    std::string error;
    const auto grammar = common_build_tool_dsl_grammar({checks}, common_tool_dsl_grammar_form::action, error);
    require(!grammar.empty(), error);
    std::string text = "open! checks";
    for (size_t i = 0; i < 12; ++i) {
        const auto path = "\nrules[" + std::to_string(i) + "]";
        text += path + ".type: numeric_range" + path + ".column: amount" + path + ".min: 0" + path + ".max: 10";
    }
    require(!matches(grammar, "open! checks\nrules[1].type: numeric_range\nrules[1].column: amount\nrules[1].min: 0"),
        "bounded object list accepted an initial index other than zero");
    require(!matches(grammar, "open! checks\nrules[0].type: numeric_range\nrules[0].column: amount\nrules[0].min: 0\nrules[0].type: numeric_range\nrules[0].column: amount\nrules[0].min: 0"),
        "bounded object list repeated a completed item index");
    require(matches(grammar, text), "overlapping array alternatives rejected a valid bounded list");
}

void projection_contracts() {
    const std::string schema = R"({"type":"object","required":["dataset","column"],"properties":{
        "dataset":{"type":"string"},"column":{"type":"string"}},"allOf":[{"required":["dataset"]}]})";
    std::string error;
    const auto projected = common_project_model_input_schema_without_bound_fields(schema, {"dataset"}, error);
    require(!projected.empty() && error.empty(), error);
    const auto value = nlohmann::ordered_json::parse(projected);
    require(!value["properties"].contains("dataset") && value["required"] == nlohmann::ordered_json::array({"column"}) &&
        value["allOf"][0]["required"].empty(), "host-owned input still required in a branch");
    const common_chat_tool tool{"lookup", "Lookup", projected, "{}"};
    const auto grammar = common_build_tool_dsl_grammar({tool}, common_tool_dsl_grammar_form::action, error);
    require(!grammar.empty() && matches(grammar, "open! lookup\ncolumn: amount") &&
        !matches(grammar, "open! lookup\ndataset: secret\ncolumn: amount"), "host-bound grammar mismatch: " + error);
    const common_chat_tool branch_array{"checks", "Checks", R"({"type":"object","properties":{
        "rules":{"type":"array","items":{"oneOf":[
        {"type":"object","required":["type","column"],"properties":{"type":{"const":"unique"},"column":{"type":"string"}}},
        {"type":"object","required":["type","column"],"properties":{"type":{"const":"not_null"},"column":{"type":"string"}}}]} }},
        "x-agent-examples":[{"rules":[{"type":"unique","column":"id"}]}]})", "{}"};
    const auto branch_rendered = common_render_compact_tool_description("checks", "Checks", branch_array.parameters, "{}", error);
    require(error.empty() && branch_rendered.find("rules[0].column") != std::string::npos &&
        branch_rendered.find("rules[0].type: unique") != std::string::npos,
        "alternative object-array contract lost field paths or example arguments: " + error);
    require(branch_rendered.find("column!: string") != std::string::npos,
        "nested required field was not identified in compact contract");
    nlohmann::ordered_json wide = {{"type", "object"}, {"properties", nlohmann::ordered_json::object()}};
    auto & nested = wide["properties"]["items"];
    nested = {{"type", "array"}, {"items", {{"type", "object"}, {"properties", nlohmann::ordered_json::object()}}}};
    for (size_t index = 0; index < 40; ++index) {
        nested["items"]["properties"]["field_" + std::to_string(index)] = {{"type", "string"}};
    }
    const auto wide_rendered = common_render_compact_tool_description("wide", "Wide", wide.dump(), "{}", error);
    require(error.empty() && wide_rendered.find("field_39?") != std::string::npos,
        "compact contract silently truncated fields after the old 32-path cap");
    const std::string search_schema = R"({"type":"object","required":["dataset","identifier_field"],"properties":{
        "dataset":{"type":"string","x-agent-type":"dataset_ref"},"identifier_field":{"type":"string"}}})";
    const auto scoped = common_project_model_input_schema_enums(search_schema,
        {{"dataset", {"dataset://allowed/events"}}, {"identifier_field", {"user_id", "host_id"}}}, error);
    require(error.empty(), "scoped source projection failed: " + error);
    const common_chat_tool scoped_tool{"collect", "Collect", scoped, "{}"};
    const auto scoped_grammar = common_build_tool_dsl_grammar({scoped_tool}, common_tool_dsl_grammar_form::action, error);
    require(error.empty() && matches(scoped_grammar, "open! collect\ndataset: dataset://allowed/events\nidentifier_field: user_id") &&
        !matches(scoped_grammar, "open! collect\ndataset: $datasets.datasets\nidentifier_field: user_id") &&
        !matches(scoped_grammar, "open! collect\ndataset: dataset://allowed/events\nidentifier_field: id"),
        "scoped grammar accepted an invented dataset or unmapped source field");
    const auto scoped_description = common_render_compact_tool_description("collect", "Collect", scoped, "{}", error);
    require(error.empty() && scoped_description.find("dataset://allowed/events") != std::string::npos &&
        scoped_description.find("user_id|host_id") != std::string::npos,
        "semantic type hid the enum enforced by the argument grammar");
    require(common_project_model_input_schema_enums(scoped, {{"identifier_field", {"unknown"}}}, error).empty() && !error.empty(),
        "host choices broadened an existing registered enum");
    require(common_project_model_input_schema_enums(scoped, {{"missing", {"value"}}}, error).empty() && !error.empty(),
        "unknown host-projected field was silently ignored");
    const auto branch_grammar = common_build_tool_dsl_grammar({branch_array}, common_tool_dsl_grammar_form::action, error);
    require(!branch_grammar.empty() && !matches(branch_grammar, "open! checks\nrules: $value") &&
        matches(branch_grammar, "open! checks\nrules[0].type: unique\nrules[0].column: id"),
        "grammar advertised unsupported whole-array bindings: " + error);
    const auto rendered = common_render_compact_tool_description("lookup", "Lookup", projected, "{}", error);
    require(error.empty() && rendered.find("dataset") == std::string::npos, "compact contract retained host-owned input");
    const auto choice = common_render_tool_choice_description("lookup", rendered, projected, "{}", error);
    require(error.empty() && choice.find("Required inputs: column.") != std::string::npos &&
        choice.find("column (required") == std::string::npos && choice.find("dataset") == std::string::npos,
        "operation choice lost prerequisites or retained the provider's argument rendering");
    require(common_project_model_input_schema_without_bound_fields("not json", {}, error).empty() && !error.empty(),
        "invalid projection silently succeeded");
}

void catalog_contracts() {
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    std::string error;
    require(catalog.bootstrap("memory", bootstrap, error), error);
    for (const auto * profile : {"code-analysis", "investigation", "analysis", "developer-read"}) {
        const auto definitions = catalog.load_profile(profile, error);
        require(error.empty() && !definitions.empty(), "missing profile fixture");
        std::vector<common_chat_tool> tools;
        for (const auto & definition : definitions) tools.push_back({definition.name, definition.description,
            common_tool_model_input_schema(definition), common_tool_model_result_schema(definition)});
        for (const auto & tool : tools) if (tool.name == "investigation.analyze") {
            const auto action_grammar = common_build_tool_dsl_grammar({tool}, common_tool_dsl_grammar_form::action, error);
            require(error.empty() && matches(action_grammar, "open! investigation.analyze\ncase_id: case-1\nanalysis: graph_metrics"), "metrics acquired spurious prerequisites: " + error);
            require(!matches(action_grammar, "open! investigation.analyze\ncase_id: case-1\nanalysis: source_followup"), "followup omitted mandatory sources");
            require(!matches(action_grammar, "open! investigation.analyze\ncase_id: case-1\nanalysis: shortest_path"), "path omitted mandatory endpoints");
            const auto description = common_render_compact_tool_description(tool.name, tool.description, tool.parameters, "{}", error);
            require(error.empty() && description.find("requires source_ids") != std::string::npos && description.find("requires goal_entity_id") != std::string::npos,
                "conditional prerequisites missing from compact contract");
            const auto scoped = common_project_model_input_schema_enums(tool.parameters, {{"source_ids", {"source-1"}}}, error);
            require(error.empty() && nlohmann::ordered_json::parse(scoped)["properties"]["source_ids"]["items"]["enum"] == nlohmann::ordered_json::array({"source-1"}), "source array enum projection failed");
            const auto scoped_grammar = common_build_tool_dsl_grammar({{tool.name, tool.description, scoped, "{}"}}, common_tool_dsl_grammar_form::action, error);
            require(error.empty() && matches(scoped_grammar, "open! investigation.analyze\ncase_id: case-1\nanalysis: source_followup\nsource_ids: source-1") &&
                !matches(scoped_grammar, "open! investigation.analyze\ncase_id: case-1\nanalysis: source_followup\nsource_ids: invented"), "scoped source grammar did not enforce host choices");
        }
        const auto grammar = common_build_tool_dsl_grammar(tools, common_tool_dsl_grammar_form::plan, error);
        require(!grammar.empty(), std::string(profile) + ": " + error);
        require(grammar.size() < 4 * 1024 * 1024, "catalog grammar exceeds its size budget");
        // Compile the real sampler even when the first tool requires inputs.
        require(!matches(grammar, "not a plan"), "catalog grammar accepted arbitrary text");
    }
    require(common_build_tool_dsl_grammar({}, common_tool_dsl_grammar_form::action, error).empty() && !error.empty(),
        "empty catalog silently enabled unconstrained generation");
    const common_chat_tool invalid{"broken", "Broken", "not-json", "{}"};
    require(common_build_tool_dsl_grammar({invalid}, common_tool_dsl_grammar_form::action, error).empty() && !error.empty(),
        "invalid schema silently enabled unconstrained generation");
}
} // namespace

int main() {
    action_contracts();
    plan_contracts();
    nested_contracts();
    typed_contracts();
    array_ambiguity_contracts();
    projection_contracts();
    catalog_contracts();
    return 0;
}
