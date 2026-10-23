#include "agent/tooling/schema/tool-dsl-grammar.h"
#include "agent/tooling/schema/tool-output-codec.h"
#include "agent/tooling/catalog/tool-catalog.h"
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
            "open! code.symbols\nbackend: true", "open! code.inspect.", "open! private.inspect"}) {
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
            "open! data.transform\nmetadata.origin: fixture"}) {
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
        const auto grammar = common_build_tool_dsl_grammar(tools, common_tool_dsl_grammar_form::plan, error);
        require(!grammar.empty(), std::string(profile) + ": " + error);
        require(matches(grammar, "plan goal=\"Inspect\"\nstep | open! " + tools.front().name), "catalog grammar could not select a declared tool");
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
    catalog_contracts();
    return 0;
}
