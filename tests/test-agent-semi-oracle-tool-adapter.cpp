#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"
#include "agent/tooling/contracts/schema-contract.h"

#include <nlohmann/json.hpp>
#include <cstdio>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); return 1; } } while (false)

int main() {
    std::string error;
    common_tool_catalog catalog;common_tool_bootstrap_result bootstrap;
    CHECK(catalog.bootstrap("semi-oracle",bootstrap,error));
    const auto definitions=catalog.load_profile("semi-oracle",error);
    CHECK(error.empty()&&definitions.size()==4);
    const auto * goal_definition = catalog.find_definition("oracle.goal.search");
    CHECK(goal_definition != nullptr);
    std::string normalized;
    CHECK(common_schema_normalize_and_validate_object(
        R"({"verdict":"UNKNOWN","ruleset_id":"test-rule","revision":1,"evaluator":"semi-oracle://evaluator/linear_algebra","evaluator_revision":"linear_astar_v1","level":"bounded","proof":null,"counterexample":null,"search_policy":{"algorithm":"exact_astar_v1","suboptimality_bound":1.0,"optimality_proven":false,"suboptimality_bound_proven":false},"reason":"budget exhausted","guidance":{"mode":"shadow","decision_count":2}})",
        goal_definition->result_schema_json, normalized, error));
    common_native_tool_bindings bindings;
    bindings.semi_oracle=[](const std::string & name,const std::string & input) {
        const auto args=nlohmann::json::parse(input,nullptr,false);
        if(!args.is_object())return common_tool_execution_result::failure("invalid",common_tool_failure_class::validation,false,"bad args");
        if(name=="oracle.ruleset.define"&&args.contains("ruleset_dsl"))return common_tool_execution_result::success("{\"active\":true}");
        if(name=="oracle.action.verify"&&args.contains("action_dsl"))return common_tool_execution_result::success("{\"verdict\":\"SATISFIED\"}");
        if(name=="oracle.goal.search"&&args.empty())return common_tool_execution_result::success("{\"verdict\":\"SATISFIED\",\"algorithm\":\"exact_astar_v1\"}");
        return common_tool_execution_result::failure("invalid",common_tool_failure_class::validation,false,"bad operation");
    };
    common_tool_registry registry;common_tool_adapter_result adapters;
    CHECK(common_register_native_tool_adapters(catalog,"semi-oracle",bindings,registry,adapters,error));
    CHECK(adapters.registered.size()==4&&registry.contains("oracle.ruleset.define")&&registry.contains("oracle.action.verify")&&registry.contains("oracle.goal.search")&&registry.contains("oracle.problem.define"));
    CHECK(!registry.is_read_only("oracle.ruleset.define")&&!registry.is_read_only("oracle.problem.define")&&
        !registry.is_read_only("oracle.action.verify")&&registry.is_read_only("oracle.goal.search"));
    const auto define=registry.execute({"oracle.ruleset.define","{\"ruleset_dsl\":\"open! oracle.ruleset.define\\nname: TTT\"}"});
    CHECK(define.ok);
    const auto verify=registry.execute({"oracle.action.verify","{\"action_dsl\":\"open! game.place_mark\\nplayer: X\\nrow: 0\\ncolumn: 0\"}"});
    CHECK(verify.ok&&verify.output.find("SATISFIED")!=std::string::npos);
    const auto search=registry.execute({"oracle.goal.search","{}"});
    CHECK(search.ok&&search.output.find("exact_astar_v1")!=std::string::npos);
    CHECK(!registry.execute({"oracle.action.verify","{}"}).ok);
    return 0;
}
