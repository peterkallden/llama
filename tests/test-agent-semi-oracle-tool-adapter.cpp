#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"

#include <nlohmann/json.hpp>

#define CHECK(x) do { if (!(x)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_tool_catalog catalog;common_tool_bootstrap_result bootstrap;
    CHECK(catalog.bootstrap("semi-oracle",bootstrap,error));
    const auto definitions=catalog.load_profile("semi-oracle",error);
    CHECK(error.empty()&&definitions.size()==2);
    common_native_tool_bindings bindings;
    bindings.semi_oracle=[](const std::string & name,const std::string & input) {
        const auto args=nlohmann::json::parse(input,nullptr,false);
        if(!args.is_object())return common_tool_execution_result::failure("invalid",common_tool_failure_class::validation,false,"bad args");
        if(name=="oracle.ruleset.define"&&args.contains("ruleset_dsl"))return common_tool_execution_result::success("{\"active\":true}");
        if(name=="oracle.action.verify"&&args.contains("action_dsl"))return common_tool_execution_result::success("{\"verdict\":\"SATISFIED\"}");
        return common_tool_execution_result::failure("invalid",common_tool_failure_class::validation,false,"bad operation");
    };
    common_tool_registry registry;common_tool_adapter_result adapters;
    CHECK(common_register_native_tool_adapters(catalog,"semi-oracle",bindings,registry,adapters,error));
    CHECK(adapters.registered.size()==2&&registry.contains("oracle.ruleset.define")&&registry.contains("oracle.action.verify"));
    const auto define=registry.execute({"oracle.ruleset.define","{\"ruleset_dsl\":\"open! oracle.ruleset.define\\nname: TTT\"}"});
    CHECK(define.ok);
    const auto verify=registry.execute({"oracle.action.verify","{\"action_dsl\":\"open! game.place_mark\\nplayer: X\\nrow: 0\\ncolumn: 0\"}"});
    CHECK(verify.ok&&verify.output.find("SATISFIED")!=std::string::npos);
    CHECK(!registry.execute({"oracle.action.verify","{}"}).ok);
    return 0;
}
