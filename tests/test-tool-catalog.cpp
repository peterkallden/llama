#include "agent/tool-catalog.h"

#include <cassert>
#include <string>

int main() {
    common_tool_catalog catalog;
    common_tool_bootstrap_result first;
    std::string error;
    assert(catalog.bootstrap("memory", first, error));
    assert(!first.definitions_created.empty());
    assert(catalog.find_definition("memory_search"));
    assert(catalog.find_definition("memory_remember")->requires_confirmation);
    assert(catalog.find_definition("memory_remember")->risk_class == common_tool_risk_class::memory_proposal);
    assert(catalog.find_definition("web_search")->executor_id == "builtin.web_search");
    assert(catalog.find_definition("web_fetch")->executor_id == "builtin.web_fetch");

    const auto * build = catalog.find_definition("development.build");
    const auto * test = catalog.find_definition("development.test");
    assert(build && test);
    assert(build->executor_id == "sandbox.development.build");
    assert(test->executor_id == "sandbox.development.test");
    assert(build->risk_class == common_tool_risk_class::sandbox_execution);
    assert(test->risk_class == common_tool_risk_class::sandbox_execution);
    assert(build->requires_confirmation && test->requires_confirmation);
    assert(build->input_schema_json.find("required\":[\"target\"]") != std::string::npos);
    assert(test->input_schema_json.find("required\":[\"target\"]") != std::string::npos);
    assert(build->policy_json.find("execution_class\":\"developer-build\"") != std::string::npos);
    assert(test->policy_json.find("filesystem\":\"workspace-write\"") != std::string::npos);

    const auto read = catalog.load_profile("memory-read", error);
    assert(error.empty());
    assert(read.size() == 8);
    for (const auto & definition : read) assert(definition.risk_class == common_tool_risk_class::local_read);

    common_tool_bootstrap_result second;
    assert(catalog.bootstrap("memory", second, error));
    assert(second.definitions_created.empty());
    assert(second.definitions_unchanged.size() >= first.definitions_created.size());
    assert(!catalog.bootstrap("not-a-profile", second, error));
    assert(error == "tool profile is unavailable: not-a-profile");
    return 0;
}
