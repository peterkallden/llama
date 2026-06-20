#include "agent/tool-catalog.h"

#include <cassert>

int main() {
    common_tool_catalog catalog;
    common_tool_bootstrap_result first;
    std::string error;
    assert(catalog.bootstrap("memory", first, error));
    assert(!first.definitions_created.empty());
    assert(catalog.find_definition("memory_search"));
    assert(catalog.find_definition("memory_remember")->requires_confirmation);
    assert(catalog.find_definition("memory_remember")->risk_class == common_tool_risk_class::memory_proposal);
    assert(catalog.find_definition("web_fetch")->executor_id == "builtin.web_fetch");

    const auto read = catalog.load_profile("memory-read", error);
    assert(error.empty());
    assert(read.size() == 7);
    for (const auto & definition : read) assert(definition.risk_class == common_tool_risk_class::local_read);

    common_tool_bootstrap_result second;
    assert(catalog.bootstrap("memory", second, error));
    assert(second.definitions_created.empty());
    assert(second.definitions_unchanged.size() >= first.definitions_created.size());
    assert(!catalog.bootstrap("not-a-profile", second, error));
    assert(error == "unknown built-in tool profile: not-a-profile");
    return 0;
}
