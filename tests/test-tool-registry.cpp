#include "agent/tooling/registry/tool-registry.h"

#include <cassert>

int main() {
    common_tool_registry registry;
    std::string error;
    common_registered_tool tool;
    tool.name = "lookup";
    tool.version = 7;
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string","minLength":3,"maxLength":8},"limit":{"type":"integer","minimum":1,"maximum":4},"kind":{"type":"string","enum":["fact","procedure"]}}})";
    tool.handler = [](const std::string &) {
        return common_tool_execution_result::success("safe result");
    };
    assert(registry.register_tool(std::move(tool), error));
    assert(registry.matches_binding("lookup", 7, "test.lookup"));
    assert(!registry.matches_binding("lookup", 1, "test.lookup"));

    auto result = registry.execute({"lookup", R"({"id":"memory:1","limit":4,"kind":"fact"})"});
    assert(result.ok);
    assert(result.output == "safe result");

    result = registry.execute({"shell", "{}"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);

    result = registry.execute({"lookup", R"({"extra":1})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    result = registry.execute({"lookup", R"({"id":7})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    result = registry.execute({"lookup", R"({"id":"x"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    result = registry.execute({"lookup", R"({"id":"memory:1","limit":5})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    result = registry.execute({"lookup", R"({"id":"memory:1","kind":"unknown"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    return 0;
}
