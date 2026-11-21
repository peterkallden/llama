#include "agent/tooling/registry/tool-registry.h"

#include <cassert>

int main() {
    common_tool_registry registry;
    std::string error;
    common_registered_tool tool;
    tool.name = "memory.lookup";
    tool.version = 7;
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string","minLength":3,"maxLength":8},"limit":{"type":"integer","minimum":1,"maximum":4},"kind":{"type":"string","enum":["fact","procedure"]}}})";
    tool.handler = [](const std::string &) {
        return common_tool_execution_result::success("safe result");
    };
    assert(registry.register_tool(std::move(tool), error));
    assert(registry.matches_binding("memory.lookup", 7, "test.lookup"));
    assert(!registry.matches_binding("memory.lookup", 1, "test.lookup"));
    assert(!registry.contains("lookup"));

    auto result = registry.execute({"memory.lookup", R"({"id":"memory:1","limit":4,"kind":"fact"})"});
    assert(result.ok);
    assert(result.output == "safe result");

    std::string normalized;
    assert(registry.normalize_and_validate(
        {"memory.lookup", R"({"id":"memory:1","limit":4,"kind":"fact"})"},
        normalized, error));
    assert(normalized == R"({"id":"memory:1","limit":4,"kind":"fact"})");

    result = registry.execute({"shell", "{}"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);

    result = registry.execute({"lookup", R"({"id":"memory:1","limit":4,"kind":"fact"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    result = registry.execute({"memory.lookup", R"({"extra":1})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    result = registry.execute({"memory.lookup", R"({"id":7})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    result = registry.execute({"memory.lookup", R"({"id":"x"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    result = registry.execute({"memory.lookup", R"({"id":"memory:1","limit":5})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    result = registry.execute({"memory.lookup", R"({"id":"memory:1","kind":"unknown"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);
    assert(!registry.normalize_and_validate(
        {"memory.lookup", R"({"id":"memory:1","limit":5})"}, normalized, error));

    return 0;
}
