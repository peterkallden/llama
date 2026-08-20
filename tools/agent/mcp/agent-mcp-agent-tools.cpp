#include "agent-mcp-agent-tools.h"

namespace {

const char * task_schema() {
    return R"({"type":"object","additionalProperties":false,"required":["task"],"properties":{"task":{"type":"string","minLength":1,"maxLength":16384},"thinking_mode":{"type":"string","enum":["reflective","deliberate","research"],"default":"reflective"},"resource_refs":{"type":"array","maxItems":32,"items":{"type":"string","minLength":1,"maxLength":1024}},"max_reflection_rounds":{"type":"integer","minimum":0,"maximum":32},"max_plan_revisions":{"type":"integer","minimum":0,"maximum":32},"max_research_iterations":{"type":"integer","minimum":0,"maximum":32},"max_tool_rounds":{"type":"integer","minimum":0,"maximum":32},"delegation_depth":{"type":"integer","minimum":0,"maximum":16}}})";
}

const char * summary_schema() {
    return R"({"type":"object","additionalProperties":false,"required":["text"],"properties":{"text":{"type":"string","minLength":1,"maxLength":16384},"thinking_mode":{"type":"string","enum":["reflective","deliberate","research"],"default":"reflective"},"delegation_depth":{"type":"integer","minimum":0,"maximum":16}}})";
}

bool register_one(
        agent_mcp_server_tool_registry & registry,
        agent_mcp_agent_tool_options options,
        const char * name,
        const char * description,
        const char * schema,
        std::string & error) {
    return registry.register_tool({
        name,
        description,
        schema,
        true,
        false,
        false,
        false,
        false,
        [name](
                const agent_mcp_json & arguments,
                agent_mcp_server_tool_result & result,
                std::string & handler_error) mutable {
            handler_error = std::string("agent tool requires the inbound MCP adapter: ") + name;
            result.failure_code = "agent.delegation_unbound";
            result.failure_class = "execution";
            result.safe_summary = handler_error;
            return false;
        },
    }, error);
}

} // namespace

bool agent_mcp_is_agent_tool(const std::string & name) {
    return name == "delegate_task" || name == "summarize" || name == "review_plan";
}

bool agent_mcp_register_agent_tools(
        agent_mcp_server_tool_registry & registry,
        agent_mcp_agent_tool_options options,
        std::string & error) {
    if (!register_one(registry, options, "delegate_task",
            "Delegate one bounded task to another agent through the host dispatcher.", task_schema(), error)) return false;
    if (!register_one(registry, options, "summarize",
            "Ask another agent to produce a bounded summary of supplied text.", summary_schema(), error)) return false;
    if (!register_one(registry, options, "review_plan",
            "Ask another agent to review a bounded plan or task description.", task_schema(), error)) return false;
    error.clear();
    return true;
}
