#pragma once

#include "agent/agent-runtime.h"
#include "agent/tooling/registry/tool-registry.h"

class test_tool_runtime_registry_adapter : public common_agent_tool_runtime {
public:
    explicit test_tool_runtime_registry_adapter(const common_tool_registry & registry)
        : registry(registry) {}

    bool is_read_only(const std::string & tool_name) const override {
        return registry.is_read_only(tool_name);
    }

    bool is_policy_gated(const std::string & tool_name) const override {
        return registry.is_policy_gated(tool_name);
    }

    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        return registry.validate(call, error);
    }

    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        return registry.execute(call);
    }

private:
    const common_tool_registry & registry;
};
