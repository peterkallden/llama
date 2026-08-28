#pragma once

#include "agent/tooling/contracts/tool-runtime-contract.h"
#include "runtime/runtime-operation.h"

#include <string>
#include <vector>

struct common_plan_tool_dataflow_contract;
struct common_plan_state;

// Host-side tool execution contract. Planning and reflection contracts live
// in their own headers so tool consumers do not inherit the full runtime.
struct common_agent_tool_repair_context {
    std::string tool_name;
    std::string validation_error;
    std::string arguments_skeleton;
    std::vector<std::string> available_tools;
    std::vector<std::string> candidate_tools;
    std::string normalized_arguments;
    bool normalization_applied = false;
    // Compact semantic contract used to guide a repair without exposing the
    // full runtime schema.  Empty means that no compact contract is available.
    std::string compact_contract;
};

class common_agent_tool_runtime {
public:
    virtual ~common_agent_tool_runtime() = default;
    virtual bool is_read_only(const std::string & tool_name) const = 0;
    virtual bool is_policy_gated(const std::string & tool_name) const = 0;
    virtual bool describe_tool_dataflow(
            const std::string &, common_plan_tool_dataflow_contract &, std::string &) const { return false; }
    virtual bool validate_plan(const common_plan_state &, std::string &) const { return true; }
    virtual bool is_available(const std::string &) const { return true; }
    virtual bool resolve_tool_name(
            const std::string &, std::string &, std::vector<std::string> &) const { return false; }
    virtual bool validate(const common_agent_tool_call & call, std::string & error) const = 0;
    virtual common_agent_tool_repair_context make_repair_context(
            const common_agent_tool_call & call,
            const std::string & validation_error) const {
        return {call.name, validation_error, {}, {}, {}, call.arguments_json, false, {}};
    }
    virtual common_tool_execution_result execute(const common_agent_tool_call & call) const = 0;
    virtual bool supports_async(const common_agent_tool_call &) const { return false; }
    virtual bool begin_async(
            const common_agent_tool_call &,
            common_runtime_operation_ref &,
            std::string & error) const {
        error = "asynchronous tool execution is unavailable";
        return false;
    }
    virtual bool poll_async(
            const common_runtime_operation_ref &,
            bool & ready,
            common_tool_execution_result &,
            std::string & error) const {
        ready = false;
        error = "asynchronous tool execution is unavailable";
        return false;
    }
    virtual bool cancel_async(
            const common_runtime_operation_ref &,
            std::string & error) const {
        error = "asynchronous tool cancellation is unavailable";
        return false;
    }
};
