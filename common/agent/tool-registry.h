#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

struct common_registered_tool {
    std::string name;
    uint32_t version = 1;
    std::string executor_id = "native.unnamed";
    std::string arguments_schema;
    bool read_only = true;
    // A policy-gated proposal may write only through a native policy callback.
    // It is not a general write capability.
    bool policy_gated = false;
    std::function<bool(const std::string & arguments_json, std::string & result, std::string & error)> handler;
};

struct common_registered_tool_call { std::string name; std::string arguments_json = "{}"; };

class common_tool_registry {
public:
    bool register_tool(common_registered_tool tool, std::string & error);
    // Validates registration and the object-shaped JSON contract without
    // invoking the handler. Runtime callers use this to distinguish a bad
    // plan/tool contract from an ordinary handler failure.
    bool validate(const common_registered_tool_call & call, std::string & error) const;
    bool execute(const common_registered_tool_call & call, std::string & result, std::string & error) const;
    bool contains(const std::string & name) const;
    bool matches_binding(const std::string & name, uint32_t version, const std::string & executor_id) const;
    bool is_read_only(const std::string & name) const;
    bool is_policy_gated(const std::string & name) const;
private:
    std::unordered_map<std::string, common_registered_tool> tools;
};
