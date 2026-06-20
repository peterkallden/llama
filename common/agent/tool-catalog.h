#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Declarative metadata only. A catalog entry never supplies executable code;
// executor_id must be matched by the native runtime's registry.
enum class common_tool_risk_class { local_read, memory_proposal, plan_proposal, network_read };

struct common_tool_definition {
    std::string name;
    uint32_t version = 1;
    std::string description;
    std::string input_schema_json;
    std::string result_schema_json;
    std::string executor_id;
    common_tool_risk_class risk_class = common_tool_risk_class::local_read;
    bool enabled = true;
    bool requires_confirmation = false;
    uint32_t timeout_ms = 1000;
    size_t max_result_bytes = 16384;
    std::string policy_json = "{}";
};

struct common_tool_profile_member {
    std::string tool_name;
    uint32_t tool_version = 1;
    bool enabled = true;
    std::string config_override_json = "{}";
};

struct common_tool_profile {
    std::string id;
    std::string description;
    bool enabled = true;
    std::vector<common_tool_profile_member> members;
};

struct common_tool_bootstrap_result {
    std::vector<std::string> definitions_created;
    std::vector<std::string> definitions_unchanged;
    std::vector<std::string> profiles_created;
    std::vector<std::string> profiles_unchanged;
};

class common_tool_catalog {
public:
    // Adds missing built-ins. Existing definitions and profiles are deliberately
    // left alone: upgrades need an explicit future migration path.
    bool bootstrap(const std::string & profile_id, common_tool_bootstrap_result & result, std::string & error);
    const common_tool_definition * find_definition(const std::string & name, uint32_t version = 1) const;
    const common_tool_profile * find_profile(const std::string & id) const;
    std::vector<common_tool_definition> load_profile(const std::string & id, std::string & error) const;

private:
    std::map<std::string, common_tool_definition> definitions;
    std::map<std::string, common_tool_profile> profiles;
};

const char * common_tool_risk_class_name(common_tool_risk_class value);

