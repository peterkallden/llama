#pragma once

#include "agent/learning/blueprint-selector.h"
#include "memory/memory-types.h"
#include <nlohmann/json.hpp>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Definition ownership is independent of the caller's execution scope.
struct common_agent_template_library_config {
    std::string id;
    std::string package;
    std::string level = "host";
    std::string namespace_id;
    std::string project_id;
};

struct common_agent_template_library_manifest {
    int schema_version = 1;
    std::string library_id;
    std::string level;
    std::string namespace_id;
    std::string project_id;
    std::string package_name;
    std::string package_version;
    std::string fingerprint;
    std::string definition_prefix;
    std::string resource_uri;
    std::map<std::string, std::string> blueprints;
    std::set<std::string> workflows;
    std::set<std::string> procedures;
    bool current = true;
};

bool common_agent_template_library_config_parse(const nlohmann::ordered_json & value,
    std::vector<common_agent_template_library_config> & out, std::string & error);
nlohmann::ordered_json common_agent_template_library_config_json(
    const std::vector<common_agent_template_library_config> & libraries);
bool common_agent_template_library_manifest_parse(const std::string & text,
    common_agent_template_library_manifest & out, std::string & error);
std::string common_agent_template_library_manifest_json(const common_agent_template_library_manifest & manifest);
bool common_agent_template_library_available(const common_agent_template_library_manifest & manifest,
    const std::string & namespace_id, const std::string & project_id);

// Constructed by the host from verified import receipts, never model input.
struct common_agent_template_library_snapshot {
    std::vector<common_agent_template_library_manifest> revisions;
    std::map<std::string, common_plan_state> definitions;
    std::map<std::string, common_memory_record> procedures;
    const common_agent_template_library_manifest * find(const std::string & persisted_id) const;
    bool permits(const std::string & persisted_id, const std::string & namespace_id,
        const std::string & project_id) const;
    bool same_revision(const std::string & left, const std::string & right) const;
};

std::optional<common_plan_state> common_agent_template_definition(common_plan_store & store,
    const std::string & id, const common_agent_template_library_snapshot * libraries, std::string & error);
bool common_agent_template_definition_allowed(const common_plan_state & plan, common_plan_scope scope,
    const std::string & namespace_id, const std::string & session_id, const std::string & project_id,
    const std::string & turn_id, const common_agent_template_library_snapshot * libraries);

// Validate a persisted task against the currently enabled, exact library origin.
bool common_agent_template_task_allowed(const common_plan_state & task,
    const std::string & namespace_id, const std::string & project_id,
    const common_agent_template_library_snapshot * libraries, std::string & error);

common_blueprint_candidate common_agent_template_candidate(const std::string & logical_id, const common_plan_state & plan);
