#pragma once

#include "memory/memory-store.h"
#include "plan/plan-store.h"
#include "plan/plan-types.h"

#include <functional>
#include <string>
#include <vector>

// Installs the curated, native starter set. Bootstrap content is owned by the
// runtime and is deliberately distinct from model-learned memory.
struct common_agent_bootstrap_config {
    std::string namespace_id;
    std::string session_id;
    std::string project_id;
    int64_t now = 0;

    bool install_procedures = true;
    bool install_blueprints = true;
};

struct common_agent_bootstrap_result {
    std::vector<std::string> installed_memory_ids;
    std::vector<std::string> existing_memory_ids;
    std::vector<std::string> installed_blueprint_ids;
    std::vector<std::string> existing_blueprint_ids;
    std::vector<std::string> installed_workflow_ids;
    std::vector<std::string> existing_workflow_ids;
    struct rejected_item { std::string kind; std::string id; std::string reason; };
    std::vector<rejected_item> rejected_items;
};

struct common_agent_bootstrap_procedure {
    std::string id;
    std::string content;
    std::string summary;
    float importance = 0.8f;
    float confidence = 1.0f;
};

// Blueprint step ids and dependencies are package-local. The installer assigns
// the persisted id, scope, session identity, timestamps, and blueprint kind.
struct common_agent_bootstrap_blueprint {
    std::string id;
    // Optional immutable source revision. The installer defaults this to the
    // enclosing package name and version.
    std::string source_revision;
    // Short, untrusted-model-safe description used only to choose among
    // already installed candidates.
    std::string selection_description;
    std::string purpose;
    std::string goal;
    std::string success_criteria;
    std::vector<common_plan_step> steps;
    std::vector<std::string> required_capabilities;
    std::vector<common_plan_constraint> constraints;
    std::vector<common_plan_assumption> assumptions;
    common_plan_workflow_policy workflow_policy = common_plan_workflow_policy::preferred;
    std::vector<std::string> procedure_refs;
    std::optional<std::string> next_action;
    std::vector<common_plan_workflow_binding> workflow_bindings;
};

struct common_agent_bootstrap_workflow {
    std::string id;
    std::string source_revision;
    common_plan_workflow_definition definition;
};

struct common_agent_bootstrap_package {
    std::string name;
    std::string version;
    std::vector<common_agent_bootstrap_procedure> procedures;
    std::vector<common_agent_bootstrap_blueprint> blueprints;
    std::vector<common_agent_bootstrap_workflow> workflows;
};

// The built-in package is a fallback. Callers may install a file/package using
// the same installer instead of depending on a second hard-coded candidate list.
common_agent_bootstrap_package common_agent_default_bootstrap_package();

using common_agent_bootstrap_embedder = std::function<bool(const std::string &, std::vector<float> &, std::string &)>;

// The supplied embedder must be the same embedder used for normal retrieval.
// Existing records are never modified: upgrading a bootstrap package requires
// a new stable id rather than silently changing established behavior.
bool common_agent_install_default_bootstrap(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const common_agent_bootstrap_config & config,
    common_agent_bootstrap_embedder embed,
    common_agent_bootstrap_result & result,
    std::string & error);

bool common_agent_install_bootstrap_package(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const common_agent_bootstrap_config & config,
    const common_agent_bootstrap_package & package,
    common_agent_bootstrap_embedder embed,
    common_agent_bootstrap_result & result,
    std::string & error);
