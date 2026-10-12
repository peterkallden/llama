#pragma once

#include "agent/learning/blueprint-selector.h"
#include "agent/agent-inference.h"
#include "agent/runtime/agent-tool-runtime.h"
#include "agent/contracts/agent-request.h"
#include "agent-runtime-tooling.h"
#include "agent/agent-scope.h"
#include "plan/plan-store.h"

#include <optional>
#include <string>
#include <vector>

enum class common_agent_route_kind {
    normal_plan,
    blueprint,
    blueprint_workflow,
};

struct common_agent_route_candidate {
    std::string id;
    common_agent_route_kind kind = common_agent_route_kind::normal_plan;
    std::string blueprint_logical_id;
    std::string blueprint_persisted_id;
    std::optional<common_plan_workflow_binding> workflow;
    std::optional<common_plan_workflow_definition> workflow_definition;
    std::string description;
    std::string blueprint_revision;
    std::string workflow_revision;
    std::string graph_revision;
    std::vector<std::string> required_capabilities;
    std::vector<std::string> resolved_optional_capabilities;
    std::vector<std::string> required_context;
    std::vector<std::string> procedure_refs;
    std::vector<std::string> resolved_tools;
};

struct common_agent_route_rejection {
    std::string blueprint_logical_id;
    std::string workflow_ref;
    std::string reason;
};

struct common_agent_route_catalog {
    std::vector<common_agent_route_candidate> candidates;
    std::vector<common_agent_route_rejection> rejections;
    std::string context_fingerprint;
    std::string catalog_fingerprint;
};

struct common_agent_execution_phase_policy {
    common_agent_execution_phase phase = common_agent_execution_phase::normal;
    bool allowed = false;
    std::vector<std::string> allowed_tools;
    std::vector<std::string> additional_capabilities;
};

// The selected route is turned into one host-owned execution envelope.  The
// envelope is the authority shared by model projection, plan validation and
// actual sync/async tool calls; a phase may narrow or explicitly expand it,
// but cannot replace the route.
struct common_agent_execution_envelope {
    std::string route_id;
    std::string blueprint_ref;
    std::string blueprint_revision;
    std::string workflow_ref;
    std::string workflow_revision;
    std::string graph_revision;
    std::vector<std::string> resolved_capabilities;
    std::vector<std::string> required_context;
    std::vector<std::string> resolved_allowed_tools;
    std::string policy_revision = "route-policy-v1";
    std::string fingerprint;
    std::vector<common_agent_execution_phase_policy> phases;
};

const common_agent_execution_phase_policy * common_agent_execution_phase_policy_for(
        const common_agent_execution_envelope & envelope,
        common_agent_execution_phase phase);

bool common_agent_execution_envelope_allows(
        const common_agent_execution_envelope & envelope,
        common_agent_execution_phase phase,
        const std::string & tool_name);

bool build_agent_execution_envelope(
        const common_agent_route_candidate & route,
        const common_agent_runtime_tooling & tooling,
        common_agent_execution_envelope & envelope,
        std::string & error);

// Resolve the workflow's semantic requirements against the current host view.
// Legacy allowed_tools are accepted as a migration fallback or an additional
// restriction, but never grant a tool absent from the active host catalog.
bool common_agent_resolve_workflow_tools(
        const common_plan_workflow_definition & workflow,
        const common_agent_runtime_tooling & tooling,
        std::vector<std::string> & resolved,
        std::string & reason);

bool common_agent_resolve_workflow_tools(
        const common_plan_workflow_definition & workflow,
        const common_agent_runtime_tooling & tooling,
        std::vector<std::string> & resolved,
        std::vector<std::string> & resolved_optional_capabilities,
        std::string & reason);

const char * common_agent_route_kind_name(common_agent_route_kind kind);

bool compile_agent_route_catalog(
        const common_agent_request & request,
        common_plan_store & plan_store,
        const common_agent_scope & scope,
        const std::vector<common_blueprint_candidate> & blueprints,
        const common_agent_runtime_tooling & tooling,
        const std::string & explicit_blueprint,
        common_agent_route_catalog & catalog,
        std::string & error);
