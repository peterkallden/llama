#include "agent-route-compiler.h"

#include <algorithm>
#include <functional>
#include <sstream>

namespace {

bool has_tool(const common_plan_workflow_definition & workflow, const std::string & name) {
    return std::find(workflow.allowed_tools.begin(), workflow.allowed_tools.end(), name) !=
        workflow.allowed_tools.end();
}

bool has_tool_name(const common_agent_runtime_tooling & tooling, const std::string & name) {
    return std::any_of(tooling.tools.begin(), tooling.tools.end(), [&](const auto & tool) {
        return tool.name == name;
    });
}

// Built-in CLI profiles currently resolve concrete tools, while configured
// hosts may additionally publish semantic capability ids.  Keep route
// eligibility deterministic in both cases: a small set of stable built-in
// capability ids can be derived from the concrete host tool view, but the
// blueprint can never add tools or capabilities to that view.
bool has_host_capability(
        const common_agent_runtime_tooling & tooling,
        const std::string & capability) {
    if (std::find(tooling.capabilities.begin(), tooling.capabilities.end(), capability) !=
            tooling.capabilities.end()) {
        return true;
    }
    if (capability == "tool.dataset") {
        return has_tool_name(tooling, "dataset.list") ||
            has_tool_name(tooling, "dataset.select") ||
            has_tool_name(tooling, "dataset.inspect") ||
            has_tool_name(tooling, "dataset.schema") ||
            has_tool_name(tooling, "dataset.sample") ||
            has_tool_name(tooling, "data.query") ||
            has_tool_name(tooling, "data.filter") ||
            has_tool_name(tooling, "data.aggregate") ||
            has_tool_name(tooling, "statistics.describe");
    }
    if (capability == "tool.repository") {
        return has_tool_name(tooling, "repository.list") ||
            has_tool_name(tooling, "repository.search") ||
            has_tool_name(tooling, "repository.read");
    }
    if (capability == "tool.workspace") {
        return has_tool_name(tooling, "workspace.list") ||
            has_tool_name(tooling, "workspace.search") ||
            has_tool_name(tooling, "workspace.read");
    }
    if (capability == "tool.openapi") {
        const auto binding = tooling.capability_tools.find("openapi.read");
        return binding != tooling.capability_tools.end() && !binding->second.empty();
    }
    return false;
}

bool workflow_capability_matches(
        const std::string & capability,
        const common_plan_workflow_definition & workflow) {
    constexpr const char * prefix = "workflow.";
    constexpr size_t prefix_size = 9;
    if (capability.size() <= prefix_size ||
            capability.compare(0, prefix_size, prefix) != 0) return false;
    return workflow.family == capability.substr(prefix_size);
}

bool blueprint_capabilities_match(
        const common_plan_state & blueprint,
        const common_agent_runtime_tooling & tooling,
        const common_plan_workflow_definition * workflow = nullptr) {
    return std::all_of(blueprint.required_capabilities.begin(),
        blueprint.required_capabilities.end(), [&](const auto & capability) {
            if (capability.rfind("workflow.", 0) == 0) {
                return workflow != nullptr && workflow_capability_matches(capability, *workflow);
            }
            return has_host_capability(tooling, capability);
        });
}

std::string route_id(const common_agent_route_candidate & candidate) {
    if (candidate.kind == common_agent_route_kind::normal_plan) return "normal-plan";
    std::string result = "blueprint:" + candidate.blueprint_logical_id;
    if (candidate.workflow) result += ":workflow:" + candidate.workflow->workflow_ref + "@" + candidate.workflow->workflow_revision;
    return result;
}

void append_rejection(common_agent_route_catalog & catalog,
        const common_blueprint_candidate & blueprint,
        const std::string & workflow_ref,
        const std::string & reason) {
    catalog.rejections.push_back({blueprint.logical_id, workflow_ref, reason});
}

std::string fingerprint(const std::string & value) {
    // The catalog is diagnostic state, not a security boundary. A stable
    // bounded hash avoids placing the full request or catalog in every trace.
    return std::to_string(std::hash<std::string>{}(value));
}

void append_unique(std::vector<std::string> & values, const std::string & value) {
    if (!value.empty() && std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

bool has_context_fact(
        const common_agent_request & request,
        const common_agent_runtime_tooling & tooling,
        const std::string & requirement) {
    if (std::find(tooling.available_context.begin(), tooling.available_context.end(), requirement) !=
            tooling.available_context.end()) return true;
    if (requirement == "context.resource.available") {
        return std::any_of(request.input_resources.begin(), request.input_resources.end(),
            [](const auto & resource) { return !resource.resource.uri.empty(); });
    }
    return false;
}

bool resolve_workflow_tools(
        const common_plan_workflow_definition & workflow,
        const common_agent_runtime_tooling & tooling,
        std::vector<std::string> & resolved,
        std::vector<std::string> & resolved_optional_capabilities,
        std::string & reason) {
    resolved.clear();
    resolved_optional_capabilities.clear();
    reason.clear();
    const auto host_has_tool = [&](const std::string & name) {
        return has_tool_name(tooling, name);
    };
    if (workflow.required_capabilities.empty()) {
        // v1 compatibility: concrete tool lists remain accepted, but are
        // filtered by the active host view and never grant unavailable tools.
        for (const auto & name : workflow.allowed_tools) {
            if (host_has_tool(name)) append_unique(resolved, name);
        }
        if (resolved.empty() && workflow.optional_capabilities.empty()) reason = "workflow resolves no registered host tools";
    }

    for (const auto & capability : workflow.required_capabilities) {
        const auto binding = tooling.capability_tools.find(capability);
        if (binding == tooling.capability_tools.end()) {
            reason = "required workflow capability is unavailable: " + capability;
            resolved.clear();
            return false;
        }
        bool resolved_capability = false;
        for (const auto & name : binding->second) {
            if (host_has_tool(name)) {
                append_unique(resolved, name);
                resolved_capability = true;
            }
        }
        if (!resolved_capability) {
            reason = "workflow capability has no tool in the active host view: " + capability;
            resolved.clear();
            return false;
        }
    }

    for (const auto & capability : workflow.optional_capabilities) {
        const auto binding = tooling.capability_tools.find(capability);
        if (binding == tooling.capability_tools.end()) continue;
        bool resolved_capability = false;
        for (const auto & name : binding->second) {
            if (host_has_tool(name)) {
                append_unique(resolved, name);
                resolved_capability = true;
            }
        }
        if (resolved_capability) append_unique(resolved_optional_capabilities, capability);
    }

    // During migration, an explicit tool list can only narrow the resolved
    // semantic capabilities; it can never widen them.
    if (!workflow.allowed_tools.empty()) {
        resolved.erase(std::remove_if(resolved.begin(), resolved.end(), [&](const auto & name) {
            return !has_tool(workflow, name);
        }), resolved.end());
        resolved_optional_capabilities.erase(std::remove_if(
            resolved_optional_capabilities.begin(), resolved_optional_capabilities.end(),
            [&](const auto & capability) {
                const auto binding = tooling.capability_tools.find(capability);
                return binding == tooling.capability_tools.end() ||
                    std::none_of(binding->second.begin(), binding->second.end(), [&](const auto & name) {
                        return std::find(resolved.begin(), resolved.end(), name) != resolved.end();
                    });
            }), resolved_optional_capabilities.end());
        if (resolved.empty()) reason = "legacy allowed_tools excludes every resolved capability tool";
    }
    return !resolved.empty();
}

common_agent_execution_phase_policy make_phase(
        common_agent_execution_phase phase,
        bool allowed,
        const std::vector<std::string> & tools) {
    common_agent_execution_phase_policy result;
    result.phase = phase;
    result.allowed = allowed;
    result.allowed_tools = tools;
    return result;
}

} // namespace

bool common_agent_resolve_workflow_tools(
        const common_plan_workflow_definition & workflow,
        const common_agent_runtime_tooling & tooling,
        std::vector<std::string> & resolved,
        std::string & reason) {
    std::vector<std::string> ignored;
    return resolve_workflow_tools(workflow, tooling, resolved, ignored, reason);
}

bool common_agent_resolve_workflow_tools(
        const common_plan_workflow_definition & workflow,
        const common_agent_runtime_tooling & tooling,
        std::vector<std::string> & resolved,
        std::vector<std::string> & resolved_optional_capabilities,
        std::string & reason) {
    return resolve_workflow_tools(
        workflow, tooling, resolved, resolved_optional_capabilities, reason);
}

const common_agent_execution_phase_policy * common_agent_execution_phase_policy_for(
        const common_agent_execution_envelope & envelope,
        common_agent_execution_phase phase) {
    for (const auto & policy : envelope.phases) {
        if (policy.phase == phase) return &policy;
    }
    return nullptr;
}

bool common_agent_execution_envelope_allows(
        const common_agent_execution_envelope & envelope,
        common_agent_execution_phase phase,
        const std::string & tool_name) {
    const auto * policy = common_agent_execution_phase_policy_for(envelope, phase);
    return policy != nullptr && policy->allowed &&
        std::find(policy->allowed_tools.begin(), policy->allowed_tools.end(), tool_name) !=
            policy->allowed_tools.end();
}

bool build_agent_execution_envelope(
        const common_agent_route_candidate & route,
        const common_agent_runtime_tooling & tooling,
        common_agent_execution_envelope & envelope,
        std::string & error) {
    envelope = {};
    error.clear();
    envelope.route_id = route.id;
    envelope.blueprint_ref = route.blueprint_logical_id;
    envelope.blueprint_revision = route.blueprint_revision;
    envelope.graph_revision = route.graph_revision;
    envelope.resolved_capabilities = route.required_capabilities;
    for (const auto & capability : route.resolved_optional_capabilities) {
        append_unique(envelope.resolved_capabilities, capability);
    }
    envelope.required_context = route.required_context;
    envelope.policy_revision = "route-policy-v1";
    if (route.workflow) {
        envelope.workflow_ref = route.workflow->workflow_ref;
        envelope.workflow_revision = route.workflow->workflow_revision;
    }

    for (const auto & tool : tooling.tools) {
        const bool route_allows = route.kind != common_agent_route_kind::blueprint_workflow ||
            std::find(route.resolved_tools.begin(), route.resolved_tools.end(), tool.name) !=
                route.resolved_tools.end();
        if (route_allows) append_unique(envelope.resolved_allowed_tools, tool.name);
    }
    if (route.kind == common_agent_route_kind::blueprint_workflow &&
            envelope.resolved_allowed_tools.empty()) {
        error = "selected workflow resolves no host tools";
        return false;
    }

    const auto base = envelope.resolved_allowed_tools;
    envelope.phases.push_back(make_phase(common_agent_execution_phase::normal, true, base));
    envelope.phases.push_back(make_phase(common_agent_execution_phase::deliberate, true, base));
    envelope.phases.push_back(make_phase(common_agent_execution_phase::reflection, true, base));

    // Research expansion is host-owned and bounded.  A workflow never gets
    // the entire provider catalog merely because the model escalated to
    // research; only these explicitly named evidence tools are added when
    // the current provider actually exposes them.
    auto research = base;
    if (route.kind == common_agent_route_kind::blueprint_workflow) {
        for (const auto & tool : tooling.tools) {
            if (tool.name == "resource.read" || tool.name == "resource_read" ||
                    tool.name == "web.search" || tool.name == "web_search" ||
                    tool.name == "web.fetch" || tool.name == "web_fetch" ||
                    tool.name == "repository.search") {
                append_unique(research, tool.name);
            }
        }
    } else {
        research = base;
    }
    auto research_phase = make_phase(
        common_agent_execution_phase::research,
        !research.empty(), research);
    if (route.kind == common_agent_route_kind::blueprint_workflow) {
        for (const auto & tool : research) {
            if (std::find(base.begin(), base.end(), tool) == base.end()) {
                research_phase.additional_capabilities.push_back(tool);
            }
        }
    }
    envelope.phases.push_back(std::move(research_phase));

    std::ostringstream identity;
    identity << envelope.route_id << "|" << envelope.blueprint_ref << "|"
        << envelope.blueprint_revision << "|" << envelope.workflow_ref << "|"
        << envelope.workflow_revision << "|" << envelope.graph_revision << "|"
        << envelope.policy_revision;
    for (const auto & capability : envelope.resolved_capabilities) identity << "|cap=" << capability;
    for (const auto & requirement : envelope.required_context) identity << "|context=" << requirement;
    for (const auto & tool : envelope.resolved_allowed_tools) identity << "|tool=" << tool;
    envelope.fingerprint = fingerprint(identity.str());
    return true;
}

const char * common_agent_route_kind_name(common_agent_route_kind kind) {
    switch (kind) {
        case common_agent_route_kind::normal_plan: return "normal_plan";
        case common_agent_route_kind::blueprint: return "blueprint";
        case common_agent_route_kind::blueprint_workflow: return "blueprint_workflow";
    }
    return "normal_plan";
}

bool compile_agent_route_catalog(
        const common_agent_request & request,
        common_plan_store & plan_store,
        const common_agent_scope & scope,
        const std::vector<common_blueprint_candidate> & blueprints,
        const common_agent_runtime_tooling & tooling,
        const std::string & explicit_blueprint,
        common_agent_route_catalog & catalog,
        std::string & error) {
    catalog = {};
    error.clear();

    std::ostringstream context;
    context << request.prompt << "|" << scope.namespace_id << "|" << scope.session_id << "|"
            << scope.project_id << "|datasets=" << tooling.available_datasets.size()
            << "|resources=" << request.input_resources.size();
    for (const auto & dataset : tooling.available_datasets) context << "|" << dataset.ref.uri;
    for (const auto & resource : request.input_resources) {
        context << "|resource=" << resource.resource.uri << ":" << resource.resource.mime_type;
    }
    for (const auto & capability : tooling.capabilities) context << "|cap=" << capability;
    for (const auto & fact : tooling.available_context) context << "|context=" << fact;
    catalog.context_fingerprint = fingerprint(context.str());

    common_agent_route_candidate normal;
    normal.id = "normal-plan";
    normal.description = "General task planning with lazy host tool acquisition.";
    catalog.candidates.push_back(std::move(normal));

    for (const auto & candidate : blueprints) {
        if (!explicit_blueprint.empty() && candidate.logical_id != explicit_blueprint) continue;
        std::string store_error;
        const auto blueprint = plan_store.get(candidate.persisted_id, store_error);
        if (!store_error.empty()) { error = store_error; return false; }
        if (!blueprint || blueprint->kind != common_plan_kind::blueprint) {
            append_rejection(catalog, candidate, {}, "persisted blueprint is unavailable");
            continue;
        }
        if (!common_plan_template_scope_matches(*blueprint, scope.plan_scope, scope.namespace_id,
                scope.session_id, scope.project_id, scope.turn_id)) {
            append_rejection(catalog, candidate, {}, "blueprint is outside the current scope");
            continue;
        }
        if (!candidate.source_revision.empty() && candidate.source_revision != blueprint->source_revision) {
            append_rejection(catalog, candidate, {}, "blueprint source revision is stale");
            continue;
        }
        if (tooling.profile_tools_active && !std::all_of(
                blueprint->required_capabilities.begin(),
                blueprint->required_capabilities.end(), [&](const auto & capability) {
                    return capability.rfind("workflow.", 0) == 0 ||
                        has_host_capability(tooling, capability);
                })) {
            append_rejection(catalog, candidate, {}, "required host capability is unavailable");
            continue;
        }

        std::vector<common_agent_route_candidate> workflow_routes;
        std::string list_error;
        const auto plans = plan_store.list(list_error);
        if (!list_error.empty()) { error = list_error; return false; }
        for (const auto & binding : blueprint->workflow_bindings) {
            const common_plan_state * workflow_plan = nullptr;
            for (const auto & plan : plans) {
                if (plan.kind != common_plan_kind::workflow || !plan.workflow_definition ||
                        !common_plan_template_scope_matches(plan, scope.plan_scope, scope.namespace_id,
                            scope.session_id, scope.project_id, scope.turn_id)) continue;
                const auto & definition = *plan.workflow_definition;
                if (definition.workflow_ref == binding.workflow_ref &&
                        definition.workflow_revision == binding.workflow_revision) {
                    workflow_plan = &plan;
                    break;
                }
            }
            if (!workflow_plan) {
                append_rejection(catalog, candidate,
                    binding.workflow_ref + "@" + binding.workflow_revision,
                    "workflow is unavailable in the current scope");
                continue;
            }
            const auto & definition = *workflow_plan->workflow_definition;
            const auto missing_context = std::find_if(
                definition.required_context.begin(), definition.required_context.end(),
                [&](const auto & requirement) {
                    return !has_context_fact(request, tooling, requirement);
                });
            if (missing_context != definition.required_context.end()) {
                append_rejection(catalog, candidate,
                    binding.workflow_ref + "@" + binding.workflow_revision,
                    "required workflow context is unavailable: " + *missing_context);
                continue;
            }
            std::vector<std::string> resolved_tools;
            std::vector<std::string> resolved_optional_capabilities;
            std::string resolution_reason;
            if (!common_agent_resolve_workflow_tools(
                    definition, tooling, resolved_tools, resolved_optional_capabilities,
                    resolution_reason)) {
                append_rejection(catalog, candidate,
                    binding.workflow_ref + "@" + binding.workflow_revision,
                    std::move(resolution_reason));
                continue;
            }
            const auto missing_capability = std::find_if(
                definition.required_capabilities.begin(), definition.required_capabilities.end(),
                [&](const auto & capability) {
                    return std::find(tooling.capabilities.begin(), tooling.capabilities.end(),
                        capability) == tooling.capabilities.end();
                });
            if (missing_capability != definition.required_capabilities.end()) {
                append_rejection(catalog, candidate,
                    binding.workflow_ref + "@" + binding.workflow_revision,
                    "required workflow capability is unavailable: " + *missing_capability);
                continue;
            }
            if (!blueprint_capabilities_match(*blueprint, tooling, &definition)) {
                append_rejection(catalog, candidate,
                    binding.workflow_ref + "@" + binding.workflow_revision,
                    "workflow does not satisfy required blueprint capabilities");
                continue;
            }
            common_agent_route_candidate route;
            route.kind = common_agent_route_kind::blueprint_workflow;
            route.blueprint_logical_id = candidate.logical_id;
            route.blueprint_persisted_id = candidate.persisted_id;
            route.workflow = binding;
            route.workflow_definition = definition;
            route.description = candidate.description.empty() ? blueprint->goal : candidate.description;
            route.blueprint_revision = blueprint->source_revision;
            route.workflow_revision = definition.workflow_revision;
            route.graph_revision = definition.graph_revision;
            route.required_capabilities = blueprint->required_capabilities;
            for (const auto & capability : definition.required_capabilities) {
                append_unique(route.required_capabilities, capability);
            }
            route.required_context = definition.required_context;
            route.resolved_optional_capabilities = std::move(resolved_optional_capabilities);
            route.procedure_refs = blueprint->procedure_refs;
            route.resolved_tools = std::move(resolved_tools);
            route.id = route_id(route);
            workflow_routes.push_back(std::move(route));
        }

        if (!workflow_routes.empty()) {
            if (blueprint->workflow_policy == common_plan_workflow_policy::preferred ||
                    blueprint->workflow_policy == common_plan_workflow_policy::required) {
                for (auto & route : workflow_routes) catalog.candidates.push_back(std::move(route));
            }
            continue;
        }
        if (blueprint->workflow_policy == common_plan_workflow_policy::required) {
            append_rejection(catalog, candidate, {}, "required workflow is not materializable");
            continue;
        }
        common_agent_route_candidate route;
        route.kind = common_agent_route_kind::blueprint;
        route.id = "blueprint:" + candidate.logical_id;
        route.blueprint_logical_id = candidate.logical_id;
        route.blueprint_persisted_id = candidate.persisted_id;
        route.description = candidate.description.empty() ? blueprint->goal : candidate.description;
        route.blueprint_revision = blueprint->source_revision;
        route.required_capabilities = blueprint->required_capabilities;
        route.procedure_refs = blueprint->procedure_refs;
        catalog.candidates.push_back(std::move(route));
    }

    std::ostringstream catalog_text;
    for (const auto & candidate : catalog.candidates) catalog_text << candidate.id << "|" << candidate.description << "\n";
    catalog.catalog_fingerprint = fingerprint(catalog_text.str());
    return true;
}
