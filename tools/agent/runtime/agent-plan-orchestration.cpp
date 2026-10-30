#include "plan/plan-context.h"
#include "agent-plan-orchestration.h"
#include "agent/template-library.h"
#include "hash/hash.h"

#include "../runtime/agent-planning-selection.h"
#include "../tooling/agent-tool-provider.h"
#include "tools/agent/cli/agent-cli-scope.h"
#include "agent-dataset-workflow-adapter.h"
#include "agent-workflow-transition.h"
#include "agent/agent-bootstrap.h"
#include "agent/artifact/artifact-contract.h"
#include "agent/learning/blueprint-selector.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <sstream>
#include <nlohmann/json.hpp>

std::string make_agent_automatic_blueprint_plan_id(const common_agent_scope & scope) {
    const auto identity = nlohmann::ordered_json::array({scope.namespace_id, scope.project_id,
        scope.session_id, scope.turn_id.empty() ? std::string("turn") : scope.turn_id}).dump();
    return "agent-blueprint:" + hash_sha256_hex(identity.data(), identity.size());
}

namespace {

std::string make_bootstrap_prefix(const common_agent_scope & scope) {
    return "bootstrap:" + scope.namespace_id + ":" +
        (scope.project_id.empty() ? "global" : "project:" + scope.project_id) + ":";
}


common_agent_request make_orchestration_selection_request(
        const common_agent_orchestration_config & config,
        const common_agent_scope & scope,
        common_agent_tool_output_format format) {
    common_agent_request request;
    request.tool_output_format = format;
    request.prompt = config.prompt;
    common_agent_scope_apply(scope, request);
    return request;
}

std::vector<common_chat_tool> workflow_action_candidates_impl(
        const std::vector<common_chat_tool> & route_tools,
        const common_agent_route_candidate & route) {
    const bool dataset_workflow = route.workflow_definition &&
        route.workflow_definition->family == "dataset";
    std::vector<common_chat_tool> selected;
    for (const auto & tool : route_tools) {
        if (dataset_workflow && tool.name.rfind("dataset.", 0) == 0) {
            // Dataset resolution and inspection are deterministic A* prefix
            // transitions; the model chooses the requested terminal operation.
            continue;
        }
        selected.push_back(tool);
    }
    return selected;
}

} // namespace

std::vector<common_chat_tool> common_agent_workflow_action_candidates(
        const std::vector<common_chat_tool> & route_tools,
        const common_agent_route_candidate & route) {
    return workflow_action_candidates_impl(route_tools, route);
}

common_agent_orchestration_config make_agent_orchestration_config(
        common_agent_orchestration_build_config config) {
    common_agent_orchestration_config result;
    result.prompt = std::move(config.prompt);
    result.agent_plan = std::move(config.agent_plan);
    result.agent_blueprint = std::move(config.agent_blueprint);
    result.agent_bootstrap = std::move(config.agent_bootstrap);
    result.agent_import = std::move(config.agent_import);
    result.agent_export = std::move(config.agent_export);
    result.blueprint_instance_materializer = std::move(config.blueprint_instance_materializer);
    return result;
}

bool maybe_install_agent_bootstrap(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const common_agent_orchestration_config & config,
    const common_agent_bootstrap_runtime_config & runtime_config,
    const common_agent_scope & scope,
    std::string & current_plan_id,
    std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
    std::string & error) {
    if (config.agent_bootstrap != "default" && config.agent_import.empty()) {
        error.clear();
        return true;
    }

    common_agent_bootstrap_config bootstrap_config;
    bootstrap_config.namespace_id = scope.namespace_id;
    bootstrap_config.session_id = scope.session_id;
    bootstrap_config.project_id = scope.project_id;
    bootstrap_config.now = std::time(nullptr);
    common_agent_bootstrap_result bootstrap_result;

    common_agent_bootstrap_package package;
    if (config.agent_import.empty()) {
        package = common_agent_default_bootstrap_package();
    } else if (!load_bootstrap_file(config.agent_import, package, error)) {
        error = "agent import failed: " + error;
        return false;
    }

    if (!common_agent_install_bootstrap_package(memory_store, plan_store, bootstrap_config, package, runtime_config.embed_procedure, bootstrap_result, error)) {
        error = "agent bootstrap failed: " + error;
        return false;
    }

    if (bootstrap_config.install_blueprints) {
        const std::string prefix = make_bootstrap_prefix(scope) + "blueprint:";
        for (const auto & blueprint : package.blueprints) {
            common_blueprint_candidate candidate;
            candidate.logical_id = blueprint.id;
            candidate.persisted_id = prefix + blueprint.id;
            candidate.source_revision = blueprint.source_revision;
            candidate.description = blueprint.selection_description.empty() ? blueprint.goal : blueprint.selection_description;
            candidate.purpose = blueprint.purpose;
            candidate.goal = blueprint.goal;
            candidate.success_criteria = blueprint.success_criteria;
            candidate.required_capabilities = blueprint.required_capabilities;
            candidate.workflow_policy = blueprint.workflow_policy;
            candidate.procedure_refs = blueprint.procedure_refs;
            candidate.workflow_bindings = blueprint.workflow_bindings;
            candidate.constraints = blueprint.constraints;
            candidate.assumptions = blueprint.assumptions;
            for (const auto & step : blueprint.steps) {
                const auto & contribution = step.intended_contribution.empty() ? step.objective : step.intended_contribution;
                if (!contribution.empty()) candidate.contributions.push_back(contribution);
            }
            installed_blueprint_candidates.push_back(std::move(candidate));
        }
    }

    fprintf(stderr, "agent bootstrap: procedures installed=%zu existing=%zu; blueprints installed=%zu existing=%zu\n",
        bootstrap_result.installed_memory_ids.size(), bootstrap_result.existing_memory_ids.size(),
        bootstrap_result.installed_blueprint_ids.size(), bootstrap_result.existing_blueprint_ids.size());

    // Blueprint selection is deliberately deferred to maybe_select_agent_route.
    // This keeps explicit and automatic selection on the same host-owned route
    // catalog and lets resume win before either path creates a new plan.
    (void) current_plan_id;

    error.clear();
    return true;
}

bool maybe_export_agent_package(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const common_agent_orchestration_config & config,
    const common_agent_scope & scope,
    bool & exported,
    std::string & error) {
    exported = false;
    if (config.agent_export.empty()) {
        error.clear();
        return true;
    }
    if (!export_agent_package(memory_store, plan_store, scope, config.agent_export, error)) {
        error = "agent export failed: " + error;
        return false;
    }
    fprintf(stderr, "agent export written: %s\n", config.agent_export.c_str());
    exported = true;
    error.clear();
    return true;
}

bool maybe_auto_select_plan(
    const common_agent_orchestration_runtime_context & context,
    std::string & error) {
    if (context.attachment_orientation) {
        // Do not resume an unrelated conversational task for the host-created
        // first-read operation. Keep only this turn's already-materialized
        // orientation plan so retries remain idempotent.
        if (!context.current_plan_id.empty()) {
            std::string lookup_error;
            const auto current = context.plan_store.get(context.current_plan_id, lookup_error);
            if (!lookup_error.empty()) {
                error = "attachment orientation plan lookup failed: " + lookup_error;
                return false;
            }
            const bool same_orientation_turn = current &&
                current->turn_id == context.scope.turn_id &&
                current->session_id == context.scope.session_id &&
                current->route_binding &&
                current->route_binding->workflow_ref == "workflow://resource/document-analysis";
            if (!same_orientation_turn) context.current_plan_id.clear();
        }
        error.clear();
        return true;
    }
    if (context.config.agent_plan != "auto" || !context.current_plan_id.empty()) {
        error.clear();
        return true;
    }

    const auto plans = context.plan_store.list(error);
    if (!error.empty()) {
        error = "failed to list plan candidates: " + error;
        return false;
    }

    std::vector<common_plan_state> candidates;
    for (const auto & plan : plans) {
        if (plan.kind != common_plan_kind::task ||
                (plan.status != common_plan_status::active && plan.status != common_plan_status::blocked)) {
            continue;
        }
        if (!common_plan_scope_matches(plan, context.scope.plan_scope, context.scope.namespace_id, context.scope.session_id, context.scope.project_id, context.scope.turn_id)) {
            continue;
        }
        candidates.push_back(plan);
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.updated_at != rhs.updated_at) {
            return lhs.updated_at > rhs.updated_at;
        }
        return lhs.id < rhs.id;
    });
    if (candidates.size() > 8) {
        candidates.resize(8);
    }

    if (!candidates.empty()) {
        const auto selection_request = make_orchestration_selection_request(context.config, context.scope, context.tool_output_format);
        std::string selection_error;
        const auto selection_result = select_agent_planning_plan_result(
            context.inference, context.generation_config, selection_request, candidates, selection_error);
        if (selection_result.plan_id) {
            context.current_plan_id = *selection_result.plan_id;
            fprintf(stderr, "agent plan auto-selected: %s\n", context.current_plan_id.c_str());
            context.pre_turn_trace.push_back({common_runtime_trace_stage::plan,
                common_runtime_trace_kind::decided, "plan selection outcome=resume", context.current_plan_id});
        } else if (selection_result.invalid_selection) {
            context.pre_turn_trace.push_back({common_runtime_trace_stage::plan,
                common_runtime_trace_kind::decided,
                "plan selection outcome=invalid_fallback_new_task; prior plans left unchanged", {}});
        } else if (!selection_error.empty()) {
            error = "plan selection failed before a decision: " + selection_error;
            return false;
        } else {
            fprintf(stderr, "agent plan auto-selection declined; creating a new plan\n");
            context.pre_turn_trace.push_back({common_runtime_trace_stage::plan,
                common_runtime_trace_kind::decided, "plan selection outcome=new_task", {}});
        }
    } else {
        context.pre_turn_trace.push_back({common_runtime_trace_stage::plan,
            common_runtime_trace_kind::decided, "plan selection outcome=no_compatible_candidate_new_task", {}});
    }

    error.clear();
    return true;
}

bool maybe_auto_select_blueprint(
        const common_agent_orchestration_runtime_context & context,
        std::string & error) {
    if (context.config.agent_blueprint != "auto") {
        error.clear();
        return true;
    }

    if (context.scope.session_id.empty()) {
        error.clear();
        return true;
    }
    if (context.installed_blueprint_candidates.empty()) {
        error.clear();
        return true;
    }

    // Family preflight and automatic plan resumption are host decisions about
    // the original request.  Only after both have had a chance to run may an
    // automatic blueprint reserve the task identity it needs for
    // instantiation.  This keeps a reservation distinct from a caller-owned
    // active plan and preserves family routing, required-tool propagation and
    // subsequent resource chunk planning on the first automatic turn.
    if (context.current_plan_id.empty()) {
        context.current_plan_id = make_agent_automatic_blueprint_plan_id(context.scope);
    }

    auto selector = make_agent_planning_blueprint_selector(context.inference, context.generation_config);
    common_blueprint_selection_config selection_config;
    selection_config.allow_keyword_fallback = false;
    if(context.tooling)selection_config.template_libraries=context.tooling->template_libraries;
    selection_config.task_plan_id = context.current_plan_id;
    selection_config.session_id = context.scope.session_id;
    selection_config.scope = context.scope.plan_scope;
    selection_config.now = std::time(nullptr);
    selection_config.materialize_instance = context.config.blueprint_instance_materializer
        ? context.config.blueprint_instance_materializer
        : make_agent_dataset_workflow_materializer(context.plan_store, context.tooling);
    if (context.tooling != nullptr && context.tooling->profile_tools_active) {
        selection_config.capabilities_resolved = true;
        for (const auto & tool : context.tooling->capabilities) {
            selection_config.available_capabilities.push_back(tool);
        }
        selection_config.blocked_constraint_ids = context.tooling->blocked_constraint_ids;
    }
    common_blueprint_selection_result selection;
    auto selection_request = make_orchestration_selection_request(context.config, context.scope, context.tool_output_format);
    if (context.policy_pack != nullptr) {
        selection_request.policy_pack = *context.policy_pack;
    }
    if (!common_agent_select_and_instantiate_blueprint(
            context.plan_store,
            selection_request,
            *selector,
            context.installed_blueprint_candidates,
            selection_config,
            selection,
            error)) {
        error = "agent blueprint selection failed: " + error;
        return false;
    }

    std::ostringstream diagnostic;
    diagnostic << "blueprint selection candidates=" << selection.candidate_count
               << " eligible=" << selection.eligible_count
               << " rejected=" << selection.rejections.size()
               << " outcome=" << static_cast<int>(selection.outcome);
    if (!selection.reason.empty()) diagnostic << " reason=" << selection.reason;
    context.pre_turn_events.push_back({
        common_agent_event_type::blueprint_selection_evaluated,
        diagnostic.str(),
        {},
        context.current_plan_id.empty()
            ? std::nullopt
            : std::optional<std::string>(context.current_plan_id),
    });
    context.pre_turn_trace.push_back({
        common_runtime_trace_stage::plan,
        common_runtime_trace_kind::decided,
        diagnostic.str(),
        context.current_plan_id,
        {}, {}, {}, selection.logical_id.value_or(std::string{}),
    });

    if (selection.outcome == common_blueprint_selection_outcome::instantiated) {
        fprintf(stderr, "agent blueprint auto-selected: %s -> %s\n", selection.logical_id->c_str(), context.current_plan_id.c_str());
        if (context.tooling != nullptr && context.tooling->profile_tools_active && context.tooling->tool_view != nullptr) {
            const auto binding_request = make_orchestration_selection_request(context.config, context.scope, context.tool_output_format);
            std::string binding_error;
            const auto binding_result = bind_agent_planning_blueprint_tools_result(
                context.inference,
                context.generation_config,
                *context.tooling->tool_view,
                binding_request,
                context.plan_store,
                context.current_plan_id,
                binding_error);
            if (!binding_result.applied) {
                fprintf(stderr, "agent blueprint binding declined safely: %s\n", binding_error.c_str());
            }
        }
    } else if (selection.outcome == common_blueprint_selection_outcome::resumed) {
        fprintf(stderr, "agent blueprint selection skipped: existing plan resumed\n");
    } else {
        fprintf(stderr, "agent blueprint auto-selection declined or failed safely; using normal plan creation\n");
    }

    error.clear();
    return true;
}

bool maybe_select_agent_route(
        const common_agent_orchestration_runtime_context & context,
        common_agent_route_candidate & selected_route,
        bool & route_selected,
        std::string & error) {
    selected_route = {};
    route_selected = false;
    error.clear();
    const bool host_orientation = context.attachment_orientation;
    if (!context.current_plan_id.empty() ||
            (context.config.agent_blueprint == "off" && !host_orientation) ||
            context.scope.session_id.empty() || context.tooling == nullptr) return true;

    common_agent_route_catalog catalog;
    if (!compile_agent_route_catalog(
            make_orchestration_selection_request(context.config, context.scope, context.tool_output_format),
            context.plan_store, context.scope, context.installed_blueprint_candidates,
            *context.tooling,
            (context.config.agent_blueprint == "auto" || host_orientation)
                ? std::string{} : context.config.agent_blueprint,
            catalog, error)) {
        error = "route compilation failed: " + error;
        return false;
    }

    const auto is_normal = [](const auto & candidate) {
        return candidate.kind == common_agent_route_kind::normal_plan;
    };
    std::optional<common_agent_route_candidate> chosen;
    std::string selection_mode = "deterministic";
    float confidence = 1.0f;
    std::string reason;
    if (context.config.agent_blueprint != "auto" && !host_orientation) {
        const auto found = std::find_if(catalog.candidates.begin(), catalog.candidates.end(),
            [&](const auto & candidate) {
                return !is_normal(candidate) && candidate.blueprint_logical_id == context.config.agent_blueprint;
            });
        if (found == catalog.candidates.end()) {
            error = "explicit blueprint is not eligible in the current host context";
            for (const auto & rejection : catalog.rejections) {
                if (rejection.blueprint_logical_id == context.config.agent_blueprint) {
                    error += ": " + rejection.reason;
                    if (!rejection.workflow_ref.empty()) error += " (" + rejection.workflow_ref + ")";
                    break;
                }
            }
            return false;
        }
        chosen = *found;
        reason = "explicit blueprint route";
    } else if (host_orientation) {
        const auto orientation_candidate = [&](const auto & candidate) {
            return candidate.kind == common_agent_route_kind::blueprint_workflow && candidate.workflow &&
                candidate.workflow->workflow_ref == "workflow://resource/document-analysis" &&
                (context.config.agent_blueprint == "auto" || context.config.agent_blueprint == "off" ||
                    candidate.blueprint_logical_id == context.config.agent_blueprint);
        };
        if (std::count_if(catalog.candidates.begin(), catalog.candidates.end(), orientation_candidate) > 1) {
            error = "attachment orientation has multiple authorized template origins; select an explicit blueprint";
            return false;
        }
        const auto found = std::find_if(catalog.candidates.begin(), catalog.candidates.end(), orientation_candidate);
        if (found == catalog.candidates.end()) {
            error = "host attachment orientation route is unavailable or not eligible";
            for (const auto & rejection : catalog.rejections) {
                if (rejection.workflow_ref == "workflow://resource/document-analysis") {
                    error += ": " + rejection.reason;
                    break;
                }
            }
            return false;
        }
        chosen = *found;
        selection_mode = "host";
        reason = "host-selected initial resource orientation";
    } else if (catalog.candidates.size() == 1) {
        chosen = catalog.candidates.front();
        reason = "single host-compiled route";
    } else {
        std::string selection_error;
        const auto route_choice = select_agent_planning_route(
            context.inference, context.generation_config,
            make_orchestration_selection_request(context.config, context.scope, context.tool_output_format),
            catalog, selection_error);
        if (route_choice.route_id) {
            const auto found = std::find_if(catalog.candidates.begin(), catalog.candidates.end(),
                [&](const auto & candidate) { return candidate.id == *route_choice.route_id; });
            if (found != catalog.candidates.end()) chosen = *found;
            selection_mode = "model";
            confidence = route_choice.confidence;
            reason = route_choice.reason;
        } else {
            chosen = *std::find_if(catalog.candidates.begin(), catalog.candidates.end(), is_normal);
            selection_mode = "fallback";
            confidence = 0.0f;
            reason = selection_error.empty() ? "route selector declined" : selection_error;
        }
    }
    if (!chosen) {
        error = "route catalog did not produce a safe fallback";
        return false;
    }
    selected_route = *chosen;
    route_selected = true;

    auto rejected_routes = nlohmann::ordered_json::array();
    for (const auto & rejection : catalog.rejections) {
        if (rejected_routes.size() == 32) break;
        rejected_routes.push_back({{"blueprint", rejection.blueprint_logical_id},
            {"workflow", rejection.workflow_ref}, {"reason", rejection.reason}});
    }
    const auto detail = nlohmann::ordered_json{
        {"context_fingerprint", catalog.context_fingerprint},
        {"catalog_fingerprint", catalog.catalog_fingerprint},
        {"candidate_count", catalog.candidates.size()},
        {"rejected_count", catalog.rejections.size()},
        {"rejected_routes", std::move(rejected_routes)},
        {"unavailable_optional_steps", selected_route.unavailable_optional_steps},
        {"selected_route_id", selected_route.id},
        {"selection_mode", selection_mode},
        {"confidence", confidence},
        {"reason", reason},
    }.dump();
    context.pre_turn_events.push_back({
        common_agent_event_type::route_selection_evaluated,
        detail, {}, context.current_plan_id.empty()
            ? std::nullopt : std::optional<std::string>(context.current_plan_id)});
    context.pre_turn_trace.push_back({
        common_runtime_trace_stage::plan,
        common_runtime_trace_kind::decided,
        detail,
        context.current_plan_id,
        {}, {}, {}, selected_route.id});

    context.route_procedure_memories.clear();
    if (!selected_route.procedure_refs.empty() && context.tooling && context.tooling->template_libraries &&
            context.tooling->template_libraries->find(selected_route.blueprint_persisted_id)) {
        const auto & libraries=*context.tooling->template_libraries;
        const auto * origin=libraries.find(selected_route.blueprint_persisted_id);
        for(const auto & ref:selected_route.procedure_refs){
            const auto id=ref.rfind(origin->definition_prefix,0)==0?ref:origin->definition_prefix+"procedure:"+ref;
            auto found=libraries.procedures.find(id);
            if(found==libraries.procedures.end() || !libraries.same_revision(selected_route.blueprint_persisted_id,id)){
                error="selected library procedure revision is unavailable";return false;
            }
            common_memory_hit hit;hit.memory=found->second;hit.provenance="verified template library procedure";hit.final_score=1.0f;
            context.route_procedure_memories.push_back(std::move(hit));
        }
    } else if (!selected_route.procedure_refs.empty()) {
        common_memory_query query;
        query.kind = common_memory_kind::procedure;
        query.scope = context.scope.project_id.empty()
            ? common_memory_scope::global : common_memory_scope::project;
        query.namespace_id = context.scope.namespace_id;
        query.session_id = context.scope.session_id;
        query.project_id = context.scope.project_id;
        query.turn_id = context.scope.turn_id;
        query.global_opt_in = context.scope.project_id.empty();
        query.limit = 64;
        std::string procedure_error;
        const auto procedures = context.memory_store.list(query, procedure_error);
        if (procedure_error.empty()) {
            for (const auto & procedure_ref : selected_route.procedure_refs) {
                const auto found = std::find_if(procedures.begin(), procedures.end(),
                    [&](const auto & procedure) {
                        return procedure.id == procedure_ref ||
                            procedure.id.size() > procedure_ref.size() + 1 &&
                            procedure.id.compare(procedure.id.size() - procedure_ref.size(),
                                procedure_ref.size(), procedure_ref) == 0 &&
                            procedure.id[procedure.id.size() - procedure_ref.size() - 1] == ':';
                    });
                if (found != procedures.end()) {
                    common_memory_hit hit;
                    hit.memory = *found;
                    hit.provenance = "blueprint procedure ref";
                    hit.final_score = 1.0f;
                    context.route_procedure_memories.push_back(std::move(hit));
                }
            }
        }
    }

    if (selected_route.kind == common_agent_route_kind::normal_plan) return true;
    if (context.current_plan_id.empty()) context.current_plan_id = make_agent_automatic_blueprint_plan_id(context.scope);

    const auto blueprint_candidate = std::find_if(
        context.installed_blueprint_candidates.begin(), context.installed_blueprint_candidates.end(),
        [&](const auto & candidate) { return candidate.logical_id == selected_route.blueprint_logical_id; });
    if (blueprint_candidate == context.installed_blueprint_candidates.end()) {
        error = "selected route does not map to an installed blueprint";
        return false;
    }
    common_explicit_blueprint_selector selector(selected_route.blueprint_logical_id);
    common_blueprint_selection_config selection_config;
    selection_config.allow_keyword_fallback = false;
    if(context.tooling)selection_config.template_libraries=context.tooling->template_libraries;
    selection_config.task_plan_id = context.current_plan_id;
    selection_config.session_id = context.scope.session_id;
    selection_config.scope = context.scope.plan_scope;
    selection_config.now = std::time(nullptr);
    selection_config.selected_workflow = selected_route.workflow;
    common_agent_execution_envelope route_envelope;
    if (!build_agent_execution_envelope(
            selected_route, *context.tooling, route_envelope, error)) {
        error = "selected route envelope failed: " + error;
        return false;
    }
    common_plan_route_binding route_binding;
    route_binding.route_id = selected_route.id;
    route_binding.blueprint_ref = selected_route.blueprint_logical_id;
    route_binding.blueprint_revision = selected_route.blueprint_revision;
    route_binding.graph_revision = selected_route.graph_revision;
    route_binding.execution_envelope_fingerprint = route_envelope.fingerprint;
    route_binding.policy_revision = route_envelope.policy_revision;
    if (selected_route.workflow) {
        route_binding.workflow_ref = selected_route.workflow->workflow_ref;
        route_binding.workflow_revision = selected_route.workflow->workflow_revision;
    }
    selection_config.route_binding = route_binding;
    selection_config.defer_unmaterialized_blueprint = selected_route.kind == common_agent_route_kind::blueprint;

    auto request = make_orchestration_selection_request(context.config, context.scope, context.tool_output_format);
    request.selected_workflow = selected_route.workflow;
    request.workflow_definition = selected_route.workflow_definition;
    request.memories = context.route_procedure_memories;
    request.available_datasets = context.tooling->available_datasets;
    request.investigation_source_kinds = context.tooling->investigation_source_kinds;
    request.investigation_context = context.tooling->investigation_context;
    if (context.policy_pack != nullptr) request.policy_pack = *context.policy_pack;
    if (context.tool_argument_bindings != nullptr) {
        request.tool_argument_bindings = *context.tool_argument_bindings;
    }
    std::optional<common_agent_tool_argument_binding> workflow_action;
    std::optional<common_plan_workflow_transition> initial_artifact_transition;
    if (selected_route.kind == common_agent_route_kind::blueprint_workflow) {
        std::vector<common_chat_tool> resolved_route_tools;
        for (const auto & tool : context.tooling->tools) {
            if (std::find(selected_route.resolved_tools.begin(),
                    selected_route.resolved_tools.end(), tool.name) !=
                    selected_route.resolved_tools.end()) {
                resolved_route_tools.push_back(tool);
            }
        }
        const auto route_tools = common_agent_workflow_action_candidates(
            resolved_route_tools, selected_route);
        std::vector<common_chat_tool> graph_entry_tools;
        if (selected_route.workflow_definition && !selected_route.workflow_definition->transitions.empty()) {
            for (const auto & transition : selected_route.workflow_definition->transitions) {
                if (transition.from_state == selected_route.workflow_definition->start_state &&
                        transition.kind == common_plan_workflow_transition_kind::artifact_compose) {
                    initial_artifact_transition = transition;
                    break;
                }
            }
            for (const auto & transition : selected_route.workflow_definition->transitions) {
                if (transition.from_state != selected_route.workflow_definition->start_state ||
                        transition.kind != common_plan_workflow_transition_kind::tool) continue;
                const auto found = std::find_if(resolved_route_tools.begin(), resolved_route_tools.end(),
                    [&](const auto & tool) { return tool.name == transition.tool_name; });
                if (found == resolved_route_tools.end()) {
                    if (transition.optional_availability) continue;
                    error = "workflow graph entry tool is outside the route envelope";
                    return false;
                }
                graph_entry_tools.push_back(*found);
            }
            if (graph_entry_tools.empty() && !initial_artifact_transition && !host_orientation) {
                error = "workflow graph has no materializable entry transition";
                return false;
            }
        }
        std::vector<std::string> host_resolved_dataset_tools;
        const auto selected_tools = graph_entry_tools.empty() ? route_tools : graph_entry_tools;
        std::string deterministic_entry_arguments;
        bool deterministic_graph_entry = false;
        if (graph_entry_tools.size() == 1 && selected_route.workflow_definition) {
            const auto transition = std::find_if(selected_route.workflow_definition->transitions.begin(),
                selected_route.workflow_definition->transitions.end(), [&](const auto & value) {
                    return value.from_state == selected_route.workflow_definition->start_state &&
                        value.kind == common_plan_workflow_transition_kind::tool &&
                        value.tool_name == graph_entry_tools.front().name;
                });
            if (transition != selected_route.workflow_definition->transitions.end()) {
                std::string argument_error;
                deterministic_graph_entry = common_agent_workflow_static_tool_arguments(
                    *transition, graph_entry_tools.front(), deterministic_entry_arguments, argument_error);
                if (!argument_error.empty()) {
                    error = std::move(argument_error);
                    return false;
                }
            }
        }
        if (host_orientation) {
            // The bounded resource reader below owns acquisition. Do not ask
            // the model to rediscover the resource or choose a read tool.
        } else if (initial_artifact_transition && graph_entry_tools.empty()) {
            // This entry is a host-owned plan operation, not a tool call.
        } else if (deterministic_graph_entry) {
            workflow_action = common_agent_tool_argument_binding{
                graph_entry_tools.front().name, deterministic_entry_arguments, "workflow-graph-entry", false};
        } else {
            std::string action_error;
            if (selected_route.workflow_definition) {
                for (const auto & transition : selected_route.workflow_definition->transitions) {
                    if (transition.from_state == selected_route.workflow_definition->start_state && !transition.model_instruction.empty())
                        request.prompt += "\n[Entry " + transition.tool_name + "]\n" + transition.model_instruction;
                }
            }
            const auto action = select_agent_planning_workflow_action(
                context.inference, context.generation_config, request, selected_tools,
                request.tool_argument_bindings, host_resolved_dataset_tools,
                context.tool_output_format, action_error);
            if (!action.action) {
                error = "workflow action selection failed: " + action_error;
                return false;
            }
            workflow_action = *action.action;
        }
        if (workflow_action) request.tool_argument_bindings.push_back(*workflow_action);
        if (workflow_action) context.pre_turn_trace.push_back({
            common_runtime_trace_stage::plan,
            common_runtime_trace_kind::decided,
            nlohmann::ordered_json{
                {"type", "workflow_action_selected"},
                {"tool", workflow_action->tool_name},
                {"selection_mode", deterministic_graph_entry ? "deterministic_graph_entry" :
                    (context.tool_output_format == common_agent_tool_output_format::compact_dsl
                        ? "compact_dsl_grammar" : "json_schema")},
                {"route_id", selected_route.id},
            }.dump(),
            context.current_plan_id,
            {}, workflow_action->tool_name, {}, selected_route.id});
    }
    if (context.config.blueprint_instance_materializer && !host_orientation) {
        selection_config.materialize_instance = context.config.blueprint_instance_materializer;
    } else if (selected_route.kind == common_agent_route_kind::blueprint_workflow) {
        selection_config.materialize_instance = make_agent_dataset_workflow_materializer(
            context.plan_store, context.tooling);
    }
    if (selected_route.kind == common_agent_route_kind::blueprint_workflow &&
            (workflow_action || initial_artifact_transition || host_orientation)) {
        const auto workflow_materializer = selection_config.materialize_instance;
        const auto action = workflow_action;
        const auto artifact_entry = initial_artifact_transition;
        const auto graph_definition = selected_route.workflow_definition;
        selection_config.materialize_instance = [workflow_materializer, action, artifact_entry, graph_definition, host_orientation](
                const common_agent_request & materialization_request,
                const common_plan_state & instance,
                common_plan_state & materialized,
                std::string & materialization_error) {
            // A declared workflow graph is the authoritative materialization
            // contract.  The dataset materializer predates graphs and is a
            // compatibility fallback for graph-less dataset workflows only;
            // letting it run first can turn a graph entry into an already
            // completed legacy step before the generic runner sees it.
            const bool has_workflow_graph = graph_definition &&
                !graph_definition->transitions.empty();
            if (!has_workflow_graph && workflow_materializer) {
                const auto outcome = workflow_materializer(
                    materialization_request, instance, materialized, materialization_error);
                if (outcome != common_blueprint_materialization_outcome::not_applicable) {
                    return outcome;
                }
            }
            common_plan_state out = instance;
            out.kind = common_plan_kind::task;
            out.observations.push_back(common_plan_make_blueprint_outline(instance, instance.id));
            out.steps.clear();
            out.active_step_id.reset();
            out.next_action.reset();
            out.status = common_plan_status::active;
            if (host_orientation) {
                common_plan_step answer;
                answer.id = instance.id + ":resource-orientation:answer";
                answer.title = "Orient to supplied resource";
                answer.objective = instance.goal;
                answer.intended_contribution = instance.success_criteria.empty()
                    ? instance.goal : instance.success_criteria;
                answer.mode = common_plan_step_mode::final_response;
                answer.status = common_plan_step_status::pending;
                answer.created_at = instance.created_at;
                answer.updated_at = instance.updated_at;
                out.steps.push_back(std::move(answer));
                out.next_action = out.steps.front().id;
                materialized = std::move(out);
                materialization_error.clear();
                return common_blueprint_materialization_outcome::applied;
            }
            if (!action && artifact_entry && graph_definition) {
                common_agent_artifact_profile profile;
                if (!common_agent_artifact_profile_from_json(artifact_entry->artifact_profile_json, profile, materialization_error))
                    return common_blueprint_materialization_outcome::failed_safely;
                const auto section = std::find_if(profile.sections.begin(), profile.sections.end(), [&](const auto & candidate) {
                    return candidate.id == artifact_entry->artifact_section_id;
                });
                if (section == profile.sections.end()) { materialization_error="artifact entry section is missing from its profile";return common_blueprint_materialization_outcome::failed_safely; }
                common_plan_step compose;compose.id=instance.id+":artifact:"+profile.id+":"+section->id;
                compose.title=section->title;compose.objective=section->purpose;compose.intended_contribution=section->purpose;
                compose.status=common_plan_step_status::pending;compose.mode=common_plan_step_mode::artifact_compose;
                compose.semantic_alias="workflow-transition:"+artifact_entry->id;
                compose.artifact_binding=common_plan_artifact_binding{artifact_entry->id,profile.id,profile.revision,section->id,"snapshot:"+out.id};
                compose.required_evidence=section->required_observation_sources;compose.created_at=instance.created_at;compose.updated_at=instance.updated_at;
                out.steps.push_back(compose);
                common_plan_step answer;answer.id=instance.id+":workflow:answer";answer.title="Answer";
                answer.objective=instance.goal.empty()?"Answer from the validated report.":instance.goal;
                answer.intended_contribution=instance.success_criteria.empty()?answer.objective:instance.success_criteria;
                answer.mode=common_plan_step_mode::final_response;answer.status=common_plan_step_status::pending;answer.depends_on={compose.id};
                answer.created_at=instance.created_at;answer.updated_at=instance.updated_at;out.steps.push_back(std::move(answer));
                out.next_action=compose.id;materialized=std::move(out);materialization_error.clear();
                return common_blueprint_materialization_outcome::applied;
            }
            if (!action) { materialization_error="workflow has no host-materializable entry action";return common_blueprint_materialization_outcome::failed_safely; }
            common_plan_step operation;
            operation.id = instance.id + ":workflow:action";
            operation.title = action->tool_name;
            operation.objective = "Execute the selected workflow action.";
            operation.intended_contribution = operation.objective;
            // The runtime scheduler owns activation.  Persisting an entry
            // transition as pending ensures it is validated and activated in
            // the same dependency gate as every later graph transition.
            operation.status = common_plan_step_status::pending;
            operation.mode = common_plan_step_mode::tool;
            operation.selected_tool = action->tool_name;
            operation.tool_call = common_plan_tool_call{action->tool_name, action->arguments_json};
            if (graph_definition && !graph_definition->transitions.empty()) {
                const auto transition = std::find_if(graph_definition->transitions.begin(), graph_definition->transitions.end(),
                    [&](const auto & value) { return value.from_state == graph_definition->start_state &&
                        value.kind == common_plan_workflow_transition_kind::tool && value.tool_name == action->tool_name; });
                if (transition == graph_definition->transitions.end()) {
                    materialization_error = "workflow graph entry action is inconsistent with the selected tool";
                    return common_blueprint_materialization_outcome::failed_safely;
                }
                operation.semantic_alias = "workflow-transition:" + transition->id;
                operation.title = transition->id;
                operation.objective = "Execute workflow transition " + transition->id + ".";
                operation.intended_contribution = operation.objective;
                common_agent_workflow_apply_blueprint_phase(out, *transition, operation);
            }
            operation.created_at = instance.created_at;
            operation.updated_at = instance.updated_at;
            out.steps.push_back(operation);
            common_plan_step answer;
            answer.id = instance.id + ":workflow:answer";
            answer.title = "Answer";
            answer.objective = instance.goal.empty()
                ? "Answer from the workflow result." : instance.goal;
            answer.intended_contribution = instance.success_criteria.empty()
                ? answer.objective : instance.success_criteria;
            answer.mode = common_plan_step_mode::final_response;
            answer.status = common_plan_step_status::pending;
            answer.depends_on = {operation.id};
            answer.created_at = instance.created_at;
            answer.updated_at = instance.updated_at;
            out.steps.push_back(std::move(answer));
            out.active_step_id.reset();
            out.next_action = operation.id;
            materialized = std::move(out);
            materialization_error.clear();
            return common_blueprint_materialization_outcome::applied;
        };
    }
    if (context.tooling->profile_tools_active) {
        selection_config.capabilities_resolved = true;
        selection_config.available_capabilities = context.tooling->capabilities;
        // The route compiler has already validated semantic built-in
        // capabilities derived from the concrete tool view and the selected
        // workflow family.  Carry that same host-owned snapshot into the
        // selector's structural eligibility check; otherwise a valid route
        // would be rejected by the older selector because it only knows the
        // raw configured capability ids.
        for (const auto & capability : selected_route.required_capabilities) {
            if (std::find(selection_config.available_capabilities.begin(),
                    selection_config.available_capabilities.end(), capability) ==
                    selection_config.available_capabilities.end()) {
                selection_config.available_capabilities.push_back(capability);
            }
        }
        selection_config.blocked_constraint_ids = context.tooling->blocked_constraint_ids;
    }
    common_blueprint_selection_result selection;
    if (!common_agent_select_and_instantiate_blueprint(
            context.plan_store, request, selector, {*blueprint_candidate}, selection_config,
            selection, error)) {
        error = "route blueprint instantiation failed: " + error;
        return false;
    }
    if (selection.outcome != common_blueprint_selection_outcome::instantiated &&
            selection.outcome != common_blueprint_selection_outcome::deferred_to_planner &&
            selection.outcome != common_blueprint_selection_outcome::resumed) {
        error = "selected route failed safely: " + selection.reason;
        return false;
    }
    return true;
}
