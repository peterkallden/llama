#include "agent-runtime-execution.h"
#include "common/agent/resource-chunk-extraction.h"
#include "agent/context-compaction.h"
#include "agent/input-resources.h"
#include "agent/tool-family-index.h"
#include "../runtime/agent-runtime-chat-driver.h"
#include "../runtime/agent-route-compiler.h"
#include "agent-dataset-workflow-adapter.h"
#include "agent-resource-workflow-continuation.h"
#include "agent-workflow-transition.h"

#include "../runtime/agent-plan-orchestration.h"
#include "../runtime/agent-runtime-assembly.h"
#include "../cli/agent-cli-selection.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <ctime>

namespace {

bool transition_driver_cognitive_mode(
        common_agent_runtime_driver_execution & execution,
        common_agent_cognitive_mode to,
        const std::string & reason,
        const std::string & source) {
    auto & state = execution.cognitive_state;
    if (state.mode == to) return true;
    const auto from = state.mode;
    const bool allowed = common_agent_cognitive_transition(state, to, reason);
    const auto detail = nlohmann::ordered_json{
        {"type", "cognitive_mode_transition"},
        {"from", common_agent_cognitive_mode_name(from)},
        {"to", common_agent_cognitive_mode_name(to)},
        {"reason", reason},
        {"source", source},
        {"allowed", allowed},
    }.dump();
    execution.pre_turn_trace.push_back({
        common_runtime_trace_stage::plan,
        allowed ? common_runtime_trace_kind::updated : common_runtime_trace_kind::failed,
        detail,
        execution.current_plan_id,
        {}, {}, {}, {}});
    return allowed;
}

bool tool_has_no_required_arguments(const common_chat_tool & tool) {
    const auto schema = nlohmann::ordered_json::parse(tool.parameters, nullptr, false);
    if (schema.is_discarded() || !schema.is_object()) return false;
    if (!schema.contains("required")) return true;
    return schema["required"].is_array() && schema["required"].empty();
}

bool restore_and_validate_bound_route(
        common_agent_runtime_driver_execution & execution,
        std::string & error) {
    if (execution.current_plan_id.empty()) return true;
    std::string plan_error;
    const auto plan = execution.plan_store.get(execution.current_plan_id, plan_error);
    if (!plan_error.empty()) { error = "resume route validation failed: " + plan_error; return false; }
    if (!plan || !plan->route_binding) return true; // legacy plans remain unbound.

    common_agent_route_candidate route;
    route.id = plan->route_binding->route_id;
    route.blueprint_logical_id = plan->route_binding->blueprint_ref;
    route.blueprint_revision = plan->route_binding->blueprint_revision;
    route.graph_revision = plan->route_binding->graph_revision;
    route.required_capabilities = plan->required_capabilities;
    if (!plan->route_binding->workflow_ref.empty()) {
        route.kind = common_agent_route_kind::blueprint_workflow;
        route.workflow = common_plan_workflow_binding{
            plan->route_binding->workflow_ref,
            plan->route_binding->workflow_revision};
        if (!plan->workflow_definition ||
                plan->workflow_definition->workflow_ref != route.workflow->workflow_ref ||
                plan->workflow_definition->workflow_revision != route.workflow->workflow_revision ||
                plan->workflow_definition->graph_revision != route.graph_revision) {
            error = "route binding is stale: persisted workflow identity changed";
            return false;
        }
        for (const auto & capability : plan->workflow_definition->required_capabilities) {
            if (std::find(route.required_capabilities.begin(), route.required_capabilities.end(),
                    capability) == route.required_capabilities.end()) {
                route.required_capabilities.push_back(capability);
            }
        }
        for (const auto & requirement : plan->workflow_definition->required_context) {
            const bool resource_context_available = requirement == "context.resource.available" &&
                std::any_of(execution.input_resources.begin(), execution.input_resources.end(),
                    [](const auto & resource) { return !resource.resource.uri.empty(); });
            if (!resource_context_available &&
                    std::find(execution.tooling.available_context.begin(), execution.tooling.available_context.end(),
                        requirement) == execution.tooling.available_context.end()) {
                error = "route binding is stale: required workflow context is unavailable: " + requirement;
                return false;
            }
            route.required_context.push_back(requirement);
        }
        std::string resolution_reason;
        std::vector<std::string> resolved_optional_capabilities;
        if (!common_agent_resolve_workflow_tools(
                *plan->workflow_definition, execution.tooling,
                route.resolved_tools, resolved_optional_capabilities, resolution_reason)) {
            error = "route binding is stale: " + resolution_reason;
            return false;
        }
        route.resolved_optional_capabilities = std::move(resolved_optional_capabilities);
    } else if (!route.blueprint_logical_id.empty()) {
        route.kind = common_agent_route_kind::blueprint;
    } else {
        route.kind = common_agent_route_kind::normal_plan;
    }

    common_agent_execution_envelope envelope;
    if (!build_agent_execution_envelope(route, execution.tooling, envelope, error)) {
        error = "route binding is stale: " + error;
        return false;
    }
    const auto & binding = *plan->route_binding;
    if (binding.blueprint_revision != envelope.blueprint_revision ||
            binding.workflow_revision != envelope.workflow_revision ||
            binding.graph_revision != envelope.graph_revision ||
            binding.policy_revision != envelope.policy_revision ||
            binding.execution_envelope_fingerprint != envelope.fingerprint) {
        error = "route binding is stale: execution envelope or policy revision changed";
        return false;
    }
    for (const auto & capability : plan->required_capabilities) {
        if (std::find(execution.tooling.capabilities.begin(), execution.tooling.capabilities.end(), capability) ==
                execution.tooling.capabilities.end()) {
            error = "route binding is stale: required capability is unavailable: " + capability;
            return false;
        }
    }
    execution.selected_route = std::move(route);
    execution.route_selected = true;
    execution.execution_envelope = std::move(envelope);
    execution.execution_envelope_ready = true;
    execution.pre_turn_events.push_back({
        common_agent_event_type::route_selection_evaluated,
        "resumed existing route after deterministic envelope validation",
        {}, execution.current_plan_id});
    if (execution.policy.deliberation_policy.mode != common_agent_thinking_mode::research &&
            !transition_driver_cognitive_mode(execution,
                common_agent_cognitive_mode::execute,
                "resume after deterministic route revalidation", "host")) {
        error = "invalid cognitive-mode transition while resuming a route";
        return false;
    }
    return true;
}

bool select_model_tool_families(
        common_agent_runtime_driver_execution & execution,
        const common_agent_request & request,
        std::string & error) {
    const bool caller_requires_tool_execution = execution.require_tool_execution;
    execution.model_tools.clear();
    const auto & generation_config = execution.runtime_config.generation_config;
    execution.family_chat_routed = false;
    execution.family_chat_result = {};
    // Family preflight may strengthen this invariant after selecting a
    // family, but must never discard an explicit caller requirement.
    execution.require_tool_execution = caller_requires_tool_execution;
    if (!generation_config.enable_tool_family_routing ||
            execution.tooling.tools.empty() ||
            !execution.current_plan_id.empty()) {
        return true;
    }

    const auto families = common_generate_tool_family_index(
        execution.tooling.tools, execution.tooling.family_descriptions);
    const std::string family_view = common_render_tool_family_index(families, 2048);
    const bool required_family_selection = caller_requires_tool_execution;
    common_chat_msg system{
        "system",
        required_family_selection
            ? common_tool_family_required_selection_prompt(families)
            : "Decide whether the user request needs external tools. Reply with exactly one line: "
              "NO_TOOLS for ordinary conversation, or TOOLS: family_id[, family_id...] when tools "
              "are needed. Current-turn attachments are optional inputs: choose the resource or "
              "dataset family when the request concerns an attached file, but keep NO_TOOLS for "
              "unrelated conversation. Do not explain, use JSON, or invent family ids. Available "
              "families:\n" + family_view,
    };
    common_chat_msg user{"user", request.prompt + common_agent_render_input_resource_context(
        request.input_resources, 2048, request.available_resources)};
    common_agent_generation_options options;
    // This is a one-line classifier, not a planning turn. A small hard cap
    // prevents a weak model from spending the whole turn elaborating before
    // the host can route it.
    options.n_predict = 64;
    options.n_threads = generation_config.n_threads;
    options.generation_trace = generation_config.generation_trace;
    const auto generate_selection = [&](const std::string & request_suffix,
                                        const std::vector<common_chat_msg> & messages,
                                        const std::string & json_schema = std::string{}) {
        return execution.inference.generate_result(
            common_agent_make_generation_request(
                common_agent_generation_purpose::tool_family_selection,
                request.turn_id + request_suffix,
                common_agent_scope_from_request(request),
                messages,
                options,
                json_schema));
    };
    const std::string required_schema = required_family_selection
        ? common_tool_family_required_selection_schema(families)
        : std::string{};
    auto generation = generate_selection(
        ":tool-family-selection", {system, user}, required_schema);
    if (!common_agent_generation_succeeded(generation)) {
        error = "tool family selection failed: " + generation.error_message;
        if (generation.error_message.empty()) error += " model did not return a completed selection";
        return false;
    }

    common_tool_family_selection selection;
    const bool parsed = required_family_selection
        ? common_parse_tool_family_required_selection(
            generation.content, families, selection, error)
        : common_parse_tool_family_selection_text(
            generation.content, families, selection, error);
    if (!parsed) {
        const std::string original_selection_error = error;
        // Keep the parser strict: a malformed first classifier response must
        // never grant a family that it did not name exactly.  Small models do
        // occasionally append prompt text after an otherwise valid `TOOLS:`
        // answer, though.  Give that one-line classifier one bounded repair
        // turn with a structured response instead of weakening the
        // authorization grammar itself.
        common_chat_msg repair{
            "system",
            required_family_selection
                ? common_tool_family_required_selection_repair_prompt()
                : "Your previous tool-family selection was invalid. Return the required JSON object only. "
                  "Families may contain only ids from the available list; do not include tool arguments, "
                  "resource ids, explanations, or any other text.",
        };
        const auto repaired = generate_selection(
            ":tool-family-selection-repair", {system, repair, user},
            required_family_selection
                ? common_tool_family_required_selection_schema(families)
                : common_tool_family_selection_schema());
        const bool repaired_ok = common_agent_generation_succeeded(repaired) &&
            (required_family_selection
                ? common_parse_tool_family_required_selection(
                    repaired.content, families, selection, error)
                : common_parse_tool_family_selection(
                    repaired.content, selection, error));
        if (repaired_ok) {
            // Repaired selection is subject to the same policy filtering as
            // the original response.
        } else {
            // A weak model can emit the required TOOLS: prefix and then stop
            // before naming a family. With exactly one current-turn resource,
            // the host can safely default to the already policy-filtered
            // resource family. This keeps the family gate intact and does not
            // guess when there are multiple resources or no resource family.
            const bool empty_tools_selection =
                original_selection_error == "tool family selection must include at least one family after TOOLS:";
            const bool has_resource_family = std::any_of(
                families.begin(), families.end(), [](const common_tool_family_index & family) {
                    return family.id == "resource";
                });
            if (!required_family_selection && empty_tools_selection &&
                    request.input_resources.size() == 1 && has_resource_family) {
                selection.needs_tools = true;
                selection.family_ids = {"resource"};
                error.clear();
            } else {
                error = "tool family selection failed: " + error;
                return false;
            }
        }
    }
    if (!selection.needs_tools && !selection.family_ids.empty()) {
        error = "tool family selection must not select families when needs_tools is false";
        return false;
    }
    const auto unknown_family = std::find_if(
        selection.family_ids.begin(), selection.family_ids.end(),
        [&families](const std::string & id) {
            return std::none_of(families.begin(), families.end(),
                [&id](const common_tool_family_index & family) { return family.id == id; });
        });
    if (unknown_family != selection.family_ids.end()) {
        error = "tool family selection returned unknown family: " + *unknown_family;
        return false;
    }
    if (!selection.needs_tools) {
        if (execution.require_tool_execution) {
            error = "tool family selection declined tools for a tool-required request";
            return false;
        }
        common_agent_runtime_tooling chat_tooling = execution.tooling;
        chat_tooling.tools.clear();
        common_agent_generation_options chat_options;
        chat_options.n_predict = generation_config.n_predict;
        chat_options.n_threads = generation_config.n_threads;
        chat_options.generation_trace = generation_config.generation_trace;
        common_agent_chat_runtime_execution chat_execution{
            execution.inference,
            request,
            chat_options,
            {execution.policy.max_tool_rounds, execution.runtime_config.max_continuations},
            chat_tooling,
            execution.execution_control,
        };
        execution.family_chat_routed = true;
        return run_agent_chat_runtime(chat_execution, execution.family_chat_result, error);
    }
    execution.model_tools = common_expand_tool_family_prerequisites(
        execution.tooling.tools, selection.family_ids);
    if (execution.model_tools.empty()) {
        error = "tool family selection did not resolve any registered tools";
        return false;
    }
    // The preflight is a host-owned intent decision. Once it says that a
    // family is needed, a planner-generated answer without a completed tool
    // would be an ungrounded fallback (for example, a placeholder answer to
    // a time query). Reuse the runtime's existing fail-closed contract.
    execution.require_tool_execution = true;

    // A selected singleton tool with no mandatory arguments does not need a
    // model-authored plan. Reuse the normal chat/tool driver, but keep the
    // request marked as tool-required so it cannot silently answer without
    // dispatching the selected tool.
    if (execution.model_tools.size() == 1 &&
            tool_has_no_required_arguments(execution.model_tools.front())) {
        common_agent_runtime_tooling chat_tooling = execution.tooling;
        chat_tooling.tools = execution.model_tools;
        common_agent_request chat_request = request;
        chat_request.require_tool_execution = true;
        common_agent_generation_options chat_options;
        chat_options.n_predict = generation_config.n_predict;
        chat_options.n_threads = generation_config.n_threads;
        chat_options.generation_trace = generation_config.generation_trace;
        common_agent_chat_runtime_execution chat_execution{
            execution.inference,
            std::move(chat_request),
            chat_options,
            {execution.policy.max_tool_rounds, execution.runtime_config.max_continuations},
            chat_tooling,
            execution.execution_control,
        };
        execution.family_chat_routed = true;
        return run_agent_chat_runtime(chat_execution, execution.family_chat_result, error);
    }
    return true;
}

bool prepare_available_resources(common_agent_runtime_driver_execution & execution) {
    if (execution.tooling.resource_runtime.store == nullptr) return true;
    std::vector<agent_resource_descriptor> listed;
    std::string list_error;
    const auto authority = make_agent_resource_read_authority(
        execution.tooling.resource_runtime, static_cast<int64_t>(std::time(nullptr)));
    // Listing is advisory context. A store/list failure must not make an
    // otherwise valid turn fail; normal binding still enforces read authority.
    if (!execution.tooling.resource_runtime.store->list(authority, listed, list_error)) return true;
    execution.available_resources.clear();
    for (const auto & descriptor : listed) {
        if (descriptor.uri.empty()) continue;
        bool current = false;
        for (auto & input : execution.input_resources) {
            if (input.resource.uri != descriptor.uri) continue;
            // Web clients normally submit only the opaque URI. Rehydrate the
            // host-owned descriptor before family selection so the model can
            // see the attachment name, MIME type and bounded metadata while
            // retaining the caller-owned role/required flags.
            const auto role = input.role;
            const bool required = input.required;
            input.resource = descriptor;
            input.role = role;
            input.required = required;
            current = true;
            break;
        }
        if (!current) execution.available_resources.push_back({descriptor, "scoped_reference", false});
    }
    return true;
}

bool prepare_resource_chunk_observations(
        common_agent_runtime_driver_execution & execution,
        std::string & error) {
    if (execution.resource_chunk_observations_prepared) {
        return true;
    }
    execution.resource_chunk_observations_prepared = true;
    execution.resource_chunk_original_inputs = execution.input_resources;
    if (execution.tooling.resource_runtime.store == nullptr ||
            execution.input_resources.empty() || execution.current_plan_id.empty()) {
        return true;
    }

    std::string plan_error;
    auto plan = execution.plan_store.get(execution.current_plan_id, plan_error);
    if (!plan) {
        error = plan_error.empty() ? "resource chunk planning requires an active plan" : plan_error;
        return false;
    }
    const auto & context_budgets = execution.runtime_config.generation_config.context_budgets;
    const common_runtime_resource_chunk_policy policy{
        context_budgets.resource_chunk_max_bytes,
        context_budgets.resource_chunk_overlap_bytes};
    const auto authority = make_agent_resource_read_authority(
        execution.tooling.resource_runtime, static_cast<int64_t>(std::time(nullptr)));
    const size_t input_count = execution.input_resources.size();
    for (size_t input_index = 0; input_index < input_count; ++input_index) {
        const auto & input = execution.input_resources[input_index];
        if (input.resource.uri.empty() ||
                common_runtime_resource_lineage_is_chunk(input.resource.lineage) ||
                input.resource.size_bytes <= policy.max_bytes) {
            continue;
        }
        agent_resource_evidence_session evidence_session;
        if (!make_agent_resource_evidence_session(
                    *execution.tooling.resource_runtime.store,
                    input.resource.uri,
                    authority,
                    policy,
                    evidence_session,
                    error)) {
            return false;
        }
        const auto & chunk_plan = evidence_session.plan;
        if (chunk_plan.ranges.empty()) {
            continue;
        }
        size_t next_chunk_index = 0;
        for (const auto & observation : plan->observations) {
            if (!common_plan_observation_is_resource_chunk(observation) ||
                    observation.resource_refs.size() != 1 ||
                    observation.resource_refs.front().lineage.parent_uri != input.resource.uri) {
                continue;
            }
            const auto & lineage = observation.resource_refs.front().lineage;
            if (lineage.chunk_index == next_chunk_index) ++next_chunk_index;
        }
        if (next_chunk_index >= chunk_plan.ranges.size()) {
            continue;
        }
        const auto first_ref = make_agent_resource_chunk_ref(
            chunk_plan, chunk_plan.ranges[next_chunk_index]);
        evidence_session.active_index = next_chunk_index;
        execution.resource_evidence_sessions.push_back(std::move(evidence_session));
        const size_t plan_index = execution.resource_evidence_sessions.size() - 1;
        execution.input_resources[input_index] = {
            first_ref, input.role.empty() ? "resource_chunk" : input.role, input.required};
        execution.active_resource_chunk_plan = plan_index;
        execution.active_resource_chunk_input = input_index;
        execution.active_resource_chunk_index = next_chunk_index;

        const std::string plan_observation_id = "resource_chunk_plan:" + input.resource.uri;
        const bool plan_observation_exists = std::any_of(
            plan->observations.begin(), plan->observations.end(),
            [&](const common_plan_observation & observation) {
                return observation.id == plan_observation_id;
            });
        if (!plan_observation_exists) {
            common_plan_operation observed;
            observed.kind = common_plan_operation_kind::record_observation;
            observed.plan_id = plan->id;
            observed.expected_version = plan->version;
            observed.reason_summary = "bounded resource chunk plan attached to the session lane";
            observed.observation = common_plan_observation{
                plan_observation_id,
                "resource_chunk_planned",
                "Planned " + std::to_string(execution.resource_evidence_sessions.back().plan.ranges.size()) +
                    " bounded text chunks; starting at chunk " + std::to_string(next_chunk_index) +
                    "; the original resource remains authoritative.",
                1.0f,
                {input.resource.uri},
                {first_ref},
                static_cast<int64_t>(std::time(nullptr)),
            };
            common_plan_state updated;
            if (!execution.plan_store.apply(observed, updated, error)) {
                return false;
            }
            *plan = std::move(updated);
            execution.pre_turn_events.push_back({
                common_agent_event_type::resource_chunk_planned,
                "bounded resource chunk plan attached chunk_count=" +
                    std::to_string(execution.resource_evidence_sessions.back().plan.ranges.size()),
                {},
                plan->id,
                {},
                observed.observation->id,
                {},
                first_ref.uri,
            });
        }
        // A turn has one active bounded input chain.  Additional oversized
        // resources are planned on a later session-lane turn, never in
        // parallel with this one.
        break;
    }
    error.clear();
    return true;
}

bool extract_active_resource_chunk(
        common_agent_runtime_driver_execution & execution,
        const std::string & task,
        common_agent_resource_chunk_context & context,
        common_agent_resource_chunk_extraction & extraction,
        std::string & error) {
    if (execution.active_resource_chunk_plan >= execution.resource_evidence_sessions.size() ||
            execution.active_resource_chunk_index >=
                execution.resource_evidence_sessions[execution.active_resource_chunk_plan].plan.ranges.size() ||
            execution.tooling.resource_runtime.store == nullptr) {
        error = "resource chunk extraction context is invalid";
        return false;
    }
    const auto & session = execution.resource_evidence_sessions[execution.active_resource_chunk_plan];
    const auto & plan = session.plan;
    const size_t index = execution.active_resource_chunk_index;
    context = {};
    const auto authority = make_agent_resource_read_authority(
        execution.tooling.resource_runtime, static_cast<int64_t>(std::time(nullptr)));
    if (!read_agent_resource_evidence_slice(*execution.tooling.resource_runtime.store,
            authority, session, context, error)) {
        error = "host could not read active resource chunk: " + error;
        return false;
    }

    const auto & config = execution.runtime_config.generation_config;
    for (size_t attempt = 0; attempt < 2; ++attempt) {
        common_agent_generation_options options;
        options.n_predict = 192;
        options.n_threads = config.n_threads;
        options.generation_trace = config.generation_trace;
        const std::string correction = attempt == 0 ? "" :
            "the previous attempt failed because " + error +
            ". Try once more with only one complete evidence sentence copied from the visible text; do not add a finding field or wrap it in quotes. "
            "Return status: no_findings if no sentence helps answer the task.";
        auto request = common_agent_make_resource_chunk_extraction_request(
            task,
            context,
            execution.scope,
            options,
            execution.scope.turn_id + ":resource_chunk:" + std::to_string(index) +
                ":attempt:" + std::to_string(attempt + 1),
            correction);
        const std::string submitted_slice_text = context.text;
        const bool prompt_contains_submitted_slice = request.messages.size() > 1 &&
            request.messages[1].content.find(submitted_slice_text) != std::string::npos;
        const auto generated = execution.inference.generate_result(request);
        if (!common_agent_generation_succeeded(generated)) {
            error = "resource evidence generation failed: " + generated.error_message;
            if (generated.error_message.empty()) error += " model did not return a completed result";
            continue;
        }
        if (common_agent_parse_resource_chunk_extraction(
                generated.content, context, extraction, error)) {
            error.clear();
            return true;
        }
        error += " (prompt_contains_slice=" +
            std::string(prompt_contains_submitted_slice ? "true" : "false") +
            ", context_unchanged=" +
            std::string(context.text == submitted_slice_text ? "true" : "false") + ")";
    }
    // Extraction is an evidence-quality outcome, not a workflow-fatal tool
    // failure. Record the missing coverage and let the host advance to the
    // next bounded slice.
    extraction = {};
    extraction.unresolved = true;
    extraction.unresolved_reason = error;
    error.clear();
    return true;
}

bool record_and_advance_resource_chunk(
        common_agent_runtime_driver_execution & execution,
        const common_agent_resource_chunk_context & context,
        const common_agent_resource_chunk_extraction & extraction,
        bool & chunk_chain_completed,
        std::string & error) {
    chunk_chain_completed = false;
    if (execution.active_resource_chunk_plan == static_cast<size_t>(-1)) return false;
    if (execution.active_resource_chunk_plan >= execution.resource_evidence_sessions.size() ||
            execution.active_resource_chunk_input >= execution.input_resources.size()) {
        error = "resource chunk session cursor is invalid";
        return false;
    }
    auto & evidence_session = execution.resource_evidence_sessions[execution.active_resource_chunk_plan];
    const auto & chunk_plan = evidence_session.plan;
    const size_t chunk_index = execution.active_resource_chunk_index;
    if (chunk_index >= chunk_plan.ranges.size()) {
        error = "resource chunk session cursor is exhausted";
        return false;
    }
    std::string plan_error;
    auto plan = execution.plan_store.get(execution.current_plan_id, plan_error);
    if (!plan) {
        error = plan_error.empty() ? "resource chunk completion requires an active plan" : plan_error;
        return false;
    }
    const auto chunk_ref = context.source;
    std::ostringstream summary;
    summary << "Resource evidence extraction status: ";
    if (extraction.unresolved) {
        summary << "unresolved; source coverage is incomplete.\n";
    } else if (extraction.no_findings) {
        summary << "no relevant findings.\n";
    } else {
        summary << "validated findings.\n";
    }
    for (const auto & finding : extraction.findings) {
        summary << "finding: " << finding.finding << "\nevidence: " << finding.evidence << "\n";
    }
    common_plan_operation observed;
    observed.kind = common_plan_operation_kind::record_observation;
    observed.plan_id = plan->id;
    observed.expected_version = plan->version;
    observed.reason_summary = "bounded resource chunk processed on the session lane";
    observed.observation = common_plan_observation{
        "resource_chunk:" + chunk_plan.parent.uri + ":" + std::to_string(chunk_index),
        extraction.unresolved ? "resource_chunk_unresolved" : "resource_chunk",
        summary.str(),
        1.0f,
        {chunk_plan.parent.uri},
        {chunk_ref},
        static_cast<int64_t>(std::time(nullptr)),
    };
    common_plan_state updated;
    if (!execution.plan_store.apply(observed, updated, error)) return false;
    *plan = std::move(updated);
    const auto transition = advance_agent_resource_evidence_session(
        evidence_session, !extraction.unresolved);
    if (transition == agent_resource_evidence_transition::invalid) {
        error = "resource evidence session could not advance its cursor";
        return false;
    }
    execution.pre_turn_events.push_back({
        common_agent_event_type::resource_chunk_processed,
        "bounded resource chunk processed chunk_index=" + std::to_string(chunk_index) +
            " chunk_count=" + std::to_string(chunk_plan.ranges.size()) +
            " extraction_status=" + (extraction.unresolved ? "unresolved" :
                extraction.no_findings ? "no_findings" : "findings") +
            (extraction.unresolved_reason.empty() ? "" :
                " extraction_error=" + extraction.unresolved_reason),
        {},
        plan->id,
        {},
        observed.observation->id,
        {},
        chunk_ref.uri,
    });
    if (transition == agent_resource_evidence_transition::complete) {
        chunk_chain_completed = true;
        execution.active_resource_chunk_plan = static_cast<size_t>(-1);
        execution.active_resource_chunk_input = static_cast<size_t>(-1);
        execution.active_resource_chunk_index = 0;
        execution.active_resource_chunk_context.reset();
        execution.input_resources.clear();
        for (const auto & input : execution.resource_chunk_original_inputs) {
            if (input.resource.uri != chunk_plan.parent.uri) execution.input_resources.push_back(input);
        }
        return false;
    }
    execution.active_resource_chunk_index = evidence_session.active_index;
    const auto next_ref = make_agent_resource_chunk_ref(
        chunk_plan, chunk_plan.ranges[execution.active_resource_chunk_index]);
    auto & active_input = execution.input_resources[execution.active_resource_chunk_input];
    active_input.resource = next_ref;
    active_input.role = "resource_chunk";
    active_input.required = true;
    return true;
}

void append_unique_strings(std::vector<std::string> & target, const std::vector<std::string> & values) {
    for (const auto & value : values) {
        if (std::find(target.begin(), target.end(), value) == target.end()) target.push_back(value);
    }
}

void append_runtime_result(common_agent_result & aggregate, const common_agent_result & slice) {
    if (!slice.response.empty()) {
        if (!aggregate.response.empty()) aggregate.response += "\n";
        aggregate.response += slice.response;
    }
    aggregate.total_decoded_tokens += slice.total_decoded_tokens;
    aggregate.response_decoded_tokens += slice.response_decoded_tokens;
    aggregate.reasoning_decoded_tokens += slice.reasoning_decoded_tokens;
    aggregate.response_generation_status = slice.response_generation_status;
    aggregate.response_stop_reason = slice.response_stop_reason;
    aggregate.plan_id = slice.plan_id;
    aggregate.plan_version = slice.plan_version;
    aggregate.reflected = aggregate.reflected || slice.reflected;
    aggregate.revised = aggregate.revised || slice.revised;
    aggregate.limit_reached = slice.limit_reached;
    aggregate.memory_learning_related_count = slice.memory_learning_related_count;
    aggregate.memory_learning_summary = slice.memory_learning_summary;
    aggregate.learned_memory_candidate = slice.learned_memory_candidate;
    aggregate.learning_signals.insert(
        aggregate.learning_signals.end(), slice.learning_signals.begin(), slice.learning_signals.end());
    append_unique_strings(aggregate.memory_ids, slice.memory_ids);
    aggregate.generation_records.insert(
        aggregate.generation_records.end(), slice.generation_records.begin(), slice.generation_records.end());
    aggregate.events.insert(aggregate.events.end(), slice.events.begin(), slice.events.end());
    aggregate.trace.insert(aggregate.trace.end(), slice.trace.begin(), slice.trace.end());
    aggregate.research_result = slice.research_result;
    aggregate.research_workspace_checkpoint = slice.research_workspace_checkpoint;
    aggregate.research_verification = slice.research_verification;
    aggregate.continuation_checkpoint = slice.continuation_checkpoint;
    aggregate.chunk_resource_request = slice.chunk_resource_request;
}

std::string make_continuation_prompt(
        const common_agent_result & slice,
        const common_plan_state & plan) {
    std::ostringstream prompt;
    prompt << "Continue the same bounded agent task from the existing plan.\n"
           << "Do not create a new plan, repeat completed work, or treat this as a new user request.\n"
           << "The previous generation reached its completion limit.\n"
           << "plan_id=" << plan.id << "\n"
           << "plan_version=" << plan.version << "\n";
    if (plan.active_step_id) prompt << "active_step_id=" << *plan.active_step_id << "\n";
    if (plan.next_action) prompt << "next_action=" << *plan.next_action << "\n";
    if (!slice.response.empty()) {
        constexpr size_t max_fragment_chars = 4096;
        const size_t offset = slice.response.size() > max_fragment_chars
            ? slice.response.size() - max_fragment_chars : 0;
        prompt << "Continue after this bounded previous output fragment:\n"
               << slice.response.substr(offset) << "\n";
    }
    prompt << "Produce only the next bounded result or the final answer.\n";
    return prompt.str();
}

bool slice_requires_context_continuation(const common_agent_result & slice) {
    return std::any_of(slice.trace.begin(), slice.trace.end(), [](const auto & entry) {
        return entry.detail.find("context pressure requires") != std::string::npos;
    });
}

bool make_continuation_checkpoint(
        const common_agent_runtime_driver_execution & execution,
        const common_agent_result & result,
        size_t sequence,
        common_agent_continuation_checkpoint & checkpoint,
        std::string & error) {
    if (result.response_stop_reason != common_agent_generation_stop_reason::limit ||
            !result.plan_id || execution.scope.turn_id.empty()) {
        return false;
    }
    const auto plan = execution.plan_store.get(*result.plan_id, error);
    if (!plan) {
        if (error.empty()) error = "continuation checkpoint requires an existing plan";
        return false;
    }
    if (!common_plan_chunk_observations_valid(plan->observations, error)) {
        return false;
    }
    checkpoint = {};
    checkpoint.checkpoint_id = "checkpoint:" + execution.scope.turn_id + ":" +
        plan->id + ":" + std::to_string(plan->version);
    checkpoint.request_id = "turn:" + execution.scope.turn_id;
    checkpoint.turn_id = execution.scope.turn_id;
    checkpoint.plan_id = plan->id;
    checkpoint.plan_version = plan->version;
    checkpoint.active_step_id = plan->active_step_id.value_or(std::string{});
    checkpoint.next_action = plan->next_action.value_or(std::string{});
    checkpoint.sequence = sequence;
    checkpoint.reason = common_agent_continuation_reason::completion_limit;
    const auto & context_budgets = execution.runtime_config.generation_config.context_budgets;
    checkpoint.working_state = make_common_agent_working_state(
        *plan, context_budgets.working_state);
    for (const auto & step : plan->steps) {
        if (step.status == common_plan_step_status::completed) checkpoint.completed_step_ids.push_back(step.id);
    }
    for (const auto & observation : plan->observations) {
        checkpoint.resource_refs.insert(
            checkpoint.resource_refs.end(), observation.resource_refs.begin(), observation.resource_refs.end());
        checkpoint.dataset_refs.insert(
            checkpoint.dataset_refs.end(), observation.dataset_refs.begin(), observation.dataset_refs.end());
        if (common_plan_observation_is_resource_chunk(observation) && observation.resource_refs.size() == 1) {
            const auto & lineage = observation.resource_refs.front().lineage;
            if (checkpoint.chunk_parent_uri.empty()) {
                checkpoint.chunk_parent_uri = lineage.parent_uri;
                checkpoint.chunk_count = lineage.chunk_count;
            }
            if (checkpoint.chunk_parent_uri == lineage.parent_uri &&
                    checkpoint.chunk_count == lineage.chunk_count &&
                    std::find(checkpoint.completed_chunk_indexes.begin(), checkpoint.completed_chunk_indexes.end(), lineage.chunk_index) == checkpoint.completed_chunk_indexes.end()) {
                checkpoint.completed_chunk_indexes.push_back(lineage.chunk_index);
            }
        }
    }
    if (!common_agent_continuation_checkpoint_valid(checkpoint, error)) return false;
    return true;
}

} // namespace

static void apply_explicit_deliberation_policy(
        const common_agent_deliberation_policy & policy,
        common_agent_request & request) {
    // Reflective remains the compatibility baseline for existing callers. The
    // deeper modes explicitly opt into the stronger runtime guarantees.
    if (policy.mode == common_agent_thinking_mode::reflective) {
        return;
    }

    request.enable_planning = true;
    request.enable_reflection = true;
    request.max_reflection_rounds = std::max<size_t>(
        request.max_reflection_rounds,
        static_cast<size_t>(std::max(0, policy.max_reflection_rounds)));
    request.max_iterations = std::max<size_t>(
        request.max_iterations,
        static_cast<size_t>(1 + std::max(0, policy.max_plan_revisions)));
    request.max_tool_batches = std::max<size_t>(
        request.max_tool_batches,
        static_cast<size_t>(std::max(0, policy.max_tool_rounds)));
}

common_agent_runtime_policy make_agent_runtime_policy(common_agent_runtime_policy_build_config options) {
    common_agent_runtime_policy policy;
    policy.deliberation_policy = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::reflective);
    policy.agent_inference_backend = std::move(options.agent_inference_backend);
    policy.tool_profile = std::move(options.tool_profile);
    policy.memory_learn = std::move(options.memory_learn);
    policy.memory_learn_show_candidate = options.memory_learn_show_candidate;
    policy.plan_show_summary = options.plan_show_summary;
    policy.agent_trace = options.agent_trace;
    policy.enable_reflection = true;
    policy.max_iterations = 2;
    policy.max_reflection_rounds = 1;
    // Zero in host/bootstrap configuration means "use the safe default", not
    // "disable every tool round". A caller that wants a smaller budget can
    // still provide an explicit positive value.
    policy.max_tool_rounds = options.max_tool_rounds > 0
        ? options.max_tool_rounds
        : 16;
    common_tool_profile_snapshot profile_snapshot;
    std::string profile_error;
    if (resolve_common_tool_profile_snapshot(
            policy.tool_profile,
            options.tool_capabilities,
            options.tool_profiles,
            profile_snapshot,
            profile_error)) {
        policy.allow_policy_gated_tool_proposals =
            profile_snapshot.allow_policy_gated_writes.value_or(false);
    }
    return policy;
}

common_agent_runtime_driver_execution make_agent_runtime_driver_execution(
    common_agent_runtime_driver_inputs & inputs,
    common_agent_inference & inference) {
    auto execution = common_agent_runtime_driver_execution{
        inputs.memory_store,
        inputs.plan_store,
        inference,
        inputs.policy,
        inputs.runtime_config,
        inputs.orchestration_config,
        inputs.current_plan_id,
        inputs.scope,
        inputs.installed_blueprint_candidates,
        inputs.policy_pack,
        inputs.memories,
        inputs.memory_scope,
        inputs.memory_enabled,
        inputs.tooling,
        {},
        {},
        false,
        inputs.input_resources,
        std::nullopt,
        inputs.research_should_stop,
        inputs.research_stop_reason,
        inputs.explicit_memory_candidate,
        inputs.explicit_memory_confirmed,
        {},
        {},
        {},
        false,
        {},
        false,
        {},
        inputs.execution_control,
    };
    execution.require_tool_execution = inputs.require_tool_execution;
    execution.tool_output_format = inputs.tool_output_format;
    execution.flydelta_activation = inputs.flydelta_activation;
    execution.tool_argument_bindings = inputs.tool_argument_bindings;
    execution.flydelta_capture = inputs.flydelta_capture;
    return execution;
}

common_agent_request make_agent_runtime_driver_request(
    const common_agent_runtime_driver_execution & execution) {
    common_agent_request request;
    request.memories = execution.memories;
    request.memories.insert(request.memories.end(), execution.route_procedure_memories.begin(),
        execution.route_procedure_memories.end());
    request.enable_memory = execution.memory_enabled;
    request.enable_planning = true;
    request.enable_reflection = execution.policy.enable_reflection;
    request.memory_scope = execution.memory_scope;
    request.plan_scope = execution.scope.plan_scope;
    request.prompt = execution.orchestration_config.prompt;
    request.input_resources = execution.input_resources;
    request.available_resources = execution.available_resources;
    request.available_datasets = execution.tooling.available_datasets;
    request.working_state = execution.compact_working_state;
    if (!execution.current_plan_id.empty()) {
        request.plan_id = execution.current_plan_id;
    }
    request.policy_pack = execution.policy_pack;
    common_agent_scope_apply(execution.scope, request);
    request.max_iterations = execution.policy.max_iterations;
    request.max_reflection_rounds = execution.policy.max_reflection_rounds;
    request.max_tool_batches = execution.tooling.profile_tools_active ? execution.policy.max_tool_rounds : 0;
    request.require_tool_execution = execution.require_tool_execution;
    request.tool_output_format = execution.tool_output_format;
    request.allow_policy_gated_tool_proposals = execution.policy.allow_policy_gated_tool_proposals;
    request.deliberation_policy = execution.policy.deliberation_policy;
    request.cognitive_state = execution.cognitive_state;
    request.research_should_stop = execution.research_should_stop;
    request.research_stop_reason = execution.research_stop_reason;
    request.explicit_memory_candidate = execution.explicit_memory_candidate;
    request.explicit_memory_confirmed = execution.explicit_memory_confirmed;
    request.flydelta_activation = execution.flydelta_activation;
    request.flydelta_capture = execution.flydelta_capture;
    request.tool_argument_bindings = execution.tool_argument_bindings;
    if (execution.route_selected &&
            execution.selected_route.kind != common_agent_route_kind::normal_plan &&
            execution.execution_envelope_ready) {
        common_plan_route_binding binding;
        binding.route_id = execution.selected_route.id;
        binding.blueprint_ref = execution.selected_route.blueprint_logical_id;
        binding.blueprint_revision = execution.selected_route.blueprint_revision;
        binding.graph_revision = execution.selected_route.graph_revision;
        binding.execution_envelope_fingerprint = execution.execution_envelope.fingerprint;
        binding.policy_revision = execution.execution_envelope.policy_revision;
        if (execution.selected_route.workflow) {
            binding.workflow_ref = execution.selected_route.workflow->workflow_ref;
            binding.workflow_revision = execution.selected_route.workflow->workflow_revision;
            request.selected_workflow = execution.selected_route.workflow;
        }
        request.route_binding = std::move(binding);
        if (!execution.selected_route.blueprint_persisted_id.empty()) {
            request.blueprint_plan_id = execution.selected_route.blueprint_persisted_id;
        }
        request.workflow_definition = execution.selected_route.workflow_definition;
    }
    if (execution.route_selected && execution.selected_route.workflow &&
            execution.selected_route.workflow->workflow_ref ==
                "workflow://resource/document-analysis") {
        request.workflow_continuation = make_agent_resource_document_workflow_continuation(
            execution.selected_route, execution.tool_argument_bindings);
    }
    // A graph with a declared tool transition is an executable host contract,
    // not an optional planning hint.  Preserve that requirement even when a
    // caller reached the route through a compatibility entry point that did
    // not carry --require-tool-execution into the driver state.
    if (execution.selected_route.workflow_definition &&
            std::any_of(execution.selected_route.workflow_definition->transitions.begin(),
                execution.selected_route.workflow_definition->transitions.end(),
                [](const common_plan_workflow_transition & transition) {
                    return transition.kind == common_plan_workflow_transition_kind::tool;
                })) {
        request.require_tool_execution = true;
    }
    apply_explicit_deliberation_policy(request.deliberation_policy, request);
    return request;
}

bool run_agent_runtime_driver_session(
    common_agent_runtime_driver_inputs & inputs,
    common_agent_runtime_session & session,
    common_agent_result & result,
    std::string & error) {
    agent_inference_backend inference_backend_kind = agent_inference_backend::cli;
    if (!parse_agent_inference_backend(inputs.policy.agent_inference_backend, inference_backend_kind)) {
        error = "unsupported --agent-inference-backend: " + inputs.policy.agent_inference_backend;
        return false;
    }

    if (!initialize_agent_runtime_session(
            inputs.inference_options,
            inference_backend_kind,
            inputs.memory_enabled,
            inputs.fallback_reason,
            session,
            error)) {
        return false;
    }

    auto * inference_session = session.active_inference_session();
    if (inference_session == nullptr || !inference_session->inference) {
        error = "runtime session failed to initialize an inference context";
        return false;
    }

    auto execution = make_agent_runtime_driver_execution(inputs, *inference_session->inference);
    return run_agent_runtime_driver(execution, result, error);
}

bool run_agent_runtime_driver(
        common_agent_runtime_driver_execution & execution,
        common_agent_result & result,
        std::string & error) {
    execution.pre_turn_events.clear();
    execution.pre_turn_trace.clear();
    execution.pre_turn_trace.push_back({
        common_runtime_trace_stage::plan,
        common_runtime_trace_kind::started,
        nlohmann::ordered_json{
            {"type", "cognitive_mode_initialized"},
            {"mode", common_agent_cognitive_mode_name(execution.cognitive_state.mode)},
            {"reason", execution.cognitive_state.reason},
            {"source", "host"},
        }.dump(),
        execution.current_plan_id,
        {}, {}, {}, {}});
    if (execution.tooling.profile_tools_active && execution.tooling.tool_view == nullptr) {
        error = "profile tool execution requires a resolved tool view";
        return false;
    }

    const common_agent_orchestration_runtime_context orchestration_context{
        execution.inference,
        execution.runtime_config.generation_config,
        execution.orchestration_config,
        execution.memory_store,
        execution.current_plan_id,
        execution.scope,
        execution.plan_store,
        execution.installed_blueprint_candidates,
        &execution.policy_pack,
        &execution.tooling,
        execution.selected_route,
        execution.route_selected,
        execution.route_procedure_memories,
        execution.pre_turn_events,
        execution.pre_turn_trace,
        execution.tool_output_format,
        &execution.tool_argument_bindings,
    };

    const std::string original_prompt = execution.orchestration_config.prompt;
    common_agent_result aggregate;
    size_t continuation_count = 0;
    const auto preserve_partial_result = [&]() {
        execution.orchestration_config.prompt = original_prompt;
        result = std::move(aggregate);
        result.events.insert(result.events.begin(), execution.pre_turn_events.begin(), execution.pre_turn_events.end());
        result.trace.insert(result.trace.begin(), execution.pre_turn_trace.begin(), execution.pre_turn_trace.end());
    };
    const auto fail_with_partial_result = [&](const std::string & failure) {
        preserve_partial_result();
        result.error = failure;
        error = failure;
        return false;
    };
    const auto stop_for_execution_control = [&]() {
        preserve_partial_result();
        result.response_generation_status = common_agent_generation_status::cancelled;
        result.response_stop_reason = common_agent_generation_stop_reason::cancelled;
        result.error = execution.execution_control.stop_reason();
        error = result.error.empty() ? "agent runtime execution stopped" : result.error;
        return false;
    };

    prepare_available_resources(execution);
    if (execution.execution_control.should_stop()) {
        return stop_for_execution_control();
    }
    if (!maybe_auto_select_plan(orchestration_context, error)) {
        return false;
    }
    if (execution.execution_control.should_stop()) {
        return stop_for_execution_control();
    }

    if (!restore_and_validate_bound_route(execution, error)) return false;

    if (!maybe_select_agent_route(orchestration_context, execution.selected_route,
            execution.route_selected, error)) {
        return false;
    }

    if (execution.route_selected) {
        if (!build_agent_execution_envelope(
                execution.selected_route, execution.tooling,
                execution.execution_envelope, error)) {
            error = "execution envelope construction failed: " + error;
            return false;
        }
        execution.execution_envelope_ready = true;
    }

    if (execution.route_selected &&
            execution.selected_route.kind == common_agent_route_kind::blueprint_workflow) {
        execution.model_tools.clear();
        for (const auto & tool : execution.tooling.tools) {
            if (std::find(execution.selected_route.resolved_tools.begin(),
                    execution.selected_route.resolved_tools.end(), tool.name) !=
                    execution.selected_route.resolved_tools.end()) {
                execution.model_tools.push_back(tool);
            }
        }
    }

    if (!execution.route_selected ||
            execution.selected_route.kind == common_agent_route_kind::normal_plan) {
        const common_agent_request routed_request = make_agent_runtime_driver_request(execution);
        if (!select_model_tool_families(execution, routed_request, error)) {
            return false;
        }
        if (execution.family_chat_routed) {
            result = std::move(execution.family_chat_result);
            error.clear();
            return true;
        }
    }

    if (!prepare_resource_chunk_observations(execution, error)) {
        return false;
    }

    while (true) {
        if (execution.execution_control.should_stop()) return stop_for_execution_control();
        if (execution.active_resource_chunk_plan != static_cast<size_t>(-1)) {
            if (continuation_count >= execution.runtime_config.max_continuations) {
                aggregate.limit_reached = true;
                aggregate.response_stop_reason = common_agent_generation_stop_reason::limit;
                result = std::move(aggregate);
                break;
            }
            common_agent_resource_chunk_context context;
            common_agent_resource_chunk_extraction extraction;
            if (!extract_active_resource_chunk(
                    execution, original_prompt, context, extraction, error)) {
                return fail_with_partial_result(error);
            }
            execution.active_resource_chunk_context = context;
            bool chunk_chain_completed = false;
            const bool has_next_chunk = record_and_advance_resource_chunk(
                execution, context, extraction, chunk_chain_completed, error);
            if (!error.empty()) return fail_with_partial_result(error);
            ++continuation_count;
            if (has_next_chunk) {
                execution.orchestration_config.prompt =
                    "Continue the same task. The host is processing the next bounded resource slice; "
                    "do not ask the model to read or locate it.\n";
                continue;
            }
            if (chunk_chain_completed) {
                if (continuation_count >= execution.runtime_config.max_continuations) {
                    aggregate.limit_reached = true;
                    aggregate.response_stop_reason = common_agent_generation_stop_reason::limit;
                    result = std::move(aggregate);
                    break;
                }
                execution.orchestration_config.prompt =
                    "Synthesize the validated resource-chunk findings from the active plan in chunk order. "
                    "Use the exact evidence and source provenance recorded by the host. The source text has "
                    "already been read and processed; do not claim it still needs to be fetched or read. "
                    "Answer the original user task below directly; do not substitute generic workflow or acquisition status. "
                    "State uncertainty or incomplete coverage if source coverage is incomplete.\n"
                    "Original user task:\n" + original_prompt + "\n";
                std::string coverage_error;
                const auto active_plan = execution.plan_store.get(
                    execution.current_plan_id, coverage_error);
                if (active_plan) {
                    common_plan_chunk_synthesis_input coverage;
                    if (common_plan_chunk_synthesis_from_observations(
                            active_plan->observations, {}, coverage, coverage_error) &&
                            !coverage.missing_chunk_indexes.empty()) {
                        execution.orchestration_config.prompt +=
                            "Host coverage status: incomplete. Give a partial answer from verified evidence "
                            "and state that some source text could not be assessed. The resource itself was "
                            "fetched and made readable; do not claim it is unavailable or was not accessed. "
                            "Do not infer missing details.\n";
                    }
                }
                continue;
            }
        }
        const auto & model_tools = execution.model_tools.empty()
            ? execution.tooling.tools
            : execution.model_tools;
        common_agent_request request = make_agent_runtime_driver_request(execution);
        if (execution.route_selected && execution.selected_route.workflow_definition &&
                !execution.selected_route.workflow_definition->transitions.empty()) {
            const auto route = execution.selected_route;
            const auto choice_request = request;
            request.workflow_continuation = make_agent_workflow_graph_continuation(route,
                [&execution, choice_request](const common_plan_observation & observation,
                        const common_plan_workflow_transition & transition,
                        std::string & candidate_id, std::string & choice_error) {
                    return select_llama_cli_workflow_candidate(execution.inference,
                        execution.runtime_config.generation_config, choice_request,
                        observation, transition, candidate_id, choice_error);
                });
        }
        auto assembly = make_agent_runtime_assembly(
            execution.memory_store,
            execution.plan_store,
            execution.inference,
            execution.runtime_config,
            model_tools,
            execution.tooling.tool_view,
            execution.execution_envelope_ready ? &execution.execution_envelope : nullptr,
            execution.policy.deliberation_policy.mode == common_agent_thinking_mode::research
                ? common_agent_execution_phase::research
                : execution.policy.deliberation_policy.mode == common_agent_thinking_mode::deliberate
                    ? common_agent_execution_phase::deliberate
                    : common_agent_execution_phase::normal);

        const auto slice = assembly.runtime->run(request);
        if (!slice.error.empty()) {
            append_runtime_result(aggregate, slice);
            if (execution.policy.agent_trace) {
                for (const auto & failure : slice.failures) {
                    fprintf(stderr,
                        "agent: failure code=%s class=%s stage=%s step=%s tool=%s evidence=%s retryable=%s summary=%s repair_context=%s\n",
                        failure.code.c_str(),
                        common_agent_failure_class_name(failure.classification),
                        failure.stage.c_str(),
                        failure.step_id.c_str(),
                        failure.tool_name.c_str(),
                        failure.evidence_id.c_str(),
                        failure.retryable ? "true" : "false",
                        failure.safe_summary.c_str(),
                        failure.repair_context_json.c_str());
                }
            }
            return fail_with_partial_result("agent runtime failed: " + slice.error);
        }
        append_runtime_result(aggregate, slice);
        if (execution.execution_control.should_stop()) return stop_for_execution_control();
        if (slice.chunk_resource_request &&
                execution.active_resource_chunk_plan != static_cast<size_t>(-1)) {
            const auto & current = execution.resource_evidence_sessions[
                execution.active_resource_chunk_plan].plan;
            if (current.parent.uri != slice.chunk_resource_request->uri) {
                return fail_with_partial_result(
                    "workflow requested a second chunked resource while another resource is active");
            }
        }
        if (slice.chunk_resource_request &&
                execution.active_resource_chunk_plan == static_cast<size_t>(-1)) {
            const auto & resource = *slice.chunk_resource_request;
            std::string plan_error;
            const auto plan_id = slice.plan_id.value_or(execution.current_plan_id);
            const auto active_plan = execution.plan_store.get(plan_id, plan_error);
            if (!active_plan) {
                return fail_with_partial_result(
                    plan_error.empty() ? "workflow chunk handoff requires an active plan" : plan_error);
            }
            execution.current_plan_id = active_plan->id;
            for (const auto & observation : active_plan->observations) {
                if (!common_plan_observation_is_resource_chunk(observation) || observation.resource_refs.size() != 1) continue;
                if (observation.resource_refs.front().lineage.parent_uri != resource.uri) {
                    return fail_with_partial_result(
                        "workflow requested a second chunked resource in one plan; start a new plan for " + resource.uri);
                }
            }
            if (execution.active_resource_chunk_plan != static_cast<size_t>(-1)) {
                const auto & current = execution.resource_evidence_sessions[
                    execution.active_resource_chunk_plan].plan;
                if (current.parent.uri != resource.uri) {
                    return fail_with_partial_result(
                        "workflow requested a second chunked resource while another resource is active");
                }
            } else {
                execution.input_resources.push_back({resource, "workflow_full_text", true});
                execution.resource_chunk_observations_prepared = false;
                if (!prepare_resource_chunk_observations(execution, error)) {
                    return fail_with_partial_result(error);
                }
            }
            if (continuation_count >= execution.runtime_config.max_continuations) {
                aggregate.limit_reached = true;
                aggregate.response_stop_reason = common_agent_generation_stop_reason::limit;
                result = std::move(aggregate);
                break;
            }
            execution.orchestration_config.prompt =
                execution.active_resource_chunk_plan != static_cast<size_t>(-1)
                    ? "Continue the same workflow using the next bounded resource slice.\n"
                      "Treat each slice as evidence; do not produce the final answer until all slices are processed.\n"
                    : "Continue the same workflow from the completed bounded resource read.\n"
                      "Use the recorded resource observation and continue the existing plan.\n";
            ++continuation_count;
            continue;
        }
        if (slice.response_stop_reason != common_agent_generation_stop_reason::limit ||
                !slice.plan_id ||
                continuation_count >= execution.runtime_config.max_continuations) {
            result = std::move(aggregate);
            break;
        }

        std::string plan_error;
        const auto plan = execution.plan_store.get(*slice.plan_id, plan_error);
        if (!plan) {
            return fail_with_partial_result(plan_error.empty()
                ? "continuation could not resolve the current plan"
                : plan_error);
        }
        execution.current_plan_id = plan->id;
        if (slice_requires_context_continuation(slice)) {
            common_agent_context_compaction_limits compaction_limits;
            compaction_limits.working_state =
                execution.runtime_config.generation_config.context_budgets.working_state;
            compaction_limits.max_input_resources =
                compaction_limits.working_state.max_resource_refs;
            const auto compacted = compact_common_agent_context(
                *plan, execution.policy_pack, execution.input_resources, compaction_limits);
            execution.compact_working_state = compacted.working_state;
            execution.policy_pack = compacted.policy_pack;
            execution.input_resources = compacted.input_resources;
            execution.orchestration_config.prompt =
                "Continue the same task from the bounded working state.\n"
                "The plan and resource stores remain authoritative; do not recreate completed work.\n"
                "Produce only the next bounded result or the final answer.\n";
        } else {
            execution.orchestration_config.prompt = make_continuation_prompt(slice, *plan);
        }
        ++continuation_count;
    }

    execution.orchestration_config.prompt = original_prompt;
    if (result.response_stop_reason == common_agent_generation_stop_reason::limit &&
            !result.continuation_checkpoint) {
        common_agent_continuation_checkpoint checkpoint;
        std::string checkpoint_error;
        if (make_continuation_checkpoint(
                execution, result, continuation_count + 1, checkpoint, checkpoint_error)) {
            result.continuation_checkpoint = std::move(checkpoint);
        } else if (!checkpoint_error.empty()) {
            return fail_with_partial_result(checkpoint_error);
        }
    }
    result.events.insert(result.events.begin(), execution.pre_turn_events.begin(), execution.pre_turn_events.end());
    result.trace.insert(result.trace.begin(), execution.pre_turn_trace.begin(), execution.pre_turn_trace.end());

    if (execution.policy.memory_learn == "post-turn") {
        const auto * candidate = result.learned_memory_candidate ? &*result.learned_memory_candidate : nullptr;
        fprintf(stderr, "audit: memory_learn summary=%s plan=%s candidate=%s confidence=%.2f reuse=%.2f related=%zu\n",
            result.memory_learning_summary.c_str(), result.plan_id ? result.plan_id->c_str() : "",
            candidate ? common_memory_kind_name(candidate->kind) : "none", candidate ? candidate->confidence : 0.0f,
            candidate ? candidate->expected_reuse : 0.0f, result.memory_learning_related_count);
        if (execution.policy.memory_learn_show_candidate && candidate) {
            fprintf(stderr, "memory_learn candidate: kind=%s content=%s rationale=%s\n",
                common_memory_kind_name(candidate->kind), candidate->content.c_str(), candidate->rationale.c_str());
        }
    }

    if (execution.policy.plan_show_summary && result.plan_id) {
        std::string plan_error;
        const auto plan = execution.plan_store.get(*result.plan_id, plan_error);
        if (plan) {
            fprintf(stderr, "plan: id=%s version=%llu steps=%zu observations=%zu reflected=%s revised=%s\n",
                plan->id.c_str(), (unsigned long long) plan->version, plan->steps.size(), plan->observations.size(),
                result.reflected ? "yes" : "no", result.revised ? "yes" : "no");
        }
    }

    if (execution.policy.agent_trace) {
        for (const auto & entry : result.trace) {
            fprintf(stderr, "agent: trace stage=%s kind=%s plan=%s step=%s tool=%s observation=%s detail=%s\n",
                common_runtime_trace_stage_name(entry.stage),
                common_runtime_trace_kind_name(entry.kind),
                entry.plan_id.c_str(),
                entry.step_id.c_str(),
                entry.tool_name.c_str(),
                entry.observation_id.c_str(),
                entry.detail.c_str());
        }
        for (const auto & event : result.events) {
            fprintf(stderr, "agent: event=%d plan=%s detail=%s\n", (int) event.type,
                event.plan_id ? event.plan_id->c_str() : "", event.detail.c_str());
        }
    }

    error.clear();
    return true;
}
