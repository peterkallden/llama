#include "agent-runtime-execution.h"
#include "common/agent/resource-chunk-extraction.h"
#include "agent/context-compaction.h"
#include "agent/input-resources.h"
#include "plan/plan-workflow-state.h"
#include "agent/tool-family-index.h"
#include "../runtime/agent-runtime-chat-driver.h"
#include "../runtime/agent-route-compiler.h"
#include "agent-dataset-workflow-adapter.h"
#include "agent-resource-workflow-continuation.h"
#include "agent-workflow-transition.h"
#include "agent/thinking/research/research-workspace.h"

#include "../runtime/agent-plan-orchestration.h"
#include "../runtime/agent-runtime-assembly.h"
#include "../cli/agent-cli-selection.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>
#include <ctime>
#include <cstdint>

namespace {

using json = nlohmann::ordered_json;

void load_route_procedure_memories(
        common_agent_runtime_driver_execution & execution,
        const std::vector<std::string> & procedure_refs,
        const std::string & provenance) {
    execution.route_procedure_memories.clear();
    if (procedure_refs.empty()) return;
    common_memory_query query;
    query.kind = common_memory_kind::procedure;
    query.scope = execution.scope.project_id.empty()
        ? common_memory_scope::global : common_memory_scope::project;
    query.namespace_id = execution.scope.namespace_id;
    query.session_id = execution.scope.session_id;
    query.project_id = execution.scope.project_id;
    query.turn_id = execution.scope.turn_id;
    query.global_opt_in = execution.scope.project_id.empty();
    query.limit = 64;
    std::string procedure_error;
    const auto procedures = execution.memory_store.list(query, procedure_error);
    if (!procedure_error.empty()) return;
    for (const auto & procedure_ref : procedure_refs) {
        const auto found = std::find_if(procedures.begin(), procedures.end(),
            [&](const auto & procedure) {
                return procedure.id == procedure_ref ||
                    (procedure.id.size() > procedure_ref.size() + 1 &&
                        procedure.id.compare(procedure.id.size() - procedure_ref.size(),
                            procedure_ref.size(), procedure_ref) == 0 &&
                        procedure.id[procedure.id.size() - procedure_ref.size() - 1] == ':');
            });
        if (found == procedures.end()) continue;
        common_memory_hit hit;
        hit.memory = *found;
        hit.provenance = provenance;
        hit.final_score = 1.0f;
        execution.route_procedure_memories.push_back(std::move(hit));
    }
}

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

    // Dataset references are durable plan observations; reconstruct the
    // transient route context when a host resumes after a workflow handoff.
    for (const auto & observation : plan->observations) {
        for (const auto & ref : observation.dataset_refs) {
            if (ref.uri.empty() || std::any_of(execution.continuation_datasets.begin(),
                    execution.continuation_datasets.end(), [&](const auto & dataset) {
                        return dataset.ref.uri == ref.uri;
                    })) continue;
            common_agent_dataset_descriptor dataset;
            dataset.ref = ref;
            dataset.origin.kind = "workflow_continuation";
            execution.continuation_datasets.push_back(std::move(dataset));
        }
    }

    common_agent_route_candidate route;
    route.id = plan->route_binding->route_id;
    route.blueprint_logical_id = plan->route_binding->blueprint_ref;
    route.blueprint_revision = plan->route_binding->blueprint_revision;
    route.graph_revision = plan->route_binding->graph_revision;
    route.required_capabilities = plan->required_capabilities;
    route.procedure_refs = plan->procedure_refs;
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
            const bool dataset_context_available = requirement == "context.dataset.available" &&
                (!execution.continuation_datasets.empty() ||
                    !execution.tooling.available_datasets.empty());
            const bool multiple_dataset_context_available =
                requirement == "context.dataset.multiple_available" && [&]() {
                    std::set<std::string> refs;
                    for (const auto & dataset : execution.tooling.available_datasets) {
                        if (!dataset.ref.uri.empty()) refs.insert(dataset.ref.uri);
                    }
                    for (const auto & dataset : execution.continuation_datasets) {
                        if (!dataset.ref.uri.empty()) refs.insert(dataset.ref.uri);
                    }
                    return refs.size() >= 2;
                }();
            if (!resource_context_available && !dataset_context_available &&
                    !multiple_dataset_context_available &&
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
    load_route_procedure_memories(execution, execution.selected_route.procedure_refs,
        "resumed workflow procedure ref");
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

    // Attachment orientation is a host-authored, read-only task. The daemon
    // has already restricted its tool catalog to resource/dataset/document
    // inspection, so asking the model to classify those same families adds a
    // fragile generation before the actual read without adding authorization
    // value. Select only those families that are present in the restricted
    // catalog and let the normal planner choose the concrete read operation.
    const bool attachment_orientation = request.resource_read_binding &&
        request.resource_read_binding->intent == common_agent_resource_read_intent::orientation;
    if (attachment_orientation) {
        const auto families = common_generate_tool_family_index(
            execution.tooling.tools, execution.tooling.family_descriptions);
        std::vector<std::string> family_ids;
        for (const auto & family : families) {
            if (family.id == "resource" || family.id == "dataset" ||
                    family.id == "document") {
                family_ids.push_back(family.id);
            }
        }
        execution.model_tools = common_expand_tool_family_prerequisites(
            execution.tooling.tools, family_ids);
        if (execution.model_tools.empty()) {
            error = "attachment orientation has no permitted resource inspection tools";
            return false;
        }
        execution.require_tool_execution = true;
        return true;
    }

    if (!generation_config.enable_tool_family_routing ||
            execution.tooling.tools.empty() ||
            (!execution.current_plan_id.empty() && !request.resource_read_binding)) {
        return true;
    }

    const auto families = common_generate_tool_family_index(
        execution.tooling.tools, execution.tooling.family_descriptions);
    std::vector<std::string> required_attachment_families;
    const bool bound_resource_read = request.resource_read_binding &&
        request.resource_read_binding->require_content_read;
    for (const auto & input : request.input_resources) {
        if (!bound_resource_read && input.role != "attachment_question" && input.role != "document_context") continue;
        required_attachment_families.push_back("resource");
        const auto & mime = input.resource.mime_type;
        if (mime == "text/csv" || mime == "text/tab-separated-values") {
            required_attachment_families.push_back("dataset");
        }
        if (mime == "text/html" ||
                mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
                mime == "application/epub+zip") {
            required_attachment_families.push_back("document");
        }
    }
    std::sort(required_attachment_families.begin(), required_attachment_families.end());
    required_attachment_families.erase(
        std::unique(required_attachment_families.begin(), required_attachment_families.end()),
        required_attachment_families.end());
    if (bound_resource_read) {
        for (const auto & family : families) {
            if ((family.id == "resource" || family.id == "dataset" || family.id == "document") &&
                    std::find(required_attachment_families.begin(), required_attachment_families.end(), family.id) ==
                        required_attachment_families.end()) {
                required_attachment_families.push_back(family.id);
            }
        }
        std::sort(required_attachment_families.begin(), required_attachment_families.end());
    }
    const std::string family_view = common_render_tool_family_index(families, 2048);
    const bool required_family_selection = caller_requires_tool_execution;
    common_chat_msg system{
        "system",
        required_family_selection
            ? common_tool_family_required_selection_prompt(families)
            : "Decide whether the user request needs external tools. Reply with exactly one line: "
              "NO_TOOLS for ordinary conversation, or TOOLS: family_id[, family_id...] when tools "
              "are needed. If the user asks about text already included in the prompt, including "
              "host-validated document context, answer from that text and return NO_TOOLS. Do not "
              "select resource or document only because the user mentions an attached file. Select "
              "those families only when more content must be fetched or extracted; select web when "
              "external research is requested, and document tools when editing is requested. "
              "For other requests, attached files are optional inputs: choose resource or dataset "
              "when their contents must be read or analyzed, and keep NO_TOOLS for unrelated conversation. "
              "Return family ids, not tool names; for example, "
              "dataset.validate belongs to the dataset family. Do not explain, use JSON, or invent family ids. Available "
              "families:\n" + family_view,
    };
    common_chat_msg user{"user", common_tool_family_selection_user_prompt(request.prompt) +
        common_agent_render_input_resource_context(
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
    // A caller-bound attachment question must actually inspect its resource;
    // a model-selected web or unrelated tool alone is not grounded. Add the
    // host-required families after model classification so the model may also
    // select web/research or other capabilities requested by the user.
    for (const auto & required_family : required_attachment_families) {
        const bool family_available = std::any_of(
            families.begin(), families.end(), [&required_family](const auto & family) {
                return family.id == required_family;
            });
        if (!family_available) {
            error = "attached resource requires unavailable tool family: " + required_family;
            return false;
        }
        if (std::find(selection.family_ids.begin(), selection.family_ids.end(), required_family) ==
                selection.family_ids.end()) {
            selection.family_ids.push_back(required_family);
        }
        selection.needs_tools = true;
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
        if (descriptor.metadata.purpose.rfind("agent.context.", 0) == 0) continue;
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
    if (!execution.resource_read_binding ||
            !execution.resource_read_binding->require_content_read ||
            !execution.resource_read_binding->require_full_coverage) {
        // Questions use the ordinary resource.read path, which can choose a
        // bounded range based on the task. Eagerly extracting chunk zero here
        // would both waste context and make a partial read look task-relevant.
        // Unbound session attachments are inventory only, not an implicit
        // instruction to read every file on every turn.
        return true;
    }
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
            std::string target_fingerprint;
            if (execution.resource_read_binding) {
                const auto target = std::find_if(execution.resource_read_binding->targets.begin(),
                    execution.resource_read_binding->targets.end(), [&](const auto & candidate) {
                        return candidate.uri == input.resource.uri || candidate.parent_uri == input.resource.uri;
                    });
                if (target != execution.resource_read_binding->targets.end()) target_fingerprint = target->content_fingerprint;
            }
            if (!common_runtime_resource_lineage_is_chunk(lineage) ||
                    !execution.resource_read_binding || !observation.resource_read_provenance ||
                    observation.resource_read_provenance->task_fingerprint != execution.resource_read_binding->task_fingerprint ||
                    target_fingerprint.empty() ||
                    observation.resource_read_provenance->source_fingerprint != target_fingerprint) continue;
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

        const std::string plan_observation_id = "resource_chunk_plan:" + input.resource.uri + ":" +
            (execution.resource_read_binding ? execution.resource_read_binding->task_fingerprint.substr(0, 16) : "unbound");
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
        "resource_chunk:" + chunk_plan.parent.uri + ":" +
            (execution.resource_read_binding ? execution.resource_read_binding->task_fingerprint.substr(0, 16) : "unbound") +
            ":" + std::to_string(chunk_index),
        extraction.unresolved ? "resource_chunk_unresolved" : "resource_chunk",
        summary.str(),
        1.0f,
        {chunk_plan.parent.uri},
        {chunk_ref},
        static_cast<int64_t>(std::time(nullptr)),
    };
    if (execution.resource_read_binding) {
        const auto target = std::find_if(execution.resource_read_binding->targets.begin(),
            execution.resource_read_binding->targets.end(), [&](const auto & candidate) {
                return candidate.uri == chunk_plan.parent.uri || candidate.parent_uri == chunk_plan.parent.uri;
            });
        if (target != execution.resource_read_binding->targets.end()) {
            observed.observation->resource_read_provenance = common_plan_resource_read_provenance{
                execution.resource_read_binding->task_fingerprint,
                common_agent_resource_read_intent_name(execution.resource_read_binding->intent),
                target->uri,
                target->content_fingerprint,
                target->representation,
                target->parent_uri};
        }
    }
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
        execution.resource_chunk_observations_prepared = false;
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
        inputs.resource_read_binding,
        {},
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

static void apply_workflow_thinking_preference(common_agent_runtime_driver_execution & execution) {
    if (!execution.selected_route.workflow_definition) return;
    const auto & workflow = *execution.selected_route.workflow_definition;
    if (workflow.preferred_thinking_mode.empty()) return;
    auto decision = resolve_common_agent_workflow_thinking_preference(
        execution.policy.deliberation_policy, workflow.preferred_thinking_mode);
    if (decision.allowed && (!execution.policy.enable_reflection || execution.policy.max_reflection_rounds == 0)) {
        decision.allowed = false;
        decision.summary = "workflow thinking preference denied: host step-review budget unavailable";
    }
    if (decision.allowed) {
        auto next = make_common_agent_escalated_policy(execution.policy.deliberation_policy, decision.to_mode);
        next.max_tool_rounds = execution.policy.deliberation_policy.max_tool_rounds;
        next.max_plan_revisions = execution.policy.deliberation_policy.max_plan_revisions;
        next.max_reflection_rounds = std::min(next.max_reflection_rounds,
            static_cast<int>(execution.policy.max_reflection_rounds));
        execution.policy.deliberation_policy = next;
        execution.workflow_preference_applied = true;
    }
    execution.pre_turn_trace.push_back({common_runtime_trace_stage::plan, common_runtime_trace_kind::decided,
        decision.summary + " workflow=" + workflow.workflow_ref + " preferred=" + workflow.preferred_thinking_mode +
        " resolved=" + common_agent_thinking_mode_name(execution.policy.deliberation_policy.mode)});
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
    request.resource_read_binding = execution.resource_read_binding;
    request.available_resources = execution.available_resources;
    request.available_datasets = execution.tooling.available_datasets;
    request.grounding_dataset_reader = execution.tooling.grounding_dataset_reader;
    for (const auto & dataset : execution.continuation_datasets) {
        if (std::none_of(request.available_datasets.begin(), request.available_datasets.end(),
                [&](const auto & existing) { return existing.ref.uri == dataset.ref.uri; })) {
            request.available_datasets.push_back(dataset);
        }
    }
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
    if (!execution.workflow_preference_applied) apply_explicit_deliberation_policy(request.deliberation_policy, request);
    return request;
}

static std::string continuation_fingerprint(const common_plan_observation & observation,
        const std::vector<std::string> & supporting_values = {}) {
    uint64_t hash = 1469598103934665603ULL;
    const auto add = [&](const std::string & text) {
        for (const unsigned char ch : text) { hash ^= ch; hash *= 1099511628211ULL; }
        hash ^= 0xff; hash *= 1099511628211ULL;
    };
    add(observation.summary);
    for (const auto & ref : observation.dataset_refs) { add(ref.uri); add(ref.content_hash); }
    for (const auto & ref : observation.resource_refs) {
        add(ref.uri);
        add(ref.mime_type);
        add(ref.metadata.processing_cache_key);
        add(std::to_string(ref.size_bytes));
    }
    for (const auto & value : supporting_values) add(value);
    return std::to_string(hash);
}

static std::string continuation_link_identity(const common_plan_workflow_continuation_link & link) {
    return link.source_transition_id + "@" + link.source_terminal_state + "->" +
        link.target_workflow_ref + "@" + link.target_workflow_revision;
}

static std::string workflow_step_transition_id(const common_plan_step & step) {
    if (!step.semantic_alias || step.semantic_alias->rfind("workflow-transition:", 0) != 0) return {};
    auto aliases = step.semantic_alias->substr(std::string("workflow-transition:").size());
    const auto comma = aliases.rfind(',');
    if (comma != std::string::npos) aliases = aliases.substr(comma + 1);
    const auto equals = aliases.find('=');
    if (equals != std::string::npos) aliases.resize(equals);
    return aliases;
}

static const common_plan_observation * successful_workflow_observation(
        const common_plan_state & plan,
        const std::string & source_transition_id,
        bool require_dataset) {
    return common_plan_workflow_observation(plan, source_transition_id, require_dataset);
}

static bool transition_to_workflow(
        common_agent_runtime_driver_execution & execution,
        const common_plan_state & plan,
        const std::string & original_prompt,
        std::string & error) {
    error.clear();
    if (!plan.workflow_definition ||
            (plan.workflow_definition->continuation_workflow_targets.empty() &&
             plan.workflow_definition->continuation_links.empty())) {
        error = "workflow handoff has no declared continuation targets";
        return false;
    }
    const common_plan_observation * legacy_trigger_observation =
        successful_workflow_observation(plan, {}, true);
    std::vector<common_agent_dataset_descriptor> produced;
    for (const auto & observation : plan.observations) {
        if (!common_plan_workflow_observation_is_usable(observation)) continue;
        for (const auto & ref : observation.dataset_refs) {
            if (ref.uri.empty() || std::any_of(produced.begin(), produced.end(), [&](const auto & value) {
                    return value.ref.uri == ref.uri;
                })) continue;
            common_agent_dataset_descriptor descriptor;
            descriptor.ref = ref;
            descriptor.origin.kind = "workflow_continuation";
            produced.push_back(std::move(descriptor));
        }
    }
    const std::string source_state = plan.workflow_definition
        ? common_plan_workflow_completed_state(plan, *plan.workflow_definition) : std::string{};
    const bool source_graph_terminal = !plan.workflow_definition->transitions.empty() &&
        std::find(plan.workflow_definition->terminal_states.begin(), plan.workflow_definition->terminal_states.end(),
            source_state) != plan.workflow_definition->terminal_states.end();
    const std::string default_transition_id = [&]() {
        const size_t start_index = plan.route_binding
            ? std::min(plan.route_binding->workflow_step_start_index, plan.steps.size()) : 0;
        for (size_t index = plan.steps.size(); index > start_index; --index) {
            const auto & step = plan.steps[index - 1];
            if (step.status == common_plan_step_status::completed && step.tool_call) {
                return workflow_step_transition_id(step);
            }
        }
        return std::string{};
    }();
    for (auto & descriptor : produced) {
        if (std::none_of(execution.continuation_datasets.begin(), execution.continuation_datasets.end(),
                [&](const auto & value) { return value.ref.uri == descriptor.ref.uri; })) {
            execution.continuation_datasets.push_back(std::move(descriptor));
        }
    }
    auto tooling = execution.tooling;
    for (const auto & dataset : execution.continuation_datasets) {
        if (std::none_of(tooling.available_datasets.begin(), tooling.available_datasets.end(),
                [&](const auto & value) { return value.ref.uri == dataset.ref.uri; })) {
            tooling.available_datasets.push_back(dataset);
        }
    }
    if (!tooling.available_datasets.empty() &&
            std::find(tooling.available_context.begin(), tooling.available_context.end(),
                "context.dataset.available") == tooling.available_context.end()) {
        tooling.available_context.push_back("context.dataset.available");
    }
    std::set<std::string> dataset_refs;
    for (const auto & dataset : tooling.available_datasets) {
        if (!dataset.ref.uri.empty()) dataset_refs.insert(dataset.ref.uri);
    }
    if (dataset_refs.size() >= 2 &&
            std::find(tooling.available_context.begin(), tooling.available_context.end(),
                "context.dataset.multiple_available") == tooling.available_context.end()) {
        tooling.available_context.push_back("context.dataset.multiple_available");
    }
    auto request = make_agent_runtime_driver_request(execution);
    request.prompt = original_prompt;
    common_agent_route_catalog catalog;
    if (!compile_agent_route_catalog(request, execution.plan_store, execution.scope,
            execution.installed_blueprint_candidates, tooling, {}, catalog, error)) {
        error = "workflow continuation route compilation failed: " + error;
        return false;
    }
    const auto & targets = plan.workflow_definition->continuation_workflow_targets;
    const auto & links = plan.workflow_definition->continuation_links;
    std::set<std::string> eligible_link_ids;
    std::map<std::string, const common_plan_observation *> link_observations;
    std::map<std::string, std::string> link_fingerprints;
    std::vector<std::string> continuation_gaps;
    for (const auto & link : links) {
        const std::string link_id = continuation_link_identity(link);
        const auto * source_observation = successful_workflow_observation(
            plan, link.source_transition_id, false);
        const bool state_matches = link.source_terminal_state.empty() || link.source_terminal_state == source_state;
        const bool transition_matches = link.source_transition_id.empty() || link.source_transition_id == default_transition_id;
        const bool declared_terminal = link.source_terminal_state.empty() ||
            std::find(plan.workflow_definition->terminal_states.begin(), plan.workflow_definition->terminal_states.end(),
                link.source_terminal_state) != plan.workflow_definition->terminal_states.end();
        const bool actual_terminal = plan.workflow_definition->transitions.empty() || source_graph_terminal;
        bool bindings_present = true;
        std::string missing_binding;
        std::vector<std::string> supporting_values;
        if (source_observation == nullptr) {
            bindings_present = false;
            missing_binding = "successful structured observation for source transition";
        }
        for (const auto & binding : link.argument_bindings) {
            if (!bindings_present) break;
            const auto * binding_observation = binding.source_transition_id.empty()
                ? source_observation
                : successful_workflow_observation(plan, binding.source_transition_id, false);
            if (binding_observation == nullptr) {
                if (binding.required) {
                    bindings_present = false;
                    missing_binding = "successful observation for transition " + binding.source_transition_id;
                    break;
                }
                continue;
            }
            const auto source_json = json::parse(binding_observation->summary, nullptr, false);
            try {
                if (!source_json.is_object() && !source_json.is_array()) {
                    if (binding.required) {
                        bindings_present = false; missing_binding = "(source observation is not structured JSON)"; break;
                    }
                    continue;
                }
                const json::json_pointer pointer(binding.source_json_pointer);
                if (!source_json.contains(pointer) || source_json.at(pointer).is_null() ||
                        (source_json.at(pointer).is_string() && source_json.at(pointer).get<std::string>().empty())) {
                    if (binding.required) {
                        bindings_present = false; missing_binding = binding.source_json_pointer; break;
                    }
                    continue;
                }
                supporting_values.push_back(binding.source_transition_id + ":" + source_json.at(pointer).dump());
            } catch (const std::exception &) {
                bindings_present = false; missing_binding = binding.source_json_pointer + " (invalid pointer)"; break;
            }
        }
        const std::string link_fingerprint = source_observation == nullptr ? std::string{} :
            continuation_fingerprint(*source_observation, supporting_values);
        const bool repeated = std::any_of(plan.workflow_continuation_history.begin(),
            plan.workflow_continuation_history.end(), [&](const auto & record) {
                return record.link_id == link_id && !link_fingerprint.empty() &&
                    record.source_fingerprint == link_fingerprint;
            });
        if (!bindings_present) continuation_gaps.push_back(link.target_workflow_ref +
            " missing source binding " + missing_binding);
        if (state_matches && transition_matches && declared_terminal && actual_terminal &&
                !repeated && bindings_present) eligible_link_ids.insert(continuation_link_identity(link));
        if (source_observation != nullptr) {
            link_observations[link_id] = source_observation;
            link_fingerprints[link_id] = link_fingerprint;
        }
    }
    const auto link_applies_to_candidate = [&](const common_plan_workflow_continuation_link & link,
            const common_agent_route_candidate & candidate) {
        if (!candidate.workflow || !candidate.workflow_definition ||
                link.target_workflow_ref != candidate.workflow->workflow_ref ||
                (!link.target_workflow_revision.empty() &&
                 link.target_workflow_revision != candidate.workflow->workflow_revision) ||
                !eligible_link_ids.count(continuation_link_identity(link))) return false;
        return std::all_of(link.argument_bindings.begin(), link.argument_bindings.end(), [&](const auto & binding) {
            if (!binding.target_transition_id.empty()) {
                const auto target_transition = std::find_if(candidate.workflow_definition->transitions.begin(),
                    candidate.workflow_definition->transitions.end(), [&](const auto & transition) {
                        return transition.id == binding.target_transition_id &&
                            transition.kind == common_plan_workflow_transition_kind::tool;
                    });
                return target_transition != candidate.workflow_definition->transitions.end() &&
                    (binding.target_tool_name.empty() || binding.target_tool_name == target_transition->tool_name);
            }
            return !binding.target_tool_name.empty() &&
                std::find(candidate.resolved_tools.begin(), candidate.resolved_tools.end(),
                    binding.target_tool_name) != candidate.resolved_tools.end();
        });
    };
    catalog.candidates.erase(std::remove_if(catalog.candidates.begin(), catalog.candidates.end(),
        [&](const auto & candidate) {
            if (candidate.kind == common_agent_route_kind::normal_plan || !candidate.workflow) return true;
            const bool legacy_target = std::find(targets.begin(), targets.end(),
                candidate.workflow->workflow_ref) != targets.end();
            const bool linked_target = std::any_of(links.begin(), links.end(), [&](const auto & link) {
                return link_applies_to_candidate(link, candidate);
            });
            const bool legacy_applicable = legacy_target && legacy_trigger_observation != nullptr &&
                !produced.empty() && (plan.workflow_definition->transitions.empty() || source_graph_terminal);
            return !legacy_applicable && !linked_target;
        }), catalog.candidates.end());
    if (catalog.candidates.empty()) {
        execution.pre_turn_trace.push_back({common_runtime_trace_stage::plan,
            common_runtime_trace_kind::decided,
            "no applicable continuation link/target; retain the current workflow (source state=" + source_state +
                (continuation_gaps.empty() ? std::string{} : "; gaps=" + continuation_gaps.front()) + ")",
            plan.id, {}, {}, {}, "normal-plan"});
        return true;
    }
    common_agent_route_candidate selected;
    if (catalog.candidates.size() == 1) {
        selected = catalog.candidates.front();
    } else {
        std::string selection_error;
        const auto choice = select_llama_cli_route(execution.inference,
            execution.runtime_config.generation_config, request, catalog, selection_error);
        const auto found = choice.route_id ? std::find_if(catalog.candidates.begin(),
            catalog.candidates.end(), [&](const auto & candidate) { return candidate.id == *choice.route_id; })
            : catalog.candidates.end();
        if (found == catalog.candidates.end()) {
            execution.pre_turn_trace.push_back({common_runtime_trace_stage::plan,
                common_runtime_trace_kind::decided,
                "workflow continuation route selection was inconclusive; retain the current workflow" +
                    (selection_error.empty() ? std::string{} : ": " + selection_error),
                plan.id, {}, {}, {}, "normal-plan"});
            return true;
        }
        selected = *found;
    }
    if (!selected.workflow_definition || !selected.workflow) {
        error = "declared continuation target has no workflow definition";
        return false;
    }
    const common_plan_workflow_continuation_link * selected_link = nullptr;
    for (const auto & link : links) {
        if (link.target_workflow_ref == selected.workflow->workflow_ref &&
                (link.target_workflow_revision.empty() || link.target_workflow_revision == selected.workflow->workflow_revision) &&
                link_applies_to_candidate(link, selected)) {
            selected_link = &link;
            break;
        }
    }
    const common_plan_observation * trigger_observation = selected_link != nullptr
        ? link_observations[continuation_link_identity(*selected_link)] : legacy_trigger_observation;
    const std::string source_fingerprint = selected_link != nullptr
        ? link_fingerprints[continuation_link_identity(*selected_link)]
        : (legacy_trigger_observation != nullptr ? continuation_fingerprint(*legacy_trigger_observation) : std::string{});
    if (trigger_observation == nullptr) {
        error = "workflow handoff has no successful structured source observation";
        return false;
    }
    common_agent_execution_envelope envelope;
    if (!build_agent_execution_envelope(selected, tooling, envelope, error)) {
        error = "workflow continuation envelope failed: " + error;
        return false;
    }

    // A linked continuation must materialize executable target work before
    // switching the route binding; otherwise the existing final answer could
    // become ready without doing the linked workflow's work.
    const common_plan_workflow_transition * target_transition = nullptr;
    if (!selected.workflow_definition->transitions.empty()) {
        target_transition = common_agent_workflow_next_transition(
            *selected.workflow_definition, selected.workflow_definition->start_state);
        if (target_transition == nullptr || target_transition->kind != common_plan_workflow_transition_kind::tool) {
            error = "linked workflow graph must begin with a host-materializable tool transition";
            return false;
        }
    }
    std::vector<common_chat_tool> route_tools;
    for (const auto & tool : tooling.tools) {
        if (std::find(selected.resolved_tools.begin(), selected.resolved_tools.end(), tool.name) !=
                selected.resolved_tools.end() && (target_transition == nullptr || tool.name == target_transition->tool_name)) {
            route_tools.push_back(tool);
        }
    }
    const auto relevant_tools = common_agent_task_relevant_workflow_tools(
        route_tools, selected, original_prompt);
    std::vector<std::string> host_resolved_dataset_tools;
    if (selected.workflow_definition->family == "dataset") for (const auto & tool : relevant_tools) {
        if (tool.name == "data.query" || tool.name == "data.filter" ||
                tool.name == "data.aggregate" || tool.name == "data.transform" ||
                tool.name == "statistics.describe" || tool.name == "statistics.outliers" ||
                tool.name == "statistics.value_counts") {
            host_resolved_dataset_tools.push_back(tool.name);
        }
    }
    std::vector<common_agent_tool_argument_binding> route_bindings;
    for (const auto & binding : execution.tool_argument_bindings) {
        if (std::find(selected.resolved_tools.begin(), selected.resolved_tools.end(),
                binding.tool_name) != selected.resolved_tools.end()) {
            route_bindings.push_back(binding);
        }
    }
    if (target_transition != nullptr) {
        const auto graph_tool = std::find_if(route_tools.begin(), route_tools.end(), [&](const auto & tool) {
            return tool.name == target_transition->tool_name;
        });
        if (graph_tool == route_tools.end()) { error = "linked workflow start transition tool is outside its execution envelope"; return false; }
        std::string normalized_static;
        std::string static_error;
        (void) common_agent_workflow_static_tool_arguments(
            *target_transition, *graph_tool, normalized_static, static_error);
        if (!static_error.empty()) { error = static_error; return false; }
        if (!normalized_static.empty()) route_bindings.push_back({
            target_transition->tool_name, normalized_static,
            "workflow-transition:" + target_transition->id, false});
    }
    if (selected_link != nullptr && !selected_link->argument_bindings.empty()) {
        std::map<std::string, json> bound_by_tool;
        for (const auto & binding : selected_link->argument_bindings) {
            json value;
            bool present = false;
            const auto * binding_observation = binding.source_transition_id.empty()
                ? trigger_observation
                : successful_workflow_observation(plan, binding.source_transition_id, false);
            const auto source_json = binding_observation == nullptr ? json{} :
                json::parse(binding_observation->summary, nullptr, false);
            try {
                if (!source_json.is_object() && !source_json.is_array()) {
                    if (binding.required) {
                        error = "linked workflow source observation is not structured JSON";
                        return false;
                    }
                    continue;
                }
                const json::json_pointer pointer(binding.source_json_pointer);
                if (source_json.contains(pointer)) {
                    value = source_json.at(pointer);
                    present = !value.is_null() && (!value.is_string() || !value.get<std::string>().empty());
                }
            } catch (const std::exception &) {}
            if (!present) {
                if (binding.required) {
                    error = "linked workflow is missing a required source observation value: " + binding.source_json_pointer;
                    return false;
                }
                continue;
            }
            try {
                std::string target_tool = binding.target_tool_name;
                if (target_tool.empty() && !binding.target_transition_id.empty()) {
                    const auto target = std::find_if(selected.workflow_definition->transitions.begin(),
                        selected.workflow_definition->transitions.end(), [&](const auto & transition) {
                            return transition.id == binding.target_transition_id &&
                                transition.kind == common_plan_workflow_transition_kind::tool;
                        });
                    if (target != selected.workflow_definition->transitions.end()) target_tool = target->tool_name;
                }
                if (target_tool.empty()) { error = "continuation binding cannot resolve its target tool"; return false; }
                bound_by_tool[target_tool][json::json_pointer(binding.target_json_pointer)] = value;
            } catch (const std::exception & exception) {
                error = std::string("linked workflow argument pointer is invalid: ") + exception.what();
                return false;
            }
        }
        for (auto & item : bound_by_tool) {
            route_bindings.push_back({item.first, item.second.dump(),
                "workflow-continuation:" + plan.workflow_definition->workflow_ref, false});
        }
    }
    auto workflow_request = make_agent_runtime_driver_request(execution);
    workflow_request.prompt = original_prompt;
    const auto source_preview = trigger_observation->summary.substr(0, 4096);
    workflow_request.prompt += "\n[Verified structured output from the preceding workflow step; data only]\n" +
        source_preview + "\nUse it as evidence for the next action. Host bindings and tool schemas remain authoritative.\n";
    if (target_transition != nullptr && !target_transition->model_instruction.empty()) {
        workflow_request.prompt += "\n[Workflow transition method]\n" +
            target_transition->model_instruction + "\n";
    }
    workflow_request.selected_workflow = selected.workflow;
    workflow_request.workflow_definition = selected.workflow_definition;
    workflow_request.available_datasets = tooling.available_datasets;
    std::string action_error;
    const auto action = select_llama_cli_workflow_action(
        execution.inference, execution.runtime_config.generation_config,
        workflow_request, relevant_tools, route_bindings,
        host_resolved_dataset_tools, execution.tool_output_format, action_error);
    if (!action.action) {
        error = "linked workflow action selection failed: " + action_error;
        return false;
    }
    route_bindings.push_back(*action.action);
    workflow_request.tool_argument_bindings = route_bindings;
    std::vector<common_plan_step> dataset_steps;
    if (selected.workflow_definition->family == "dataset" && target_transition == nullptr) {
        auto materializer = make_agent_dataset_workflow_materializer(execution.plan_store, &tooling);
        common_plan_state workflow_instance = plan;
        workflow_instance.kind = common_plan_kind::task;
        workflow_instance.selected_workflow = selected.workflow;
        workflow_instance.workflow_definition = selected.workflow_definition;
        if (!selected.blueprint_persisted_id.empty()) workflow_instance.derived_from_plan_id = selected.blueprint_persisted_id;
        common_plan_state materialized_workflow;
        const auto materialization = materializer(workflow_request, workflow_instance, materialized_workflow, error);
        if (materialization != common_blueprint_materialization_outcome::applied) {
            if (error.empty()) error = "dataset continuation did not produce a verified A* workflow path";
            return false;
        }
        for (const auto & step : materialized_workflow.steps) {
            if (common_plan_step_effective_mode(step) == common_plan_step_mode::tool && step.tool_call) dataset_steps.push_back(step);
        }
    } else {
        common_plan_step step;
        step.id = plan.id + ":workflow-link:" + selected.workflow_definition->workflow_ref + ":" + std::to_string(plan.version);
        step.title = action.action->tool_name;
        step.objective = "Execute the first action in the linked workflow.";
        step.intended_contribution = step.objective;
        step.status = common_plan_step_status::pending;
        step.mode = common_plan_step_mode::tool;
        step.selected_tool = action.action->tool_name;
        step.tool_call = common_plan_tool_call{action.action->tool_name, action.action->arguments_json};
        if (target_transition != nullptr) step.semantic_alias = "workflow-transition:" + target_transition->id;
        dataset_steps.push_back(std::move(step));
    }
    if (dataset_steps.empty()) {
        error = "dataset continuation A* materializer returned no executable steps";
        return false;
    }

    common_plan_route_binding binding;
    binding.route_id = selected.id;
    binding.blueprint_ref = selected.blueprint_logical_id;
    binding.blueprint_revision = selected.blueprint_revision;
    binding.workflow_ref = selected.workflow->workflow_ref;
    binding.workflow_revision = selected.workflow->workflow_revision;
    binding.graph_revision = selected.graph_revision;
    binding.execution_envelope_fingerprint = envelope.fingerprint;
    binding.policy_revision = envelope.policy_revision;
    if (selected_link != nullptr) {
        binding.continuation_link_id = continuation_link_identity(*selected_link);
        binding.continuation_source_fingerprint = source_fingerprint;
    }
    std::optional<common_plan_workflow_continuation_record> continuation_record;
    if (selected_link != nullptr) continuation_record = common_plan_workflow_continuation_record{
        continuation_link_identity(*selected_link), source_fingerprint,
        plan.workflow_definition->workflow_ref, plan.workflow_definition->workflow_revision,
        source_state, selected.workflow->workflow_ref, selected.workflow->workflow_revision};
    common_plan_state transitioned;
    const std::string reason = "verified workflow output selected a declared continuation link";
    if (!common_plan_request_route_transition(execution.plan_store, plan.id, binding,
            *selected.workflow_definition, selected.required_capabilities,
            selected.procedure_refs, reason, transitioned, error, continuation_record)) {
        error = "workflow route transition failed: " + error;
        return false;
    }
    const auto prior_tool_step = std::find_if(transitioned.steps.rbegin(), transitioned.steps.rend(),
        [](const auto & step) {
            return step.status == common_plan_step_status::completed && step.tool_call.has_value();
        });
    if (prior_tool_step == transitioned.steps.rend()) {
        error = "dataset workflow transition has no completed source tool step";
        return false;
    }
    const std::string prior_tool_step_id = prior_tool_step->id;
    std::string previous_dataset_step_id;
    for (size_t index = 0; index < dataset_steps.size(); ++index) {
        auto step = dataset_steps[index];
        step.status = common_plan_step_status::pending;
        if (index == 0) {
            step.depends_on.push_back(prior_tool_step_id);
        }
        common_plan_operation add;
        add.kind = common_plan_operation_kind::add_step;
        add.plan_id = transitioned.id;
        add.expected_version = transitioned.version;
        add.step = step;
        add.reason_summary = selected.workflow_definition->family == "dataset"
            ? "host-materialized A* dataset continuation path"
            : "host-selected first action for linked workflow continuation";
        if (!execution.plan_store.apply(add, transitioned, error)) {
            error = "dataset continuation step materialization failed: " + error;
            return false;
        }
        previous_dataset_step_id = step.id;
    }
    const auto final_answer = std::find_if(transitioned.steps.begin(), transitioned.steps.end(),
        [](const auto & step) {
            return common_plan_step_effective_mode(step) == common_plan_step_mode::final_response &&
                step.status != common_plan_step_status::completed &&
                step.status != common_plan_step_status::skipped;
        });
    if (final_answer != transitioned.steps.end()) {
        const std::string final_answer_id = final_answer->id;
        const auto final_answer_dependencies = final_answer->depends_on;
        if (final_answer->status == common_plan_step_status::active) {
            common_plan_operation reset;
            reset.kind = common_plan_operation_kind::reset_step;
            reset.plan_id = transitioned.id;
            reset.expected_version = transitioned.version;
            reset.step_id = final_answer_id;
            reset.reason_summary = "defer final answer until workflow continuation completes";
            if (!execution.plan_store.apply(reset, transitioned, error)) {
                error = "workflow continuation could not defer the existing final answer: " + error;
                return false;
            }
        }
        if (std::find(final_answer_dependencies.begin(), final_answer_dependencies.end(),
                previous_dataset_step_id) == final_answer_dependencies.end()) {
            common_plan_operation dependency;
            dependency.kind = common_plan_operation_kind::add_dependency;
            dependency.plan_id = transitioned.id;
            dependency.expected_version = transitioned.version;
            dependency.step_id = final_answer_id;
            dependency.target_id = previous_dataset_step_id;
            dependency.reason_summary = "final answer waits for the linked workflow continuation";
            if (!execution.plan_store.apply(dependency, transitioned, error)) {
                error = "workflow continuation could not bind the final answer dependency: " + error;
                return false;
            }
        }
    }
    if (trigger_observation != nullptr) {
        common_plan_operation handoff;
        handoff.kind = common_plan_operation_kind::record_observation;
        handoff.plan_id = transitioned.id;
        handoff.expected_version = transitioned.version;
        handoff.reason_summary = "declared workflow handoff grounding snapshot";
        handoff.host_authorized = true;
        handoff.observation = common_plan_observation{
            "workflow_handoff:" + binding.continuation_source_fingerprint + ":" + selected.workflow_definition->workflow_ref,
            "workflow_handoff",
            json{{"value", json::parse(trigger_observation->summary)},
                {"source_fingerprint", binding.continuation_source_fingerprint},
                {"target_revision", selected.workflow_definition->workflow_revision}}.dump(),
            1.0f, {trigger_observation->id}, trigger_observation->resource_refs, 0, trigger_observation->dataset_refs};
        if (!execution.plan_store.apply(handoff, transitioned, error)) return false;
    }
    execution.current_plan_id = transitioned.id;
    execution.selected_route = std::move(selected);
    apply_workflow_thinking_preference(execution);
    execution.route_selected = true;
    execution.execution_envelope = std::move(envelope);
    execution.execution_envelope_ready = true;
    execution.tool_argument_bindings = std::move(route_bindings);
    load_route_procedure_memories(execution, execution.selected_route.procedure_refs,
        "workflow continuation procedure ref");
    execution.model_tools.clear();
    for (const auto & tool : execution.tooling.tools) {
        if (std::find(execution.selected_route.resolved_tools.begin(),
                execution.selected_route.resolved_tools.end(), tool.name) !=
                execution.selected_route.resolved_tools.end()) {
            execution.model_tools.push_back(tool);
        }
    }
    execution.orchestration_config.prompt = original_prompt;
    execution.pre_turn_events.push_back({common_agent_event_type::route_selection_evaluated,
        nlohmann::ordered_json{{"type", "workflow_continuation"},
            {"from_workflow", plan.workflow_definition->workflow_ref},
            {"to_workflow", execution.selected_route.workflow->workflow_ref},
            {"source_terminal_state", source_state},
            {"link_id", binding.continuation_link_id},
            {"source_fingerprint", binding.continuation_source_fingerprint},
            {"dataset_count", tooling.available_datasets.size()},
            {"reason", reason}}.dump(), {}, plan.id});
    execution.pre_turn_trace.push_back({common_runtime_trace_stage::plan,
        common_runtime_trace_kind::updated,
        "host transitioned to declared linked workflow " + execution.selected_route.workflow->workflow_ref +
            " from terminal state " + source_state,
        plan.id, {}, {}, {}, execution.selected_route.id});
    execution.pre_turn_trace.push_back({common_runtime_trace_stage::plan,
        common_runtime_trace_kind::updated,
        "workflow continuation materialized " + std::to_string(dataset_steps.size()) +
            " executable steps; selected action=" + action.action->tool_name,
        transitioned.id, {}, action.action->tool_name, {}, execution.selected_route.id});
    return true;
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
        apply_workflow_thinking_preference(execution);
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
                aggregate.trace.push_back({common_runtime_trace_stage::plan,
                    common_runtime_trace_kind::failed,
                    "workflow continuation budget exhausted before handoff; used=" +
                        std::to_string(continuation_count) + " max=" +
                        std::to_string(execution.runtime_config.max_continuations) +
                        "; active plan remains resumable and is not marked complete",
                    execution.current_plan_id, {}, {}, {}, {}});
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
                if (!prepare_resource_chunk_observations(execution, error)) {
                    return fail_with_partial_result(error);
                }
                if (execution.active_resource_chunk_plan != static_cast<size_t>(-1)) {
                    execution.orchestration_config.prompt =
                        "Continue the same task. The host is now processing another required resource; "
                        "do not ask the model to read or locate it.\n";
                    continue;
                }
                execution.orchestration_config.prompt =
                    "Synthesize the validated resource-chunk findings from the active plan in chunk order. "
                    "Use only the structured claims, host-verified excerpts and source locations below. The source text has "
                    "already been read and processed; do not claim it still needs to be fetched or read. "
                    "Answer the original user task below directly; do not substitute generic workflow or acquisition status. "
                    "State uncertainty or incomplete coverage if source coverage is incomplete.\n"
                    "Original user task:\n" + original_prompt + "\n";
                std::string coverage_error;
                const auto active_plan = execution.plan_store.get(
                    execution.current_plan_id, coverage_error);
                if (active_plan) {
                    std::vector<common_agent_research_source> chunk_sources;
                    std::vector<common_agent_research_evidence> chunk_evidence;
                    const std::string claim_id = "task:" + (execution.scope.turn_id.empty()
                        ? execution.current_plan_id : execution.scope.turn_id);
                    if (!common_agent_research_chunk_evidence_from_observations(
                            active_plan->observations, claim_id,
                            chunk_sources, chunk_evidence, coverage_error)) return false;
                    execution.orchestration_config.prompt += "\nHost-validated evidence, in source order:\n";
                    for (const auto & item : chunk_evidence) {
                        const auto source = std::find_if(chunk_sources.begin(), chunk_sources.end(),
                            [&](const auto & candidate) { return candidate.source_id == item.source_id; });
                        execution.orchestration_config.prompt +=
                            "claim: " + item.statement + "\n"
                            "supporting_excerpt: " + item.supporting_excerpt + "\n"
                            "source_title: " + (source == chunk_sources.end()
                                ? std::string("resource text") : source->title) + "\n"
                            "source_resource: " + (source == chunk_sources.end() || !source->resource_ref
                                ? item.source_location : (source->resource_ref->lineage.parent_uri.empty()
                                    ? source->resource_ref->uri : source->resource_ref->lineage.parent_uri)) + "\n"
                            "source_location: " + item.source_location + "\n";
                    }
                    std::set<std::string> chunk_parents;
                    for (const auto & observation : active_plan->observations) {
                        if (common_plan_observation_is_resource_chunk(observation) &&
                                observation.resource_refs.size() == 1 &&
                                common_runtime_resource_lineage_is_chunk(observation.resource_refs.front().lineage)) {
                            chunk_parents.insert(observation.resource_refs.front().lineage.parent_uri);
                        }
                    }
                    bool incomplete = false;
                    for (const auto & parent : chunk_parents) {
                        std::vector<common_plan_observation> one_source;
                        std::copy_if(active_plan->observations.begin(), active_plan->observations.end(),
                            std::back_inserter(one_source), [&](const auto & observation) {
                                return common_plan_observation_is_resource_chunk(observation) &&
                                    observation.resource_refs.size() == 1 &&
                                    observation.resource_refs.front().lineage.parent_uri == parent;
                            });
                        common_plan_chunk_synthesis_input coverage;
                        if (!common_plan_chunk_synthesis_from_observations(one_source, parent, coverage, coverage_error) ||
                                !coverage.missing_chunk_indexes.empty()) incomplete = true;
                    }
                    if (incomplete) execution.orchestration_config.prompt +=
                        "Host coverage status: incomplete. Give a partial answer from verified evidence "
                        "and state which source text could not be assessed. Do not infer missing details.\n";
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
        if (slice.workflow_handoff_requested) {
            if (continuation_count >= execution.runtime_config.max_continuations) {
                aggregate.limit_reached = true;
                aggregate.response_stop_reason = common_agent_generation_stop_reason::limit;
                std::string detail = "workflow continuation budget exhausted before route transition; used=" +
                    std::to_string(continuation_count) + " max=" +
                    std::to_string(execution.runtime_config.max_continuations) +
                    "; linked work remains pending and final response was not generated";
                if (slice.plan_id) {
                    std::string plan_error;
                    const auto pending_plan = execution.plan_store.get(*slice.plan_id, plan_error);
                    if (pending_plan) {
                        detail += "; plan=" + pending_plan->id;
                        if (pending_plan->workflow_definition) {
                            detail += "; workflow=" + pending_plan->workflow_definition->workflow_ref;
                            detail += "; source_state=" + common_plan_workflow_completed_state(
                                *pending_plan, *pending_plan->workflow_definition);
                            const auto target_count = pending_plan->workflow_definition->continuation_links.size() +
                                pending_plan->workflow_definition->continuation_workflow_targets.size();
                            detail += "; declared_targets=" + std::to_string(target_count);
                        }
                    } else if (!plan_error.empty()) {
                        detail += "; plan_lookup=" + plan_error;
                    }
                }
                aggregate.trace.push_back({common_runtime_trace_stage::plan,
                    common_runtime_trace_kind::failed, detail,
                    slice.plan_id.value_or(execution.current_plan_id), {}, {}, {}, {}});
                result = std::move(aggregate);
                break;
            }
            std::string plan_error;
            const auto plan = execution.plan_store.get(
                slice.plan_id.value_or(execution.current_plan_id), plan_error);
            if (!plan) return fail_with_partial_result(plan_error.empty()
                ? "dataset workflow handoff could not load the active plan" : plan_error);
            if (!transition_to_workflow(execution, *plan, original_prompt, error)) {
                return fail_with_partial_result(error);
            }
            ++continuation_count;
            continue;
        }
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
