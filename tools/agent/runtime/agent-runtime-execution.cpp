#include "agent-runtime-execution.h"
#include "agent/context-compaction.h"
#include "agent/input-resources.h"
#include "agent/tool-family-index.h"
#include "../runtime/agent-runtime-chat-driver.h"

#include "../runtime/agent-plan-orchestration.h"
#include "../runtime/agent-runtime-assembly.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <ctime>

namespace {

bool tool_has_no_required_arguments(const common_chat_tool & tool) {
    const auto schema = nlohmann::ordered_json::parse(tool.parameters, nullptr, false);
    if (schema.is_discarded() || !schema.is_object()) return false;
    if (!schema.contains("required")) return true;
    return schema["required"].is_array() && schema["required"].empty();
}

bool select_model_tool_families(
        common_agent_runtime_driver_execution & execution,
        const common_agent_request & request,
        std::string & error) {
    execution.model_tools.clear();
    const auto & generation_config = execution.runtime_config.generation_config;
    execution.family_chat_routed = false;
    execution.family_chat_result = {};
    execution.require_tool_execution = false;
    if (!generation_config.enable_tool_family_routing ||
            execution.tooling.tools.empty() ||
            !execution.current_plan_id.empty()) {
        return true;
    }

    const auto families = common_generate_tool_family_index(execution.tooling.tools);
    const std::string family_view = common_render_tool_family_index(families, 2048);
    common_chat_msg system{
        "system",
        "Decide whether the user request needs external tools. Reply with exactly one line: "
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
    const auto generation = execution.inference.generate_result(
        common_agent_make_generation_request(
            common_agent_generation_purpose::tool_family_selection,
            request.turn_id + ":tool-family-selection",
            common_agent_scope_from_request(request),
            {system, user},
            options,
            {}));
    if (!common_agent_generation_succeeded(generation)) {
        error = "tool family selection failed: " + generation.error_message;
        if (generation.error_message.empty()) error += " model did not return a completed selection";
        return false;
    }

    common_tool_family_selection selection;
    if (!common_parse_tool_family_selection_text(generation.content, families, selection, error)) {
        // A weak model can emit the required TOOLS: prefix and then stop
        // before naming a family. With exactly one current-turn resource,
        // the host can safely default to the already policy-filtered
        // resource family. This keeps the family gate intact and does not
        // guess when there are multiple resources or no resource family.
        const bool empty_tools_selection =
            error == "tool family selection must include at least one family after TOOLS:";
        const bool has_resource_family = std::any_of(
            families.begin(), families.end(), [](const common_tool_family_index & family) {
                return family.id == "resource";
            });
        if (empty_tools_selection && request.input_resources.size() == 1 && has_resource_family) {
            selection.needs_tools = true;
            selection.family_ids = {"resource"};
            error.clear();
        } else {
            error = "tool family selection failed: " + error;
            return false;
        }
    }
    if (!selection.needs_tools && !selection.family_ids.empty()) {
        error = "tool family selection must not select families when needs_tools is false";
        return false;
    }
    if (!selection.needs_tools) {
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
    execution.model_tools = common_filter_tools_by_families(
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
        if (input.resource.uri.empty() || !input.resource.lineage.parent_uri.empty() ||
                input.resource.size_bytes <= policy.max_bytes) {
            continue;
        }
        agent_resource_chunk_plan chunk_plan;
        if (!plan_agent_resource_text_chunks(
                    *execution.tooling.resource_runtime.store,
                    input.resource.uri,
                    authority,
                    policy,
                    chunk_plan,
                    error)) {
            return false;
        }
        if (chunk_plan.ranges.empty()) {
            continue;
        }
        size_t next_chunk_index = 0;
        for (const auto & observation : plan->observations) {
            if (observation.source != "resource_chunk" ||
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
        execution.resource_chunk_plans.push_back(std::move(chunk_plan));
        const size_t plan_index = execution.resource_chunk_plans.size() - 1;
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
                "Planned " + std::to_string(execution.resource_chunk_plans.back().ranges.size()) +
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
                "bounded resource chunk plan attached",
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

bool record_and_advance_resource_chunk(
        common_agent_runtime_driver_execution & execution,
        const common_agent_result & slice,
        bool & chunk_chain_completed,
        std::string & error) {
    chunk_chain_completed = false;
    if (execution.active_resource_chunk_plan == static_cast<size_t>(-1)) return false;
    if (execution.active_resource_chunk_plan >= execution.resource_chunk_plans.size() ||
            execution.active_resource_chunk_input >= execution.input_resources.size()) {
        error = "resource chunk session cursor is invalid";
        return false;
    }
    const auto & chunk_plan = execution.resource_chunk_plans[execution.active_resource_chunk_plan];
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
    const auto chunk_ref = make_agent_resource_chunk_ref(chunk_plan, chunk_plan.ranges[chunk_index]);
    std::string summary = slice.response;
    constexpr size_t max_summary_chars = 1024;
    if (summary.size() > max_summary_chars) summary.resize(max_summary_chars);
    if (summary.empty()) summary = "The bounded resource slice was presented to the runtime.";
    common_plan_operation observed;
    observed.kind = common_plan_operation_kind::record_observation;
    observed.plan_id = plan->id;
    observed.expected_version = plan->version;
    observed.reason_summary = "bounded resource chunk processed on the session lane";
    observed.observation = common_plan_observation{
        "resource_chunk:" + chunk_plan.parent.uri + ":" + std::to_string(chunk_index),
        "resource_chunk",
        summary,
        1.0f,
        {chunk_plan.parent.uri},
        {chunk_ref},
        static_cast<int64_t>(std::time(nullptr)),
    };
    common_plan_state updated;
    if (!execution.plan_store.apply(observed, updated, error)) return false;
    *plan = std::move(updated);
    execution.pre_turn_events.push_back({
        common_agent_event_type::resource_chunk_processed,
        "bounded resource chunk processed",
        {},
        plan->id,
        {},
        observed.observation->id,
        {},
        chunk_ref.uri,
    });
    if (chunk_index + 1 >= chunk_plan.ranges.size()) {
        chunk_chain_completed = true;
        execution.active_resource_chunk_plan = static_cast<size_t>(-1);
        execution.active_resource_chunk_input = static_cast<size_t>(-1);
        execution.active_resource_chunk_index = 0;
        execution.input_resources = execution.resource_chunk_original_inputs;
        return false;
    }
    ++execution.active_resource_chunk_index;
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
        if (observation.source == "resource_chunk" && observation.resource_refs.size() == 1) {
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
    return {
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
        inputs.execution_control,
    };
}

common_agent_request make_agent_runtime_driver_request(
    const common_agent_runtime_driver_execution & execution) {
    common_agent_request request;
    request.memories = execution.memories;
    request.enable_memory = execution.memory_enabled;
    request.enable_planning = true;
    request.enable_reflection = execution.policy.enable_reflection;
    request.memory_scope = execution.memory_scope;
    request.plan_scope = execution.scope.plan_scope;
    request.prompt = execution.orchestration_config.prompt;
    request.input_resources = execution.input_resources;
    request.available_resources = execution.available_resources;
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
    request.allow_policy_gated_tool_proposals = execution.policy.allow_policy_gated_tool_proposals;
    request.deliberation_policy = execution.policy.deliberation_policy;
    request.research_should_stop = execution.research_should_stop;
    request.research_stop_reason = execution.research_stop_reason;
    request.explicit_memory_candidate = execution.explicit_memory_candidate;
    request.explicit_memory_confirmed = execution.explicit_memory_confirmed;
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
    if (execution.tooling.profile_tools_active && execution.tooling.tool_view == nullptr) {
        error = "profile tool execution requires a resolved tool view";
        return false;
    }

    const common_agent_orchestration_runtime_context orchestration_context{
        execution.inference,
        execution.runtime_config.generation_config,
        execution.orchestration_config,
        execution.current_plan_id,
        execution.scope,
        execution.plan_store,
        execution.installed_blueprint_candidates,
        &execution.policy_pack,
        &execution.tooling,
        execution.pre_turn_events,
        execution.pre_turn_trace,
    };

    if (!maybe_auto_select_plan(orchestration_context, error)) {
        return false;
    }

    if (!maybe_auto_select_blueprint(orchestration_context, error)) {
        return false;
    }

    if (!prepare_resource_chunk_observations(execution, error)) {
        return false;
    }

    const std::string original_prompt = execution.orchestration_config.prompt;
    common_agent_result aggregate;
    size_t continuation_count = 0;
    const auto stop_for_execution_control = [&]() {
        execution.orchestration_config.prompt = original_prompt;
        result = std::move(aggregate);
        result.response_generation_status = common_agent_generation_status::cancelled;
        result.response_stop_reason = common_agent_generation_stop_reason::cancelled;
        result.error = execution.execution_control.stop_reason();
        error = result.error.empty() ? "agent runtime execution stopped" : result.error;
        return false;
    };
    prepare_available_resources(execution);
    const common_agent_request initial_request = make_agent_runtime_driver_request(execution);
    if (execution.execution_control.should_stop()) {
        return stop_for_execution_control();
    }
    if (!select_model_tool_families(execution, initial_request, error)) {
        return false;
    }
    if (execution.execution_control.should_stop()) {
        return stop_for_execution_control();
    }
    if (execution.family_chat_routed) {
        result = std::move(execution.family_chat_result);
        error.clear();
        return true;
    }
    while (true) {
        if (execution.execution_control.should_stop()) return stop_for_execution_control();
        const auto & model_tools = execution.model_tools.empty()
            ? execution.tooling.tools
            : execution.model_tools;
        const common_agent_request request = make_agent_runtime_driver_request(execution);
        auto assembly = make_agent_runtime_assembly(
            execution.memory_store,
            execution.plan_store,
            execution.inference,
            execution.runtime_config,
            model_tools,
            execution.tooling.tool_view);

        const auto slice = assembly.runtime->run(request);
        if (!slice.error.empty()) {
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
            error = "agent runtime failed: " + slice.error;
            return false;
        }
        append_runtime_result(aggregate, slice);
        if (execution.execution_control.should_stop()) return stop_for_execution_control();
        bool resource_chunk_chain_completed = false;
        const bool has_next_resource_chunk = record_and_advance_resource_chunk(
            execution, slice, resource_chunk_chain_completed, error);
        if (!error.empty()) return false;

        if (has_next_resource_chunk) {
            if (continuation_count >= execution.runtime_config.max_continuations) {
                aggregate.limit_reached = true;
                aggregate.response_stop_reason = common_agent_generation_stop_reason::limit;
                result = std::move(aggregate);
                break;
            }
            execution.orchestration_config.prompt =
                "Continue the same task using the next bounded resource slice.\n"
                "Treat the previous slice observation as evidence; do not restart the plan.\n"
                "Inspect and synthesize the currently attached resource_chunk, then produce the next bounded result.\n";
            ++continuation_count;
            continue;
        }

        if (resource_chunk_chain_completed) {
            if (continuation_count >= execution.runtime_config.max_continuations) {
                aggregate.limit_reached = true;
                aggregate.response_stop_reason = common_agent_generation_stop_reason::limit;
                result = std::move(aggregate);
                break;
            }
            execution.orchestration_config.prompt =
                "Synthesize the completed bounded resource observations in deterministic chunk order.\n"
                "Use the plan observations as working evidence and retain references to the authoritative resource.\n"
                "Do not treat an individual chunk response as the final answer; produce the bounded synthesis result.\n";
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
            result = std::move(aggregate);
            result.error = plan_error.empty()
                ? "continuation could not resolve the current plan"
                : plan_error;
            error = result.error;
            return false;
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
            result.error = checkpoint_error;
            error = result.error;
            return false;
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
