#include "agent/agent-runtime.h"
#include "agent/agent-runtime-context.h"
#include "agent/context-pressure.h"
#include "agent/learning/memory-learning.h"
#include "agent/thinking/research/research-runner.h"
#include "agent/runtime-json-contracts.h"
#include "agent/tooling/routing/tool-navigation.h"
#include "plan/plan-bindings.h"
#include "plan/plan-json.h"
#include "plan/plan-goal.h"
#include "plan/plan-memory.h"
#include "plan/plan-scheduler.h"
#include "plan/plan-context.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <set>

using json = nlohmann::ordered_json;

static bool common_agent_reflection_context_overflow(const std::string & error) {
    return error.find("exceeds the available context size") != std::string::npos ||
        error.find("context size") != std::string::npos;
}

static bool request_has_active_resource_chunk(const common_agent_request & request) {
    return std::any_of(request.input_resources.begin(), request.input_resources.end(),
        [](const common_agent_input_resource & input) {
            return !input.resource.lineage.parent_uri.empty();
        });
}

static size_t estimate_common_agent_context_tokens(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_context_budget_config & budgets) {
    size_t characters = request.prompt.size();
    common_plan_context_config plan_config;
    plan_config.char_budget = budgets.plan_chars;
    plan_config.include_observations = false;
    common_plan_context_config observation_config;
    observation_config.char_budget = budgets.tool_observation_chars;
    if (request.working_state) {
        characters += render_common_agent_working_state(*request.working_state, budgets.plan_chars).size();
        characters += common_plan_render_tool_observations(plan, observation_config).size();
        for (const auto & input : request.input_resources) {
            characters += input.resource.uri.size() + input.resource.name.size();
        }
        return (characters + 3) / 4;
    }
    characters += common_plan_render_context(plan, plan_config).size();
    characters += common_plan_render_tool_observations(plan, observation_config).size();
    for (const auto & input : request.input_resources) {
        characters += input.resource.name.size() + input.resource.description.size() + input.resource.uri.size();
    }
    // This is deliberately a conservative host-side estimate. The model
    // adapter remains responsible for exact tokenizer accounting.
    return (characters + 3) / 4;
}

static void append_trace(
        common_agent_result & result,
        common_runtime_trace_stage stage,
        common_runtime_trace_kind kind,
        std::string detail,
        std::string plan_id = {},
        std::string step_id = {},
        std::string tool_name = {},
        std::string observation_id = {},
        std::string related_id = {}) {
    result.trace.push_back({
        stage,
        kind,
        std::move(detail),
        std::move(plan_id),
        std::move(step_id),
        std::move(tool_name),
        std::move(observation_id),
        std::move(related_id),
    });
}

static common_agent_deliberation_policy policy_after_escalation(
        const common_agent_deliberation_policy & current,
        common_agent_thinking_mode target) {
    return make_common_agent_escalated_policy(current, target);
}

static void append_event(
        common_agent_result & result,
        const common_agent_request & request,
        common_agent_event_type type,
        std::string detail,
        std::string memory_id = {},
        std::optional<std::string> plan_id = std::nullopt,
        std::string step_id = {},
        std::string observation_id = {},
        std::string tool_name = {},
        std::string resource_uri = {}) {
    common_agent_event event{
        type,
        std::move(detail),
        std::move(memory_id),
        std::move(plan_id),
        std::move(step_id),
        std::move(observation_id),
        std::move(tool_name),
        std::move(resource_uri),
    };
    result.events.push_back(event);
    if (request.event_sink) {
        request.event_sink(event);
    }
}

static void append_event(
        common_agent_result & result,
        const common_agent_request & request,
        common_agent_event event) {
    result.events.push_back(event);
    if (request.event_sink) {
        request.event_sink(event);
    }
}

static void append_observation_and_resource_events(
        common_agent_result & result,
        const common_agent_request & request,
        const std::string & detail,
        const std::string & plan_id,
        const std::string & step_id,
        const std::string & observation_id,
        const std::string & tool_name,
        const std::vector<common_runtime_resource_ref> & resource_refs) {
    append_event(
        result,
        request,
        common_agent_event_type::observation_recorded,
        detail,
        {},
        plan_id,
        step_id,
        observation_id,
        tool_name);
    for (const auto & resource_ref : resource_refs) {
        if (resource_ref.uri.rfind("agent-resource://", 0) == 0) {
            append_event(
                result,
                request,
                common_agent_event_type::resource_created,
                "host-owned resource recorded for observation",
                {},
                plan_id,
                step_id,
                observation_id,
                tool_name,
                resource_ref.uri);
        }
        append_event(
            result,
            request,
            common_agent_event_type::resource_attached,
            "resource attached to observation",
            {},
            plan_id,
            step_id,
            observation_id,
            tool_name,
            resource_ref.uri);
    }
}

static bool apply_request_objective(const common_agent_request & request, common_plan_state & plan, std::string & error) {
    const auto bounded = [](const std::string & value) { return value.size() <= 512; };
    const auto purpose = request.objective && !request.objective->purpose.empty() ? request.objective->purpose : request.prompt;
    const auto outcome = request.objective && !request.objective->desired_outcome.empty() ? request.objective->desired_outcome : request.prompt;
    if (purpose.empty() || outcome.empty() || !bounded(purpose) || !bounded(outcome)) {
        error = "agent objective requires bounded purpose and desired outcome";
        return false;
    }
    if (request.objective) {
        if (request.objective->success_criteria.size() > 8 || request.objective->constraints.size() > 8) {
            error = "agent objective exceeds bounded criteria or constraints";
            return false;
        }
        for (const auto & value : request.objective->success_criteria) if (value.empty() || !bounded(value)) { error = "agent objective contains invalid success criteria"; return false; }
        for (const auto & value : request.objective->constraints) if (value.empty() || !bounded(value)) { error = "agent objective contains invalid constraints"; return false; }
    }
    // Purpose is owned by the caller, rather than an untrusted planner
    // proposal. A planner may refine the executable goal, but cannot replace
    // why the turn was requested.
    plan.purpose = purpose;
    if (plan.goal.empty()) plan.goal = outcome;
    if (request.objective && !request.objective->success_criteria.empty()) {
        plan.success_criteria.clear();
        for (const auto & criterion : request.objective->success_criteria) {
            if (!plan.success_criteria.empty()) plan.success_criteria += "; ";
            plan.success_criteria += criterion;
        }
        if (plan.success_criteria.size() > 1024) { error = "agent objective success criteria are too long"; return false; }
    } else if (plan.success_criteria.empty()) {
        plan.success_criteria = "Provide a grounded, concise response.";
    }
    if (request.objective) {
        for (size_t index = 0; index < request.objective->constraints.size(); ++index) {
            plan.constraints.push_back({"objective-constraint-" + std::to_string(index + 1), request.objective->constraints[index], true});
        }
    }
    for (auto & step : plan.steps) if (step.intended_contribution.empty()) step.intended_contribution = step.objective;
    error.clear();
    return true;
}

static common_agent_failure tool_failure(
        const std::string & tool_name,
        const std::string & step_id,
        const std::string & evidence_id,
        const std::string & code,
        common_agent_failure_class classification,
        bool retryable,
        const std::string & safe_summary) {
    return {code, classification, "tool_execution", tool_name, step_id, evidence_id, retryable, safe_summary};
}

static bool plan_has_completed_tool_step(const common_plan_state & plan) {
    return std::any_of(plan.steps.begin(), plan.steps.end(), [](const common_plan_step & step) {
        return common_plan_step_effective_mode(step) == common_plan_step_mode::tool &&
            step.status == common_plan_step_status::completed &&
            step.tool_call.has_value();
    });
}

static bool plan_has_pending_mandatory_tool_step(const common_plan_state & plan) {
    return std::any_of(plan.steps.begin(), plan.steps.end(), [](const common_plan_step & step) {
        return common_plan_step_effective_mode(step) == common_plan_step_mode::tool &&
            !step.optional &&
            (step.status == common_plan_step_status::pending ||
             step.status == common_plan_step_status::active);
    });
}

static bool plan_has_failed_mandatory_tool_step(const common_plan_state & plan) {
    return std::any_of(plan.steps.begin(), plan.steps.end(), [](const common_plan_step & step) {
        return common_plan_step_effective_mode(step) == common_plan_step_mode::tool &&
            !step.optional &&
            step.status == common_plan_step_status::failed;
    });
}

static common_agent_failure structured_tool_failure(const std::string & tool_name, const std::string & step_id,
        const std::string & evidence_id, const common_tool_execution_result & result) {
    const auto classification = result.failure_class == common_tool_failure_class::validation ? common_agent_failure_class::validation :
        result.failure_class == common_tool_failure_class::policy ? common_agent_failure_class::policy :
        result.failure_class == common_tool_failure_class::not_found ? common_agent_failure_class::not_found :
        result.failure_class == common_tool_failure_class::timeout ? common_agent_failure_class::timeout :
        result.failure_class == common_tool_failure_class::network ? common_agent_failure_class::network :
        result.failure_class == common_tool_failure_class::limit ? common_agent_failure_class::limit : common_agent_failure_class::execution;
    return tool_failure(tool_name, step_id, evidence_id, result.failure_code.empty() ? "tool.execution_failed" : result.failure_code,
        classification, result.retryable, result.safe_summary.empty() ? "The tool failed." : result.safe_summary);
}

static std::string tool_observation_payload(const common_tool_execution_result & result) {
    // Structured output is the binding source. A summary-only result is still
    // recorded as JSON, but deliberately has no typed fields to autowire.
    if (!result.output.empty()) return result.output;
    json fallback = json::object();
    if (!result.content_summary.empty()) fallback["summary"] = result.content_summary;
    else if (!result.safe_summary.empty()) fallback["summary"] = result.safe_summary;
    return fallback.dump();
}

static std::string tool_repair_context_json(const common_agent_tool_repair_context & context) {
    json available = json::array();
    for (const auto & name : context.available_tools) available.push_back(name);
    json candidates = json::array();
    for (const auto & name : context.candidate_tools) candidates.push_back(name);
    json value = {
        {"tool", context.tool_name},
        {"error", context.validation_error},
        {"arguments_skeleton", context.arguments_skeleton.empty()
            ? json(nullptr)
            : json::parse(context.arguments_skeleton, nullptr, false)},
        {"available_tools", std::move(available)},
        {"candidate_tools", std::move(candidates)},
        {"normalized_arguments", context.normalized_arguments.empty()
            ? json(nullptr)
            : json::parse(context.normalized_arguments, nullptr, false)},
        {"normalization_applied", context.normalization_applied},
        {"compact_contract", context.compact_contract},
    };
    return value.dump();
}

static bool normalize_agent_tool_call(
        common_agent_tool_call & call,
        const common_agent_tool_runtime * tools = nullptr,
        bool * name_normalized = nullptr,
        std::vector<std::string> * name_candidates = nullptr) {
    if (name_normalized) *name_normalized = false;
    if (name_candidates) name_candidates->clear();
    if (tools) {
        std::string resolved_name;
        std::vector<std::string> candidates;
        if (tools->resolve_tool_name(call.name, resolved_name, candidates)) {
            call.name = std::move(resolved_name);
            if (name_normalized) *name_normalized = true;
        }
        if (name_candidates) *name_candidates = std::move(candidates);
    }
    const auto original = call.arguments_json;
    std::string normalized;
    std::string ignored_error;
    if (!common_plan_normalize_tool_arguments_json(
            call.name,
            call.arguments_json,
            normalized,
            ignored_error)) {
        return false;
    }
    call.arguments_json = std::move(normalized);
    return call.arguments_json != original;
}

static bool is_research_acquisition_tool(const std::string & tool_name) {
    return tool_name == "resource.read" || tool_name == "resource_read" ||
        tool_name == "repository.search" ||
        tool_name == "web.search" || tool_name == "web_search" ||
        tool_name == "web.fetch" || tool_name == "web_fetch" ||
        tool_name == "memory.get" || tool_name == "memory_get" ||
        tool_name == "memory.search" || tool_name == "memory_search";
}

static bool merge_reflection_tool_repair_arguments(
        const common_plan_state & plan,
        common_plan_operation & operation,
        std::string & error) {
    if (!operation.step || !operation.step->tool_call) return false;

    const common_plan_step * failed_base = nullptr;
    for (auto it = plan.steps.rbegin(); it != plan.steps.rend(); ++it) {
        if (it->status == common_plan_step_status::failed && it->tool_call &&
                it->tool_call->name == operation.step->tool_call->name) {
            failed_base = &*it;
            break;
        }
    }
    if (!failed_base) return false;

    std::string merged;
    if (!common_plan_merge_tool_arguments_json(
            failed_base->tool_call->arguments_json,
            operation.step->tool_call->arguments_json,
            merged,
            error)) {
        return false;
    }
    operation.step->tool_call->arguments_json = std::move(merged);
    return true;
}

static bool reflection_operation_has_active_tool_failure(
        const common_plan_state & plan,
        const common_plan_operation & operation) {
    const auto failed_step = [&](const std::string & step_id) {
        for (const auto & step : plan.steps) {
            if (step.id == step_id) {
                return step.status == common_plan_step_status::failed &&
                    common_plan_step_effective_mode(step) == common_plan_step_mode::tool;
            }
        }
        return false;
    };

    if (operation.kind == common_plan_operation_kind::add_step) {
        if (!operation.step || !operation.step->tool_call) return true;
        bool completed_same_tool = false;
        for (const auto & step : plan.steps) {
            if (step.status == common_plan_step_status::failed && step.tool_call &&
                    step.tool_call->name == operation.step->tool_call->name) {
                return true;
            }
            completed_same_tool = completed_same_tool ||
                (step.status == common_plan_step_status::completed && step.tool_call &&
                 step.tool_call->name == operation.step->tool_call->name);
        }
        // A new tool operation can be a legitimate reflection extension. It
        // becomes stale only when reflection tries to schedule another call
        // for a tool that has already completed successfully.
        return !completed_same_tool;
    }

    if (operation.kind == common_plan_operation_kind::replace_step && operation.step) {
        return failed_step(operation.step->id);
    }
    if (operation.kind == common_plan_operation_kind::activate_step ||
            operation.kind == common_plan_operation_kind::reset_step ||
            operation.kind == common_plan_operation_kind::unblock_step) {
        return operation.step_id && failed_step(*operation.step_id);
    }
    return true;
}

static void discard_stale_reflection_repairs(
        const common_plan_state & plan,
        common_reflection_result & reflection,
        common_agent_result & result) {
    std::vector<common_plan_operation> retained;
    retained.reserve(reflection.proposed_plan_operations.size());
    for (auto & operation : reflection.proposed_plan_operations) {
        const bool repair_operation = operation.kind == common_plan_operation_kind::add_step ||
            operation.kind == common_plan_operation_kind::replace_step ||
            operation.kind == common_plan_operation_kind::activate_step ||
            operation.kind == common_plan_operation_kind::reset_step ||
            operation.kind == common_plan_operation_kind::unblock_step;
        if (repair_operation && !reflection_operation_has_active_tool_failure(plan, operation)) {
            append_trace(result, common_runtime_trace_stage::reflection,
                common_runtime_trace_kind::skipped,
                "stale reflection repair ignored: no active failed tool step", plan.id,
                operation.step_id.value_or(operation.step ? operation.step->id : std::string()));
            continue;
        }
        retained.push_back(std::move(operation));
    }
    reflection.proposed_plan_operations = std::move(retained);
}

static std::string next_tool_observation_id(const common_plan_state & plan, const std::string & step_id, const std::string & tool_name) {
    const std::string base = "tool:" + step_id + ":" + tool_name;
    const std::string attempt_prefix = base + ":attempt:";
    bool seen_base = false;
    size_t next_attempt = 2;
    for (const auto & observation : plan.observations) {
        if (observation.id == base) {
            seen_base = true;
            continue;
        }
        if (observation.id.rfind(attempt_prefix, 0) != 0) continue;
        seen_base = true;
        try {
            next_attempt = std::max(next_attempt, static_cast<size_t>(std::stoull(observation.id.substr(attempt_prefix.size())) + 1));
        } catch (const std::exception &) {
            next_attempt = std::max(next_attempt, static_cast<size_t>(3));
        }
    }
    return seen_base ? attempt_prefix + std::to_string(next_attempt) : base;
}

static std::string tool_argument_keys(const std::string & arguments_json) {
    const auto arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object()) return "invalid";
    std::string keys;
    for (const auto & item : arguments.items()) {
        if (!keys.empty()) keys += ",";
        keys += item.key();
        if (keys.size() >= 256) {
            keys.resize(256);
            keys += "...";
            break;
        }
    }
    return keys.empty() ? "<none>" : keys;
}

static std::string tool_argument_resource_ref(const std::string & arguments_json) {
    const auto arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object() || !arguments.contains("resource") ||
            !arguments["resource"].is_string()) {
        return {};
    }
    std::string value = arguments["resource"].get<std::string>();
    if (value.size() > 256) value.resize(256);
    return " resource_ref=" + value;
}

static std::string tool_trace_diagnostic(const std::string & diagnostic) {
    if (diagnostic.empty()) return {};
    std::string value;
    value.reserve(std::min<size_t>(diagnostic.size(), 256));
    for (const unsigned char ch : diagnostic) {
        if (value.size() >= 256) break;
        value += (ch >= 0x20 && ch != 0x7f) ? static_cast<char>(ch) : ' ';
    }
    return " diagnostic=" + value;
}

static bool tool_argument_key_is_sensitive(const std::string & key) {
    std::string normalized;
    normalized.reserve(key.size());
    for (const unsigned char ch : key) {
        if (std::isalnum(ch)) normalized += static_cast<char>(std::tolower(ch));
    }
    return normalized.find("secret") != std::string::npos ||
        normalized.find("token") != std::string::npos ||
        normalized.find("password") != std::string::npos ||
        normalized.find("authorization") != std::string::npos ||
        normalized.find("apikey") != std::string::npos ||
        normalized.find("bearer") != std::string::npos;
}

static json bounded_tool_trace_arguments(const json & value, size_t depth = 0) {
    if (depth >= 3) return "<nested>";
    if (value.is_object()) {
        json sanitized = json::object();
        size_t count = 0;
        for (const auto & item : value.items()) {
            if (count++ >= 16) {
                sanitized["..."] = "<truncated>";
                break;
            }
            sanitized[item.key()] = tool_argument_key_is_sensitive(item.key())
                ? json("<redacted>")
                : bounded_tool_trace_arguments(item.value(), depth + 1);
        }
        return sanitized;
    }
    if (value.is_array()) {
        json sanitized = json::array();
        const size_t limit = std::min<size_t>(value.size(), 16);
        for (size_t index = 0; index < limit; ++index) {
            sanitized.push_back(bounded_tool_trace_arguments(value[index], depth + 1));
        }
        if (value.size() > limit) sanitized.push_back("<truncated>");
        return sanitized;
    }
    if (value.is_string()) {
        std::string text = value.get<std::string>();
        if (text.size() > 256) text = text.substr(0, 256) + "...";
        return text;
    }
    return value;
}

static std::string tool_trace_attempt_arguments(
        const std::string & step_id,
        const std::string & model_arguments_json,
        const std::string & normalized_arguments_json) {
    if (step_id.rfind("repair", 0) != 0) return {};
    const auto model_arguments = json::parse(model_arguments_json, nullptr, false);
    const auto normalized_arguments = json::parse(normalized_arguments_json, nullptr, false);
    if (model_arguments.is_discarded() || normalized_arguments.is_discarded()) {
        return " model_args=<invalid-json>";
    }
    const auto bounded_model = bounded_tool_trace_arguments(model_arguments).dump();
    const auto bounded_normalized = bounded_tool_trace_arguments(normalized_arguments).dump();
    std::string result = " model_args=" + bounded_model;
    if (bounded_normalized != bounded_model) result += " normalized_args=" + bounded_normalized;
    if (result.size() > 1536) result.resize(1536);
    return result;
}

static bool is_incomplete_tool_call(
        const std::string & tool_name,
        const json & arguments,
        std::string & validation_error) {
    if (!arguments.is_object()) {
        return false;
    }

    if (validation_error == "required contract field is missing" && arguments.empty()) {
        if (tool_name == "memory.get" || tool_name == "memory_get") {
            validation_error = "memory_get requires an id from a prior memory_search or recorded memory reference";
        }
        return true;
    }

    if ((tool_name == "memory.get" || tool_name == "memory_get") && !arguments.contains("id")) {
        validation_error = "memory_get requires an id from a prior memory_search or recorded memory reference";
        return true;
    }

    return false;
}

common_agent_runtime::common_agent_runtime(common_plan_store & store, common_planner & planner, common_action_executor & executor, common_reflection_engine & reflector, const common_agent_tool_runtime * tools, common_memory_post_turn_learner * memory_learner, const common_agent_research_answer_verifier * research_verifier, common_agent_context_budget_config context_budgets, size_t context_size_tokens, size_t reserved_output_tokens, common_agent_context_token_estimator context_token_estimator) : store(store), planner(planner), executor(executor), reflector(reflector), tools(tools), memory_learner(memory_learner), research_verifier(research_verifier), context_budgets(std::move(context_budgets)), context_size_tokens(context_size_tokens), reserved_output_tokens(reserved_output_tokens), context_token_estimator(std::move(context_token_estimator)) {}

common_agent_result common_agent_runtime::run(const common_agent_request & input_request) {
    common_agent_request request = input_request;
    common_agent_result result;
    std::string error;
    common_agent_runtime_turn_context turn{
        request,
        result,
        error,
        tools,
        reflector,
        {},
        {},
        {},
        {},
        {},
        false,
        false,
        [&result, &request](common_agent_event_type type, std::string detail) {
            append_event(result, request, type, std::move(detail));
        },
        [&result, &request](common_agent_event event) {
            append_event(result, request, std::move(event));
        },
        [&result](
                common_runtime_trace_stage stage,
                common_runtime_trace_kind kind,
                std::string detail,
                std::string plan_id,
                std::string related_id) {
            append_trace(
                result,
                stage,
                kind,
                std::move(detail),
                std::move(plan_id),
                {},
                {},
                {},
                std::move(related_id));
        },
        [](
                const common_agent_deliberation_policy & current,
                common_agent_thinking_mode target) {
            return policy_after_escalation(current, target);
        },
    };
    append_trace(result, common_runtime_trace_stage::turn, common_runtime_trace_kind::started,
        "agent runtime turn started", {}, {}, {}, {}, request.turn_id);
    common_agent_escalation_signals escalation_signals;
    escalation_signals.multiple_constraints = request.objective &&
        request.objective->constraints.size() > 1;
    escalation_signals.resource_comparison_required = request.input_resources.size() > 1;
    escalation_signals.user_requested_verification = request.prompt.find("verify") != std::string::npos ||
        request.prompt.find("compare") != std::string::npos;
    escalation_signals.external_uncertainty = request.prompt.find("latest") != std::string::npos ||
        request.prompt.find("current") != std::string::npos ||
        request.prompt.find("uncertain") != std::string::npos;
    const auto escalation = resolve_common_agent_escalation(
        request.deliberation_policy, escalation_signals);
    if (escalation.escalation_requested && escalation.allowed) {
        request.deliberation_policy = policy_after_escalation(
            request.deliberation_policy, escalation.to_mode);
        append_event(result, request, common_agent_event_type::thinking_escalation_allowed,
            escalation.summary + " reason=" + common_agent_escalation_reason_name(escalation.reason));
    } else if (escalation.escalation_requested) {
        append_event(result, request, common_agent_event_type::thinking_escalation_denied,
            escalation.summary + " reason=" + common_agent_escalation_reason_name(escalation.reason));
    }
    append_event(result, request, common_agent_event_type::thinking_mode_resolved,
        std::string("thinking mode resolved to ") +
            common_agent_thinking_mode_name(request.deliberation_policy.mode));
    if (request.deliberation_policy.require_plan && !request.enable_planning) {
        result.error = "deliberation mode requires planning";
        return result;
    }
    if (request.deliberation_policy.require_step_review &&
            (!request.enable_reflection || request.max_reflection_rounds == 0)) {
        result.error = "deliberation mode requires bounded step review";
        return result;
    }
    if (!request.enable_planning) { result.error = "planning is disabled"; return result; }
    const auto normalize_planned_tool_step = [&](common_plan_step & step, bool degrade_on_any_invalid, std::string & detail) {
        detail.clear();
        if (!step.tool_call) return false;
        common_agent_tool_call call{step.tool_call->name, step.tool_call->arguments_json};
        bool name_normalization_applied = false;
        std::vector<std::string> name_candidates;
        normalize_agent_tool_call(call, tools, &name_normalization_applied, &name_candidates);
        common_agent_runtime_apply_safe_tool_defaults(request, call);
        step.tool_call->name = call.name;
        if (name_normalization_applied) step.selected_tool = call.name;
        step.tool_call->arguments_json = call.arguments_json;
        if (!tools) {
            if (!degrade_on_any_invalid) return false;
            step.selected_tool.reset();
            step.tool_call.reset();
            step.mode = common_plan_step_mode::reasoning;
            step.required_evidence.clear();
            detail = "registered tool execution is unavailable";
            return true;
        }
        if (request.deliberation_policy.mode == common_agent_thinking_mode::research &&
                is_research_acquisition_tool(call.name)) {
            step.selected_tool.reset();
            step.tool_call.reset();
            step.mode = common_plan_step_mode::reasoning;
            step.required_evidence.clear();
            detail = "research acquisition is owned by the research controller";
            return true;
        }
        std::string validation_error;
        const bool policy_allowed = tools->is_read_only(call.name) || (request.allow_policy_gated_tool_proposals && tools->is_policy_gated(call.name));
        const bool valid_tool_call = policy_allowed && tools->validate(call, validation_error);
        if (valid_tool_call) return false;
        if (validation_error.empty()) validation_error = "tool is not approved by policy";
        const auto arguments = json::parse(call.arguments_json, nullptr, false);
        const bool incomplete_tool_call = is_incomplete_tool_call(
            call.name,
            arguments,
            validation_error);
        // Preserve ambiguous name matches as tool steps so the runtime can
        // expose the bounded candidate list to repair/reflection. Only an
        // unambiguous, schema-invalid call is degraded to reasoning here.
        if (!degrade_on_any_invalid && !incomplete_tool_call) return false;
        if (!name_candidates.empty()) return false;
        // A unique fuzzy name match is already a host-approved canonical tool
        // name. Preserve it for ordinary schema repair instead of losing the
        // semantic tool intent by degrading the step to reasoning.
        if (name_normalization_applied) return false;
        step.selected_tool.reset();
        step.tool_call.reset();
        step.mode = common_plan_step_mode::reasoning;
        step.required_evidence.clear();
        detail = validation_error;
        return true;
    };
    for (const auto & hit : request.memories) {
        result.memory_ids.push_back(hit.memory.id);
        append_event(result, request, common_agent_event_type::memory_retrieved, "memory supplied to agent runtime", hit.memory.id);
        append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
            "memory supplied to runtime", {}, {}, {}, hit.memory.id);
    }
    common_plan_state plan;
    if (request.plan_id && !request.plan_id->empty()) {
        const auto existing = store.get(*request.plan_id, error);
        if (!error.empty()) { result.error = error; return result; }
        if (existing) {
            if (!common_plan_scope_matches(*existing, request.plan_scope, request.namespace_id, request.session_id, request.project_id, request.turn_id)) {
                result.error = "existing plan identity does not match requested scope";
                return result;
            }
            plan = *existing;
            append_event(result, request, common_agent_event_type::plan_updated, "existing plan resumed", {}, plan.id);
            append_trace(result, common_runtime_trace_stage::plan, common_runtime_trace_kind::updated,
                "existing plan resumed", plan.id);
        }
    }
    if (plan.id.empty()) {
        auto proposal = planner.create_plan_result(request, error);
        if (!error.empty()) { result.error = error; return result; }
        if (proposal.generation) {
            result.generation_records.push_back(common_agent_generation_record_from_result(
                common_agent_generation_stage::planner,
                *proposal.generation));
        }
        if (!apply_request_objective(request, proposal.plan, error)) { result.error = error; return result; }
        if (!proposal.operations.empty() && !proposal.plan.steps.empty()) {
            proposal.plan.steps.clear();
            proposal.plan.active_step_id.reset();
        }
        std::vector<std::string> initial_guardrail_events;
        for (auto & step : proposal.plan.steps) {
            std::string detail;
            if (normalize_planned_tool_step(step, false, detail)) {
                initial_guardrail_events.push_back("initial tool step degraded to reasoning: " + detail);
            }
        }
        for (auto & operation : proposal.operations) {
            if (operation.step && operation.step->intended_contribution.empty()) {
                operation.step->intended_contribution = operation.step->objective;
            }
            if (operation.step) {
                std::string detail;
                if (normalize_planned_tool_step(*operation.step, false, detail)) {
                    initial_guardrail_events.push_back("initial tool step degraded to reasoning: " + detail);
                }
            }
        }
        if (request.plan_id) proposal.plan.id = *request.plan_id;
        proposal.plan.scope = request.plan_scope;
        proposal.plan.namespace_id = request.namespace_id;
        if (proposal.plan.session_id.empty()) proposal.plan.session_id = request.session_id;
        proposal.plan.project_id = request.project_id;
        proposal.plan.turn_id = request.turn_id;
        if (!store.create(proposal.plan, error)) { result.error = error; return result; }
        plan = proposal.plan;
        append_event(result, request, common_agent_event_type::plan_created, "plan created", {}, plan.id);
        append_trace(result, common_runtime_trace_stage::plan, common_runtime_trace_kind::started,
            "plan created", plan.id);
        for (const auto & detail : initial_guardrail_events) {
            append_event(result, request, {common_agent_event_type::tool_rejected, detail, {}, plan.id});
            append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::failed,
                detail, plan.id);
        }
        for (auto op : proposal.operations) {
            common_plan_bind_memory_provenance(op, request.memories);
            op.plan_id = plan.id;
            op.expected_version = plan.version;
            if (!store.apply(op, plan, error)) { result.error = error; return result; }
            append_event(result, request, common_agent_event_type::plan_updated, "initial plan operation applied", {}, plan.id);
            append_trace(result, common_runtime_trace_stage::plan, common_runtime_trace_kind::updated,
                "initial plan operation applied", plan.id,
                op.step_id.value_or(op.step ? op.step->id : std::string()));
        }
    }
    if (tools && !tools->validate_plan(plan, error)) {
        result.error = "tool workflow validation failed: " + error;
        return result;
    }
    turn.plan_store = &store;
    turn.outer_plan = &plan;
    if (request.deliberation_policy.mode == common_agent_thinking_mode::research) {
        request.plan_id = plan.id;
        if (!run_common_agent_research_phase(turn)) return result;
    }
    result.plan_id = plan.id;
    if (!turn.research_synthesis_context.empty()) {
        request.prompt += "\n\nHost-approved research synthesis context:\n";
        request.prompt += turn.research_synthesis_context;
        if (request.prompt.size() > 8192) request.prompt.resize(8192);
    }

    if (request.user_correction) {
        const auto & correction = *request.user_correction;
        if (correction.source_turn_id.empty() || correction.statement.empty() || correction.statement.size() > 512) {
            result.error = "explicit user correction requires bounded source turn and statement";
            return result;
        }
        const std::string observation_id = "feedback:correction:" + std::to_string(plan.version);
        common_plan_operation observed;
        observed.kind = common_plan_operation_kind::record_observation;
        observed.plan_id = plan.id;
        observed.expected_version = plan.version;
        observed.reason_summary = "explicit user correction";
        observed.observation = common_plan_observation{
            observation_id,
            "user_correction",
                    common_agent_runtime_user_correction_json(
                        correction.source_turn_id,
                        correction.statement),
                    1.0f,
                    {},
                    {},
            0};
        if (!store.apply(observed, plan, error)) { result.error = error; return result; }
        result.learning_signals.push_back({common_learning_signal_type::user_correction, plan.id, {}, {}, observation_id,
            "explicit user correction supplied by the caller"});
        append_event(result, request, common_agent_event_type::plan_updated, "explicit user correction recorded", {}, plan.id);
        append_observation_and_resource_events(
            result,
            request,
            "explicit user correction recorded",
            plan.id,
            {},
            observation_id,
            "user_correction",
            {});
        append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
            "explicit user correction recorded", plan.id, {}, {}, observation_id);
    }

    const auto activate_next_ready_step = [&]() -> bool {
        bool has_active_step = false;
        for (const auto & step : plan.steps) if (step.status == common_plan_step_status::active) { has_active_step = true; break; }
        if (has_active_step) return true;

        const auto schedule = common_plan_schedule(plan);
        if (schedule.complete && plan.status == common_plan_status::active) {
            auto completed_candidate = plan;
            completed_candidate.status = common_plan_status::completed;
            const auto evaluation = common_plan_evaluate_goal(completed_candidate);
            if (!evaluation.evidence_sufficient) {
                result.error = "plan goal is not sufficiently evidenced";
                if (!evaluation.unmet_criteria.empty()) result.error += ": " + evaluation.unmet_criteria.front();
                return false;
            }
            common_plan_operation complete_plan;
            complete_plan.kind = common_plan_operation_kind::complete_plan;
            complete_plan.plan_id = plan.id;
            complete_plan.expected_version = plan.version;
            complete_plan.reason_summary = "all mandatory plan steps completed";
            if (!store.apply(complete_plan, plan, error)) return false;
            append_event(result, request, {common_agent_event_type::plan_updated, "plan completed by scheduler", {}, plan.id});
            return false;
        }
        if (schedule.ready_step_ids.empty()) return false;

        common_plan_operation activate;
        activate.kind = common_plan_operation_kind::activate_step;
        activate.plan_id = plan.id;
        activate.expected_version = plan.version;
        activate.step_id = schedule.ready_step_ids.front();
        activate.reason_summary = "scheduler selected dependency-ready step";
        if (!store.apply(activate, plan, error)) return false;
        append_event(result, request, {common_agent_event_type::plan_updated, "scheduler activated plan step", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::step, common_runtime_trace_kind::started,
            "scheduler activated plan step", plan.id, *activate.step_id);
        return true;
    };

    const auto block_unrepaired_tool_failure = [&]() -> bool {
        for (const auto & step : plan.steps) {
            if (step.status == common_plan_step_status::failed &&
                    common_plan_step_effective_mode(step) == common_plan_step_mode::tool) {
                result.error = "final synthesis is blocked by an unrepaired failed tool step: " + step.id;
                error = result.error;
                append_trace(result, common_runtime_trace_stage::response,
                    common_runtime_trace_kind::failed,
                    "final response blocked until failed tool step is repaired and rerun",
                    plan.id, step.id, step.selected_tool.value_or(std::string()));
                return true;
            }
        }
        return false;
    };

    const auto has_unresolved_required_tool_step = [&]() -> bool {
        for (const auto & step : plan.steps) {
            if (common_plan_step_effective_mode(step) != common_plan_step_mode::tool || step.optional) continue;
            if (step.status != common_plan_step_status::completed &&
                    step.status != common_plan_step_status::skipped) {
                return true;
            }
        }
        return false;
    };

    const auto complete_active_synthesis_step = [&]() -> bool {
        if (!plan.active_step_id) return true;
        common_plan_step * active = nullptr;
        for (auto & step : plan.steps) if (step.id == *plan.active_step_id) { active = &step; break; }
        if (!active || active->status != common_plan_step_status::active || common_plan_step_effective_mode(*active) != common_plan_step_mode::final_response) return true;

        // A final response cannot turn a failed tool-backed step into
        // evidence.  Reflection must reset/replace and rerun the failed step
        // before synthesis is allowed to complete.
        if (block_unrepaired_tool_failure()) return false;

        // A bounded resource slice is an intermediate inference result.  The
        // resource driver records that slice after this runtime returns and
        // then runs one synthesis slice over the complete plan evidence.  Do
        // not complete the final-response step while a parent-linked chunk is
        // still the active model input.
        if (request_has_active_resource_chunk(request)) return true;

        common_plan_chunk_synthesis_input chunk_synthesis;
        std::string chunk_error;
        if (!common_plan_chunk_synthesis_from_observations(
                    plan.observations, {}, chunk_synthesis, chunk_error)) {
            common_plan_operation block;
            block.kind = common_plan_operation_kind::block_step;
            block.plan_id = plan.id;
            block.expected_version = plan.version;
            block.step_id = active->id;
            block.reason_summary = "resource synthesis blocked by conflicting chunk observations";
            if (!store.apply(block, plan, error)) return false;
            append_event(result, request, common_agent_event_type::plan_updated,
                "resource synthesis step blocked by conflicting chunk observations", {}, plan.id, active->id);
            result.error = "resource synthesis is blocked by conflicting chunk observations: " + chunk_error;
            error = result.error;
            return false;
        }
        if (chunk_synthesis.chunk_count > 0 &&
                chunk_synthesis.status != common_plan_chunk_synthesis_status::complete) {
            result.error = "resource synthesis is incomplete; missing chunk observations remain";
            error = result.error;
            return false;
        }

        common_plan_operation complete;
        complete.kind = common_plan_operation_kind::complete_step;
        complete.plan_id = plan.id;
        complete.expected_version = plan.version;
        complete.step_id = active->id;
        complete.reason_summary = "final response synthesis completed";
        const std::string completed_step_id = active->id;
        for (const auto & observation : plan.observations) {
            complete.evidence_ids.push_back(observation.id);
            complete.evidence_ids.insert(complete.evidence_ids.end(), observation.evidence_ids.begin(), observation.evidence_ids.end());
        }
        if (!store.apply(complete, plan, error)) return false;
        append_event(result, request, {common_agent_event_type::plan_updated, "final synthesis step completed", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::step, common_runtime_trace_kind::completed,
            "final synthesis step completed", plan.id, completed_step_id);
        activate_next_ready_step();
        return error.empty();
    };

    std::vector<std::string> guidance;
    std::set<std::string> executed_step_ids;
    common_agent_tool_navigation_context tool_navigation;
    bool tool_navigation_active = false;
    bool executed_request_tool = false;
    size_t plan_revision_count = 0;
    size_t runtime_iteration_limit = request.max_iterations;
    for (size_t iteration = 0; iteration < runtime_iteration_limit; ++iteration) {
        size_t tool_batches = 0;
        bool defer_draft_for_required_tools = false;
        // Execute the contiguous, dependency-ready tool chain before drafting.
        // This makes normal plan progression deterministic; reflection remains
        // reserved for repair or replanning.
        while (true) {
            std::optional<common_agent_tool_call> tool_call;
            std::string tool_step_id = "request";
            if (plan.active_step_id) for (const auto & step : plan.steps) if (step.id == *plan.active_step_id && step.status == common_plan_step_status::active && common_plan_step_effective_mode(step) == common_plan_step_mode::reasoning && !executed_step_ids.count(step.id)) {
                const auto reasoning_result = executor.generate_reasoning_result(request, plan, step, error);
                std::string reasoning = reasoning_result.content;
                if (!error.empty()) { result.error = "reasoning step failed: " + error; return result; }
                result.generation_records.push_back(common_agent_generation_record_from_result(
                    common_agent_generation_stage::reasoning,
                    reasoning_result));
                result.reasoning_decoded_tokens += reasoning_result.decoded_tokens;
                result.total_decoded_tokens += reasoning_result.decoded_tokens;
                // Reasoning is evidence only. Small local models occasionally
                // ignore the requested JSON envelope; preserve that bounded
                // output as explicitly unstructured evidence instead of
                // failing an otherwise valid blueprint execution.
                reasoning = common_agent_runtime_normalize_reasoning_observation_json(reasoning);
                if (reasoning.size() > 4096) { result.error = "reasoning step result is too large"; return result; }
                common_plan_operation observed;
                observed.kind = common_plan_operation_kind::record_observation;
                observed.plan_id = plan.id;
                observed.expected_version = plan.version;
                observed.reason_summary = "reasoning step result";
                observed.observation = common_plan_observation{"reasoning:" + step.id, "reasoning", reasoning, 1.0f, {}, {}, 0};
                if (!store.apply(observed, plan, error)) { result.error = error; return result; }
                common_plan_operation complete;
                complete.kind = common_plan_operation_kind::complete_step;
                complete.plan_id = plan.id;
                complete.expected_version = plan.version;
                complete.step_id = step.id;
                complete.reason_summary = "reasoning step completed";
                if (!store.apply(complete, plan, error)) { result.error = error; return result; }
                executed_step_ids.insert(step.id);
                append_event(result, request, common_agent_event_type::plan_updated, "reasoning observation recorded", {}, plan.id);
                append_observation_and_resource_events(
                    result,
                    request,
                    "reasoning observation recorded",
                    plan.id,
                    step.id,
                    "reasoning:" + step.id,
                    "reasoning",
                    {});
                append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
                    "reasoning observation recorded", plan.id, step.id, {}, "reasoning:" + step.id);
                append_trace(result, common_runtime_trace_stage::step, common_runtime_trace_kind::completed,
                    "reasoning step completed", plan.id, step.id);
                if (!activate_next_ready_step()) break;
                continue;
            }
            if (plan.active_step_id) for (const auto & step : plan.steps) if (step.id == *plan.active_step_id && step.status == common_plan_step_status::active && common_plan_step_effective_mode(step) == common_plan_step_mode::tool && !executed_step_ids.count(step.id)) {
                if (step.selected_tool && *step.selected_tool != step.tool_call->name) { result.error = "active step selected tool does not match its tool call"; return result; }
                tool_call = common_agent_tool_call{step.tool_call->name, step.tool_call->arguments_json};
                const common_plan_tool_dataflow_contract_resolver dataflow_resolver =
                    tools ? [this](const std::string & tool_name,
                            common_plan_tool_dataflow_contract & contract,
                            std::string & contract_error) {
                        return tools->describe_tool_dataflow(tool_name, contract, contract_error);
                    } : common_plan_tool_dataflow_contract_resolver{};
                if (!common_plan_materialize_tool_arguments(
                        plan, step, tool_call->arguments_json, tool_call->arguments_json,
                        error, dataflow_resolver)) { append_event(result, request, {common_agent_event_type::tool_rejected, error, {}, plan.id}); result.error = "tool argument binding failed: " + error; return result; }
                tool_step_id = step.id;
                break;
            }
            if (!tool_call && request.tool_call && !executed_request_tool) tool_call = request.tool_call;
            if (!tool_call) {
                if (plan.active_step_id || !activate_next_ready_step()) break;
                continue;
            }
            if (tool_batches >= request.max_tool_batches) break;
            if (!tools || request.max_tool_batches == 0) { result.failures.push_back(tool_failure(tool_call->name, tool_step_id, {}, "tool.unavailable", common_agent_failure_class::execution, false, "Registered tool execution is unavailable.")); append_event(result, request, {common_agent_event_type::tool_rejected, "registered tool execution is unavailable", {}, plan.id}); result.error = "registered tool execution is unavailable"; return result; }
            bool name_normalization_applied = false;
            std::vector<std::string> name_candidates;
            const bool normalization_applied = normalize_agent_tool_call(
                *tool_call, tools, &name_normalization_applied, &name_candidates);
            if (!tools->is_available(tool_call->name)) {
                auto repair = tools->make_repair_context(*tool_call, "tool is unavailable in the effective runtime view");
                repair.candidate_tools = std::move(name_candidates);
                repair.normalized_arguments = tool_call->arguments_json;
                repair.normalization_applied = normalization_applied || name_normalization_applied;
                const std::string failure_observation_id = next_tool_observation_id(plan, tool_step_id, tool_call->name);
                auto failure = tool_failure(tool_call->name, tool_step_id, failure_observation_id, "tool.unavailable", common_agent_failure_class::not_found, false, "The requested tool is not available in the effective runtime view.");
                failure.repair_context_json = tool_repair_context_json(repair);
                result.failures.push_back(failure);
                common_plan_operation observed;
                observed.kind = common_plan_operation_kind::record_observation;
                observed.plan_id = plan.id;
                observed.expected_version = plan.version;
                observed.reason_summary = "unavailable tool repair context recorded";
                observed.observation = common_plan_observation{failure_observation_id, tool_call->name, common_agent_runtime_failure_observation_json(failure), 0.0f, {}, {}, 0};
                if (!store.apply(observed, plan, error)) { result.error = error; return result; }
                common_plan_operation failed;
                failed.kind = common_plan_operation_kind::fail_step;
                failed.plan_id = plan.id;
                failed.expected_version = plan.version;
                failed.step_id = tool_step_id;
                failed.reason_summary = "requested tool was unavailable";
                if (!store.apply(failed, plan, error)) { result.error = error; return result; }
                append_event(result, request, common_agent_event_type::tool_repair_context_created, "unavailable tool repair context recorded", {}, plan.id, tool_step_id, failure_observation_id, tool_call->name);
                append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::failed, "unavailable tool repair context recorded", plan.id, tool_step_id, tool_call->name, failure_observation_id);
                break;
            }
            if (!tools->is_read_only(tool_call->name) && !(request.allow_policy_gated_tool_proposals && tools->is_policy_gated(tool_call->name))) { result.failures.push_back(tool_failure(tool_call->name, tool_step_id, {}, "tool.policy_denied", common_agent_failure_class::policy, false, "The tool is not approved by the active policy.")); append_event(result, request, {common_agent_event_type::tool_rejected, "tool is not approved for this batch", {}, plan.id}); result.error = "planned tool is not approved for this batch"; return result; }
            if (tool_step_id != "request") {
                std::vector<std::string> required_evidence;
                for (const auto & step : plan.steps) {
                    if (step.id == tool_step_id) {
                        required_evidence = step.required_evidence;
                        break;
                    }
                }
                if (!tool_navigation_active || tool_navigation.step_id != tool_step_id) {
                    if (!common_agent_tool_navigation_begin(
                            tool_navigation,
                            request.deliberation_policy.mode,
                            request.turn_id.empty() ? plan.id : request.turn_id,
                            plan.id,
                            tool_step_id,
                            tool_call->name,
                            required_evidence,
                            error)) {
                        result.error = "tool navigation setup failed: " + error;
                        return result;
                    }
                    tool_navigation_active = true;
                } else if (!common_agent_tool_navigation_select_tool(tool_navigation, tool_call->name, error)) {
                    result.error = "tool navigation selection failed: " + error;
                    return result;
                }
            }
            const std::string model_arguments_json = tool_call->arguments_json;
            const bool defaults_applied = common_agent_runtime_apply_safe_tool_defaults(request, *tool_call);
            append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::started,
                "tool call prepared args=" + tool_argument_keys(tool_call->arguments_json) +
                    tool_argument_resource_ref(tool_call->arguments_json) +
                    (tool_navigation_active && tool_navigation.step_id == tool_step_id
                        ? " family=" + tool_navigation.current_family
                        : std::string()) +
                    " name_normalized=" + (name_normalization_applied ? "true" : "false") +
                    " defaults_applied=" + (defaults_applied ? "true" : "false") +
                    tool_trace_attempt_arguments(tool_step_id, model_arguments_json, tool_call->arguments_json),
                plan.id, tool_step_id, tool_call->name);
            if (!tools->validate(*tool_call, error)) {
                const std::string failure_observation_id = next_tool_observation_id(plan, tool_step_id, tool_call->name);
                const auto validation_error = error;
                auto failure = tool_failure(tool_call->name, tool_step_id, failure_observation_id, "tool.invalid_arguments", common_agent_failure_class::validation, false, "Tool arguments do not satisfy the registered contract: " + validation_error);
                auto repair = tools->make_repair_context(*tool_call, validation_error);
                repair.candidate_tools = std::move(name_candidates);
                repair.normalized_arguments = tool_call->arguments_json;
                repair.normalization_applied = normalization_applied || name_normalization_applied || defaults_applied;
                failure.repair_context_json = tool_repair_context_json(repair);
                result.failures.push_back(failure);
                common_plan_operation observed;
                observed.kind = common_plan_operation_kind::record_observation;
                observed.plan_id = plan.id;
                observed.expected_version = plan.version;
                observed.reason_summary = "registered tool validation failure";
                observed.observation = common_plan_observation{
                    failure_observation_id,
                    tool_call->name,
                    common_agent_runtime_failure_observation_json(failure),
                    0.0f,
                    {},
                    {},
                    0};
                if (!store.apply(observed, plan, error)) { result.error = error; return result; }
                common_plan_operation failed;
                failed.kind = common_plan_operation_kind::fail_step;
                failed.plan_id = plan.id;
                failed.expected_version = plan.version;
                failed.step_id = tool_step_id;
                failed.reason_summary = "registered tool validation failed";
                if (!store.apply(failed, plan, error)) { result.error = error; return result; }
                result.learning_signals.push_back({common_learning_signal_type::tool_failure, plan.id, tool_step_id,
                    tool_call->name, failure_observation_id, "registered tool validation failed"});
                append_event(result, request, {common_agent_event_type::tool_rejected, validation_error, {}, plan.id});
                append_event(result, request, common_agent_event_type::tool_repair_context_created, "schema-derived tool repair context recorded", {}, plan.id, tool_step_id, failure_observation_id, tool_call->name);
                append_event(result, request, {common_agent_event_type::plan_updated, "tool validation failure recorded for repair", {}, plan.id});
                append_observation_and_resource_events(
                    result,
                    request,
                    "tool validation failure recorded",
                    plan.id,
                    tool_step_id,
                    failure_observation_id,
                    tool_call->name,
                    {});
                append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::failed,
                    "validation_error=" + validation_error + " args=" + tool_argument_keys(tool_call->arguments_json),
                    plan.id, tool_step_id, tool_call->name, failure_observation_id);
                append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::started,
                    "repair_required failure_code=tool.invalid_arguments retryable=false",
                    plan.id, tool_step_id, tool_call->name, failure_observation_id);
                break;
            }
            const auto execution = tools->execute(*tool_call);
            if (!execution.ok) {
                if (tool_step_id == "request") { append_event(result, request, {common_agent_event_type::tool_rejected, execution.safe_summary, {}, plan.id}); result.error = "registered request tool failed: " + execution.safe_summary; return result; }
                const std::string failure_observation_id = next_tool_observation_id(plan, tool_step_id, tool_call->name);
                const auto failure = structured_tool_failure(tool_call->name, tool_step_id, failure_observation_id, execution);
                result.failures.push_back(failure);
                common_plan_operation observed;
                observed.kind = common_plan_operation_kind::record_observation;
                observed.plan_id = plan.id;
                observed.expected_version = plan.version;
                observed.reason_summary = "registered tool failure";
                observed.observation = common_plan_observation{
                    failure_observation_id,
                    tool_call->name,
                    common_agent_runtime_failure_observation_json(failure),
                    0.0f,
                    {},
                    execution.resource_refs,
                    0};
                if (!store.apply(observed, plan, error)) { result.error = error; return result; }
                common_plan_operation failed;
                failed.kind = common_plan_operation_kind::fail_step;
                failed.plan_id = plan.id;
                failed.expected_version = plan.version;
                failed.step_id = tool_step_id;
                failed.reason_summary = "registered tool failed";
                if (!store.apply(failed, plan, error)) { result.error = error; return result; }
                result.learning_signals.push_back({common_learning_signal_type::tool_failure, plan.id, tool_step_id,
                    tool_call->name, failure_observation_id, "registered tool failed"});
                append_event(result, request, {common_agent_event_type::tool_rejected, execution.safe_summary, {}, plan.id});
                append_event(result, request, {common_agent_event_type::plan_updated, "tool failure recorded for repair", {}, plan.id});
                append_observation_and_resource_events(
                    result,
                    request,
                    "tool failure recorded",
                    plan.id,
                    tool_step_id,
                    failure_observation_id,
                    tool_call->name,
                    execution.resource_refs);
                append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::failed,
                    "failure_code=" + execution.failure_code + " " + execution.safe_summary +
                        tool_trace_diagnostic(execution.raw_diagnostic) +
                        tool_argument_resource_ref(tool_call->arguments_json),
                    plan.id, tool_step_id, tool_call->name, failure_observation_id);
                append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::started,
                    "repair_required failure_code=" + failure.code +
                        " retryable=" + std::string(failure.retryable ? "true" : "false"),
                    plan.id, tool_step_id, tool_call->name, failure_observation_id);
                break;
            }
            std::string tool_result = tool_observation_payload(execution);
            if (tool_result.size() > context_budgets.tool_observation_chars) tool_result.resize(context_budgets.tool_observation_chars);
            const std::string tool_observation_id = next_tool_observation_id(plan, tool_step_id, tool_call->name);
            common_plan_operation observed;
            observed.kind = common_plan_operation_kind::record_observation;
            observed.plan_id = plan.id;
            observed.expected_version = plan.version;
            observed.reason_summary = "registered tool result";
            observed.observation = common_plan_observation{tool_observation_id, tool_call->name, tool_result, 1.0f, {}, execution.resource_refs, 0};
            if (!store.apply(observed, plan, error)) { result.error = error; return result; }
            if (tool_step_id == "request") executed_request_tool = true; else {
                executed_step_ids.insert(tool_step_id);
                common_plan_operation complete;
                complete.kind = common_plan_operation_kind::complete_step;
                complete.plan_id = plan.id;
                complete.expected_version = plan.version;
                complete.step_id = tool_step_id;
                complete.reason_summary = "registered tool completed";
                complete.evidence_ids.push_back(tool_observation_id);
                complete.evidence_ids.insert(
                    complete.evidence_ids.end(),
                    observed.observation->evidence_ids.begin(),
                    observed.observation->evidence_ids.end());
                if (!store.apply(complete, plan, error)) { result.error = error; return result; }
                append_event(result, request, {common_agent_event_type::plan_updated, "tool step completed", {}, plan.id});
                append_trace(result, common_runtime_trace_stage::step, common_runtime_trace_kind::completed,
                    "tool step completed", plan.id, tool_step_id, tool_call->name);
                activate_next_ready_step();
                if (!error.empty()) { result.error = error; return result; }
            }
            ++tool_batches;
            append_event(result, request, {common_agent_event_type::tool_executed, "registered tool result recorded", {}, plan.id});
            append_event(result, request, common_agent_event_type::plan_updated, "tool observation recorded", {}, plan.id);
            append_observation_and_resource_events(
                result,
                request,
                "tool observation recorded",
                plan.id,
                tool_step_id,
                observed.observation->id,
                tool_call->name,
                execution.resource_refs);
            append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::succeeded,
                "registered tool result recorded", plan.id, tool_step_id, tool_call->name, observed.observation->id);
            append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
                "tool observation recorded", plan.id, tool_step_id, tool_call->name, observed.observation->id);
        }

        if (request.require_tool_execution) {
            if (!plan_has_completed_tool_step(plan) &&
                    !plan_has_failed_mandatory_tool_step(plan)) {
                result.error = "required tool execution did not complete before draft fallback";
                error = result.error;
                append_trace(result, common_runtime_trace_stage::tool,
                    common_runtime_trace_kind::failed,
                    result.error,
                    plan.id);
                return result;
            }
            if (plan_has_pending_mandatory_tool_step(plan)) {
                result.limit_reached = true;
                result.response_generation_status = common_agent_generation_status::completed;
                result.response_stop_reason = common_agent_generation_stop_reason::limit;
                append_trace(result, common_runtime_trace_stage::tool,
                    common_runtime_trace_kind::decided,
                    "required tool execution is incomplete; draft deferred until pending tool steps complete",
                    plan.id);
                defer_draft_for_required_tools = true;
                break;
            }
        }

        if (defer_draft_for_required_tools) continue;

        if (context_size_tokens > 0) {
            const size_t estimated_context_tokens = context_token_estimator
                ? context_token_estimator(request, plan).value_or(estimate_common_agent_context_tokens(request, plan, context_budgets))
                : estimate_common_agent_context_tokens(request, plan, context_budgets);
            const auto context_evaluation = evaluate_common_agent_context_pressure({
                context_size_tokens,
                estimated_context_tokens,
                reserved_output_tokens,
                request.max_tool_batches * 128,
                256,
            });
            if (!context_evaluation.valid ||
                    context_evaluation.pressure == common_agent_context_pressure::compact_required ||
                    context_evaluation.pressure == common_agent_context_pressure::continuation_required) {
                result.plan_id = plan.id;
                result.plan_version = plan.version;
                result.limit_reached = true;
                result.response_generation_status = common_agent_generation_status::completed;
                result.response_stop_reason = common_agent_generation_stop_reason::limit;
                append_trace(result, common_runtime_trace_stage::turn,
                    common_runtime_trace_kind::decided,
                    "context pressure requires a bounded continuation before draft inference",
                    plan.id);
                break;
            }
            if (context_evaluation.pressure == common_agent_context_pressure::compact_recommended) {
                append_trace(result, common_runtime_trace_stage::turn,
                    common_runtime_trace_kind::decided,
                    "context pressure recommends bounded prompt compaction",
                    plan.id);
            }
        }

        const auto draft_result = executor.generate_draft_result(request, plan, guidance, error);
        auto draft = draft_result.content;
        if (!error.empty()) { result.error = error; return result; }
        result.generation_records.push_back(common_agent_generation_record_from_result(
            common_agent_generation_stage::draft,
            draft_result));
        result.response_decoded_tokens = draft_result.decoded_tokens;
        result.total_decoded_tokens += draft_result.decoded_tokens;
        result.response_generation_status = draft_result.status;
        result.response_stop_reason = draft_result.stop_reason;
        append_trace(result, common_runtime_trace_stage::response, common_runtime_trace_kind::recorded,
            "draft generated", plan.id);
        if (result.research_result) {
            const auto & verifier = research_verifier ? *research_verifier : default_research_verifier;
            common_agent_research_verification_context verification_context;
            verification_context.plan = &plan;
            verification_context.memories = &request.memories;
            for (const auto & input : request.input_resources) {
                verification_context.input_resources.push_back(input.resource);
            }
            auto verification = verifier.verify(
                *result.research_result, draft, verification_context, error);
            if (!error.empty()) {
                result.error = "research answer verification failed safely: " + error;
                result.response = draft;
                break;
            }
            result.research_verification = verification;
            append_event(result, request, {
                common_agent_event_type::answer_reviewed,
                "research answer verification: " + verification.summary,
                {}, plan.id});
            append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::completed,
                verification.summary, plan.id);
            if (verification.decision == common_agent_research_verification_decision::gather_evidence) {
                if (turn.research_reopened || !turn.research_workspace || !turn.research_runner || !turn.research_adapter) {
                    result.error = "research answer requires more evidence after the bounded reopen limit";
                    result.response = draft;
                    result.limit_reached = true;
                    break;
                }
                const std::string reopen_gap_id =
                    turn.research_workspace->workspace_id + ":verification-gap:1";
                const std::string reopen_question =
                    "Gather additional evidence for the verified answer: " + verification.summary;
                if (!common_agent_research_add_gap(*turn.research_workspace, {
                        reopen_gap_id,
                        reopen_question,
                        "answer verification requested more evidence",
                        "evidence must directly address the research question",
                        2}, error)) {
                    result.error = "research reopen failed safely: " + error;
                    result.response = draft;
                    break;
                }
                if (!turn.research_workspace->gaps.empty() && result.research_result) {
                    turn.research_workspace->gaps.back().evidence_ids =
                        result.research_result->critical_evidence_ids;
                }
                turn.research_workspace->budget.max_iterations =
                    turn.research_workspace->iterations_completed + 1;
                turn.research_workspace->budget.max_tasks =
                    static_cast<int>(turn.research_workspace->tasks.size() + 2);
                turn.research_workspace->budget.max_tool_calls = turn.research_workspace->tool_calls + 1;
                append_event(result, request, common_agent_event_type::research_reopened,
                    "research reopened after answer verification", plan.id);
                append_trace(result, common_runtime_trace_stage::research, common_runtime_trace_kind::started,
                    "research reopened after answer verification", plan.id, {}, {}, {},
                    turn.research_workspace->workspace_id);
                std::string reopen_error;
                const auto reopened_result = turn.research_runner->run(
                    *turn.research_workspace,
                    *turn.research_adapter,
                    reopen_error,
                    request.research_should_stop,
                    request.research_stop_reason,
                    make_common_agent_research_lifecycle_sink(turn));
                result.research_result = reopened_result;
                if (!reopen_error.empty() || !reopened_result.complete) {
                    result.error = reopen_error.empty()
                        ? "research reopen did not meet its bounded evidence requirement"
                        : "research reopen failed safely: " + reopen_error;
                    result.response = draft;
                    result.limit_reached = true;
                    break;
                }
                turn.research_synthesis_context = reopened_result.synthesis_context;
                request.prompt += "\n\nUpdated host-approved research synthesis context:\n";
                request.prompt += turn.research_synthesis_context;
                if (request.prompt.size() > 8192) request.prompt.resize(8192);
                turn.research_reopened = true;
                runtime_iteration_limit = request.max_iterations + 1;
                append_event(result, request, common_agent_event_type::research_completed,
                    "reopened research completed with sufficient evidence coverage", plan.id);
                append_trace(result, common_runtime_trace_stage::research, common_runtime_trace_kind::completed,
                    "reopened research completed", plan.id, {}, {}, {}, turn.research_workspace->workspace_id);
                continue;
            }
            if (verification.decision == common_agent_research_verification_decision::revise_answer) {
                guidance.push_back("Research assurance requires a revised answer: " + verification.summary);
            }
            if (verification.decision == common_agent_research_verification_decision::fail_with_uncertainty) {
                result.limit_reached = true;
            }
        }
        if (!request.enable_reflection || iteration >= request.max_reflection_rounds) {
            if (block_unrepaired_tool_failure()) break;
            if (!complete_active_synthesis_step()) { result.error = error; return result; }
            result.response = draft;
            result.limit_reached = request.enable_reflection;
            append_trace(result, common_runtime_trace_stage::response, common_runtime_trace_kind::completed,
                "response accepted without reflection revision", plan.id);
            break;
        }
        common_reflection_result reflection;
        if (!evaluate_common_agent_reflection_phase(
                turn, plan, draft, executed_step_ids, reflection)) {
            if (common_agent_reflection_context_overflow(error) &&
                    !has_unresolved_required_tool_step()) {
                if (!complete_active_synthesis_step()) {
                    result.error = error;
                    return result;
                }
                result.response = draft;
                error.clear();
                result.error.clear();
                append_trace(result, common_runtime_trace_stage::reflection,
                    common_runtime_trace_kind::failed,
                    "reflection skipped because its context exceeded the model budget",
                    plan.id);
                append_trace(result, common_runtime_trace_stage::response,
                    common_runtime_trace_kind::completed,
                    "response accepted without reflection after context-budget guard",
                    plan.id);
                break;
            }
            result.response = draft;
            if (common_agent_reflection_context_overflow(error)) {
                // Do not leak an unverified draft when a mandatory tool step
                // is still unresolved. The caller gets the bounded reason
                // and can retry with a larger context or repair the step.
                result.response.clear();
                result.error = "reflection skipped safely: model context budget exceeded while required tool work remains unresolved";
            } else {
                result.error = "reflection failed safely: " + error;
            }
            break;
        }
        const bool tool_execution_closed = common_plan_tool_execution_closed(plan);
        if (tool_execution_closed) {
            bool attempted_reopen = reflection.decision == common_reflection_decision::replan;
            bool policy_violation = false;
            for (const auto & op : reflection.proposed_plan_operations) {
                if (op.kind == common_plan_operation_kind::request_replan ||
                        op.kind == common_plan_operation_kind::add_step ||
                        op.kind == common_plan_operation_kind::replace_step ||
                        op.kind == common_plan_operation_kind::activate_step ||
                        op.kind == common_plan_operation_kind::unblock_step ||
                        op.kind == common_plan_operation_kind::reset_step ||
                        op.kind == common_plan_operation_kind::remove_step) {
                    attempted_reopen = true;
                }
                if ((op.kind == common_plan_operation_kind::add_step ||
                        op.kind == common_plan_operation_kind::replace_step) &&
                        op.step && tools && !tools->is_read_only(op.step->tool_call ? op.step->tool_call->name : std::string()) &&
                        !(request.allow_policy_gated_tool_proposals &&
                          op.step->tool_call && tools->is_policy_gated(op.step->tool_call->name))) {
                    policy_violation = true;
                }
            }
            if (attempted_reopen && !policy_violation) {
                append_event(result, request, {
                    common_agent_event_type::plan_revision_requested,
                    "reflection plan revision rejected after tool execution closed", {}, plan.id});
                append_trace(result, common_runtime_trace_stage::reflection,
                    common_runtime_trace_kind::failed,
                    "tool execution closed; reflection cannot add or restart tool steps",
                    plan.id);
                if (block_unrepaired_tool_failure()) break;
                if (!complete_active_synthesis_step()) { result.error = error; return result; }
                result.response = draft;
                result.revised = false;
                append_trace(result, common_runtime_trace_stage::response,
                    common_runtime_trace_kind::completed,
                    "response accepted from verified tool observations after closed-plan guard",
                    plan.id);
                break;
            }
        }
        discard_stale_reflection_repairs(plan, reflection, result);
        const bool deeper_deliberation =
            request.deliberation_policy.mode != common_agent_thinking_mode::reflective;
        const auto reflection_escalation = handle_common_agent_reflection_escalation(
            turn,
            plan,
            reflection,
            draft,
            guidance,
            iteration,
            runtime_iteration_limit);
        const bool reflection_requested_escalation = reflection_escalation.requested;
        const bool reflection_escalation_denied = reflection_escalation.denied;
        if (reflection_escalation.stop_turn) break;
        if (reflection_escalation.continue_turn) continue;
        if (reflection.learning_hint) {
            const auto & hint = *reflection.learning_hint;
            common_plan_operation observed;
            observed.kind = common_plan_operation_kind::record_observation;
            observed.plan_id = plan.id;
            observed.expected_version = plan.version;
            observed.reason_summary = "reflection learning hint";
            const std::string observation_id = "reflection:learning:" + std::to_string(plan.version) + ":" + std::to_string(iteration);
            observed.observation = common_plan_observation{
                observation_id,
                "reflection_hint",
                common_agent_runtime_reflection_learning_hint_json(hint),
                reflection.confidence,
                {},
                {},
                0};
            if (store.apply(observed, plan, error)) {
                result.learning_signals.push_back({common_learning_signal_type::reflection_hint, plan.id, {}, {}, observation_id,
                    "reflection supplied a bounded reusable learning hint"});
                append_observation_and_resource_events(
                    result,
                    request,
                    "reflection learning hint recorded",
                    plan.id,
                    {},
                    observation_id,
                    "reflection_hint",
                    {});
                append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
                    "reflection learning hint recorded", plan.id, {}, {}, observation_id);
            } else {
                error.clear();
            }
        }
        const bool requests_plan_revision =
            !reflection.proposed_plan_operations.empty() ||
            reflection.decision == common_reflection_decision::replan;
        if (deeper_deliberation && requests_plan_revision) {
            const size_t revision_limit = static_cast<size_t>(std::max(
                0, request.deliberation_policy.max_plan_revisions));
            if (plan_revision_count >= revision_limit) {
                result.response = draft;
                result.limit_reached = true;
                append_event(result, request, {
                    common_agent_event_type::plan_revision_limit_reached,
                    "deliberation plan revision limit reached", {}, plan.id});
                append_trace(result, common_runtime_trace_stage::plan,
                    common_runtime_trace_kind::failed,
                    "deliberation plan revision limit reached", plan.id);
                append_trace(result, common_runtime_trace_stage::response,
                    common_runtime_trace_kind::completed,
                    "response returned at deliberation limit", plan.id);
                break;
            }
            append_event(result, request, {
                common_agent_event_type::plan_revision_requested,
                "deliberation requested a plan revision", {}, plan.id});
            ++plan_revision_count;
        }
        std::vector<std::string> reflection_work_step_ids;
        for (const auto & op : reflection.proposed_plan_operations) {
            const common_plan_step * step = nullptr;
            if (op.step) {
                step = &*op.step;
            } else if (op.step_id) {
                for (const auto & candidate : plan.steps) {
                    if (candidate.id == *op.step_id) {
                        step = &candidate;
                        break;
                    }
                }
            }
            if (step && common_plan_step_effective_mode(*step) != common_plan_step_mode::final_response &&
                    (op.kind == common_plan_operation_kind::add_step ||
                     op.kind == common_plan_operation_kind::replace_step ||
                     op.kind == common_plan_operation_kind::activate_step ||
                     op.kind == common_plan_operation_kind::unblock_step ||
                     op.kind == common_plan_operation_kind::reset_step)) {
                reflection_work_step_ids.push_back(step->id);
            }
        }
        const std::optional<std::string> active_final_step_id = [&]() -> std::optional<std::string> {
            if (!plan.active_step_id) return std::nullopt;
            for (const auto & step : plan.steps) {
                if (step.id == *plan.active_step_id && step.status == common_plan_step_status::active &&
                        common_plan_step_effective_mode(step) == common_plan_step_mode::final_response) {
                    return step.id;
                }
            }
            return std::nullopt;
        }();
        if (!reflection_work_step_ids.empty() && active_final_step_id) {
            common_plan_operation reset_answer;
            reset_answer.kind = common_plan_operation_kind::reset_step;
            reset_answer.plan_id = plan.id;
            reset_answer.expected_version = plan.version;
            reset_answer.step_id = *active_final_step_id;
            reset_answer.reason_summary = "final answer suspended for reflection repair";
            if (!store.apply(reset_answer, plan, error)) {
                result.error = error;
                return result;
            }
            append_event(result, request, {common_agent_event_type::plan_updated,
                "final answer suspended for reflection repair", {}, plan.id});
            append_trace(result, common_runtime_trace_stage::response,
                common_runtime_trace_kind::updated,
                "final answer suspended for pending reflection repair", plan.id,
                *active_final_step_id);
        }
        std::vector<std::string> applied_reflection_work_step_ids;
        for (auto op : reflection.proposed_plan_operations) {
            common_plan_bind_memory_provenance(op, request.memories);
            if (op.kind == common_plan_operation_kind::add_step && op.step && op.step->tool_call) {
                const auto & proposed_call = *op.step->tool_call;
                const bool reflection_policy_allowed = tools != nullptr &&
                    (tools->is_read_only(proposed_call.name) ||
                     (request.allow_policy_gated_tool_proposals && tools->is_policy_gated(proposed_call.name)));
                if (!reflection_policy_allowed) {
                    const std::string detail = "reflection repair proposed a tool that is not allowed: " + proposed_call.name;
                    append_event(result, request, {common_agent_event_type::tool_rejected, detail, {}, plan.id});
                    append_trace(result, common_runtime_trace_stage::reflection,
                        common_runtime_trace_kind::failed, detail, plan.id,
                        op.step->id, proposed_call.name);
                    result.error = detail;
                    return result;
                }
                std::string repair_merge_error;
                if (merge_reflection_tool_repair_arguments(plan, op, repair_merge_error)) {
                    append_trace(result, common_runtime_trace_stage::plan, common_runtime_trace_kind::updated,
                        "reflection tool repair arguments merged with failed call", plan.id,
                        op.step->id, op.step->tool_call->name);
                }
                std::string validation_error;
                if (normalize_planned_tool_step(*op.step, true, validation_error)) {
                    append_event(result, request, {common_agent_event_type::tool_rejected,
                        "reflection-added tool step degraded to reasoning: " + validation_error, {}, plan.id});
                }
            }
            op.plan_id = plan.id;
            op.expected_version = plan.version;
            if (!store.apply(op, plan, error)) { error.clear(); continue; }
            append_event(result, request, {common_agent_event_type::plan_updated, "reflection plan operation applied", {}, plan.id});
            append_trace(result, common_runtime_trace_stage::plan, common_runtime_trace_kind::updated,
                "reflection plan operation applied", plan.id,
                op.step_id.value_or(op.step ? op.step->id : std::string()));
            const std::string applied_step_id = op.step_id.value_or(op.step ? op.step->id : std::string());
            if (std::find(reflection_work_step_ids.begin(), reflection_work_step_ids.end(), applied_step_id) != reflection_work_step_ids.end() &&
                    applied_step_id != active_final_step_id.value_or(std::string())) {
                applied_reflection_work_step_ids.push_back(applied_step_id);
                append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::updated,
                    "repair_scheduled step=" + applied_step_id,
                    plan.id, applied_step_id);
            }
        }
        if (!applied_reflection_work_step_ids.empty() && active_final_step_id) {
            for (const auto & work_step_id : applied_reflection_work_step_ids) {
                bool already_dependency = false;
                for (const auto & step : plan.steps) {
                    if (step.id == *active_final_step_id) {
                        already_dependency = std::find(step.depends_on.begin(), step.depends_on.end(), work_step_id) != step.depends_on.end();
                        break;
                    }
                }
                if (already_dependency) continue;
                common_plan_operation dependency;
                dependency.kind = common_plan_operation_kind::add_dependency;
                dependency.plan_id = plan.id;
                dependency.expected_version = plan.version;
                dependency.step_id = *active_final_step_id;
                dependency.target_id = work_step_id;
                dependency.reason_summary = "final answer waits for reflection repair";
                if (!store.apply(dependency, plan, error)) {
                    result.error = error;
                    return result;
                }
                append_event(result, request, {common_agent_event_type::plan_updated,
                    "final answer dependency added for reflection repair", {}, plan.id});
                append_trace(result, common_runtime_trace_stage::plan,
                    common_runtime_trace_kind::updated,
                    "final answer dependency added for reflection repair", plan.id,
                    *active_final_step_id, {}, work_step_id);
            }
            if (!activate_next_ready_step()) {
                result.error = error.empty() ? "reflection repair step could not be scheduled" : error;
                return result;
            }
            result.revised = true;
            append_event(result, request, {common_agent_event_type::response_revised,
                "reflection repair scheduled before final answer", {}, plan.id});
            append_trace(result, common_runtime_trace_stage::reflection,
                common_runtime_trace_kind::decided,
                "reflection scheduled repair before accepting response", plan.id);
            continue;
        }
        if (!reflection_escalation_denied &&
                !reflection_requested_escalation &&
                (reflection.ready_to_answer || reflection.decision == common_reflection_decision::accept)) {
            if (block_unrepaired_tool_failure()) break;
            if (!complete_active_synthesis_step()) { result.error = error; return result; }
            result.response = draft;
            append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::decided,
                "reflection accepted response", plan.id);
            append_trace(result, common_runtime_trace_stage::response, common_runtime_trace_kind::completed,
                "response accepted after reflection", plan.id);
            break;
        }
        if (reflection.decision == common_reflection_decision::abort) {
            result.error = "reflection aborted answer";
            append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::failed,
                "reflection aborted answer", plan.id);
            break;
        }
        if (reflection_escalation_denied && iteration + 1 >= runtime_iteration_limit) {
            if (!complete_active_synthesis_step()) { result.error = error; return result; }
            result.response = draft;
            result.limit_reached = true;
            append_trace(result, common_runtime_trace_stage::response, common_runtime_trace_kind::completed,
                "response returned after denied reflection escalation", plan.id);
            break;
        }
        guidance = reflection.revision_guidance;
        for (const auto & issue : reflection.issues) {
            if (!issue.description.empty()) guidance.push_back(issue.description);
        }
        result.revised = true;
        append_event(result, request, {common_agent_event_type::response_revised, "reflection requested revision", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::decided,
            "reflection requested response revision", plan.id);
    }
    if (result.response.empty() && result.error.empty() && !result.limit_reached) {
        result.error = "agent loop reached its iteration limit";
        append_trace(result, common_runtime_trace_stage::turn, common_runtime_trace_kind::failed,
            result.error, plan.id);
    }
    if (result.error.empty() && !result.response.empty() && plan.status == common_plan_status::active) {
        for (size_t pass = 0; pass < 3 && plan.status == common_plan_status::active; ++pass) {
            const auto before_version = plan.version;
            if (!complete_active_synthesis_step()) { result.error = error; return result; }
            if (plan.status != common_plan_status::active) break;
            activate_next_ready_step();
            if (!error.empty()) { result.error = error; return result; }
            if (plan.version == before_version) break;
        }
    }
    result.plan_version = plan.version;
    if (result.error.empty() && !result.response.empty() && plan.status == common_plan_status::completed) {
        const auto failure = std::find_if(result.learning_signals.begin(), result.learning_signals.end(), [](const auto & signal) {
            return signal.type == common_learning_signal_type::tool_failure;
        });
        if (failure != result.learning_signals.end()) result.learning_signals.push_back({common_learning_signal_type::successful_recovery, plan.id, failure->step_id,
            failure->tool_name, failure->evidence_id, "plan completed after a recorded tool failure"});
    }
    if (memory_learner && result.error.empty() && !result.response.empty() && plan.status == common_plan_status::completed) {
        const auto learning = memory_learner->learn(request, plan, result);
        if (learning.generation) {
            result.generation_records.push_back(common_agent_generation_record_from_result(
                common_agent_generation_stage::memory_learning,
                *learning.generation));
        }
        result.learned_memory_candidate = learning.candidate;
        result.memory_learning_summary = std::string(common_memory_learning_decision_name(learning.decision)) + ": " + learning.reason;
        result.memory_learning_related_count = learning.related_count;
        if ((learning.decision == common_memory_learning_decision::accepted || learning.decision == common_memory_learning_decision::duplicate) &&
                learning.candidate && learning.candidate->kind == common_memory_kind::procedure && learning.stored_memory_id) {
            const auto promotion = memory_learner->promote_completed_procedure(request, plan, store, *learning.stored_memory_id);
            if (promotion.blueprint_id) {
                append_event(result, request, {common_agent_event_type::blueprint_promoted,
                    "procedure promoted after " + std::to_string(promotion.verified_uses) + " verified uses",
                    *learning.stored_memory_id, *promotion.blueprint_id});
                append_trace(result, common_runtime_trace_stage::memory_learning, common_runtime_trace_kind::updated,
                    "procedure promoted after verified uses", plan.id, {}, {}, *learning.stored_memory_id, *promotion.blueprint_id);
            }
        }
        append_trace(result, common_runtime_trace_stage::memory_learning, common_runtime_trace_kind::summary,
            std::string(common_memory_learning_decision_name(learning.decision)) + ": " + learning.reason, plan.id);
        if (learning.decision == common_memory_learning_decision::accepted) {
            append_event(result, request, {request.explicit_memory_candidate
                    ? common_agent_event_type::memory_capture_confirmed
                    : common_agent_event_type::memory_remembered,
                request.explicit_memory_candidate
                    ? "explicit memory candidate passed confirmation and native policy"
                    : "post-turn candidate stored",
                learning.stored_memory_id.value_or(""), plan.id});
        } else if (learning.decision == common_memory_learning_decision::awaiting_confirmation) {
            append_event(result, request, {common_agent_event_type::memory_capture_confirmation_required,
                "explicit memory candidate requires confirmation before persistence", {}, plan.id});
        } else if (learning.decision == common_memory_learning_decision::no_candidate) {
            append_event(result, request, {common_agent_event_type::memory_candidate_extracted, "post-turn no candidate", {}, plan.id});
        } else {
            append_event(result, request, {common_agent_event_type::memory_candidate_not_stored, "post-turn candidate not stored: " + learning.reason, {}, plan.id});
        }
    } else if (memory_learner && result.error.empty() && !result.response.empty()) {
        result.memory_learning_summary = "skipped: plan did not complete";
        append_event(result, request, {common_agent_event_type::memory_candidate_not_stored, "post-turn learning skipped because the plan did not complete", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::memory_learning, common_runtime_trace_kind::skipped,
            "post-turn learning skipped because the plan did not complete", plan.id);
    }
    if (!result.response.empty()) {
        append_trace(result, common_runtime_trace_stage::turn, common_runtime_trace_kind::completed,
            "agent runtime turn completed", plan.id);
    } else if (!result.error.empty()) {
        append_trace(result, common_runtime_trace_stage::turn, common_runtime_trace_kind::failed,
            result.error, plan.id);
    }
    return result;
}
