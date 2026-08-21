#pragma once
#include "agent/agent-contract.h"
#include "agent/agent-context-budgets.h"
#include "agent/tool-runtime-contract.h"
#include "plan/plan-store.h"
#include "runtime/runtime-operation.h"
#include <memory>

class common_memory_post_turn_learner;
struct common_plan_tool_dataflow_contract;

struct common_agent_tool_repair_context {
    std::string tool_name;
    std::string validation_error;
    std::string arguments_skeleton;
    std::vector<std::string> available_tools;
    // Host-ranked candidates for an unavailable or invalid tool name. These
    // are always drawn from the immutable effective tool view.
    std::vector<std::string> candidate_tools;
    std::string normalized_arguments;
    bool normalization_applied = false;
    // Compact model-facing contract for the affected tool only. Repair must
    // not need the complete effective tool catalog.
    std::string compact_contract;
};

class common_agent_tool_runtime {
public:
    virtual ~common_agent_tool_runtime() = default;
    virtual bool is_read_only(const std::string & tool_name) const = 0;
    virtual bool is_policy_gated(const std::string & tool_name) const = 0;
    // Optional typed input/output metadata used by the plan binding seam.
    // Runtime implementations without result contracts simply return false;
    // they remain usable, but are never selected for implicit dataflow.
    virtual bool describe_tool_dataflow(
            const std::string &, common_plan_tool_dataflow_contract &, std::string &) const { return false; }
    // Tool availability is distinct from read/write policy. The default keeps
    // older runtime implementations source-compatible; provider-backed
    // runtimes override it with the immutable resolved tool view.
    virtual bool is_available(const std::string &) const { return true; }
    // Resolve a model-produced tool name against the host-owned effective view.
    // A true result means the match is deterministic and safe to apply. A
    // false result may still populate candidates for bounded repair.
    virtual bool resolve_tool_name(
            const std::string &,
            std::string &,
            std::vector<std::string> &) const { return false; }
    virtual bool validate(const common_agent_tool_call & call, std::string & error) const = 0;
    virtual common_agent_tool_repair_context make_repair_context(
            const common_agent_tool_call & call,
            const std::string & validation_error) const {
        return {call.name, validation_error, {}, {}, {}, call.arguments_json, false, {}};
    }
    virtual common_tool_execution_result execute(const common_agent_tool_call & call) const = 0;

    // Optional asynchronous tool seam. Existing runtimes remain synchronous
    // until the session lane elects to suspend and resume a turn.
    virtual bool supports_async(const common_agent_tool_call &) const { return false; }
    virtual bool begin_async(
            const common_agent_tool_call &,
            common_runtime_operation_ref &,
            std::string & error) const {
        error = "asynchronous tool execution is unavailable";
        return false;
    }
    virtual bool poll_async(
            const common_runtime_operation_ref &,
            bool & ready,
            common_tool_execution_result &,
            std::string & error) const {
        ready = false;
        error = "asynchronous tool execution is unavailable";
        return false;
    }

    virtual bool cancel_async(
            const common_runtime_operation_ref &,
            std::string & error) const {
        error = "asynchronous tool cancellation is unavailable";
        return false;
    }
};

enum class common_reflection_decision { accept, revise, request_action, replan, abort };
struct common_reflection_issue { std::string kind, description, correction; float severity = 0.5f; };
struct common_reflection_learning_hint { std::string category, statement; float expected_reuse = 0.5f; };
struct common_reflection_result { common_reflection_decision decision = common_reflection_decision::accept; std::vector<common_reflection_issue> issues; std::vector<common_plan_operation> proposed_plan_operations; std::optional<std::string> requested_action; std::vector<std::string> revision_guidance; std::optional<common_reflection_learning_hint> learning_hint; common_agent_reflection_next_action next_action = common_agent_reflection_next_action::accept; bool ready_to_answer = false; float confidence = 0.5f; std::optional<common_agent_generated_text_result> generation; };
struct common_plan_proposal { common_plan_state plan; std::vector<common_plan_operation> operations; std::optional<common_agent_generated_text_result> generation; };
class common_planner {
public:
    virtual ~common_planner() = default;
    // Optional host runtime seam for planners that need a bounded producer
    // result before rendering a later model-facing slot. Ordinary planners
    // remain pure proposal builders.
    virtual void set_tool_runtime(const common_agent_tool_runtime *) {}
    virtual common_plan_proposal create_plan(const common_agent_request & request, std::string & error) = 0;
    virtual common_plan_proposal create_plan_result(const common_agent_request & request, std::string & error) {
        return create_plan(request, error);
    }
};
class common_action_executor {
public:
    virtual ~common_action_executor() = default;
    virtual std::string generate_draft(const common_agent_request & request, const common_plan_state & plan, const std::vector<std::string> & guidance, std::string & error) = 0;
    virtual common_agent_generated_text_result generate_draft_result(
            const common_agent_request & request,
            const common_plan_state & plan,
            const std::vector<std::string> & guidance,
            std::string & error) {
        common_agent_generated_text_result result;
        result.content = generate_draft(request, plan, guidance, error);
        if (!error.empty()) {
            result.status = common_agent_generation_status::errored;
            result.stop_reason = common_agent_generation_stop_reason::error;
            result.error_message = error;
            return result;
        }
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::none;
        return result;
    }
    // Reasoning output is an observation, never user-visible text. Existing
    // executors remain source-compatible until they opt into this capability.
    virtual std::string generate_reasoning(const common_agent_request &, const common_plan_state &, const common_plan_step &, std::string & error) {
        error = "reasoning step generation is unavailable";
        return {};
    }
    virtual common_agent_generated_text_result generate_reasoning_result(
            const common_agent_request & request,
            const common_plan_state & plan,
            const common_plan_step & step,
            std::string & error) {
        common_agent_generated_text_result result;
        result.content = generate_reasoning(request, plan, step, error);
        if (!error.empty()) {
            result.status = common_agent_generation_status::errored;
            result.stop_reason = common_agent_generation_stop_reason::error;
            result.error_message = error;
            return result;
        }
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::none;
        return result;
    }
};
class common_reflection_engine {
public:
    virtual ~common_reflection_engine() = default;
    virtual common_reflection_result evaluate(const common_agent_request & request, const common_plan_state & plan, const std::string & draft, std::string & error) = 0;
    virtual common_reflection_result evaluate_result(
            const common_agent_request & request,
            const common_plan_state & plan,
            const std::string & draft,
            std::string & error) {
        return evaluate(request, plan, draft, error);
    }
};
using common_agent_context_token_estimator = std::function<std::optional<size_t>(
    const common_agent_request &, const common_plan_state &)>;

class common_agent_runtime { public: common_agent_runtime(common_plan_store & store, common_planner & planner, common_action_executor & executor, common_reflection_engine & reflector, const common_agent_tool_runtime * tools = nullptr, common_memory_post_turn_learner * memory_learner = nullptr, const common_agent_research_answer_verifier * research_verifier = nullptr, common_agent_context_budget_config context_budgets = {}, size_t context_size_tokens = 0, size_t reserved_output_tokens = 0, common_agent_context_token_estimator context_token_estimator = {}); common_agent_result run(const common_agent_request & request); private: common_plan_store & store; common_planner & planner; common_action_executor & executor; common_reflection_engine & reflector; const common_agent_tool_runtime * tools; common_memory_post_turn_learner * memory_learner; const common_agent_research_answer_verifier * research_verifier; common_agent_context_budget_config context_budgets; size_t context_size_tokens = 0; size_t reserved_output_tokens = 0; common_agent_context_token_estimator context_token_estimator; common_agent_research_bounded_verifier default_research_verifier; };
