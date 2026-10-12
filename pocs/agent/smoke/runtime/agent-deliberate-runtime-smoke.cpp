#include "agent/agent-runtime.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cstdio>

class scripted_deliberate_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "deliberate-runtime-plan";
        proposal.plan.goal = "Inspect and verify the bounded runtime";
        proposal.plan.success_criteria = "The reasoning observation is recorded before the answer.";
        proposal.plan.status = common_plan_status::active;

        common_plan_step reasoning{"inspect", "Inspect", "Record the scripted inspection result"};
        reasoning.status = common_plan_step_status::active;
        reasoning.mode = common_plan_step_mode::reasoning;

        common_plan_step answer{"answer", "Answer", "Return the verified result"};
        answer.mode = common_plan_step_mode::final_response;
        answer.depends_on = {"inspect"};

        proposal.plan.steps = {reasoning, answer};
        proposal.plan.active_step_id = reasoning.id;
        if (request.prompt.find("incomplete chunk") != std::string::npos) {
            common_runtime_resource_ref chunk;
            chunk.uri = "agent-resource://chunk/0";
            chunk.lineage = {"agent-resource://parent", 0, 2, 0, 16, 0, "text-chunk"};
            proposal.plan.observations.push_back({
                "resource_chunk:agent-resource://parent:0",
                "resource_chunk",
                "Only the first bounded chunk was observed.",
                1.0f,
                {"agent-resource://parent"},
                {chunk},
                0});
        } else if (request.prompt.find("complete chunk") != std::string::npos) {
            for (size_t index = 0; index < 2; ++index) {
                common_runtime_resource_ref chunk;
                chunk.uri = "agent-resource://chunk/" + std::to_string(index);
                chunk.lineage = {"agent-resource://parent", index, 2, index * 16, 16, 0, "text-chunk"};
                proposal.plan.observations.push_back({
                    "resource_chunk:agent-resource://parent:complete:" + std::to_string(index),
                    "resource_chunk",
                    "Complete bounded chunk observation " + std::to_string(index) + ".",
                    1.0f,
                    {"agent-resource://parent"},
                    {chunk},
                    0});
            }
        } else if (request.prompt.find("conflicting chunk") != std::string::npos) {
            for (size_t duplicate = 0; duplicate < 2; ++duplicate) {
                common_runtime_resource_ref chunk;
                chunk.uri = "agent-resource://chunk/" + std::to_string(duplicate);
                chunk.lineage = {"agent-resource://parent", 0, 2, duplicate * 16, 16, 0, "text-chunk"};
                proposal.plan.observations.push_back({
                    "resource_chunk:agent-resource://parent:duplicate:" + std::to_string(duplicate),
                    "resource_chunk",
                    "Conflicting duplicate chunk evidence.",
                    1.0f,
                    {"agent-resource://parent"},
                    {chunk},
                    0});
            }
        }
        return proposal;
    }
};

class scripted_deliberate_executor final : public common_action_executor {
public:
    std::string generate_reasoning(const common_agent_request &, const common_plan_state &, const common_plan_step &, std::string & error) override {
        error.clear();
        return R"({"summary":"bounded inspection completed"})";
    }

    std::string generate_draft(const common_agent_request &, const common_plan_state &, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "deliberate-answer";
    }
};

class accepting_deliberate_reflector final : public common_reflection_engine {
public:
    int calls = 0;

    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        ++calls;
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

class late_escalation_reflector final : public common_reflection_engine {
public:
    int calls = 0;

    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &,
            const std::string &, std::string & error) override {
        error.clear();
        ++calls;
        common_reflection_result result;
        if (calls == 1) {
            result.decision = common_reflection_decision::revise;
            result.next_action = common_agent_reflection_next_action::escalate_deliberate;
            result.issues.push_back({"missing_structure", "The draft needs deliberate review.", {}, 0.8f});
            return result;
        }
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

int main() {
    std::string error;
    common_plan_in_memory_store store;
    if (!store.open("", error)) {
        std::fprintf(stderr, "deliberate runtime smoke setup failed: %s\n", error.c_str());
        return 1;
    }

    scripted_deliberate_planner planner;
    scripted_deliberate_executor executor;
    accepting_deliberate_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector);

    common_agent_request request;
    request.prompt = "Verify the bounded deliberate runtime path";
    request.namespace_id = "deliberate-smoke";
    request.session_id = "deliberate-session";
    request.turn_id = "deliberate-turn";
    request.max_iterations = 1;
    request.max_reflection_rounds = 2;
    request.deliberation_policy = make_common_agent_deliberation_policy(common_agent_thinking_mode::deliberate);
    size_t live_events = 0;
    request.event_sink = [&live_events](const common_agent_event &) { ++live_events; };

    const auto result = runtime.run(request);
    bool step_reviewed = false;
    bool answer_reviewed = false;
    bool reflection_completed = false;
    bool reflection_trace = false;
    bool response_completed_trace = false;
    for (const auto & event : result.events) {
        step_reviewed = step_reviewed || event.type == common_agent_event_type::step_reviewed;
        answer_reviewed = answer_reviewed || event.type == common_agent_event_type::answer_reviewed;
        reflection_completed = reflection_completed || event.type == common_agent_event_type::reflection_completed;
    }
    for (const auto & trace : result.trace) {
        reflection_trace = reflection_trace ||
            trace.stage == common_runtime_trace_stage::reflection &&
            trace.kind == common_runtime_trace_kind::completed;
        response_completed_trace = response_completed_trace ||
            trace.stage == common_runtime_trace_stage::response &&
            trace.kind == common_runtime_trace_kind::completed;
    }

    const auto plan = store.get("deliberate-runtime-plan", error);
    if (!error.empty() || !result.error.empty() || result.response != "deliberate-answer" ||
            !result.reflected || reflector.calls != 1 || !step_reviewed || !answer_reviewed ||
            !reflection_completed || !reflection_trace || !response_completed_trace ||
            !plan || plan->status != common_plan_status::completed ||
            plan->observations.empty() || live_events == 0) {
        std::fprintf(stderr, "deliberate runtime smoke failed: %s\n", error.empty() ? result.error.c_str() : error.c_str());
        return 1;
    }

    std::printf("deliberate_runtime=ok response=%s reflections=%d\n", result.response.c_str(), reflector.calls);

    common_plan_in_memory_store late_store;
    if (!late_store.open("", error)) return 1;
    late_escalation_reflector late_reflector;
    common_agent_runtime late_runtime(late_store, planner, executor, late_reflector);
    auto late_request = request;
    late_request.prompt = "Return a bounded answer";
    late_request.turn_id = "late-escalation-turn";
    late_request.deliberation_policy = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::reflective);
    late_request.max_reflection_rounds = 1;
    const auto late_result = late_runtime.run(late_request);
    bool requested = false;
    bool allowed = false;
    bool resolved_deliberate = false;
    for (const auto & event : late_result.events) {
        requested = requested || event.type == common_agent_event_type::thinking_escalation_requested;
        allowed = allowed || event.type == common_agent_event_type::thinking_escalation_allowed;
        resolved_deliberate = resolved_deliberate ||
            event.type == common_agent_event_type::thinking_mode_resolved &&
            event.detail.find("deliberate") != std::string::npos;
    }
    if (!late_result.error.empty() || late_result.response != "deliberate-answer" ||
            late_reflector.calls != 2 || !requested || !allowed || !resolved_deliberate) {
        std::fprintf(stderr, "late deliberate escalation smoke failed: error=%s response=%s calls=%d requested=%d allowed=%d resolved=%d events=%zu\n",
            late_result.error.c_str(), late_result.response.c_str(), late_reflector.calls,
            requested ? 1 : 0, allowed ? 1 : 0, resolved_deliberate ? 1 : 0, late_result.events.size());
        for (const auto & event : late_result.events) {
            std::fprintf(stderr, "  event=%s detail=%s\n",
                common_agent_event_type_name(event.type), event.detail.c_str());
        }
        return 1;
    }
    std::printf("late_reflective_to_deliberate=ok\n");

    common_plan_in_memory_store denied_store;
    if (!denied_store.open("", error)) return 1;
    late_escalation_reflector denied_reflector;
    common_agent_runtime denied_runtime(denied_store, planner, executor, denied_reflector);
    auto denied_request = late_request;
    denied_request.turn_id = "late-escalation-denied-turn";
    denied_request.deliberation_policy.maximum_mode = common_agent_thinking_mode::reflective;
    const auto denied_result = denied_runtime.run(denied_request);
    bool denied = false;
    for (const auto & event : denied_result.events) {
        denied = denied || event.type == common_agent_event_type::thinking_escalation_denied;
    }
    if (!denied_result.error.empty() || denied_result.response != "deliberate-answer" ||
            denied_reflector.calls != 1 || !denied) {
        std::fprintf(stderr, "late escalation denial smoke failed: %s\n", denied_result.error.c_str());
        return 1;
    }
    std::printf("late_reflective_escalation_denied=ok\n");

    common_plan_in_memory_store incomplete_store;
    if (!incomplete_store.open("", error)) return 1;
    accepting_deliberate_reflector incomplete_reflector;
    common_agent_runtime incomplete_runtime(
        incomplete_store, planner, executor, incomplete_reflector);
    auto incomplete_request = request;
    incomplete_request.prompt = "Verify an incomplete chunk synthesis";
    incomplete_request.turn_id = "incomplete-chunk-turn";
    incomplete_request.enable_reflection = false;
    incomplete_request.deliberation_policy = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::reflective);
    const auto incomplete_result = incomplete_runtime.run(incomplete_request);
    const auto incomplete_plan = incomplete_store.get("deliberate-runtime-plan", error);
    if (incomplete_result.error.find("resource synthesis is incomplete") == std::string::npos ||
            incomplete_result.response != "" || !incomplete_plan || incomplete_plan->steps.size() < 2 ||
            incomplete_plan->steps.back().status == common_plan_step_status::completed) {
        std::fprintf(stderr, "incomplete chunk synthesis was not gated: %s\n",
            incomplete_result.error.c_str());
        return 1;
    }
    std::printf("incomplete_chunk_synthesis_gated=ok\n");

    common_plan_in_memory_store complete_store;
    if (!complete_store.open("", error)) return 1;
    accepting_deliberate_reflector complete_reflector;
    common_agent_runtime complete_runtime(
        complete_store, planner, executor, complete_reflector);
    auto complete_request = incomplete_request;
    complete_request.prompt = "Verify a complete chunk synthesis";
    complete_request.turn_id = "complete-chunk-turn";
    const auto complete_result = complete_runtime.run(complete_request);
    const auto complete_plan = complete_store.get("deliberate-runtime-plan", error);
    if (!complete_result.error.empty() || complete_result.response != "deliberate-answer" ||
            !complete_plan || complete_plan->status != common_plan_status::completed) {
        std::fprintf(stderr, "complete chunk synthesis was not accepted: %s\n",
            complete_result.error.c_str());
        return 1;
    }
    std::printf("complete_chunk_synthesis_allowed=ok\n");

    common_plan_in_memory_store conflict_store;
    if (!conflict_store.open("", error)) return 1;
    accepting_deliberate_reflector conflict_reflector;
    common_agent_runtime conflict_runtime(conflict_store, planner, executor, conflict_reflector);
    auto conflict_request = incomplete_request;
    conflict_request.prompt = "Verify a conflicting chunk synthesis";
    conflict_request.turn_id = "conflicting-chunk-turn";
    const auto conflict_result = conflict_runtime.run(conflict_request);
    const auto conflict_plan = conflict_store.get("deliberate-runtime-plan", error);
    if (conflict_result.error.find("conflicting chunk observations") == std::string::npos ||
            !conflict_plan || conflict_plan->steps.size() < 2 ||
            conflict_plan->steps.back().status != common_plan_step_status::blocked) {
        std::fprintf(stderr, "conflicting chunk synthesis was not blocked: %s\n",
            conflict_result.error.c_str());
        return 1;
    }
    std::printf("conflicting_chunk_synthesis_blocked=ok\n");
    return 0;
}
