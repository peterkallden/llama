#include "agent/agent-runtime.h"
#include "agent/thinking/cognitive-mode.h"
#include "plan/plan-in-memory.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <nlohmann/json.hpp>

namespace {

class planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "cognitive-mode-plan";
        proposal.plan.session_id = request.session_id;
        proposal.plan.goal = request.prompt;
        proposal.plan.success_criteria = "Provide a grounded answer.";
        proposal.plan.status = common_plan_status::active;
        common_plan_step answer;
        answer.id = "answer";
        answer.title = "Answer";
        answer.objective = "Prepare the final response.";
        answer.mode = common_plan_step_mode::final_response;
        answer.status = common_plan_step_status::active;
        proposal.plan.steps.push_back(std::move(answer));
        proposal.plan.active_step_id = "answer";
        return proposal;
    }
};

class executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state &,
            const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "Grounded answer.";
    }
};

class reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &,
            const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

} // namespace

int main() {
    using mode = common_agent_cognitive_mode;
    assert(common_agent_cognitive_transition_allowed(mode::frame, mode::execute));
    assert(common_agent_cognitive_transition_allowed(mode::frame, mode::explore));
    assert(common_agent_cognitive_transition_allowed(mode::execute, mode::reflect));
    assert(common_agent_cognitive_transition_allowed(mode::reflect, mode::execute));
    assert(common_agent_cognitive_transition_allowed(mode::reflect, mode::frame));
    assert(common_agent_cognitive_transition_allowed(mode::reflect, mode::explore));
    assert(common_agent_cognitive_transition_allowed(mode::explore, mode::frame));
    assert(!common_agent_cognitive_transition_allowed(mode::execute, mode::explore));
    assert(!common_agent_cognitive_transition_allowed(mode::explore, mode::execute));

    common_agent_cognitive_state state;
    assert(state.mode == mode::frame);
    assert(common_agent_cognitive_transition(state, mode::execute, "route selected"));
    assert(state.mode == mode::execute && state.reason == "route selected");
    assert(!common_agent_cognitive_transition(state, mode::explore, "not host-authorized"));
    assert(state.mode == mode::execute && state.reason == "route selected");
    assert(common_agent_cognitive_mode_instruction(mode::reflect).find("verified evidence") != std::string::npos);

    common_plan_in_memory_store store;
    std::string error;
    assert(store.open("", error));
    planner planner_impl;
    executor executor_impl;
    reflector reflector_impl;
    common_agent_runtime runtime(store, planner_impl, executor_impl, reflector_impl);
    common_agent_request request;
    request.prompt = "Answer from the available evidence.";
    request.enable_reflection = true;
    request.max_reflection_rounds = 1;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "Grounded answer.");

    std::vector<std::pair<std::string, std::string>> transitions;
    for (const auto & trace : result.trace) {
        const auto detail = nlohmann::json::parse(trace.detail, nullptr, false);
        if (detail.is_object() && detail.value("type", std::string()) == "cognitive_mode_transition") {
            transitions.emplace_back(detail.value("from", std::string()),
                detail.value("to", std::string()));
        }
    }
    assert(transitions.size() == 2);
    assert(transitions[0] == std::make_pair(std::string("frame"), std::string("execute")));
    assert(transitions[1] == std::make_pair(std::string("execute"), std::string("reflect")));
}
