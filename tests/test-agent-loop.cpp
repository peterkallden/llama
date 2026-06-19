#include "agent/agent-runtime.h"
#include "plan/plan-in-memory.h"
#include <cassert>

class planner final : public common_planner { public: common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override { error.clear(); common_plan_proposal p; p.plan.id = "turn-1"; p.plan.session_id = request.session_id; p.plan.goal = request.prompt; p.plan.success_criteria = "answer"; return p; } };
class executor final : public common_action_executor { public: std::string generate_draft(const common_agent_request &, const common_plan_state &, const std::vector<std::string> &, std::string & error) override { error.clear(); return "draft"; } };
class reflector final : public common_reflection_engine { public: common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override { error.clear(); common_reflection_result r; r.decision = common_reflection_decision::accept; r.ready_to_answer = true; return r; } };
int main() { common_plan_in_memory_store store; std::string error; assert(store.open("", error)); planner p; executor e; reflector r; common_agent_runtime runtime(store, p, e, r); common_agent_request request; request.prompt = "answer"; request.session_id = "s"; auto result = runtime.run(request); assert(result.error.empty()); assert(result.answer == "draft"); assert(result.plan_id == "turn-1"); return 0; }
