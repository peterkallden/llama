#include "plan/plan-blueprint.h"

#include <cassert>

int main() {
    common_plan_state blueprint;
    blueprint.id = "research-blueprint";
    blueprint.kind = common_plan_kind::blueprint;
    blueprint.goal = "Research a topic";
    blueprint.success_criteria = "Grounded answer";
    common_plan_step search{"search", "Search", "Find sources"};
    search.selected_tool = "web_search";
    search.tool_call = common_plan_tool_call{"web_search", R"({"query":"topic"})"};
    common_plan_step answer{"answer", "Answer", "Write answer"};
    answer.depends_on = {"search"};
    blueprint.steps = {search, answer};
    common_plan_state instance;
    std::string error;
    assert(common_plan_instantiate_blueprint(blueprint, "research-1", "session-a", instance, error, common_plan_scope::project, 42));
    assert(instance.kind == common_plan_kind::task && instance.derived_from_plan_id && *instance.derived_from_plan_id == blueprint.id);
    assert(instance.scope == common_plan_scope::project && instance.steps[0].id == "research-1:search");
    assert(instance.steps[0].status == common_plan_step_status::active && instance.steps[1].depends_on[0] == "research-1:search");
    assert(!instance.steps[0].selected_tool && !instance.steps[0].tool_call);
    blueprint.kind = common_plan_kind::task;
    assert(!common_plan_instantiate_blueprint(blueprint, "research-2", "session-a", instance, error));
    return 0;
}
