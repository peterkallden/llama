#include "tools/agent/tooling/agent-tool-runtime-adapter.h"

#include <cassert>

namespace {

class permissive_runtime final : public common_agent_tool_runtime {
public:
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool is_available(const std::string &) const override { return true; }
    bool validate(const common_agent_tool_call &, std::string & error) const override {
        error.clear();
        return true;
    }
    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        last_tool = call.name;
        return common_tool_execution_result::success("{}", "ok");
    }
    mutable std::string last_tool;
};

common_agent_execution_envelope make_envelope() {
    common_agent_execution_envelope envelope;
    envelope.route_id = "blueprint:dataset-analysis:workflow:local@v1";
    envelope.policy_revision = "route-policy-v1";
    envelope.resolved_allowed_tools = {"dataset.inspect"};
    envelope.phases = {
        {common_agent_execution_phase::normal, true, {"dataset.inspect"}, {}},
        {common_agent_execution_phase::reflection, true, {"dataset.inspect"}, {}},
        {common_agent_execution_phase::research, true, {"dataset.inspect", "web.fetch"}, {"web.fetch"}},
    };
    envelope.fingerprint = "test-envelope";
    return envelope;
}

} // namespace

int main() {
    auto provider = std::make_unique<permissive_runtime>();
    [[maybe_unused]] auto * provider_ptr = provider.get();
    auto runtime = make_scoped_agent_tool_runtime(
        std::move(provider), make_envelope(), common_agent_execution_phase::normal);

    std::string error;
    assert(runtime->has_execution_envelope());
    assert(runtime->validate({"dataset.inspect", "{}"}, error));
    assert(!runtime->validate({"web.fetch", "{}"}, error));
    assert(runtime->execute({"dataset.inspect", "{}"}).ok);
    assert(provider_ptr->last_tool == "dataset.inspect");
    assert(!runtime->execute({"web.fetch", "{}"}).ok);

    common_plan_state plan;
    plan.id = "plan";
    common_plan_step step;
    step.id = "fetch";
    step.title = "Fetch";
    step.objective = "Fetch evidence";
    step.mode = common_plan_step_mode::tool;
    step.selected_tool = "web.fetch";
    step.tool_call = common_plan_tool_call{"web.fetch", "{}"};
    plan.steps.push_back(step);
    assert(!runtime->validate_plan(plan, error));

    assert(runtime->set_execution_phase(common_agent_execution_phase::research));
    assert(runtime->validate({"web.fetch", "{}"}, error));
    assert(runtime->execute({"web.fetch", "{}"}).ok);
    assert(provider_ptr->last_tool == "web.fetch");
    return 0;
}
