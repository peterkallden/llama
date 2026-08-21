#include "agent/agent-runtime.h"
#include "agent/tool-registry.h"
#include "plan/plan-in-memory.h"
#include "test-tool-runtime-registry-adapter.h"

#include <cassert>

class repair_test_runtime final : public test_tool_runtime_registry_adapter {
public:
    explicit repair_test_runtime(const common_tool_registry & registry)
        : test_tool_runtime_registry_adapter(registry) {}

    bool is_available(const std::string & name) const override {
        return name == "lookup";
    }

    common_agent_tool_repair_context make_repair_context(
            const common_agent_tool_call & call,
            const std::string & validation_error) const override {
        return {
            call.name,
            validation_error,
            call.name == "lookup" ? R"({"id":""})" : "",
            {"lookup"},
            {},
            call.arguments_json,
            false,
            "lookup\nargs: id:string\nexample: args:{id:first}",
        };
    }
};

class planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "two-rounds";
        proposal.plan.goal = "Use two bounded lookups";
        common_plan_step first{"first", "First lookup", "Get first fact"};
        first.status = common_plan_step_status::active;
        first.selected_tool = "lookup";
        first.tool_call = common_plan_tool_call{"lookup", R"({"id":"first"})"};
        first.required_evidence = {"tool:first:lookup"};
        common_plan_step second{"second", "Second lookup", "Get follow-up fact"};
        second.selected_tool = "lookup";
        second.tool_call = common_plan_tool_call{"lookup", R"({"id":"second"})"};
        second.depends_on = {"first"};
        common_plan_step answer{"answer", "Answer", "Synthesize the verified result"};
        answer.depends_on = {"second"};
        proposal.plan.steps = {first, second, answer};
        proposal.plan.active_step_id = "first";
        proposal.plan.status = common_plan_status::active;
        return proposal;
    }
};

class executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state & plan, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "observations=" + std::to_string(plan.observations.size());
    }
};

class reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state & plan, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        assert(plan.observations.size() == 2);
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

int main() {
    common_plan_in_memory_store store;
    std::string error;
    assert(store.open("", error));
    common_tool_registry registry;
    common_registered_tool tool;
    tool.name = "lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string & input) {
        return common_tool_execution_result::success(input.find("second") == std::string::npos ? "first result" : "second result");
    };
    assert(registry.register_tool(std::move(tool), error));
    planner p; executor e; reflector r;
    test_tool_runtime_registry_adapter tool_runtime(registry);
    common_agent_runtime runtime(store, p, e, r, &tool_runtime);
    common_agent_request request;
    request.prompt = "two rounds";
    request.max_iterations = 2;
    request.max_reflection_rounds = 2;
    request.max_tool_batches = 2;
    const auto result = runtime.run(request);
    assert(result.error.empty());
    assert(result.response == "observations=2");
    auto plan = store.get("two-rounds", error);
    assert(plan && plan->status == common_plan_status::completed && plan->observations.size() == 2 && plan->steps[0].status == common_plan_step_status::completed && plan->steps[1].status == common_plan_step_status::completed && plan->steps[2].status == common_plan_step_status::completed);

    class repair_planner final : public common_planner {
    public:
        explicit repair_planner(std::string id, std::string tool_name, std::string args)
            : id(std::move(id)), tool_name(std::move(tool_name)), args(std::move(args)) {}
        common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
            common_plan_proposal proposal;
            proposal.plan.id = id;
            proposal.plan.goal = "Repair one tool call";
            common_plan_step step{"repair", "Repair tool call", "Use the registered tool"};
            step.status = common_plan_step_status::active;
            step.selected_tool = tool_name;
            step.tool_call = common_plan_tool_call{tool_name, args};
            common_plan_step answer{"answer", "Answer", "Return a bounded result"};
            answer.mode = common_plan_step_mode::final_response;
            answer.depends_on = {"repair"};
            proposal.plan.steps = {step, answer};
            proposal.plan.active_step_id = step.id;
            proposal.plan.status = common_plan_status::active;
            error.clear();
            return proposal;
        }
    private:
        std::string id;
        std::string tool_name;
        std::string args;
    };

    class repair_reflector final : public common_reflection_engine {
    public:
        explicit repair_reflector(std::string expected) : expected(std::move(expected)) {}
        common_reflection_result evaluate(const common_agent_request &, const common_plan_state & plan, const std::string &, std::string & error) override {
            assert(!plan.observations.empty());
            const auto & observation = plan.observations.back().summary;
            assert(observation.find("repair_context") != std::string::npos);
            assert(observation.find("lookup\\nargs: id:string") != std::string::npos);
            assert(observation.find(expected) != std::string::npos);
            common_reflection_result result;
            result.decision = common_reflection_decision::accept;
            result.ready_to_answer = true;
            error.clear();
            return result;
        }
    private:
        std::string expected;
    };

    const auto run_repair_case = [&](common_agent_thinking_mode mode, const std::string & id, const std::string & tool_name, const std::string & args, const std::string & expected) {
        common_plan_in_memory_store repair_store;
        assert(repair_store.open("", error));
        repair_planner repair_plan(id, tool_name, args);
        executor repair_executor;
        repair_reflector repair_reflector_instance(expected);
        repair_test_runtime repair_tools(registry);
        common_agent_runtime repair_runtime(repair_store, repair_plan, repair_executor, repair_reflector_instance, &repair_tools);
        common_agent_request repair_request;
        repair_request.prompt = "repair tool call";
        repair_request.deliberation_policy = make_common_agent_deliberation_policy(mode);
        repair_request.max_iterations = 1;
        repair_request.max_reflection_rounds = 1;
        repair_request.max_tool_batches = 1;
        const auto repair_result = repair_runtime.run(repair_request);
        // An accepting reflector without a repair operation must not allow a
        // final response to hide the failed tool-backed step.
        assert(!repair_result.error.empty());
        assert(repair_result.error.find("unrepaired failed tool step") != std::string::npos);
        bool repair_event = false;
        bool repair_arguments_trace = false;
        bool blocked_final_response = false;
        for (const auto & event : repair_result.events) repair_event = repair_event || event.type == common_agent_event_type::tool_repair_context_created;
        for (const auto & trace : repair_result.trace) {
            repair_arguments_trace = repair_arguments_trace || trace.detail.find("model_args=") != std::string::npos;
            blocked_final_response = blocked_final_response || trace.detail.find("final response blocked until failed tool step is repaired and rerun") != std::string::npos;
            if (args.find("secret") != std::string::npos && trace.detail.find("model_args=") != std::string::npos) {
                assert(trace.detail.find("<redacted>") != std::string::npos);
                assert(trace.detail.find("do-not-log") == std::string::npos);
            }
        }
        assert(repair_event);
        assert(blocked_final_response);
        if (args.find("secret") != std::string::npos) assert(repair_arguments_trace);
    };

    // The same runtime repair path is available to every thinking mode.
    run_repair_case(common_agent_thinking_mode::reflective, "repair-reflective", "lookup", R"({"wrong":true})", "\"id\":\"\"");
    run_repair_case(common_agent_thinking_mode::deliberate, "repair-deliberate", "lookup", R"({"wrong":true})", "\"id\":\"\"");
    run_repair_case(common_agent_thinking_mode::reflective, "repair-redaction", "lookup", R"({"token":"do-not-log","wrong":true})", "\"id\":\"\"");
    run_repair_case(common_agent_thinking_mode::reflective, "repair-unavailable", "missing", R"({})", "\"lookup\"");

    common_registered_tool write_tool;
    write_tool.name = "write_tool";
    write_tool.executor_id = "test.write";
    write_tool.read_only = false;
    write_tool.arguments_schema = R"({"type":"object","required":["value"],"properties":{"value":{"type":"string"}}})";
    write_tool.handler = [](const std::string &) { return common_tool_execution_result::success("must not run"); };
    assert(registry.register_tool(std::move(write_tool), error));
    class policy_reflector final : public common_reflection_engine {
    public:
        common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
            error.clear();
            common_reflection_result result;
            result.decision = common_reflection_decision::revise;
            common_plan_operation add;
            add.kind = common_plan_operation_kind::add_step;
            common_plan_step step{"write", "Write", "Attempt a policy-denied write"};
            step.selected_tool = "write_tool";
            step.tool_call = common_plan_tool_call{"write_tool", R"({"value":"blocked"})"};
            add.step = std::move(step);
            result.proposed_plan_operations.push_back(std::move(add));
            return result;
        }
    } reflector;
    common_plan_in_memory_store policy_store;
    assert(policy_store.open("", error));
    repair_planner policy_planner("policy-reflection", "lookup", R"({"id":"first"})");
    executor policy_executor;
    repair_test_runtime policy_tools(registry);
    common_agent_runtime policy_runtime(policy_store, policy_planner, policy_executor, reflector, &policy_tools);
    common_agent_request policy_request;
    policy_request.prompt = "reject policy-denied reflection repair";
    policy_request.max_iterations = 2;
    policy_request.max_reflection_rounds = 1;
    policy_request.max_tool_batches = 1;
    const auto policy_result = policy_runtime.run(policy_request);
    assert(!policy_result.error.empty() && policy_result.error.find("not allowed") != std::string::npos);
    bool saw_policy_failure = false;
    for (const auto & trace : policy_result.trace) {
        saw_policy_failure = saw_policy_failure ||
            (trace.stage == common_runtime_trace_stage::reflection &&
             trace.kind == common_runtime_trace_kind::failed &&
             trace.detail.find("write_tool") != std::string::npos);
    }
    assert(saw_policy_failure);
    return 0;
}
