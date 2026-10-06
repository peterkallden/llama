#include "agent/agent-runtime.h"
#include "agent/tooling/registry/tool-registry.h"
#include "plan/plan-in-memory.h"
#include "test-tool-runtime-registry-adapter.h"

#include <cassert>
#include <algorithm>

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

class one_lookup_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "workflow-continuation";
        proposal.plan.goal = "Continue a selected workflow after evidence arrives";
        proposal.plan.selected_workflow = common_plan_workflow_binding{"workflow://resource/document-analysis", "v1"};
        proposal.plan.workflow_definition = common_plan_workflow_definition{
            "workflow://resource/document-analysis", "v1", "resource", "graph-v1", {},
            {"document.inspect"}, {}, {"context.resource.available"}};
        proposal.plan.route_binding = common_plan_route_binding{
            "route:document-analysis", "document-analysis", "bp-v1",
            "workflow://resource/document-analysis", "v1", "graph-v1", "env-v1", "policy-v1"};
        common_plan_step lookup{"lookup", "Inspect document", "Inspect the selected table"};
        lookup.status = common_plan_step_status::active;
        lookup.selected_tool = "lookup";
        lookup.tool_call = common_plan_tool_call{"lookup", R"({"id":"first"})"};
        common_plan_step answer{"answer", "Answer", "Return the verified result"};
        answer.depends_on = {"lookup"};
        proposal.plan.steps = {lookup, answer};
        proposal.plan.active_step_id = "lookup";
        proposal.plan.status = common_plan_status::active;
        return proposal;
    }
};

class executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state & plan, const std::vector<std::string> &, std::string & error) override {
        ++draft_calls;
        error.clear();
        return "observations=" + std::to_string(plan.observations.size());
    }

    size_t draft_calls = 0;
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

class reopening_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state & plan, const std::string &, std::string & error) override {
        error.clear();
        assert(plan.observations.size() == 2);
        common_reflection_result result;
        result.decision = common_reflection_decision::replan;
        common_plan_operation add;
        add.kind = common_plan_operation_kind::add_step;
        common_plan_step step{"rediscovery", "Rediscover", "Incorrectly restart tool acquisition"};
        step.selected_tool = "lookup";
        step.tool_call = common_plan_tool_call{"lookup", R"({"id":"duplicate"})"};
        add.step = std::move(step);
        result.proposed_plan_operations.push_back(std::move(add));
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

    common_plan_in_memory_store continuation_store;
    assert(continuation_store.open("", error));
    one_lookup_planner continuation_plan;
    executor continuation_executor;
    common_agent_runtime continuation_runtime(
        continuation_store, continuation_plan, continuation_executor, r, &tool_runtime);
    common_agent_request continuation_request;
    continuation_request.prompt = "continue the document analysis";
    continuation_request.max_iterations = 2;
    continuation_request.max_reflection_rounds = 1;
    continuation_request.max_tool_batches = 2;
    size_t continuation_calls = 0;
    continuation_request.workflow_continuation = [&continuation_calls](
            const common_plan_state &, const std::string & completed_step_id,
            const common_plan_observation &, std::vector<common_plan_step> & steps,
            std::string & continuation_error) {
        continuation_error.clear();
        ++continuation_calls;
        if (completed_step_id == "lookup") {
            common_plan_step step{"aggregate", "Aggregate table", "Run the host-bound aggregate"};
            step.selected_tool = "lookup";
            step.tool_call = common_plan_tool_call{"lookup", R"({"id":"second"})"};
            steps.push_back(std::move(step));
        }
        return true;
    };
    const auto continuation_result = continuation_runtime.run(continuation_request);
    assert(continuation_result.error.empty());
    assert(continuation_calls == 2);
    const auto continued_plan = continuation_store.get("workflow-continuation", error);
    assert(continued_plan && continued_plan->status == common_plan_status::completed);
    assert(continued_plan->steps.size() == 3);
    assert(continued_plan->steps[1].id == "answer" &&
        continued_plan->steps[1].status == common_plan_step_status::completed &&
        std::find(continued_plan->steps[1].depends_on.begin(),
            continued_plan->steps[1].depends_on.end(), "aggregate") !=
            continued_plan->steps[1].depends_on.end());
    assert(continued_plan->steps[2].id == "aggregate" &&
        continued_plan->steps[2].status == common_plan_step_status::completed &&
        continued_plan->steps[2].depends_on == std::vector<std::string>{"lookup"});

    // A caller that requires tool execution must never receive a plain draft
    // when the runtime has no tool budget left to execute the planned step.
    common_plan_in_memory_store required_store;
    assert(required_store.open("", error));
    common_agent_runtime required_runtime(required_store, p, e, r, &tool_runtime);
    common_agent_request required_request;
    required_request.prompt = "tool execution is required";
    required_request.max_iterations = 1;
    required_request.max_reflection_rounds = 0;
    required_request.max_tool_batches = 0;
    required_request.require_tool_execution = true;
    const auto required_result = required_runtime.run(required_request);
    assert(!required_result.error.empty());
    assert(required_result.error.find("required tool execution") != std::string::npos);
    assert(required_result.response.empty());

    common_plan_in_memory_store deferred_store;
    assert(deferred_store.open("", error));
    common_agent_runtime deferred_runtime(deferred_store, p, e, r, &tool_runtime);
    required_request.max_tool_batches = 1;
    const auto deferred_result = deferred_runtime.run(required_request);
    assert(deferred_result.error.empty());
    assert(deferred_result.limit_reached);
    assert(deferred_result.response.empty());

    common_plan_in_memory_store closed_store;
    assert(closed_store.open("", error));
    reopening_reflector reopening;
    common_agent_runtime closed_runtime(closed_store, p, e, reopening, &tool_runtime);
    common_agent_request closed_request;
    closed_request.prompt = "two rounds with a closed tool phase";
    closed_request.max_iterations = 2;
    closed_request.max_reflection_rounds = 2;
    closed_request.max_tool_batches = 2;
    const auto closed_result = closed_runtime.run(closed_request);
    assert(closed_result.error.empty());
    assert(closed_result.response == "observations=2");
    auto closed_plan = closed_store.get("two-rounds", error);
    assert(closed_plan && closed_plan->steps.size() == 3);
    bool saw_closed_guard = false;
    for (const auto & trace : closed_result.trace) {
        saw_closed_guard = saw_closed_guard ||
            (trace.stage == common_runtime_trace_stage::reflection &&
             trace.detail.find("tool execution closed") != std::string::npos);
    }
    assert(saw_closed_guard);

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

    class retrying_repair_reflector final : public common_reflection_engine {
    public:
        common_reflection_result evaluate(
                const common_agent_request &, const common_plan_state & plan,
                const std::string &, std::string & error) override {
            common_reflection_result result;
            const auto failed = std::find_if(plan.steps.begin(), plan.steps.end(), [](const common_plan_step & step) {
                return step.id == "repair" && step.status == common_plan_step_status::failed;
            });
            if (failed != plan.steps.end()) {
                common_plan_operation replace;
                replace.kind = common_plan_operation_kind::replace_step;
                replace.step_id = "repair";
                common_plan_step corrected{"repair", "Repair tool call", "Retry with host-required arguments"};
                corrected.selected_tool = "lookup";
                corrected.tool_call = common_plan_tool_call{"lookup", R"({"id":"repaired"})"};
                corrected.mode = common_plan_step_mode::tool;
                replace.step = std::move(corrected);
                result.proposed_plan_operations.push_back(std::move(replace));
                result.decision = common_reflection_decision::revise;
            } else {
                result.decision = common_reflection_decision::accept;
                result.ready_to_answer = true;
            }
            error.clear();
            return result;
        }
    } retrying_repair_reflector_instance;
    common_plan_in_memory_store retry_repair_store;
    assert(retry_repair_store.open("", error));
    repair_planner retry_repair_plan("repair-retryable-validation", "lookup", R"({"wrong":true})");
    executor retry_repair_executor;
    repair_test_runtime retry_repair_tools(registry);
    common_agent_runtime retry_repair_runtime(
        retry_repair_store, retry_repair_plan, retry_repair_executor,
        retrying_repair_reflector_instance, &retry_repair_tools);
    common_agent_request retry_repair_request;
    retry_repair_request.prompt = "repair retryable validation failure";
    retry_repair_request.deliberation_policy = make_common_agent_deliberation_policy(common_agent_thinking_mode::reflective);
    retry_repair_request.max_iterations = 3;
    retry_repair_request.max_reflection_rounds = 2;
    retry_repair_request.max_tool_batches = 2;
    const auto retry_repair_result = retry_repair_runtime.run(retry_repair_request);
    assert(retry_repair_result.error.empty());
    auto retry_repair_plan_result = retry_repair_store.get("repair-retryable-validation", error);
    assert(retry_repair_plan_result && retry_repair_plan_result->status == common_plan_status::completed);
    assert(retry_repair_plan_result->steps[0].status == common_plan_step_status::completed);
    assert(retry_repair_plan_result->observations.size() == 2);
    // The failed attempt is repaired before the first user-facing draft.
    // Without the runtime repair gate this case would draft once while the
    // failed step is unresolved and once again after the retry.
    assert(retry_repair_executor.draft_calls == 1);

    class wrong_identity_repair_reflector final : public common_reflection_engine {
    public:
        common_reflection_result evaluate(
                const common_agent_request &, const common_plan_state & plan,
                const std::string &, std::string & error) override {
            common_reflection_result result;
            const auto failed = std::find_if(plan.steps.begin(), plan.steps.end(), [](const common_plan_step & step) {
                return step.id == "repair" && step.status == common_plan_step_status::failed;
            });
            if (failed != plan.steps.end()) {
                common_plan_operation replace;
                replace.kind = common_plan_operation_kind::replace_step;
                replace.step_id = "not-the-failed-step";
                common_plan_step wrong{"not-the-failed-step", "Wrong repair", "Must not replace another step"};
                wrong.selected_tool = "lookup";
                wrong.tool_call = common_plan_tool_call{"lookup", R"({"id":"repaired"})"};
                wrong.mode = common_plan_step_mode::tool;
                replace.step = std::move(wrong);
                result.proposed_plan_operations.push_back(std::move(replace));
                result.decision = common_reflection_decision::revise;
            } else {
                result.decision = common_reflection_decision::accept;
                result.ready_to_answer = true;
            }
            error.clear();
            return result;
        }
    } wrong_identity_reflector;
    common_plan_in_memory_store wrong_identity_store;
    assert(wrong_identity_store.open("", error));
    repair_planner wrong_identity_plan("repair-wrong-identity", "lookup", R"({"wrong":true})");
    executor wrong_identity_executor;
    repair_test_runtime wrong_identity_tools(registry);
    common_agent_runtime wrong_identity_runtime(
        wrong_identity_store, wrong_identity_plan, wrong_identity_executor,
        wrong_identity_reflector, &wrong_identity_tools);
    common_agent_request wrong_identity_request;
    wrong_identity_request.prompt = "reject repair for another step";
    wrong_identity_request.deliberation_policy = make_common_agent_deliberation_policy(common_agent_thinking_mode::reflective);
    wrong_identity_request.max_iterations = 2;
    wrong_identity_request.max_reflection_rounds = 1;
    wrong_identity_request.max_tool_batches = 1;
    const auto wrong_identity_result = wrong_identity_runtime.run(wrong_identity_request);
    assert(!wrong_identity_result.error.empty());
    assert(wrong_identity_result.error.find("unrepaired failed tool step") != std::string::npos);

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

    // A dependent step may arrive already active in a restored/model-authored
    // plan.  The runtime must defer it before execution, repair the failed
    // producer, and only then execute the dependent step.
    class dependency_repair_planner final : public common_planner {
    public:
        common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
            common_plan_proposal proposal;
            proposal.plan.id = "dependency-repair";
            proposal.plan.goal = "Repair the producer before its dependent lookup";
            common_plan_step producer{"producer", "Producer lookup", "Fetch the source record"};
            producer.status = common_plan_step_status::active;
            producer.selected_tool = "lookup";
            producer.tool_call = common_plan_tool_call{"lookup", R"({"wrong":true})"};
            common_plan_step consumer{"consumer", "Consumer lookup", "Use the source record"};
            consumer.status = common_plan_step_status::active;
            consumer.selected_tool = "lookup";
            consumer.tool_call = common_plan_tool_call{"lookup", R"({"id":"second"})"};
            consumer.depends_on = {"producer"};
            common_plan_step answer{"answer", "Answer", "Return the verified result"};
            answer.mode = common_plan_step_mode::final_response;
            answer.depends_on = {"consumer"};
            proposal.plan.steps = {producer, consumer, answer};
            proposal.plan.active_step_id = "producer";
            proposal.plan.status = common_plan_status::active;
            error.clear();
            return proposal;
        }
    } dependency_repair_plan;

    class dependency_repair_reflector final : public common_reflection_engine {
    public:
        common_reflection_result evaluate(
                const common_agent_request &, const common_plan_state & plan,
                const std::string &, std::string & error) override {
            common_reflection_result result;
            const auto failed = std::find_if(plan.steps.begin(), plan.steps.end(), [](const common_plan_step & step) {
                return step.id == "producer" && step.status == common_plan_step_status::failed;
            });
            if (failed != plan.steps.end()) {
                const auto consumer = std::find_if(plan.steps.begin(), plan.steps.end(), [](const common_plan_step & step) {
                    return step.id == "consumer";
                });
                assert(consumer != plan.steps.end());
                assert(consumer->status == common_plan_step_status::pending);
                const bool saw_consumer_observation = std::any_of(plan.observations.begin(), plan.observations.end(), [](const common_plan_observation & observation) {
                    return observation.source == "lookup" && observation.summary.find("second result") != std::string::npos;
                });
                assert(!saw_consumer_observation);
                common_plan_operation replace;
                replace.kind = common_plan_operation_kind::replace_step;
                replace.step_id = "producer";
                common_plan_step corrected{"producer", "Producer lookup", "Fetch the source record"};
                corrected.selected_tool = "lookup";
                corrected.tool_call = common_plan_tool_call{"lookup", R"({"id":"first"})"};
                replace.step = std::move(corrected);
                result.proposed_plan_operations.push_back(std::move(replace));
                result.decision = common_reflection_decision::revise;
            } else {
                result.decision = common_reflection_decision::accept;
                result.ready_to_answer = true;
            }
            error.clear();
            return result;
        }
    } dependency_repair_reflection;

    common_plan_in_memory_store dependency_store;
    assert(dependency_store.open("", error));
    executor dependency_executor;
    repair_test_runtime dependency_tools(registry);
    common_agent_runtime dependency_runtime(
        dependency_store, dependency_repair_plan, dependency_executor,
        dependency_repair_reflection, &dependency_tools);
    common_agent_request dependency_request;
    dependency_request.prompt = "repair producer before dependent tool";
    dependency_request.deliberation_policy = make_common_agent_deliberation_policy(common_agent_thinking_mode::reflective);
    dependency_request.max_iterations = 3;
    dependency_request.max_reflection_rounds = 2;
    dependency_request.max_tool_batches = 3;
    const auto dependency_result = dependency_runtime.run(dependency_request);
    assert(dependency_result.error.empty());
    assert(dependency_result.response == "observations=3");
    auto dependency_plan_result = dependency_store.get("dependency-repair", error);
    assert(dependency_plan_result && dependency_plan_result->status == common_plan_status::completed);
    assert(dependency_plan_result->steps[0].status == common_plan_step_status::completed);
    assert(dependency_plan_result->steps[1].status == common_plan_step_status::completed);
    assert(dependency_plan_result->steps[2].status == common_plan_step_status::completed);
    bool saw_dependency_gate = false;
    for (const auto & trace : dependency_result.trace) {
        saw_dependency_gate = saw_dependency_gate ||
            (trace.stage == common_runtime_trace_stage::step &&
             trace.detail.find("dependency gate deferred active step") != std::string::npos);
    }
    assert(saw_dependency_gate);
    return 0;
}
