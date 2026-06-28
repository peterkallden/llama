#include "agent/agent-inference.h"
#include "agent/tool-registry.h"
#include "memory/memory-candidate.h"
#include "plan/plan-in-memory.h"
#include "agent-cli-runtime.h"
#include "agent-cli-selection.h"

#include <cassert>
#include <deque>

struct queued_generation {
    std::string content;
    int decoded_tokens = 0;
};

class fake_agent_inference final : public common_agent_inference {
public:
    std::deque<queued_generation> queued;
    std::vector<common_agent_generation_request> seen;

    bool generate(
            const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        seen.push_back(request);
        if (queued.empty()) {
            return false;
        }
        const auto next = queued.front();
        queued.pop_front();
        result = {};
        result.content = next.content;
        result.decoded_tokens = next.decoded_tokens;
        return true;
    }
};

static args make_test_args() {
    args options;
    options.n_predict = 64;
    return options;
}

static common_agent_request make_request() {
    common_agent_request request;
    request.prompt = "Check status";
    request.session_id = "session-42";
    request.turn_id = "turn-7";
    request.namespace_id = "tenant-a";
    return request;
}

static void test_runtime_generation_metadata() {
    fake_agent_inference inference;
    inference.queued = {
        {R"(not-json)"},
        {"draft-content", 11},
        {R"({"summary":"facts"})"},
        {R"({"decision":"accept","ready_to_answer":true})"},
        {R"({"candidate":{"kind":"procedure","content":"Verify status before replying.","rationale":"Stable workflow.","importance":0.8,"confidence":0.9,"expected_reuse":0.7,"evidence_ids":["obs-1"],"source_plan_step_ids":["inspect"]},"reason":"Reusable explicit procedure"})"},
    };

    const args options = make_test_args();
    const common_agent_request request = make_request();

    auto planner = make_llama_cli_planner(inference, options, {});
    std::string error;
    const auto proposal = planner->create_plan(request, error);
    assert(error.empty());
    assert(!proposal.plan.id.empty());

    auto executor = make_llama_cli_action_executor(inference, options);
    common_plan_state plan;
    plan.id = "plan-1";
    plan.goal = request.prompt;
    plan.success_criteria = "Reply clearly";
    const auto draft = executor->generate_draft(request, plan, {"Be concise"}, error);
    assert(error.empty() && draft == "draft-content");

    common_plan_step step{"inspect", "Inspect", "Inspect request"};
    step.mode = common_plan_step_mode::reasoning;
    const auto reasoning = executor->generate_reasoning(request, plan, step, error);
    assert(error.empty() && reasoning == R"({"summary":"facts"})");

    auto reflector = make_llama_cli_reflection_engine(inference, options);
    const auto reflection = reflector->evaluate(request, plan, draft, error);
    assert(error.empty());
    assert(reflection.decision == common_reflection_decision::accept);
    assert(reflection.ready_to_answer);

    auto extractor = make_llama_cli_memory_candidate_extractor(inference, options);
    common_agent_result agent_result;
    agent_result.response = "Final answer";
    const auto memory_candidate = extractor->extract(request, plan, agent_result, error);
    assert(error.empty());
    assert(memory_candidate.candidate);
    assert(memory_candidate.candidate->kind == common_memory_kind::procedure);

    assert(inference.seen.size() == 5);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::draft);
    assert(inference.seen[2].purpose == common_agent_generation_purpose::reasoning);
    assert(inference.seen[3].purpose == common_agent_generation_purpose::reflection);
    assert(inference.seen[4].purpose == common_agent_generation_purpose::memory_learning);
    assert(inference.seen[0].trace_id && *inference.seen[0].trace_id == "turn-7:planner");
    assert(inference.seen[1].trace_id && *inference.seen[1].trace_id == "turn-7:draft");
    assert(inference.seen[2].trace_id && *inference.seen[2].trace_id == "turn-7:reasoning");
    assert(inference.seen[3].trace_id && *inference.seen[3].trace_id == "turn-7:reflection");
    assert(inference.seen[4].trace_id && *inference.seen[4].trace_id == "turn-7:memory_learning");
}

static void test_selection_generation_metadata() {
    fake_agent_inference inference;
    inference.queued = {
        {R"({"decision":"resume","plan_id":"plan-1","confidence":0.95})"},
        {R"({"decision":"instantiate","blueprint_id":"repo-change","confidence":0.82})"},
        {R"({"bindings":[{"step_id":"inspect","tool":{"name":"lookup","arguments":{"id":"status"}}}]})"},
    };

    const args options = make_test_args();
    common_agent_request request = make_request();
    request.turn_id.clear();

    common_plan_state first;
    first.id = "plan-1";
    first.goal = "Resume";
    first.next_action = "Continue";
    std::string error;
    const auto selected = select_llama_cli_plan(inference, options, request, {first}, error);
    assert(error.empty() && selected && *selected == "plan-1");

    std::vector<common_blueprint_candidate> candidates = {
        {"repo-change", "bootstrap:repo-change", "Repository change workflow"},
    };
    auto selector = make_llama_cli_blueprint_selector(inference, options);
    const auto blueprint = selector->select(request, candidates, error);
    assert(error.empty());
    assert(blueprint.decision == common_blueprint_selection_decision::instantiate);
    assert(blueprint.logical_id && *blueprint.logical_id == "repo-change");

    common_plan_in_memory_store store;
    assert(store.open("", error));
    common_plan_state plan;
    plan.id = "task-1";
    plan.session_id = request.session_id;
    plan.derived_from_plan_id = "bootstrap:repo-change";
    plan.status = common_plan_status::active;
    common_plan_step step{"inspect", "Inspect", "Inspect repository state"};
    step.mode = common_plan_step_mode::reasoning;
    step.status = common_plan_step_status::active;
    plan.steps.push_back(step);
    plan.active_step_id = "inspect";
    assert(store.create(plan, error));

    common_tool_registry registry;
    common_registered_tool tool;
    tool.name = "lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string &) { return common_tool_execution_result::success("ok"); };
    assert(registry.register_tool(std::move(tool), error));

    assert(bind_llama_cli_blueprint_tools(inference, options, registry, request, store, "task-1", error));
    const auto updated = store.get("task-1", error);
    assert(updated && updated->steps.size() == 1);
    assert(updated->steps[0].tool_call);
    assert(updated->steps[0].tool_call->name == "lookup");

    assert(inference.seen.size() == 3);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::plan_selection);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::blueprint_selection);
    assert(inference.seen[2].purpose == common_agent_generation_purpose::blueprint_binding);
    assert(inference.seen[0].trace_id && *inference.seen[0].trace_id == "session-42:plan_selection");
    assert(inference.seen[1].trace_id && *inference.seen[1].trace_id == "session-42:blueprint_selection");
    assert(inference.seen[2].trace_id && *inference.seen[2].trace_id == "session-42:blueprint_binding");
}

int main() {
    test_runtime_generation_metadata();
    test_selection_generation_metadata();
    return 0;
}
