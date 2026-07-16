#include "agent/agent-inference.h"
#include "agent/tool-registry.h"
#include "memory/memory-candidate.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "tools/agent/cli/agent-cli-host-adapter.h"
#include "tools/agent/cli/agent-cli-runtime.h"
#include "tools/agent/cli/agent-cli-selection.h"
#include "tools/agent/runtime/agent-runtime-assembly.h"
#include "tools/agent/runtime/agent-runtime-chat-driver.h"
#include "tools/agent/runtime/agent-runtime-execution.h"
#include "tools/agent/runtime/agent-runtime-host.h"
#include "chat-peg-parser.h"
#include "common/cli-scope.h"

#include <cassert>
#include <deque>
#include <string>

#include <nlohmann/json.hpp>

struct queued_generation {
    bool ok = true;
    std::string content;
    int decoded_tokens = 0;
    common_agent_generation_status status = common_agent_generation_status::completed;
    common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::none;
    std::string error_message;
    std::optional<common_chat_params> chat_params;
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
        result.status = next.status;
        result.stop_reason = next.stop_reason;
        result.error_message = next.error_message;
        result.chat_params = next.chat_params;
        return next.ok;
    }
};

static queued_generation make_success(
        const std::string & content,
        int decoded_tokens = 0,
        common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::none,
        std::optional<common_chat_params> chat_params = std::nullopt) {
    queued_generation result;
    result.ok = true;
    result.content = content;
    result.decoded_tokens = decoded_tokens;
    result.status = common_agent_generation_status::completed;
    result.stop_reason = stop_reason;
    result.chat_params = std::move(chat_params);
    return result;
}

static queued_generation make_failure(
        common_agent_generation_status status,
        common_agent_generation_stop_reason stop_reason,
        const std::string & error_message) {
    queued_generation result;
    result.ok = false;
    result.status = status;
    result.stop_reason = stop_reason;
    result.error_message = error_message;
    return result;
}

class counting_agent_inference final : public common_agent_inference {
public:
    int calls = 0;

    bool generate(
            const common_agent_generation_request &,
            common_agent_generation_result & result) override {
        ++calls;
        result = {};
        result.content = "counted";
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::none;
        return true;
    }
};

static args make_test_args() {
    args options;
    options.n_predict = 64;
    return options;
}

static common_agent_generation_config make_test_generation_config() {
    return make_agent_generation_config(make_test_args());
}

static void test_generation_contract_helpers() {
    common_agent_generation_options options;
    options.n_predict = 64;
    options.t_max_prompt_ms = 1500;
    options.t_max_predict_ms = 2500;

    const auto overridden = common_agent_generation_options_with_n_predict(options, 128);
    assert(overridden.n_predict == 128);
    assert(overridden.t_max_prompt_ms && *overridden.t_max_prompt_ms == 1500);
    assert(overridden.t_max_predict_ms && *overridden.t_max_predict_ms == 2500);

    common_agent_scope scope;
    scope.namespace_id = "tenant-a";
    scope.session_id = "session-42";

    common_chat_msg user{"user", "Hello"};
    const auto request = common_agent_make_generation_request(
        common_agent_generation_purpose::draft,
        std::string("trace-1"),
        scope,
        {user},
        overridden,
        R"({"type":"object"})");

    assert(request.purpose == common_agent_generation_purpose::draft);
    assert(request.trace_id && *request.trace_id == "trace-1");
    assert(request.scope && request.scope->namespace_id == "tenant-a");
    assert(request.scope && request.scope->session_id == "session-42");
    assert(request.messages.size() == 1);
    assert(request.messages[0].role == "user");
    assert(request.messages[0].content == "Hello");
    assert(request.options.n_predict == 128);
    assert(request.options.t_max_prompt_ms && *request.options.t_max_prompt_ms == 1500);
    assert(request.options.t_max_predict_ms && *request.options.t_max_predict_ms == 2500);
    assert(request.json_schema == R"({"type":"object"})");
}

static void test_cli_scope_helpers() {
    args project_args;
    project_args.memory_scope = "project";
    project_args.memory_namespace = "tenant-a";
    project_args.memory_session = "session-42";
    project_args.memory_project = "repo-1";
    const auto project_scope = common_cli_make_agent_scope_with_matching_plan_scope(project_args);
    assert(project_scope.memory_scope == common_memory_scope::project);
    assert(project_scope.plan_scope == common_plan_scope::project);
    assert(common_cli_supports_bootstrap_package_scope(project_scope));

    args global_args;
    global_args.memory_scope = "global";
    global_args.memory_global_opt_in = true;
    const auto global_scope = common_cli_make_agent_scope_with_matching_plan_scope(global_args);
    assert(global_scope.memory_scope == common_memory_scope::global);
    assert(global_scope.plan_scope == common_plan_scope::global);
    assert(!common_cli_supports_bootstrap_package_scope(global_scope));

    const auto explicit_scope = common_cli_make_agent_scope(project_args, common_plan_scope::session);
    assert(explicit_scope.memory_scope == common_memory_scope::project);
    assert(explicit_scope.plan_scope == common_plan_scope::session);
}

static void test_runtime_assembly_helpers() {
    agent_inference_backend backend = agent_inference_backend::server_context;
    assert(parse_agent_inference_backend("cli", backend));
    assert(backend == agent_inference_backend::cli);
    assert(parse_agent_inference_backend("server-context", backend));
    assert(backend == agent_inference_backend::server_context);
    assert(!parse_agent_inference_backend("invalid", backend));

    common_agent_inference_session cli_session;
    std::string error;
    auto * fake_model = reinterpret_cast<llama_model *>(0x1);
    auto * fake_templates = reinterpret_cast<const common_chat_templates *>(0x2);
    assert(make_agent_inference_session(make_agent_inference_options(make_test_args()), agent_inference_backend::cli, fake_model, fake_templates, cli_session, error));
    assert(cli_session.backend == agent_inference_backend::cli);
    assert(cli_session.model == fake_model);
    assert(cli_session.templates == fake_templates);
    assert(!cli_session.keepalive);
    assert(cli_session.inference);

    fake_agent_inference inference;
    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    error.clear();
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.memory_learn = "off";
    auto runtime_config = make_agent_runtime_config(options);
    assert(runtime_config.generation_config.n_predict == 64);
    assert(!runtime_config.enable_memory_learning);
    auto assembly = make_agent_runtime_assembly(memories, plans, inference, runtime_config, {}, nullptr);
    assert(assembly.planner);
    assert(assembly.executor);
    assert(assembly.reflector);
    assert(!assembly.candidate_extractor);
    assert(!assembly.memory_learner);
    assert(assembly.runtime);

    options.memory_learn = "post-turn";
    options.memory_learn_min_confidence = 0.6f;
    options.memory_learn_min_reuse = 0.4f;
    auto learning_config = make_agent_runtime_config(options);
    assert(learning_config.enable_memory_learning);
    assert(learning_config.memory_learning_config.min_confidence == 0.6f);
    assert(learning_config.memory_learning_config.min_expected_reuse == 0.4f);
    auto learning_assembly = make_agent_runtime_assembly(memories, plans, inference, learning_config, {}, nullptr);
    assert(learning_assembly.candidate_extractor);
    assert(learning_assembly.memory_learner);
    assert(learning_assembly.runtime);
}

static common_agent_request make_request() {
    common_agent_request request;
    request.prompt = "Check status";
    request.session_id = "session-42";
    request.turn_id = "turn-7";
    request.namespace_id = "tenant-a";
    return request;
}

static common_chat_params make_tool_call_chat_params(const std::vector<common_chat_tool> & tools) {
    common_chat_params params;
    params.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    const auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto tool_call = p.standard_json_tools(
            "<tool_calls>[",
            "]</tool_calls>",
            common_chat_tools_to_json_oaicompat(tools),
            false,
            false);
        return p.content(p.until("<tool_calls>")) + p.optional(tool_call) + p.end();
    });
    params.parser = parser.save();
    return params;
}

static void test_runtime_generation_metadata() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"(not-json)"),
        make_success("draft-content", 11),
        make_success(R"({"summary":"facts"})"),
        make_success(R"({"decision":"accept","ready_to_answer":true})", 0, common_agent_generation_stop_reason::json_schema),
        make_success(R"({"candidate":{"kind":"procedure","content":"Verify status before replying.","rationale":"Stable workflow.","importance":0.8,"confidence":0.9,"expected_reuse":0.7,"evidence_ids":["obs-1"],"source_plan_step_ids":["inspect"]},"reason":"Reusable explicit procedure"})", 0, common_agent_generation_stop_reason::json_schema),
    };

    const args options = make_test_args();
    const common_agent_request request = make_request();

    auto planner = make_llama_cli_planner(inference, make_agent_generation_config(options), {});
    std::string error;
    const auto proposal = planner->create_plan_result(request, error);
    assert(error.empty());
    assert(!proposal.plan.id.empty());
    assert(proposal.generation);
    assert(proposal.generation->status == common_agent_generation_status::completed);

    auto executor = make_llama_cli_action_executor(inference, make_agent_generation_config(options));
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

    auto reflector = make_llama_cli_reflection_engine(inference, make_agent_generation_config(options));
    const auto reflection = reflector->evaluate_result(request, plan, draft, error);
    assert(error.empty());
    assert(reflection.decision == common_reflection_decision::accept);
    assert(reflection.ready_to_answer);
    assert(reflection.generation);
    assert(reflection.generation->stop_reason == common_agent_generation_stop_reason::json_schema);

    auto extractor = make_llama_cli_memory_candidate_extractor(inference, make_agent_generation_config(options));
    common_agent_result agent_result;
    agent_result.response = "Final answer";
    const auto memory_candidate = extractor->extract_result(request, plan, agent_result, error);
    assert(error.empty());
    assert(memory_candidate.candidate);
    assert(memory_candidate.candidate->kind == common_memory_kind::procedure);
    assert(memory_candidate.generation);
    assert(memory_candidate.generation->stop_reason == common_agent_generation_stop_reason::json_schema);

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
    assert(inference.seen[0].scope && inference.seen[0].scope->namespace_id == "tenant-a");
    assert(inference.seen[0].scope && inference.seen[0].scope->session_id == "session-42");
    assert(inference.seen[0].scope && inference.seen[0].scope->turn_id == "turn-7");
    assert(inference.seen[0].scope && inference.seen[0].scope->plan_scope == common_plan_scope::turn);
    assert(inference.seen[0].scope && inference.seen[0].scope->memory_scope == common_memory_scope::session);
    assert(inference.seen[0].options.n_predict == 512);
    assert(inference.seen[1].options.n_predict == 64);
    assert(inference.seen[2].options.n_predict == 64);
    assert(inference.seen[3].options.n_predict == 256);
    assert(inference.seen[4].options.n_predict == 256);
    assert(inference.queued.empty());
}

static void test_selection_generation_metadata() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"({"decision":"resume","plan_id":"plan-1","confidence":0.95})", 0, common_agent_generation_stop_reason::json_schema),
        make_success(R"({"decision":"instantiate","blueprint_id":"repo-change","confidence":0.82})", 0, common_agent_generation_stop_reason::json_schema),
        make_success(R"({"bindings":[{"step_id":"inspect","tool":{"name":"lookup","arguments":{"id":"status"}}}]})"),
    };

    const args options = make_test_args();
    common_agent_request request = make_request();
    request.turn_id.clear();

    common_plan_state first;
    first.id = "plan-1";
    first.goal = "Resume";
    first.next_action = "Continue";
    std::string error;
    const auto selected = select_llama_cli_plan_result(inference, make_agent_generation_config(options), request, {first}, error);
    assert(error.empty() && selected.plan_id && *selected.plan_id == "plan-1");
    assert(selected.generation);
    assert(selected.generation->stop_reason == common_agent_generation_stop_reason::json_schema);

    std::vector<common_blueprint_candidate> candidates = {
        {"repo-change", "bootstrap:repo-change", "Repository change workflow"},
    };
    auto selector = make_llama_cli_blueprint_selector(inference, make_agent_generation_config(options));
    const auto blueprint = selector->select_result(request, candidates, error);
    assert(error.empty());
    assert(blueprint.decision == common_blueprint_selection_decision::instantiate);
    assert(blueprint.logical_id && *blueprint.logical_id == "repo-change");
    assert(blueprint.generation);
    assert(blueprint.generation->decoded_tokens == 0);

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

    const auto binding = bind_llama_cli_blueprint_tools_result(inference, make_agent_generation_config(options), registry, request, store, "task-1", error);
    assert(error.empty());
    assert(binding.applied);
    assert(binding.bound_steps == 1);
    assert(binding.generation);
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
    assert(inference.seen[0].scope && inference.seen[0].scope->namespace_id == "tenant-a");
    assert(inference.seen[0].scope && inference.seen[0].scope->session_id == "session-42");
    assert(inference.seen[0].scope && inference.seen[0].scope->turn_id.empty());
    assert(inference.seen[0].scope && inference.seen[0].scope->plan_scope == common_plan_scope::turn);
    assert(inference.seen[0].options.n_predict == 96);
    assert(inference.seen[1].options.n_predict == 96);
    assert(inference.seen[2].options.n_predict == 64);
}

static void test_runtime_generation_failure_metadata() {
    fake_agent_inference inference;
    inference.queued = {
        make_failure(common_agent_generation_status::cancelled, common_agent_generation_stop_reason::cancelled, "resident host shutting down"),
        make_failure(common_agent_generation_status::errored, common_agent_generation_stop_reason::error, "schema stream aborted"),
    };

    const args options = make_test_args();
    const common_agent_request request = make_request();

    auto planner = make_llama_cli_planner(inference, make_agent_generation_config(options), {});
    std::string error;
    const auto proposal = planner->create_plan(request, error);
    assert(!proposal.plan.id.empty());
    assert(error == "model planner generation failed (status=cancelled, stop=cancelled): resident host shutting down");

    auto executor = make_llama_cli_action_executor(inference, make_agent_generation_config(options));
    common_plan_state plan;
    plan.id = "plan-1";
    plan.goal = request.prompt;
    plan.success_criteria = "Reply clearly";
    common_plan_step step{"inspect", "Inspect", "Inspect request"};
    step.mode = common_plan_step_mode::reasoning;
    const auto reasoning = executor->generate_reasoning(request, plan, step, error);
    assert(reasoning.empty());
    assert(error == "model reasoning generation failed (status=errored, stop=error): schema stream aborted");
}

static void test_selection_generation_failure_metadata() {
    fake_agent_inference inference;
    inference.queued = {
        make_failure(common_agent_generation_status::cancelled, common_agent_generation_stop_reason::cancelled, "resident inference cancelled"),
        make_failure(common_agent_generation_status::errored, common_agent_generation_stop_reason::error, "tool binding stream failed"),
    };

    const args options = make_test_args();
    common_agent_request request = make_request();
    request.turn_id.clear();
    std::string error;

    common_plan_state first;
    first.id = "plan-1";
    first.goal = "Resume";
    first.next_action = "Continue";
    const auto selected = select_llama_cli_plan_result(inference, make_agent_generation_config(options), request, {first}, error);
    assert(!selected.plan_id);
    assert(selected.generation);
    assert(error == "plan selector generation failed (status=cancelled, stop=cancelled): resident inference cancelled");

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

    const auto binding = bind_llama_cli_blueprint_tools_result(inference, make_agent_generation_config(options), registry, request, store, "task-1", error);
    assert(!binding.applied);
    assert(binding.generation);
    assert(error == "blueprint binding generation failed (status=errored, stop=error): tool binding stream failed");
}

static void test_mini_runtime_smoke() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"(not-json)"),
        make_success("draft-content", 7),
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.memory_learn = "off";
    options.reflection_mode = "off";
    options.agent_plan = "off";
    options.agent_blueprint = "off";

    common_agent_scope scope;
    scope.namespace_id = "tenant-a";
    scope.session_id = "session-42";
    scope.turn_id = "turn-7";
    scope.memory_scope = common_memory_scope::session;
    scope.plan_scope = common_plan_scope::turn;

    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    const std::vector<common_chat_tool> tools;
    std::string current_plan_id;
    common_agent_runtime_driver_execution execution{
        memories,
        plans,
        inference,
        make_agent_runtime_policy(options),
        make_agent_runtime_config(options),
        make_agent_orchestration_config(options),
        current_plan_id,
        scope,
        blueprints,
        hits,
        common_memory_scope::session,
        true,
        tools,
        false,
        nullptr,
    };

    common_agent_result result;
    assert(run_agent_runtime_driver(execution, result, error));
    assert(error.empty());
    assert(result.error.empty());
    assert(result.response == "draft-content");
    assert(result.plan_id);
    assert(!result.plan_id->empty());
    assert(inference.seen.size() == 2);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::draft);
}

static void test_runtime_request_builder() {
    fake_agent_inference inference;
    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.plan_id = "plan-1";
    options.reflection_mode = "always";
    options.max_tool_rounds = 3;
    options.tool_profile = "research";

    common_agent_scope scope;
    scope.namespace_id = "tenant-a";
    scope.session_id = "session-42";
    scope.turn_id = "turn-7";
    scope.project_id = "repo-1";
    scope.memory_scope = common_memory_scope::project;
    scope.plan_scope = common_plan_scope::session;

    common_memory_record record;
    record.id = "mem-1";
    record.kind = common_memory_kind::fact;
    record.content = "Remember status endpoint";
    common_memory_hit hit;
    hit.memory = record;
    hit.final_score = 0.9f;

    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits = {hit};
    const std::vector<common_chat_tool> tools;
    std::string current_plan_id = "plan-1";
    const common_agent_runtime_driver_execution execution{
        memories,
        plans,
        inference,
        make_agent_runtime_policy(options),
        make_agent_runtime_config(options),
        make_agent_orchestration_config(options),
        current_plan_id,
        scope,
        blueprints,
        hits,
        common_memory_scope::project,
        true,
        tools,
        true,
        nullptr,
    };

    const auto request = make_agent_runtime_driver_request(execution);
    assert(request.prompt == "Check status");
    assert(request.plan_id && *request.plan_id == "plan-1");
    assert(request.enable_memory);
    assert(request.enable_planning);
    assert(request.enable_reflection);
    assert(request.memory_scope == common_memory_scope::project);
    assert(request.plan_scope == common_plan_scope::session);
    assert(request.namespace_id == "tenant-a");
    assert(request.session_id == "session-42");
    assert(request.turn_id == "turn-7");
    assert(request.project_id == "repo-1");
    assert(request.max_iterations == 2);
    assert(request.max_reflection_rounds == 1);
    assert(request.max_tool_batches == 3);
    assert(request.allow_policy_gated_tool_proposals);
    assert(request.memories.size() == 1);
    assert(request.memories[0].memory.id == "mem-1");

    options.reflection_mode = "off";
    options.max_tool_rounds = 4;
    options.tool_profile = "safe";
    std::string no_tools_plan_id = "plan-1";
    const common_agent_runtime_driver_execution no_tools_execution{
        memories,
        plans,
        inference,
        make_agent_runtime_policy(options),
        make_agent_runtime_config(options),
        make_agent_orchestration_config(options),
        no_tools_plan_id,
        scope,
        blueprints,
        hits,
        common_memory_scope::project,
        false,
        tools,
        false,
        nullptr,
    };

    const auto no_tools_request = make_agent_runtime_driver_request(no_tools_execution);
    assert(!no_tools_request.enable_memory);
    assert(!no_tools_request.enable_reflection);
    assert(no_tools_request.max_iterations == 1);
    assert(no_tools_request.max_reflection_rounds == 0);
    assert(no_tools_request.max_tool_batches == 0);
    assert(!no_tools_request.allow_policy_gated_tool_proposals);
}

static void test_runtime_execution_builder() {
    fake_agent_inference inference;
    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.plan_id = "plan-1";
    options.reflection_mode = "always";
    options.max_tool_rounds = 3;
    options.tool_profile = "research";
    options.memory_learn = "post-turn";
    options.memory_learn_show_candidate = true;
    options.plan_show_summary = true;
    options.agent_trace = true;

    common_agent_scope scope;
    scope.namespace_id = "tenant-a";
    scope.session_id = "session-42";
    scope.turn_id = "turn-7";
    scope.memory_scope = common_memory_scope::project;
    scope.plan_scope = common_plan_scope::session;

    common_memory_record record;
    record.id = "mem-1";
    record.kind = common_memory_kind::fact;
    common_memory_hit hit;
    hit.memory = record;
    hit.final_score = 0.9f;

    const std::vector<common_blueprint_candidate> blueprints = {
        {"repo-change", "bootstrap:repo-change", "Repository change workflow"},
    };
    const std::vector<common_memory_hit> hits = {hit};
    const std::vector<common_chat_tool> tools = {
        {"lookup", "Look up a record", R"({"type":"object"})"},
    };
    const std::string fallback_reason = "embedding disabled";
    std::string current_plan_id = options.plan_id;
    common_agent_runtime_driver_inputs inputs{
        memories,
        plans,
        make_agent_inference_options(options),
        make_agent_runtime_policy(options),
        make_agent_runtime_config(options),
        make_agent_orchestration_config(options),
        current_plan_id,
        scope,
        blueprints,
        hits,
        common_memory_scope::project,
        true,
        fallback_reason,
        tools,
        true,
        nullptr,
    };

    const auto execution = make_agent_runtime_driver_execution(inputs, inference);
    assert(&execution.memory_store == &memories);
    assert(&execution.plan_store == &plans);
    assert(&execution.inference == &inference);
    assert(execution.orchestration_config.prompt == "Check status");
    assert(execution.current_plan_id == "plan-1");
    assert(execution.policy.enable_reflection);
    assert(execution.policy.max_iterations == 2);
    assert(execution.policy.max_reflection_rounds == 1);
    assert(execution.policy.max_tool_rounds == 3);
    assert(execution.policy.allow_policy_gated_tool_proposals);
    assert(execution.policy.memory_learn == "post-turn");
    assert(execution.policy.memory_learn_show_candidate);
    assert(execution.policy.plan_show_summary);
    assert(execution.policy.agent_trace);
    assert(execution.runtime_config.generation_config.n_predict == 64);
    assert(execution.runtime_config.enable_memory_learning);
    assert(execution.runtime_config.embed_memory);
    assert(&execution.scope == &scope);
    assert(&execution.installed_blueprint_candidates == &blueprints);
    assert(&execution.memories == &hits);
    assert(execution.memory_scope == common_memory_scope::project);
    assert(execution.memory_enabled);
    assert(&execution.tools == &tools);
    assert(execution.profile_tools_active);
    assert(!execution.tool_registry);
}

static void test_chat_runtime_driver_smoke() {
    fake_agent_inference inference;
    const std::vector<common_chat_tool> tools = {
        {"memory_search", "Search memory", R"({"type":"object","additionalProperties":false})"},
    };
    inference.queued = {
        make_success(
            R"(Inspecting first.<tool_calls>[{"name":"memory_search","arguments":{"query":"status"}}]</tool_calls>)",
            5,
            common_agent_generation_stop_reason::none,
            make_tool_call_chat_params(tools)),
        make_success("Status is green.", 7),
    };

    common_agent_request request = make_request();
    request.messages = {{"user", "Check status"}};
    common_agent_generation_options options;
    options.n_predict = 64;

    common_agent_chat_runtime_execution execution{
        inference,
        request,
        options,
        {2},
        tools,
        false,
        nullptr,
        [](const common_chat_tool_call & call) {
            assert(call.name == "memory_search");
            return std::string(R"({"ok":true,"result":{"items":[{"id":"mem-1"}]}})");
        },
    };

    common_agent_result result;
    std::string error;
    assert(run_agent_chat_runtime(execution, result, error));
    assert(error.empty());
    assert(result.response == "Status is green.");
    assert(result.total_decoded_tokens == 12);
    assert(result.response_decoded_tokens == 7);
    assert(result.response_generation_status == common_agent_generation_status::completed);
    assert(result.response_stop_reason == common_agent_generation_stop_reason::none);
    assert(inference.seen.size() == 2);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::conversation);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::tool_followup);
    assert(inference.seen[0].trace_id && *inference.seen[0].trace_id == "turn-7:conversation");
    assert(inference.seen[1].trace_id && *inference.seen[1].trace_id == "turn-7:tool_followup");
    assert(inference.seen[1].messages.size() == 3);
    assert(inference.seen[1].messages[1].role == "assistant");
    assert(inference.seen[1].messages[1].tool_calls.size() == 1);
    assert(inference.seen[1].messages[2].role == "tool");
    assert(inference.seen[1].messages[2].tool_name == "memory_search");
}

static void test_runtime_host_chat_smoke() {
    fake_agent_inference inference;
    const std::vector<common_chat_tool> tools = {
        {"memory_search", "Search memory", R"({"type":"object","additionalProperties":false})"},
    };
    inference.queued = {
        make_success(
            R"(Inspecting first.<tool_calls>[{"name":"memory_search","arguments":{"query":"status"}}]</tool_calls>)",
            5,
            common_agent_generation_stop_reason::none,
            make_tool_call_chat_params(tools)),
        make_success("Status is green.", 7),
    };

    common_memory_in_memory_store memories;
    std::string error;
    assert(memories.open("", error));

    common_agent_request request = make_request();
    request.messages = {{"user", "Check status"}};
    common_agent_generation_options options;
    options.n_predict = 64;
    common_agent_runtime_turn_request turn_request;
    turn_request.request = request;
    turn_request.generation_options = options;
    turn_request.memory_enabled = true;

    common_agent_runtime_host_inputs inputs{
        common_agent_runtime_host_mode::chat,
        memories,
        nullptr,
        turn_request,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        tools,
        false,
        nullptr,
        [](const common_chat_tool_call & call) {
            assert(call.name == "memory_search");
            return std::string(R"({"ok":true,"result":{"items":[{"id":"mem-1"}]}})");
        },
    };

    auto execution = make_agent_runtime_host_execution(inputs, inference);
    common_agent_result result;
    assert(run_agent_runtime_host(execution, result, error));
    assert(error.empty());
    assert(result.response == "Status is green.");
    assert(inference.seen.size() == 2);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::conversation);
}

static void test_runtime_host_mini_smoke() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"(not-json)"),
        make_success("draft-content", 7),
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.memory_learn = "off";
    options.reflection_mode = "off";
    options.agent_plan = "off";
    options.agent_blueprint = "off";

    common_agent_scope scope;
    scope.namespace_id = "tenant-a";
    scope.session_id = "session-42";
    scope.turn_id = "turn-7";
    scope.memory_scope = common_memory_scope::session;
    scope.plan_scope = common_plan_scope::turn;

    common_agent_request request = make_request();
    request.enable_memory = true;
    common_agent_runtime_turn_request turn_request;
    turn_request.request = request;
    turn_request.scope = scope;
    turn_request.policy = make_agent_runtime_policy(options);
    turn_request.runtime_config = make_agent_runtime_config(options);
    turn_request.orchestration_config = make_agent_orchestration_config(options);
    turn_request.memory_scope = common_memory_scope::session;
    turn_request.memory_enabled = true;

    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    const std::vector<common_chat_tool> tools;
    std::string current_plan_id;
    common_agent_runtime_host_inputs inputs{
        common_agent_runtime_host_mode::mini,
        memories,
        &plans,
        turn_request,
        &current_plan_id,
        &scope,
        &blueprints,
        &hits,
        tools,
        false,
        nullptr,
        {},
    };

    auto execution = make_agent_runtime_host_execution(inputs, inference);
    common_agent_result result;
    assert(run_agent_runtime_host(execution, result, error));
    assert(error.empty());
    assert(result.response == "draft-content");
    assert(result.plan_id);
    assert(inference.seen.size() == 2);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::planner);
}

static void test_runtime_host_input_builders() {
    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.plan_id = "plan-1";
    options.reflection_mode = "always";
    options.max_tool_rounds = 3;
    options.tool_profile = "research";

    common_agent_scope scope;
    scope.namespace_id = "tenant-a";
    scope.session_id = "session-42";
    scope.turn_id = "turn-7";
    scope.project_id = "repo-1";
    scope.memory_scope = common_memory_scope::project;
    scope.plan_scope = common_plan_scope::session;

    std::string current_plan_id = "plan-1";
    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    const std::vector<common_chat_tool> tools = {
        {"lookup", "Look up a record", R"({"type":"object"})"},
    };
    common_agent_request request;
    request.messages = {{"user", "Check status"}};
    common_agent_generation_options generation_options;
    common_agent_runtime_turn_request turn_request = make_agent_cli_runtime_turn_request(
        options,
        scope,
        make_agent_orchestration_config(options),
        common_memory_scope::project,
        true,
        error,
        std::move(request),
        generation_options);

    common_agent_runtime_host_build_context build_context{
        memories,
        &plans,
        turn_request,
        &current_plan_id,
        &blueprints,
        hits,
        tools,
        true,
        nullptr,
        {},
    };

    auto chat_inputs = make_agent_runtime_host_chat_inputs(build_context);
    assert(chat_inputs.mode == common_agent_runtime_host_mode::chat);
    assert(chat_inputs.plan_store == nullptr);
    assert(chat_inputs.turn_request.request.prompt == "Check status");
    assert(chat_inputs.turn_request.request.enable_memory);
    assert(chat_inputs.turn_request.request.namespace_id == "tenant-a");
    assert(chat_inputs.turn_request.request.project_id == "repo-1");
    assert(chat_inputs.turn_request.policy.enable_reflection);
    assert(chat_inputs.turn_request.policy.max_tool_rounds == 3);

    common_agent_runtime_turn_request mini_turn_request = make_agent_cli_runtime_turn_request(
        options,
        scope,
        make_agent_orchestration_config(options),
        common_memory_scope::project,
        true,
        error);
    common_agent_runtime_host_build_context mini_context{
        memories,
        &plans,
        mini_turn_request,
        &current_plan_id,
        &blueprints,
        hits,
        tools,
        true,
        nullptr,
        {},
    };

    const auto orchestration_config = make_agent_orchestration_config(options);
    auto mini_inputs = make_agent_runtime_host_mini_inputs(mini_context, orchestration_config);
    assert(mini_inputs.mode == common_agent_runtime_host_mode::mini);
    assert(mini_inputs.plan_store == &plans);
    assert(mini_inputs.current_plan_id == &current_plan_id);
    assert(mini_inputs.scope == &mini_inputs.turn_request.scope);
    assert(mini_inputs.installed_blueprint_candidates == &blueprints);
    assert(mini_inputs.turn_request.request.prompt == "Check status");
    assert(mini_inputs.turn_request.request.plan_scope == common_plan_scope::session);
    assert(mini_inputs.turn_request.orchestration_config.prompt == "Check status");
}

static void test_runtime_host_turn_completion() {
    common_memory_in_memory_store memories;
    std::string error;
    assert(memories.open("", error));

    const std::vector<common_chat_tool> tools;
    common_agent_request request;
    request.messages = {{"user", "Check status"}};

    common_agent_runtime_host_inputs inputs{
        common_agent_runtime_host_mode::chat,
        memories,
        nullptr,
        {},
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        tools,
        false,
        nullptr,
        {},
        true,
        {},
    };

    int post_run_calls = 0;
    inputs.post_run = [&post_run_calls](const common_agent_result & result, std::string & hook_error) {
        ++post_run_calls;
        assert(result.response == "Status is green.");
        hook_error.clear();
        return true;
    };

    common_agent_runtime_session session;
    session.inference_session.inference = std::make_unique<fake_agent_inference>();
    common_agent_result result;
    result.response = "Status is green.";

    assert(complete_agent_runtime_host_turn(inputs, session, result, error));
    assert(error.empty());
    assert(post_run_calls == 1);
    assert(session.inference_session.inference == nullptr);
}

static void test_runtime_resident_host_multi_turn_smoke() {
    common_memory_in_memory_store memories;
    std::string error;
    assert(memories.open("", error));

    fake_agent_inference inference;
    inference.queued = {
        make_success("First answer", 5),
        make_success("Second answer", 6),
    };

    common_agent_runtime_resident_host host;
    host.session().inference_session.backend = agent_inference_backend::cli;
    host.session().inference_session.inference = std::make_unique<fake_agent_inference>(std::move(inference));
    auto * inference_ptr = static_cast<fake_agent_inference *>(host.session().inference_session.inference.get());
    host.session().initialized = true;
    host.session().initialized_backend = agent_inference_backend::cli;
    host.session().initialized_options = {};

    const std::vector<common_chat_tool> tools;

    common_agent_runtime_turn_request first_turn;
    first_turn.request = make_request();
    first_turn.request.messages = {{"user", "Turn one"}};
    first_turn.request.prompt = "Turn one";
    first_turn.generation_options.n_predict = 32;
    common_agent_runtime_host_inputs first_inputs{
        common_agent_runtime_host_mode::chat,
        memories,
        nullptr,
        first_turn,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        tools,
        false,
        nullptr,
        {},
        true,
        {},
    };

    common_agent_result first_result;
    assert(host.run_turn(first_inputs, first_result, error));
    assert(error.empty());
    assert(first_result.response == "First answer");
    assert(host.session().inference_session.inference.get() == inference_ptr);
    assert(host.session().initialized);

    common_agent_runtime_turn_request second_turn;
    second_turn.request = make_request();
    second_turn.request.turn_id = "turn-8";
    second_turn.request.messages = {{"user", "Turn two"}};
    second_turn.request.prompt = "Turn two";
    second_turn.generation_options.n_predict = 32;
    common_agent_runtime_host_inputs second_inputs{
        common_agent_runtime_host_mode::chat,
        memories,
        nullptr,
        second_turn,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        tools,
        false,
        nullptr,
        {},
        true,
        {},
    };

    common_agent_result second_result;
    assert(host.run_turn(second_inputs, second_result, error));
    assert(error.empty());
    assert(second_result.response == "Second answer");
    assert(host.session().inference_session.inference.get() == inference_ptr);
    assert(inference_ptr->seen.size() == 2);
    assert(inference_ptr->seen[0].trace_id && *inference_ptr->seen[0].trace_id == "turn-7:conversation");
    assert(inference_ptr->seen[1].trace_id && *inference_ptr->seen[1].trace_id == "turn-8:conversation");

    host.reset();
    assert(!host.session().initialized);
}

static void test_runtime_resident_chat_host_builder() {
    common_memory_in_memory_store memories;
    std::string error;
    assert(memories.open("", error));

    common_agent_runtime_turn_request base_turn_request;
    base_turn_request.request.session_id = "resident-chat";
    base_turn_request.request.namespace_id = "tenant-a";
    base_turn_request.scope.session_id = "resident-chat";
    base_turn_request.scope.namespace_id = "tenant-a";
    base_turn_request.inference_options = {};
    base_turn_request.generation_options.n_predict = 24;

    common_agent_runtime_resident_chat_host host({
        memories,
        base_turn_request,
    });

    auto inference = std::make_unique<counting_agent_inference>();
    auto * inference_ptr = inference.get();
    host.runtime_host().session().inference_session.backend = agent_inference_backend::cli;
    host.runtime_host().session().inference_session.inference = std::move(inference);
    host.runtime_host().session().initialized = true;
    host.runtime_host().session().initialized_backend = agent_inference_backend::cli;
    host.runtime_host().session().initialized_options = {};

    common_agent_result result;
    assert(host.run_prompt("Reply with TEST only.", "turn-9", result, error));
    assert(error.empty());
    assert(result.response == "counted");
    assert(inference_ptr->calls == 1);

    const auto request = make_agent_runtime_resident_turn_request(base_turn_request, "Prompt text", "turn-10");
    assert(request.request.prompt == "Prompt text");
    assert(request.request.messages.size() == 1);
    assert(request.request.messages[0].content == "Prompt text");
    assert(request.request.turn_id == "turn-10");
    assert(request.scope.turn_id == "turn-10");
    assert(request.orchestration_config.prompt == "Prompt text");
}

static void test_runtime_resident_request_builders() {
    const auto base_turn_request = make_agent_runtime_resident_base_turn_request({
        "Reply with OK only.",
        "resident-session",
        "tenant-a",
        "repo-1",
        "model.gguf",
        64,
        2,
        false,
        "server-context",
        common_memory_scope::project,
        common_plan_scope::session,
    });

    assert(base_turn_request.request.prompt == "Reply with OK only.");
    assert(base_turn_request.request.messages.size() == 1);
    assert(base_turn_request.request.messages[0].role == "user");
    assert(base_turn_request.request.messages[0].content == "Reply with OK only.");
    assert(base_turn_request.request.session_id == "resident-session");
    assert(base_turn_request.request.namespace_id == "tenant-a");
    assert(base_turn_request.request.project_id == "repo-1");
    assert(base_turn_request.scope.session_id == "resident-session");
    assert(base_turn_request.scope.namespace_id == "tenant-a");
    assert(base_turn_request.scope.project_id == "repo-1");
    assert(base_turn_request.scope.memory_scope == common_memory_scope::project);
    assert(base_turn_request.scope.plan_scope == common_plan_scope::session);
    assert(base_turn_request.inference_options.model == "model.gguf");
    assert(base_turn_request.inference_options.n_predict == 64);
    assert(base_turn_request.inference_options.n_gpu_layers == 2);
    assert(!base_turn_request.inference_options.fit_params);
    assert(base_turn_request.policy.agent_inference_backend == "server-context");
    assert(base_turn_request.generation_options.n_predict == 64);

    common_memory_in_memory_store memories;
    std::string error;
    assert(memories.open("", error));
    auto runtime_config = make_agent_runtime_resident_runtime_config(
        memories,
        nullptr,
        base_turn_request);
    assert(&runtime_config.memory_store == &memories);
    assert(runtime_config.plan_store == nullptr);
    assert(runtime_config.base_turn_request.request.session_id == "resident-session");

    common_agent_runtime_daemon_turn_request daemon_request;
    daemon_request.mode = common_agent_runtime_host_mode::mini;
    daemon_request.prompt = "Check status";
    daemon_request.session_id = "resident-session";
    daemon_request.namespace_id = "tenant-a";
    daemon_request.project_id = "repo-1";
    daemon_request.turn_id = "turn-3";
    daemon_request.memory_scope = common_memory_scope::project;
    daemon_request.plan_scope = common_plan_scope::project;
    daemon_request.n_predict = 32;
    assert(daemon_request.mode == common_agent_runtime_host_mode::mini);
    assert(daemon_request.project_id == "repo-1");
    assert(daemon_request.turn_id == "turn-3");
    assert(daemon_request.memory_scope == common_memory_scope::project);
    assert(daemon_request.plan_scope == common_plan_scope::project);

    common_agent_runtime_daemon_turn_result daemon_result;
    daemon_result.ok = true;
    daemon_result.response = "All set";
    daemon_result.plan_id = "plan-1";
    daemon_result.total_decoded_tokens = 12;
    assert(daemon_result.ok);
    assert(daemon_result.response == "All set");
    assert(daemon_result.plan_id == "plan-1");
    assert(daemon_result.total_decoded_tokens == 12);
}

static void test_runtime_resident_runtime_builder() {
    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.memory_learn = "off";
    options.reflection_mode = "off";
    options.agent_plan = "off";
    options.agent_blueprint = "off";

    auto base_turn_request = make_agent_runtime_resident_base_turn_request({
        options.prompt,
        "session-42",
        "tenant-a",
        {},
        "model.gguf",
        0,
        0,
        true,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
    });
    base_turn_request.policy = make_agent_runtime_policy(options);
    base_turn_request.runtime_config = make_agent_runtime_config(options);
    base_turn_request.orchestration_config = make_agent_orchestration_config(options);
    base_turn_request.memory_scope = common_memory_scope::session;
    base_turn_request.memory_enabled = true;
    base_turn_request.generation_options.n_predict = 24;

    common_agent_runtime_resident_runtime runtime(make_agent_runtime_resident_runtime_config(
        memories,
        &plans,
        base_turn_request));

    fake_agent_inference inference;
    inference.queued = {
        make_success("chat-response", 3),
        make_success(R"(not-json)"),
        make_success("draft-content", 7),
    };
    runtime.runtime_host().session().inference_session.backend = agent_inference_backend::cli;
    runtime.runtime_host().session().inference_session.inference = std::make_unique<fake_agent_inference>(std::move(inference));
    auto * inference_ptr = static_cast<fake_agent_inference *>(runtime.runtime_host().session().inference_session.inference.get());
    runtime.runtime_host().session().initialized = true;
    runtime.runtime_host().session().initialized_backend = agent_inference_backend::cli;
    runtime.runtime_host().session().initialized_options = {};

    common_agent_result chat_result;
    assert(runtime.run_chat_prompt("Reply with TEST only.", "turn-9", chat_result, error));
    assert(error.empty());
    assert(chat_result.response == "chat-response");

    common_agent_result mini_result;
    assert(runtime.run_mini_prompt("Check status", "turn-11", mini_result, error));
    assert(error.empty());
    assert(mini_result.response == "draft-content");
    assert(mini_result.plan_id);
    assert(runtime.current_plan_id() == *mini_result.plan_id);
    assert(inference_ptr->seen.size() == 3);
    assert(inference_ptr->seen[0].trace_id && *inference_ptr->seen[0].trace_id == "turn-9:conversation");
    assert(inference_ptr->seen[1].trace_id && *inference_ptr->seen[1].trace_id == "turn-11:planner");
    assert(inference_ptr->seen[2].trace_id && *inference_ptr->seen[2].trace_id == "turn-11:draft");
}

static void test_runtime_resident_mini_host_builder() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"(not-json)"),
        make_success("draft-content", 7),
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.memory_learn = "off";
    options.reflection_mode = "off";
    options.agent_plan = "off";
    options.agent_blueprint = "off";

    auto base_turn_request = make_agent_runtime_resident_base_turn_request({
        options.prompt,
        "session-42",
        "tenant-a",
        {},
        "model.gguf",
        0,
        0,
        true,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
    });
    base_turn_request.policy = make_agent_runtime_policy(options);
    base_turn_request.runtime_config = make_agent_runtime_config(options);
    base_turn_request.orchestration_config = make_agent_orchestration_config(options);
    base_turn_request.memory_scope = common_memory_scope::session;
    base_turn_request.memory_enabled = true;

    common_agent_runtime_resident_mini_host host({
        memories,
        plans,
        base_turn_request,
        {},
        {},
        {},
        false,
        nullptr,
    });

    host.runtime_host().session().inference_session.backend = agent_inference_backend::cli;
    host.runtime_host().session().inference_session.inference = std::make_unique<fake_agent_inference>(std::move(inference));
    auto * inference_ptr = static_cast<fake_agent_inference *>(host.runtime_host().session().inference_session.inference.get());
    host.runtime_host().session().initialized = true;
    host.runtime_host().session().initialized_backend = agent_inference_backend::cli;
    host.runtime_host().session().initialized_options = {};

    common_agent_result result;
    assert(host.run_prompt("Check status", "turn-11", result, error));
    assert(error.empty());
    assert(result.response == "draft-content");
    assert(result.plan_id);
    assert(host.current_plan_id() == *result.plan_id);
    assert(inference_ptr->seen.size() == 2);
    assert(inference_ptr->seen[0].trace_id && *inference_ptr->seen[0].trace_id == "turn-11:planner");
    assert(inference_ptr->seen[1].trace_id && *inference_ptr->seen[1].trace_id == "turn-11:draft");
}

static void test_cli_runtime_host_adapter_chat_inputs() {
    common_memory_in_memory_store memories;
    std::string error;
    assert(memories.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.memory_namespace = "tenant-a";
    options.memory_session = "session-42";
    options.memory_turn = "turn-7";
    options.memory_project = "repo-1";
    options.memory_scope = "project";
    options.plan_scope = "session";

    const std::vector<common_chat_msg> messages = {
        {"user", "Check status"},
    };
    const std::vector<common_memory_hit> hits;
    const std::vector<common_chat_tool> tools = {
        {"memory_search", "Search memory", R"({"type":"object"})"},
    };

    auto inputs = make_agent_cli_runtime_host_chat_inputs(
        memories,
        options,
        messages,
        common_memory_scope::project,
        hits,
        true,
        error,
        tools,
        false,
        nullptr,
        {});

    assert(inputs.mode == common_agent_runtime_host_mode::chat);
    assert(inputs.reset_session_on_completion);
    assert(inputs.turn_request.request.prompt == "Check status");
    assert(inputs.turn_request.request.namespace_id == "tenant-a");
    assert(inputs.turn_request.request.project_id == "repo-1");
    assert(inputs.turn_request.request.plan_scope == common_plan_scope::session);
    assert(inputs.turn_request.request.enable_memory);
    assert(inputs.tool_handler);
}

static void test_runtime_session_reuse() {
    common_agent_runtime_session session;
    auto * fake_model = reinterpret_cast<llama_model *>(0x1);
    auto * fake_templates = reinterpret_cast<const common_chat_templates *>(0x2);
    auto inference = std::make_unique<counting_agent_inference>();
    auto * inference_ptr = inference.get();

    session.model = fake_model;
    session.inference_session.backend = agent_inference_backend::cli;
    session.inference_session.model = fake_model;
    session.inference_session.templates = fake_templates;
    session.inference_session.inference = std::move(inference);
    session.initialized = true;
    session.initialized_backend = agent_inference_backend::cli;
    session.initialized_options = make_agent_inference_options(make_test_args());

    std::string error;
    assert(initialize_agent_runtime_session(
        make_agent_inference_options(make_test_args()),
        agent_inference_backend::cli,
        true,
        {},
        session,
        error));
    assert(error.empty());
    assert(session.model == fake_model);
    assert(session.inference_session.inference.get() == inference_ptr);
    session.model = nullptr;
    session.inference_session = {};
    session.initialized = false;
}

static bool run_named_test(const std::string & name) {
    if (name == "generation-contract") {
        test_generation_contract_helpers();
    } else if (name == "cli-scope") {
        test_cli_scope_helpers();
    } else if (name == "runtime-assembly") {
        test_runtime_assembly_helpers();
    } else if (name == "runtime-metadata") {
        test_runtime_generation_metadata();
    } else if (name == "selection-metadata") {
        test_selection_generation_metadata();
    } else if (name == "runtime-failure") {
        test_runtime_generation_failure_metadata();
    } else if (name == "selection-failure") {
        test_selection_generation_failure_metadata();
    } else if (name == "mini-runtime-smoke") {
        test_mini_runtime_smoke();
    } else if (name == "runtime-request-builder") {
        test_runtime_request_builder();
    } else if (name == "runtime-execution-builder") {
        test_runtime_execution_builder();
    } else if (name == "chat-runtime-driver-smoke") {
        test_chat_runtime_driver_smoke();
    } else if (name == "runtime-host-chat-smoke") {
        test_runtime_host_chat_smoke();
    } else if (name == "runtime-host-mini-smoke") {
        test_runtime_host_mini_smoke();
    } else if (name == "runtime-host-input-builders") {
        test_runtime_host_input_builders();
    } else if (name == "runtime-host-turn-completion") {
        test_runtime_host_turn_completion();
    } else if (name == "runtime-resident-host-multi-turn") {
        test_runtime_resident_host_multi_turn_smoke();
    } else if (name == "runtime-resident-chat-host-builder") {
        test_runtime_resident_chat_host_builder();
    } else if (name == "runtime-resident-request-builders") {
        test_runtime_resident_request_builders();
    } else if (name == "runtime-resident-runtime-builder") {
        test_runtime_resident_runtime_builder();
    } else if (name == "runtime-resident-mini-host-builder") {
        test_runtime_resident_mini_host_builder();
    } else if (name == "cli-runtime-host-adapter-chat") {
        test_cli_runtime_host_adapter_chat_inputs();
    } else if (name == "runtime-session-reuse") {
        test_runtime_session_reuse();
    } else {
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        if (!run_named_test(argv[1])) {
            return 2;
        }
        return 0;
    }

    const char * tests[] = {
        "generation-contract",
        "cli-scope",
        "runtime-assembly",
        "runtime-metadata",
        "selection-metadata",
        "runtime-failure",
        "selection-failure",
        "mini-runtime-smoke",
        "runtime-request-builder",
        "runtime-execution-builder",
        "chat-runtime-driver-smoke",
        "runtime-host-chat-smoke",
        "runtime-host-mini-smoke",
        "runtime-host-input-builders",
        "runtime-host-turn-completion",
        "runtime-resident-host-multi-turn",
        "runtime-resident-chat-host-builder",
        "runtime-resident-request-builders",
        "runtime-resident-runtime-builder",
        "runtime-resident-mini-host-builder",
        "cli-runtime-host-adapter-chat",
        "runtime-session-reuse",
    };
    for (const char * name : tests) {
        if (!run_named_test(name)) {
            return 2;
        }
    }
    return 0;
}
