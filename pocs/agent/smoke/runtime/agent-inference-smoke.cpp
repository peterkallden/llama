#include "agent/agent-inference.h"
#include "agent/tooling/registry/tool-registry.h"
#include "memory/memory-candidate.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "tools/agent/cli/agent-cli-host-adapter.h"
#include "tools/agent/cli/agent-cli-run-adapter.h"
#include "tools/agent/cli/agent-cli-runtime.h"
#include "tools/agent/cli/agent-cli-selection.h"
#include "tools/agent/runtime/agent-runtime-assembly.h"
#include "tools/agent/runtime/agent-runtime-chat-driver.h"
#include "tools/agent/runtime/agent-runtime-execution.h"
#include "tools/agent/runtime/agent-runtime-host.h"
#include "tools/agent/runtime/agent-runtime-resident.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/agent/runtime/agent-runtime-tooling.h"
#include "tools/agent/tooling/agent-tool-provider.h"
#include "chat-peg-parser.h"
#include "tools/agent/cli/agent-cli-scope.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
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

class test_agent_tool_view final : public agent_tool_view {
public:
    std::vector<common_chat_tool> tools;
    std::function<agent_tool_result(const agent_tool_call &, std::string &)> handler;

    const std::vector<common_chat_tool> & chat_tools() const override {
        return tools;
    }

    bool exposes_tool(const std::string & name) const override {
        for (const auto & tool : tools) {
            if (tool.name == name) {
                return true;
            }
        }
        return false;
    }

    bool is_read_only(const std::string &) const override {
        return true;
    }

    bool is_policy_gated(const std::string &) const override {
        return false;
    }

    bool validate(const agent_tool_call & call, std::string & error) const override {
        if (!exposes_tool(call.name)) {
            error = "tool not exposed: " + call.name;
            return false;
        }
        error.clear();
        return true;
    }

    agent_tool_result call(
            const agent_tool_call & call,
            std::string & error) override {
        if (!validate(call, error)) {
            return {};
        }
        if (handler) {
            return handler(call, error);
        }
        error.clear();
        agent_tool_result result;
        result.ok = true;
        result.tool_call_id = call.id;
        result.tool_name = call.name;
        result.content_json = R"({"ok":true})";
        return result;
    }

    bool supports_async_call(const std::string &) const override {
        return false;
    }

    bool begin_call_async(
            const agent_tool_call &,
            agent_tool_pending_call &,
            std::string & error) override {
        error = "async tools are not supported in test_agent_tool_view";
        return false;
    }

    bool poll_call_async(
            const agent_tool_pending_call &,
            bool &,
            agent_tool_result &,
            std::string & error) override {
        error = "async tools are not supported in test_agent_tool_view";
        return false;
    }

    bool cancel_call_async(
            const agent_tool_pending_call &,
            std::string & error) override {
        error = "async tools are not supported in test_agent_tool_view";
        return false;
    }
};

static args make_test_args() {
    args options;
    options.n_predict = 64;
    return options;
}

static common_agent_inference_options make_agent_inference_options(const args & options) {
    common_agent_inference_options result;
    result.model = options.model;
    result.n_predict = options.n_predict;
    result.n_gpu_layers = options.n_gpu_layers;
    result.fit_params = true;
    return result;
}

static common_agent_generation_config make_agent_generation_config(const args & options) {
    return {options.n_predict};
}

static common_agent_runtime_policy make_agent_runtime_policy(const args & options) {
    return ::make_agent_runtime_policy({
        options.agent_inference_backend,
        options.tool_profile,
        options.memory_learn,
        options.memory_learn_show_candidate,
        options.plan_show_summary,
        options.agent_trace,
        options.max_tool_rounds,
    });
}

static common_agent_runtime_config make_agent_runtime_config(const args & options) {
    common_agent_runtime_build_config config;
    config.generation_config = make_agent_generation_config(options);
    config.enable_memory_learning = options.memory_learn == "post-turn";
    config.memory_learning_config = {
        options.memory_learn_min_confidence,
        options.memory_learn_min_reuse,
    };
    return ::make_agent_runtime_config(std::move(config));
}

static common_agent_orchestration_config make_test_agent_orchestration_config(const args & options) {
    return ::make_agent_orchestration_config({
        options.prompt,
        options.agent_plan,
        options.agent_blueprint,
        options.agent_bootstrap,
        options.agent_import,
        options.agent_export,
    });
}

static bool make_agent_inference_session(
        const common_agent_inference_options & options,
        agent_inference_backend backend,
        llama_model * model,
        const common_chat_templates * templates,
        common_agent_inference_session & session,
        std::string & error) {
    session = {};
    session.backend = backend;
    session.model = model;
    session.templates = templates;
    if (backend == agent_inference_backend::server_context) {
        error = "test helper does not build server_context sessions";
        return false;
    }
    session.inference = std::make_unique<counting_agent_inference>();
    error.clear();
    return true;
}

static common_agent_runtime_tooling make_runtime_tooling(
        const std::vector<common_chat_tool> & tools,
        agent_tool_view * tool_view = nullptr,
        bool profile_tools_active = false) {
    common_agent_runtime_tooling tooling;
    tooling.tools = tools;
    tooling.profile_tools_active = profile_tools_active;
    tooling.tool_view = tool_view;
    return tooling;
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
    auto request = common_agent_make_generation_request(
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

    common_agent_generation_resource resource;
    resource.resource.uri = "agent-resource://turn/turn-7/image.png";
    resource.resource.mime_type = "image/png";
    resource.role = "user_attachment";
    resource.required = true;
    request.input_resources.push_back(std::move(resource));
    common_agent_generation_resource audio_resource;
    audio_resource.resource.uri = "agent-resource://turn/turn-7/audio.mp3";
    audio_resource.resource.mime_type = "audio/mpeg";
    audio_resource.role = "user_attachment";
    audio_resource.required = false;
    request.input_resources.push_back(std::move(audio_resource));
    assert(request.input_resources.size() == 2);
    assert(request.input_resources[0].resource.uri == "agent-resource://turn/turn-7/image.png");
    assert(request.input_resources[0].role == "user_attachment");
    assert(request.input_resources[0].required);
    assert(request.input_resources[1].resource.uri == "agent-resource://turn/turn-7/audio.mp3");
    assert(request.input_resources[1].resource.mime_type == "audio/mpeg");
    assert(request.input_resources[1].role == "user_attachment");
    assert(!request.input_resources[1].required);
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
        make_success(R"({"goal":"Check status","steps":[{"id":"inspect","mode":"reasoning","objective":"Inspect the current status"}]})"),
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
    assert(inference.seen[0].messages.size() == 2);
    assert(inference.seen[0].messages[0].content.find("required: goal:string; steps:object[]") != std::string::npos);
    assert(inference.seen[0].messages[0].content.find("steps: step[]") != std::string::npos);
    assert(inference.seen[0].messages[0].content.find(
        "Each step normally contains only {tool?,args?,as?,mode?}") != std::string::npos);

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
    assert(inference.seen[4].messages.size() == 2);
    assert(inference.seen[4].messages[0].content.find("For dataset repair, use dataset.list/select") != std::string::npos);
    assert(inference.seen[4].messages[1].content.find("<reflection_plan>") != std::string::npos);
    assert(inference.seen[4].messages[1].content.find("[Draft]") != std::string::npos);

    auto extractor = make_llama_cli_memory_candidate_extractor(inference, make_agent_generation_config(options));
    common_agent_result agent_result;
    agent_result.response = "Final answer";
    const auto memory_candidate = extractor->extract_result(request, plan, agent_result, error);
    assert(error.empty());
    assert(memory_candidate.candidate);
    assert(memory_candidate.candidate->kind == common_memory_kind::procedure);
    assert(memory_candidate.generation);
    assert(memory_candidate.generation->stop_reason == common_agent_generation_stop_reason::json_schema);

    assert(inference.seen.size() == 6);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[1].messages.size() == 2);
    assert(inference.seen[1].messages[1].content.find("[Regeneration]") != std::string::npos);
    assert(inference.seen[2].purpose == common_agent_generation_purpose::draft);
    assert(inference.seen[3].purpose == common_agent_generation_purpose::reasoning);
    assert(inference.seen[4].purpose == common_agent_generation_purpose::reflection);
    assert(inference.seen[5].purpose == common_agent_generation_purpose::memory_learning);
    assert(inference.seen[0].trace_id && *inference.seen[0].trace_id == "turn-7:planner");
    assert(inference.seen[1].trace_id && *inference.seen[1].trace_id == "turn-7:planner");
    assert(inference.seen[2].trace_id && *inference.seen[2].trace_id == "turn-7:draft");
    assert(inference.seen[3].trace_id && *inference.seen[3].trace_id == "turn-7:reasoning");
    assert(inference.seen[4].trace_id && *inference.seen[4].trace_id == "turn-7:reflection");
    assert(inference.seen[5].trace_id && *inference.seen[5].trace_id == "turn-7:memory_learning");
    assert(inference.seen[0].scope && inference.seen[0].scope->namespace_id == "tenant-a");
    assert(inference.seen[0].scope && inference.seen[0].scope->session_id == "session-42");
    assert(inference.seen[0].scope && inference.seen[0].scope->turn_id == "turn-7");
    assert(inference.seen[0].scope && inference.seen[0].scope->plan_scope == common_plan_scope::turn);
    assert(inference.seen[0].scope && inference.seen[0].scope->memory_scope == common_memory_scope::session);
    assert(inference.seen[0].options.n_predict == 512);
    assert(inference.seen[1].options.n_predict == 512);
    assert(inference.seen[2].options.n_predict == 64);
    assert(inference.seen[3].options.n_predict == 64);
    assert(inference.seen[4].options.n_predict == 384);
    assert(inference.seen[5].options.n_predict == 256);
    assert(inference.queued.empty());
}

static void test_planner_regenerates_truncated_json() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"({"goal":"incomplete")", 512, common_agent_generation_stop_reason::limit),
        make_success(R"({"goal":"Inspect the request","steps":[{"id":"inspect","mode":"reasoning","objective":"Inspect the request"}]})"),
    };

    const auto options = make_test_args();
    const common_agent_request request = make_request();
    auto planner = make_llama_cli_planner(inference, make_agent_generation_config(options), {});
    std::string error;
    const auto proposal = planner->create_plan_result(request, error);
    assert(error.empty());
    assert(proposal.plan.goal == "Inspect the request");
    assert(proposal.plan.steps.empty());
    assert(proposal.operations.size() == 2);
    assert(proposal.operations[0].step && proposal.operations[0].step->id == "inspect");
    assert(proposal.operations[1].step && proposal.operations[1].step->id == "answer");
    assert(inference.seen.size() == 2);
    assert(inference.seen[1].messages.size() == 2);
    assert(inference.seen[1].messages[1].content.find("Regeneration") != std::string::npos);
}

static void test_planner_repairs_invalid_resource_binding() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"({"goal":"Inspect file","steps":[{"tool":"dataset.sample","mode":"tool","args":{"resource":"$datasets.datasets[]"}}]})"),
        make_success(R"({"goal":"Inspect file","steps":[{"tool":"dataset.sample","mode":"tool","args":{"resource":"r1","rows":20}}]})"),
    };

    const auto options = make_test_args();
    common_agent_request request = make_request();
    request.prompt = "Tell me what the attached CSV contains";
    request.require_tool_execution = true;
    common_agent_input_resource resource;
    resource.resource.name = "data.csv";
    resource.resource.mime_type = "text/csv";
    resource.resource.uri = "agent-resource://resource/data.csv";
    request.input_resources.push_back(std::move(resource));
    const std::vector<common_chat_tool> tools = {
        {"dataset.sample", "Sample rows from an attached dataset.",
            R"({"type":"object","additionalProperties":false,"properties":{"resource":{"type":"string"},"rows":{"type":"integer"}}})"},
    };

    auto planner = make_llama_cli_planner(inference, make_agent_generation_config(options), tools);
    std::string error;
    const auto proposal = planner->create_plan_result(request, error);
    assert(error.empty());
    assert(proposal.operations.size() == 1);
    assert(proposal.operations[0].step && proposal.operations[0].step->tool_call);
    const auto arguments = nlohmann::json::parse(proposal.operations[0].step->tool_call->arguments_json);
    assert(arguments["resource"] == "r1");
    assert(inference.seen.size() == 2);
    const auto & repair_prompt = inference.seen[1].messages[1].content;
    assert(repair_prompt.find("plan.binding.") != std::string::npos);
    assert(repair_prompt.find("current attachment choices are: r1") != std::string::npos);
    assert(repair_prompt.find("resource:'r1'") != std::string::npos);
}

static void test_planner_resolves_host_dataset_inventory() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"({"goal":"Join registered datasets","steps":[
            {"tool":"data.query","mode":"tool","args":{"dataset":"$orders.dataset"},"as":"orders"},
            {"tool":"data.query","mode":"tool","args":{"dataset":"$datasets.datasets[1]"},"as":"customers"},
            {"tool":"data.join","mode":"tool","args":{"left":"$orders.dataset","right":"$customers.dataset","on":[{"left":"customer_id","right":"customer_id"}]},"as":"joined"}
        ]})")
    };

    const auto options = make_test_args();
    common_agent_request request = make_request();
    request.prompt = "Join the orders and customers datasets";
    request.require_tool_execution = true;
    common_agent_dataset_descriptor orders;
    orders.ref.name = "orders";
    orders.ref.uri = "dataset://local/orders";
    common_agent_dataset_descriptor customers;
    customers.ref.name = "customers";
    customers.ref.uri = "dataset://local/customers";
    request.available_datasets = {orders, customers};
    const std::vector<common_chat_tool> tools = {
        {"data.query", "Query a dataset.",
            R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string"}}})"},
        {"data.join", "Join two datasets.",
            R"({"type":"object","additionalProperties":false,"properties":{"left":{"type":"string"},"right":{"type":"string"},"on":{"type":"array"}}})"},
    };

    auto planner = make_llama_cli_planner(inference, make_agent_generation_config(options), tools);
    std::string error;
    const auto proposal = planner->create_plan_result(request, error);
    assert(error.empty());
    assert(proposal.operations.size() == 3);
    const auto first_arguments = nlohmann::json::parse(proposal.operations[0].step->tool_call->arguments_json);
    const auto second_arguments = nlohmann::json::parse(proposal.operations[1].step->tool_call->arguments_json);
    assert(first_arguments.value("dataset", "") == "dataset://local/orders");
    assert(second_arguments.value("dataset", "") == "dataset://local/customers");
}

static void test_planner_resolves_compact_dataset_handles() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"({"goal":"Analyze datasets","steps":[
            {"tool":"data.join","args":{"left":"d1","right":"d2","on":{"left":"customer_id","right":"customer_id"}},"as":"joined","mode":"tool"},
            {"tool":"data.aggregate","args":{"dataset":"joined","measures":[{"function":"sum","column":"amount"}]},"as":"aggregated","mode":"tool"}
        ]})")
    };

    const auto options = make_test_args();
    common_agent_request request = make_request();
    request.require_tool_execution = true;
    common_agent_dataset_descriptor orders;
    orders.ref.name = "orders";
    orders.ref.uri = "dataset://seed/orders";
    common_agent_dataset_descriptor customers;
    customers.ref.name = "customers";
    customers.ref.uri = "dataset://seed/customers";
    request.available_datasets = {orders, customers};
    const std::vector<common_chat_tool> tools = {
        {"data.join", "Join datasets.", R"({"type":"object","properties":{"left":{"type":"string"},"right":{"type":"string"}}})"},
        {"data.aggregate", "Aggregate a dataset.", R"({"type":"object","properties":{"dataset":{"type":"string"},"measures":{"type":"array"}}})"},
    };

    auto planner = make_llama_cli_planner(inference, make_agent_generation_config(options), tools);
    std::string error;
    const auto proposal = planner->create_plan_result(request, error);
    assert(error.empty());
    assert(proposal.operations.size() == 2);
    const auto join_arguments = nlohmann::json::parse(proposal.operations[0].step->tool_call->arguments_json);
    const auto aggregate_arguments = nlohmann::json::parse(proposal.operations[1].step->tool_call->arguments_json);
    assert(join_arguments.value("left", "") == "dataset://seed/orders");
    assert(join_arguments.value("right", "") == "dataset://seed/customers");
    assert(join_arguments["on"].is_array() && join_arguments["on"].size() == 1);
    assert(join_arguments.value("materialize", false));
    assert(join_arguments.value("result_dataset", "").rfind("dataset://agent/turn/", 0) == 0);
    assert(aggregate_arguments.value("dataset", "") == "dataset://seed/orders");
}

static void test_planner_normalizes_unfiltered_query_shape() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"({"goal":"Inspect the dataset","steps":[
            {"tool":"data.query","args":{"dataset":"d1","where":"true"},"as":"rows","mode":"tool"}
        ]})")
    };

    const auto options = make_test_args();
    common_agent_request request = make_request();
    request.require_tool_execution = true;
    common_agent_dataset_descriptor dataset;
    dataset.ref.name = "orders";
    dataset.ref.uri = "dataset://seed/orders";
    request.available_datasets = {dataset};
    const std::vector<common_chat_tool> tools = {
        {"data.query", "Query a dataset.", R"({"type":"object","properties":{"dataset":{"type":"string"}}})"},
    };

    auto planner = make_llama_cli_planner(inference, make_agent_generation_config(options), tools);
    std::string error;
    const auto proposal = planner->create_plan_result(request, error);
    assert(error.empty());
    assert(proposal.operations.size() == 1);
    const auto arguments = nlohmann::json::parse(proposal.operations[0].step->tool_call->arguments_json);
    assert(arguments.value("dataset", "") == "dataset://seed/orders");
    assert(!arguments.contains("where"));
}

static void test_required_planner_failure_preserves_diagnostics() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"({"goal":"Inspect the file","steps":[{"tool":"missing.tool","mode":"tool","args":{}}]})"),
        make_success(R"({"goal":"Inspect the file","steps":[{"tool":"missing.tool","mode":"tool","args":{}}]})"),
    };

    const auto options = make_test_args();
    common_agent_request request = make_request();
    request.prompt = "Tell me what the attached CSV contains";
    request.require_tool_execution = true;
    common_agent_input_resource resource;
    resource.resource.name = "data.csv";
    resource.resource.mime_type = "text/csv";
    resource.resource.uri = "agent-resource://resource/data.csv";
    request.input_resources.push_back(std::move(resource));
    const std::vector<common_chat_tool> tools = {
        {"dataset.sample", "Sample rows from an attached dataset.",
            R"({"type":"object","additionalProperties":false,"properties":{"resource":{"type":"string"}}})"},
    };

    auto planner = make_llama_cli_planner(inference, make_agent_generation_config(options), tools);
    std::string error;
    const auto proposal = planner->create_plan_result(request, error);
    assert(proposal.operations.empty());
    assert(error.find("planner failed after bounded regeneration") != std::string::npos);
    assert(error.find("attempt 1") != std::string::npos);
    assert(error.find("missing.tool") != std::string::npos);
    assert(inference.seen.size() == 2);
    assert(inference.seen[1].messages[1].content.find("current attachment choices are: r1") != std::string::npos);
}

static void test_reflection_regenerates_invalid_json() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"({"decision":"accept")", 256, common_agent_generation_stop_reason::limit),
        make_success(R"({"decision":"accept","ready_to_answer":true})"),
    };

    const auto options = make_test_args();
    const common_agent_request request = make_request();
    common_plan_state plan;
    plan.id = "plan-reflection-regeneration";
    plan.goal = request.prompt;
    plan.success_criteria = "Reply clearly";
    auto reflector = make_llama_cli_reflection_engine(inference, make_agent_generation_config(options));
    std::string error;
    const auto reflection = reflector->evaluate_result(request, plan, "partial draft", error);
    assert(error.empty());
    assert(reflection.decision == common_reflection_decision::accept);
    assert(reflection.ready_to_answer);
    assert(inference.seen.size() == 2);
    assert(inference.seen[1].messages[1].content.find("Regeneration") != std::string::npos);
}

static void test_memory_learning_regenerates_invalid_json() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"({"candidate":)", 256, common_agent_generation_stop_reason::limit),
        make_success(R"({"candidate":{"kind":"procedure","content":"Verify the build before replying.","rationale":"Completed verification showed this is reusable.","importance":0.8,"confidence":0.9,"expected_reuse":0.7,"evidence_ids":["obs-1"],"source_plan_step_ids":["inspect"]},"reason":"Reusable verification procedure"})"),
    };

    const auto options = make_test_args();
    const common_agent_request request = make_request();
    common_plan_state plan;
    plan.id = "plan-memory-regeneration";
    plan.goal = request.prompt;
    plan.success_criteria = "Reply clearly";
    common_agent_result agent_result;
    agent_result.response = "Final answer";
    auto extractor = make_llama_cli_memory_candidate_extractor(inference, make_agent_generation_config(options));
    std::string error;
    const auto candidate = extractor->extract_result(request, plan, agent_result, error);
    assert(error.empty());
    assert(candidate.candidate);
    assert(candidate.candidate->kind == common_memory_kind::procedure);
    assert(inference.seen.size() == 2);
    assert(inference.seen[1].messages[1].content.find("Regeneration") != std::string::npos);
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

    common_blueprint_candidate repository_candidate{
        "repo-change", "bootstrap:repo-change", "Repository change workflow",
        "Safely modify a repository while preserving intended behavior.",
        "Implement and verify the requested repository change.",
        "The requested change is implemented and relevant checks pass.",
    };
    repository_candidate.constraints.push_back({"scope", "Use only host-approved workspace operations.", true});
    repository_candidate.assumptions.push_back({"workspace", "A controlled repository workspace is available.", 0.9f, true, {}});
    repository_candidate.contributions = {"inspect current state", "apply a bounded change", "verify the result"};
    std::vector<common_blueprint_candidate> candidates = {repository_candidate};
    auto selector = make_llama_cli_blueprint_selector(inference, make_agent_generation_config(options));
    const auto blueprint = selector->select_result(request, candidates, error);
    assert(error.empty());
    assert(blueprint.decision == common_blueprint_selection_decision::instantiate);
    assert(blueprint.logical_id && *blueprint.logical_id == "repo-change");
    assert(blueprint.generation);
    assert(blueprint.generation->decoded_tokens == 0);
    const auto & blueprint_prompt = inference.seen[1].messages[1].content;
    assert(blueprint_prompt.find("purpose: Safely modify a repository") != std::string::npos);
    assert(blueprint_prompt.find("goal: Implement and verify") != std::string::npos);
    assert(blueprint_prompt.find("success criteria: The requested change") != std::string::npos);
    assert(blueprint_prompt.find("constraint: Use only host-approved") != std::string::npos);
    assert(blueprint_prompt.find("assumption: A controlled repository") != std::string::npos);
    assert(blueprint_prompt.find("contribution: inspect current state") != std::string::npos);

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

    test_agent_tool_view tool_view;
    tool_view.tools = {
        {"lookup", "Look up a record", R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})"},
    };
    const auto binding = bind_llama_cli_blueprint_tools_result(inference, make_agent_generation_config(options), tool_view, request, store, "task-1", error);
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

    test_agent_tool_view tool_view;
    tool_view.tools = {
        {"lookup", "Look up a record", R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})"},
    };
    const auto binding = bind_llama_cli_blueprint_tools_result(inference, make_agent_generation_config(options), tool_view, request, store, "task-1", error);
    assert(!binding.applied);
    assert(binding.generation);
    assert(error == "blueprint binding generation failed (status=errored, stop=error): tool binding stream failed");
}

static void test_agent_runtime_smoke() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"(not-json)"),
        make_success(R"(still-not-json)"),
        make_success("draft-content", 7),
        make_success(R"({"decision":"accept"})", 3),
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    if (!memories.open("", error) || !plans.open("", error)) {
        std::fprintf(stderr, "runtime inventory smoke setup failed: %s\n", error.c_str());
        std::abort();
    }

    args options = make_test_args();
    options.prompt = "Check status";
    options.memory_learn = "off";
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
    auto tooling = make_runtime_tooling(tools);
    common_agent_dataset_descriptor scoped_dataset;
    scoped_dataset.ref.name = "orders";
    scoped_dataset.ref.uri = "dataset://seed/orders";
    tooling.available_datasets.push_back(scoped_dataset);
    common_agent_runtime_driver_execution execution{
        memories,
        plans,
        inference,
        make_agent_runtime_policy(options),
        make_agent_runtime_config(options),
        make_test_agent_orchestration_config(options),
        current_plan_id,
        scope,
        blueprints,
        std::nullopt,
        hits,
        common_memory_scope::session,
        true,
        tooling,
    };

    common_agent_result result;
    if (!run_agent_runtime_driver(execution, result, error)) {
        std::fprintf(stderr, "runtime inventory smoke failed to run: %s\n", error.c_str());
        std::abort();
    }
    assert(error.empty());
    assert(result.error.empty());
    assert(result.response == "draft-content");
    assert(result.plan_id);
    assert(!result.plan_id->empty());
    if (inference.seen.size() < 1 || inference.seen[0].messages.size() < 2 ||
            inference.seen[0].messages[1].content.find("<runtime_dataset_inventory>") == std::string::npos ||
            inference.seen[0].messages[1].content.find("ref=$datasets.datasets[0].dataset") == std::string::npos ||
            inference.seen[0].messages[1].content.find("uri=dataset://seed/orders") == std::string::npos) {
        std::fprintf(stderr, "runtime planner request did not contain the host dataset inventory\n");
        std::abort();
    }
    assert(inference.seen.size() == 4);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[0].messages.size() == 2);
    assert(inference.seen[0].messages[1].content.find("<runtime_dataset_inventory>") != std::string::npos);
    assert(inference.seen[0].messages[1].content.find("name=orders") != std::string::npos);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[1].messages[1].content.find("Regeneration") != std::string::npos);
    assert(inference.seen[2].purpose == common_agent_generation_purpose::draft);
    assert(inference.seen[3].purpose == common_agent_generation_purpose::reflection);
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
    const auto tooling = make_runtime_tooling(tools, nullptr, true);
    const common_agent_runtime_driver_execution execution{
        memories,
        plans,
        inference,
        make_agent_runtime_policy(options),
        make_agent_runtime_config(options),
        make_test_agent_orchestration_config(options),
        current_plan_id,
        scope,
        blueprints,
        std::nullopt,
        hits,
        common_memory_scope::project,
        true,
        tooling,
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
    assert(request.deliberation_policy.mode == common_agent_thinking_mode::reflective);
    assert(request.deliberation_policy.max_reflection_rounds == 1);
    assert(request.memories.size() == 1);
    assert(request.memories[0].memory.id == "mem-1");

    options.max_tool_rounds = 4;
    options.tool_profile = "safe";
    std::string no_tools_plan_id = "plan-1";
    const auto no_tools_tooling = make_runtime_tooling(tools);
    const common_agent_runtime_driver_execution no_tools_execution{
        memories,
        plans,
        inference,
        make_agent_runtime_policy(options),
        make_agent_runtime_config(options),
        make_test_agent_orchestration_config(options),
        no_tools_plan_id,
        scope,
        blueprints,
        std::nullopt,
        hits,
        common_memory_scope::project,
        false,
        no_tools_tooling,
    };

    const auto no_tools_request = make_agent_runtime_driver_request(no_tools_execution);
    assert(!no_tools_request.enable_memory);
    assert(no_tools_request.enable_reflection);
    assert(no_tools_request.max_iterations == 2);
    assert(no_tools_request.max_reflection_rounds == 1);
    assert(no_tools_request.max_tool_batches == 0);
    assert(!no_tools_request.allow_policy_gated_tool_proposals);
    assert(no_tools_request.deliberation_policy.mode == common_agent_thinking_mode::reflective);

    auto deliberate_execution = execution;
    deliberate_execution.policy.deliberation_policy =
        make_common_agent_deliberation_policy(common_agent_thinking_mode::deliberate);
    deliberate_execution.policy.enable_reflection = false;
    deliberate_execution.policy.max_iterations = 1;
    deliberate_execution.policy.max_reflection_rounds = 0;
    deliberate_execution.policy.max_tool_rounds = 0;
    const auto deliberate_request = make_agent_runtime_driver_request(deliberate_execution);
    assert(deliberate_request.deliberation_policy.mode == common_agent_thinking_mode::deliberate);
    assert(deliberate_request.enable_planning);
    assert(deliberate_request.enable_reflection);
    assert(deliberate_request.max_iterations == 3);
    assert(deliberate_request.max_reflection_rounds == 2);
    assert(deliberate_request.max_tool_batches == 16);

    auto research_execution = execution;
    research_execution.policy.deliberation_policy =
        make_common_agent_deliberation_policy(common_agent_thinking_mode::research);
    research_execution.policy.enable_reflection = false;
    research_execution.policy.max_iterations = 1;
    research_execution.policy.max_reflection_rounds = 0;
    research_execution.policy.max_tool_rounds = 0;
    const auto research_request = make_agent_runtime_driver_request(research_execution);
    assert(research_request.deliberation_policy.mode == common_agent_thinking_mode::research);
    assert(research_request.enable_planning);
    assert(research_request.enable_reflection);
    assert(research_request.max_iterations == 4);
    assert(research_request.max_reflection_rounds == 2);
    assert(research_request.max_tool_batches == 16);
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
    const auto tooling = make_runtime_tooling(tools);
    common_agent_runtime_driver_inputs inputs{
        memories,
        plans,
        make_agent_inference_options(options),
        make_agent_runtime_policy(options),
        make_agent_runtime_config(options),
        make_test_agent_orchestration_config(options),
        current_plan_id,
        scope,
        blueprints,
        std::nullopt,
        hits,
        common_memory_scope::project,
        true,
        fallback_reason,
        tooling,
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
    assert(!execution.runtime_config.embed_memory);
    assert(execution.scope.namespace_id == scope.namespace_id);
    assert(execution.scope.session_id == scope.session_id);
    assert(execution.scope.project_id == scope.project_id);
    assert(&execution.installed_blueprint_candidates == &blueprints);
    assert(&execution.memories == &hits);
    assert(execution.memory_scope == common_memory_scope::project);
    assert(execution.memory_enabled);
    assert(&execution.tooling == &tooling);
    assert(!execution.tooling.profile_tools_active);
    assert(execution.tooling.tool_view == nullptr);
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
    test_agent_tool_view tool_view;
    tool_view.tools = tools;
    tool_view.handler = [](const agent_tool_call & call, std::string & error) {
        assert(call.name == "memory_search");
        error.clear();
        agent_tool_result result;
        result.ok = true;
        result.tool_call_id = call.id;
        result.tool_name = call.name;
        result.content_json = R"({"ok":true,"result":{"items":[{"id":"mem-1"}]}})";
        return result;
    };
    const auto tooling = make_runtime_tooling(tools, &tool_view, false);

    common_agent_chat_runtime_execution execution{
        inference,
        request,
        options,
        {1},
        tooling,
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

static void test_tool_family_singleton_fast_path() {
    fake_agent_inference inference;
    const std::vector<common_chat_tool> tools = {
        {"time_now", "Return the current UTC time.", R"({"type":"object","additionalProperties":false,"properties":{"timezone":{"type":"string","enum":["UTC"]}}})"},
    };
    inference.queued = {
        // Family routing is intentionally a separate, bounded preflight.
        make_success("TOOLS: time"),
        make_success(
            R"(<tool_calls>[{"name":"time_now","arguments":{}}]</tool_calls>)",
            5,
            common_agent_generation_stop_reason::none,
            make_tool_call_chat_params(tools)),
        make_success("The current UTC time is 12:34:56."),
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "What time is it?";
    options.agent_plan = "off";
    options.agent_blueprint = "off";
    options.memory_learn = "off";
    options.max_tool_rounds = 1;

    common_agent_scope scope;
    scope.namespace_id = "tenant-a";
    scope.session_id = "session-42";
    scope.turn_id = "turn-time";
    scope.memory_scope = common_memory_scope::session;
    scope.plan_scope = common_plan_scope::turn;

    size_t dispatch_count = 0;
    test_agent_tool_view tool_view;
    tool_view.tools = tools;
    tool_view.handler = [&dispatch_count](const agent_tool_call & call, std::string & error) {
        ++dispatch_count;
        assert(call.name == "time_now");
        error.clear();
        agent_tool_result result;
        result.ok = true;
        result.tool_call_id = call.id;
        result.tool_name = call.name;
        result.content_json = R"({"timezone":"UTC","time":"12:34:56"})";
        return result;
    };

    auto tooling = make_runtime_tooling(tools, &tool_view, false);
    auto runtime_config = make_agent_runtime_config(options);
    runtime_config.generation_config.enable_tool_family_routing = true;
    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    std::string current_plan_id;
    common_agent_runtime_driver_execution execution{
        memories,
        plans,
        inference,
        make_agent_runtime_policy(options),
        runtime_config,
        make_test_agent_orchestration_config(options),
        current_plan_id,
        scope,
        blueprints,
        std::nullopt,
        hits,
        common_memory_scope::session,
        true,
        tooling,
    };

    common_agent_result result;
    assert(run_agent_runtime_driver(execution, result, error));
    assert(error.empty());
    assert(result.error.empty());
    assert(result.response == "The current UTC time is 12:34:56.");
    assert(dispatch_count == 1);
    assert(execution.family_chat_routed);
    assert(execution.require_tool_execution);

    // The singleton fast path must retain the ordinary chat/tool lifecycle:
    // preflight, required tool call, then a follow-up grounded in the result.
    // It must not silently fall back to a planner or an answer-only response.
    assert(inference.seen.size() == 3);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::tool_family_selection);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::conversation);
    assert(inference.seen[1].tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED);
    assert(inference.seen[1].tools.size() == 1);
    assert(inference.seen[1].tools[0].name == "time_now");
    assert(inference.seen[2].purpose == common_agent_generation_purpose::tool_followup);
    assert(inference.seen[2].tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE);
    assert(inference.seen[2].messages.size() == 3);
    assert(inference.seen[2].messages[2].role == "tool");
    assert(inference.seen[2].messages[2].tool_name == "time_now");
}

static void test_chat_runtime_rejects_truncated_output() {
    fake_agent_inference inference;
    inference.queued = {
        make_success("partial answer", 64, common_agent_generation_stop_reason::limit),
    };

    common_agent_request request = make_request();
    request.messages = {{"user", "Explain the result"}};
    common_agent_generation_options options;
    options.n_predict = 64;
    const auto tooling = make_runtime_tooling({}, nullptr, false);
    common_agent_chat_runtime_execution execution{
        inference,
        request,
        options,
        {1, 0},
        tooling,
    };

    common_agent_result result;
    std::string error;
    assert(!run_agent_chat_runtime(execution, result, error));
    assert(result.response == "partial answer");
    assert(result.limit_reached);
    assert(result.response_stop_reason == common_agent_generation_stop_reason::limit);
    assert(error.find("truncated") != std::string::npos);
    assert(!result.trace.empty());
    assert(result.trace.back().detail.find("output was truncated") != std::string::npos);
}

static void test_chat_runtime_continues_text_at_message_boundary() {
    fake_agent_inference inference;
    inference.queued = {
        make_success("first segment ", 64, common_agent_generation_stop_reason::limit),
        make_success("second segment", 32),
    };

    common_agent_request request = make_request();
    request.messages = {{"user", "Write a bounded answer"}};
    common_agent_generation_options options;
    options.n_predict = 64;
    const auto tooling = make_runtime_tooling({}, nullptr, false);
    common_agent_chat_runtime_execution execution{
        inference,
        request,
        options,
        {1, 1},
        tooling,
    };

    common_agent_result result;
    std::string error;
    assert(run_agent_chat_runtime(execution, result, error));
    assert(error.empty());
    assert(result.response == "first segment second segment");
    assert(!result.limit_reached);
    assert(result.response_stop_reason == common_agent_generation_stop_reason::none);
    assert(inference.seen.size() == 2);
    assert(inference.seen[1].messages.size() == 3);
    assert(inference.seen[1].messages[1].role == "assistant");
    assert(inference.seen[1].messages[2].role == "user");
}

static void test_truncated_tool_call_is_not_dispatched() {
    fake_agent_inference inference;
    const std::vector<common_chat_tool> tools = {
        {"memory_search", "Search memory", R"({"type":"object","additionalProperties":false})"},
    };
    inference.queued = {
        make_success(
            R"(<tool_calls>[{"name":"memory_search","arguments":{"query":"partial"}}]</tool_calls>)",
            64,
            common_agent_generation_stop_reason::limit,
            make_tool_call_chat_params(tools)),
    };

    common_agent_request request = make_request();
    request.messages = {{"user", "Use memory"}};
    common_agent_generation_options options;
    options.n_predict = 64;
    size_t dispatch_count = 0;
    test_agent_tool_view tool_view;
    tool_view.tools = tools;
    tool_view.handler = [&dispatch_count](const agent_tool_call &, std::string & error) {
        ++dispatch_count;
        error.clear();
        return agent_tool_result{};
    };
    const auto tooling = make_runtime_tooling(tools, &tool_view, false);
    common_agent_chat_runtime_execution execution{
        inference,
        request,
        options,
        {1},
        tooling,
    };

    common_agent_result result;
    std::string error;
    assert(!run_agent_chat_runtime(execution, result, error));
    assert(result.limit_reached);
    assert(result.response.find("tool_calls") != std::string::npos);
    assert(dispatch_count == 0);
    assert(error.find("before parsing or dispatch") != std::string::npos);
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
    turn_request.policy.max_tool_rounds = 1;
    test_agent_tool_view tool_view;
    tool_view.tools = tools;
    tool_view.handler = [](const agent_tool_call & call, std::string & error) {
        assert(call.name == "memory_search");
        error.clear();
        agent_tool_result result;
        result.ok = true;
        result.tool_call_id = call.id;
        result.tool_name = call.name;
        result.content_json = R"({"ok":true,"result":{"items":[{"id":"mem-1"}]}})";
        return result;
    };
    const auto tooling = make_runtime_tooling(tools, &tool_view, false);

    common_agent_runtime_host_inputs inputs{
        common_agent_runtime_host_mode::chat,
        memories,
        nullptr,
        turn_request,
        nullptr,
        nullptr,
        nullptr,
        tooling,
        false,
        {},
    };

    auto execution = make_agent_runtime_host_execution(inputs, inference);
    common_agent_result result;
    assert(run_agent_runtime_host(execution, result, error));
    assert(error.empty());
    assert(result.response == "Status is green.");
    assert(inference.seen.size() == 2);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::conversation);
}

static void test_runtime_host_agent_smoke() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"(not-json)"),
        make_success(R"(still-not-json)"),
        make_success("draft-content", 7),
        make_success(R"({"decision":"accept"})", 3),
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.memory_learn = "off";
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
    turn_request.orchestration_config = make_test_agent_orchestration_config(options);
    turn_request.memory_scope = common_memory_scope::session;
    turn_request.memory_enabled = true;

    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    const std::vector<common_chat_tool> tools;
    std::string current_plan_id;
    const auto tooling = make_runtime_tooling(tools);
    common_agent_runtime_host_inputs inputs{
        common_agent_runtime_host_mode::agent,
        memories,
        &plans,
        turn_request,
        &current_plan_id,
        &blueprints,
        &hits,
        tooling,
        false,
        {},
    };

    auto execution = make_agent_runtime_host_execution(inputs, inference);
    common_agent_result result;
    assert(run_agent_runtime_host(execution, result, error));
    assert(error.empty());
    assert(result.response == "draft-content");
    assert(result.plan_id);
    assert(inference.seen.size() == 4);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[2].purpose == common_agent_generation_purpose::draft);
    assert(inference.seen[3].purpose == common_agent_generation_purpose::reflection);
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
        make_test_agent_orchestration_config(options),
        common_memory_scope::project,
        true,
        {},
        nullptr,
        std::move(request),
        generation_options);
    const auto tooling = make_runtime_tooling(tools);

    common_agent_runtime_host_build_context build_context{
        memories,
        &plans,
        turn_request,
        &current_plan_id,
        &blueprints,
        hits,
        tooling,
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

    common_agent_runtime_turn_request agent_turn_request = make_agent_cli_runtime_turn_request(
        options,
        scope,
        make_test_agent_orchestration_config(options),
        common_memory_scope::project,
        true,
        {});
    common_agent_runtime_host_build_context agent_context{
        memories,
        &plans,
        agent_turn_request,
        &current_plan_id,
        &blueprints,
        hits,
        tooling,
    };

    const auto orchestration_config = make_test_agent_orchestration_config(options);
    auto agent_inputs = make_agent_runtime_host_agent_inputs(agent_context, orchestration_config);
    assert(agent_inputs.mode == common_agent_runtime_host_mode::agent);
    assert(agent_inputs.plan_store == &plans);
    assert(agent_inputs.current_plan_id == &current_plan_id);
    assert(agent_inputs.turn_request.scope.session_id == "session-42");
    assert(agent_inputs.turn_request.scope.project_id == "repo-1");
    assert(agent_inputs.installed_blueprint_candidates == &blueprints);
    assert(agent_inputs.turn_request.request.prompt == "Check status");
    assert(agent_inputs.turn_request.request.plan_scope == common_plan_scope::session);
    assert(agent_inputs.turn_request.orchestration_config.prompt == "Check status");
}

static void test_runtime_host_turn_completion() {
    common_memory_in_memory_store memories;
    std::string error;
    assert(memories.open("", error));

    const std::vector<common_chat_tool> tools;
    const auto tooling = make_runtime_tooling(tools);
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
        tooling,
        false,
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
    session.inference_context.initialized = true;
    session.inference_context.session.inference = std::make_unique<fake_agent_inference>();
    common_agent_result result;
    result.response = "Status is green.";

    assert(complete_agent_runtime_host_turn(inputs, session, result, error));
    assert(error.empty());
    assert(post_run_calls == 1);
    assert(session.active_inference_session() != nullptr);
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
    host.session().inference_context.session.backend = agent_inference_backend::cli;
    host.session().inference_context.session.inference = std::make_unique<fake_agent_inference>(std::move(inference));
    auto * inference_ptr = static_cast<fake_agent_inference *>(host.session().inference_context.session.inference.get());
    host.session().inference_context.initialized = true;

    const std::vector<common_chat_tool> tools;
    const auto tooling = make_runtime_tooling(tools);

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
        tooling,
        false,
        {},
    };

    common_agent_result first_result;
    assert(host.run_turn(first_inputs, first_result, error));
    assert(error.empty());
    assert(first_result.response == "First answer");
    assert(host.session().inference_context.session.inference.get() == inference_ptr);
    assert(host.session().inference_context.initialized);

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
        tooling,
        false,
        {},
    };

    common_agent_result second_result;
    assert(host.run_turn(second_inputs, second_result, error));
    assert(error.empty());
    assert(second_result.response == "Second answer");
    assert(host.session().inference_context.session.inference.get() == inference_ptr);
    assert(inference_ptr->seen.size() == 2);
    assert(inference_ptr->seen[0].trace_id && *inference_ptr->seen[0].trace_id == "turn-7:conversation");
    assert(inference_ptr->seen[1].trace_id && *inference_ptr->seen[1].trace_id == "turn-8:conversation");

    host.reset();
    assert(!host.session().inference_context.initialized);
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

    common_agent_runtime_resident_runtime host(
        make_agent_runtime_resident_runtime_config(memories, nullptr, base_turn_request));

    auto inference = std::make_unique<counting_agent_inference>();
    auto * inference_ptr = inference.get();
    host.runtime_host().session().inference_context.session.backend = agent_inference_backend::cli;
    host.runtime_host().session().inference_context.session.inference = std::move(inference);
    host.runtime_host().session().inference_context.initialized = true;

    common_agent_result result;
    assert(host.run_chat_prompt("Reply with TEST only.", "turn-9", 24, result, error));
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
        std::nullopt,
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
}

static void test_runtime_server_context_host_rejects_invalid_model_paths() {
    common_agent_server_context_host host;
    common_agent_server_context_host_config config;
    config.context_key.load_key.model.clear();
    config.context_key.load_key.n_gpu_layers = 0;
    config.context_key.load_key.fit_params = false;
    config.context_key.n_parallel = 1;
    config.context_key.n_sequences = 1;
    config.context_key.n_ctx = 128;
    config.context_key.n_threads = 1;

    std::string error;
    assert(!host.start(config, error));
    assert(error == "resident server_context model path is empty");

    config.context_key.load_key.model = "__llama_agent_missing_model__.gguf";
    error.clear();
    assert(!host.start(config, error));
    assert(error == "resident server_context model does not exist: __llama_agent_missing_model__.gguf");
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
    options.agent_plan = "off";
    options.agent_blueprint = "off";

    auto base_turn_request = make_agent_runtime_resident_base_turn_request({
        options.prompt,
        "session-42",
        "tenant-a",
        {},
        std::nullopt,
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
    base_turn_request.orchestration_config = make_test_agent_orchestration_config(options);
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
        make_success(R"(still-not-json)"),
        make_success("draft-content", 7),
        make_success(R"({"decision":"accept"})", 3),
    };
    auto & runtime_session = runtime.runtime_host().session();
    auto * fake_model = reinterpret_cast<llama_model *>(0x11);
    runtime_session.loaded_model.model = fake_model;
    runtime_session.loaded_model.loaded = true;
    runtime_session.loaded_model.backend = agent_inference_backend::cli;
    runtime_session.loaded_model.key = {
        base_turn_request.inference_options.model,
        base_turn_request.inference_options.n_gpu_layers,
        base_turn_request.inference_options.fit_params,
    };
    runtime_session.inference_context.session.backend = agent_inference_backend::cli;
    runtime_session.inference_context.session.model = fake_model;
    runtime_session.inference_context.session.inference = std::make_unique<fake_agent_inference>(std::move(inference));
    runtime_session.inference_context.key = {
        agent_inference_backend::cli,
        runtime_session.loaded_model.key,
    };
    auto * inference_ptr = static_cast<fake_agent_inference *>(runtime_session.inference_context.session.inference.get());
    runtime_session.inference_context.initialized = true;

    common_agent_result chat_result;
    assert(runtime.run_chat_prompt("Reply with TEST only.", "turn-9", 24, chat_result, error));
    assert(error.empty());
    assert(chat_result.response == "chat-response");

    common_agent_result agent_result;
    assert(runtime.run_agent_prompt("Check status", "turn-11", 24, agent_result, error));
    assert(error.empty());
    assert(agent_result.response == "draft-content");
    assert(agent_result.plan_id);
    assert(runtime.current_plan_id() == *agent_result.plan_id);
    assert(inference_ptr->seen.size() == 5);
    assert(inference_ptr->seen[0].trace_id && *inference_ptr->seen[0].trace_id == "turn-9:conversation");
    assert(inference_ptr->seen[1].trace_id && *inference_ptr->seen[1].trace_id == "turn-11:planner");
    assert(inference_ptr->seen[2].trace_id && *inference_ptr->seen[2].trace_id == "turn-11:planner");
    assert(inference_ptr->seen[3].trace_id && *inference_ptr->seen[3].trace_id == "turn-11:draft");
    assert(inference_ptr->seen[4].trace_id && *inference_ptr->seen[4].trace_id == "turn-11:reflection");
    runtime_session.loaded_model.model = nullptr;
    runtime_session.loaded_model.loaded = false;
}

static void test_runtime_resident_agent_host_builder() {
    fake_agent_inference inference;
    inference.queued = {
        make_success(R"(not-json)"),
        make_success(R"(still-not-json)"),
        make_success("draft-content", 7),
        make_success(R"({"decision":"accept"})", 3),
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    args options = make_test_args();
    options.prompt = "Check status";
    options.memory_learn = "off";
    options.agent_plan = "off";
    options.agent_blueprint = "off";

    auto base_turn_request = make_agent_runtime_resident_base_turn_request({
        options.prompt,
        "session-42",
        "tenant-a",
        {},
        std::nullopt,
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
    base_turn_request.orchestration_config = make_test_agent_orchestration_config(options);
    base_turn_request.memory_scope = common_memory_scope::session;
    base_turn_request.memory_enabled = true;

    common_agent_runtime_resident_runtime host(
        make_agent_runtime_resident_runtime_config(memories, &plans, base_turn_request));

    auto & runtime_session = host.runtime_host().session();
    auto * fake_model = reinterpret_cast<llama_model *>(0x12);
    runtime_session.loaded_model.model = fake_model;
    runtime_session.loaded_model.loaded = true;
    runtime_session.loaded_model.backend = agent_inference_backend::cli;
    runtime_session.loaded_model.key = {
        base_turn_request.inference_options.model,
        base_turn_request.inference_options.n_gpu_layers,
        base_turn_request.inference_options.fit_params,
    };
    runtime_session.inference_context.session.backend = agent_inference_backend::cli;
    runtime_session.inference_context.session.model = fake_model;
    runtime_session.inference_context.session.inference = std::make_unique<fake_agent_inference>(std::move(inference));
    runtime_session.inference_context.key = {
        agent_inference_backend::cli,
        runtime_session.loaded_model.key,
    };
    auto * inference_ptr = static_cast<fake_agent_inference *>(runtime_session.inference_context.session.inference.get());
    runtime_session.inference_context.initialized = true;

    common_agent_result result;
    assert(host.run_agent_prompt("Check status", "turn-11", 24, result, error));
    assert(error.empty());
    assert(result.response == "draft-content");
    assert(result.plan_id);
    assert(host.current_plan_id() == *result.plan_id);
    assert(inference_ptr->seen.size() == 4);
    assert(inference_ptr->seen[0].trace_id && *inference_ptr->seen[0].trace_id == "turn-11:planner");
    assert(inference_ptr->seen[1].trace_id && *inference_ptr->seen[1].trace_id == "turn-11:planner");
    assert(inference_ptr->seen[2].trace_id && *inference_ptr->seen[2].trace_id == "turn-11:draft");
    assert(inference_ptr->seen[3].trace_id && *inference_ptr->seen[3].trace_id == "turn-11:reflection");
    runtime_session.loaded_model.model = nullptr;
    runtime_session.loaded_model.loaded = false;
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
    const auto tooling = make_runtime_tooling(tools);

    auto inputs = make_agent_cli_runtime_host_chat_inputs(
        memories,
        options,
        messages,
        common_memory_scope::project,
        hits,
        true,
        {},
        tooling,
        nullptr,
        {});

    assert(inputs.mode == common_agent_runtime_host_mode::chat);
    assert(inputs.reset_session_on_completion);
    assert(inputs.turn_request.request.prompt == "Check status");
    assert(inputs.turn_request.request.namespace_id == "tenant-a");
    assert(inputs.turn_request.request.project_id == "repo-1");
    assert(inputs.turn_request.scope.plan_scope == common_plan_scope::project);
    assert(inputs.turn_request.request.enable_memory);
    assert(&inputs.tooling == &tooling);
    assert(inputs.tooling.tool_view == nullptr);
}

static void test_agent_runtime_gets_turn_identity_for_project_scope() {
    args options = make_test_args();
    options.agent_runtime = true;
    options.plan_scope = "project";
    options.memory_scope = "project";
    options.memory_turn.clear();
    std::string error;
    assert(prepare_agent_cli_args(options, error));
    assert(error.empty());
    assert(!options.memory_turn.empty());
    assert(options.memory_turn.rfind("implicit-", 0) == 0);
}

static void test_memory_learning_profile_defaults() {
    std::string error;

    args reflective = make_test_args();
    reflective.agent_runtime = true;
    assert(prepare_agent_cli_args(reflective, error));
    assert(error.empty() && reflective.thinking_mode == "reflective" && reflective.memory_learn == "post-turn");

    args deliberate = make_test_args();
    deliberate.agent_runtime = true;
    deliberate.thinking_mode = "deliberate";
    deliberate.thinking_mode_explicit = true;
    assert(prepare_agent_cli_args(deliberate, error));
    assert(error.empty() && deliberate.thinking_mode == "deliberate" && deliberate.memory_learn == "post-turn");

    args research = make_test_args();
    research.agent_runtime = true;
    research.agent_profile = "research";
    research.agent_profile_explicit = true;
    assert(prepare_agent_cli_args(research, error));
    assert(error.empty() && research.thinking_mode == "research" && research.memory_learn == "off");

    args research_learning = research;
    research_learning.memory_learn = "post-turn";
    research_learning.memory_learn_explicit = true;
    assert(prepare_agent_cli_args(research_learning, error));
    assert(error.empty() && research_learning.memory_learn == "post-turn");

    args disabled = make_test_args();
    disabled.agent_runtime = true;
    disabled.memory_learn = "off";
    disabled.memory_learn_explicit = true;
    assert(prepare_agent_cli_args(disabled, error));
    assert(error.empty() && disabled.memory_learn == "off");
}

static void test_runtime_session_reuse() {
    common_agent_runtime_session session;
    auto * fake_model = reinterpret_cast<llama_model *>(0x1);
    auto * fake_templates = reinterpret_cast<const common_chat_templates *>(0x2);
    auto inference = std::make_unique<counting_agent_inference>();
    auto * inference_ptr = inference.get();

    const auto options = make_agent_inference_options(make_test_args());
    session.loaded_model.model = fake_model;
    session.loaded_model.loaded = true;
    session.loaded_model.backend = agent_inference_backend::cli;
    session.loaded_model.key = {
        options.model,
        options.n_gpu_layers,
        options.fit_params,
    };
    session.inference_context.session.backend = agent_inference_backend::cli;
    session.inference_context.session.model = fake_model;
    session.inference_context.session.templates = fake_templates;
    session.inference_context.session.inference = std::move(inference);
    session.inference_context.initialized = true;
    session.inference_context.key = {
        agent_inference_backend::cli,
        session.loaded_model.key,
    };

    std::string error;
    assert(initialize_agent_runtime_session(
        options,
        agent_inference_backend::cli,
        true,
        {},
        session,
        error));
    assert(error.empty());
    assert(session.loaded_model.model == fake_model);
    assert(session.active_inference_session() != nullptr);
    assert(session.active_inference_session()->inference.get() == inference_ptr);
    session.loaded_model.model = nullptr;
    session.loaded_model.loaded = false;
    session.inference_context.session = {};
    session.inference_context.initialized = false;
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
    } else if (name == "planner-regeneration") {
        test_planner_regenerates_truncated_json();
    } else if (name == "planner-resource-binding-repair") {
        test_planner_repairs_invalid_resource_binding();
    } else if (name == "planner-host-dataset-inventory") {
        test_planner_resolves_host_dataset_inventory();
    } else if (name == "planner-compact-dataset-handles") {
        test_planner_resolves_compact_dataset_handles();
    } else if (name == "planner-unfiltered-query-shape") {
        test_planner_normalizes_unfiltered_query_shape();
    } else if (name == "planner-required-failure-diagnostics") {
        test_required_planner_failure_preserves_diagnostics();
    } else if (name == "reflection-regeneration") {
        test_reflection_regenerates_invalid_json();
    } else if (name == "memory-regeneration") {
        test_memory_learning_regenerates_invalid_json();
    } else if (name == "selection-metadata") {
        test_selection_generation_metadata();
    } else if (name == "runtime-failure") {
        test_runtime_generation_failure_metadata();
    } else if (name == "selection-failure") {
        test_selection_generation_failure_metadata();
    } else if (name == "agent-runtime-smoke") {
        test_agent_runtime_smoke();
    } else if (name == "runtime-request-builder") {
        test_runtime_request_builder();
    } else if (name == "runtime-execution-builder") {
        test_runtime_execution_builder();
    } else if (name == "chat-runtime-driver-smoke") {
        test_chat_runtime_driver_smoke();
        test_tool_family_singleton_fast_path();
        test_chat_runtime_rejects_truncated_output();
        test_chat_runtime_continues_text_at_message_boundary();
        test_truncated_tool_call_is_not_dispatched();
    } else if (name == "runtime-host-chat-smoke") {
        test_runtime_host_chat_smoke();
    } else if (name == "runtime-host-agent-smoke") {
        test_runtime_host_agent_smoke();
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
    } else if (name == "runtime-server-context-host-invalid-model-paths") {
        test_runtime_server_context_host_rejects_invalid_model_paths();
    } else if (name == "runtime-resident-runtime-builder") {
        test_runtime_resident_runtime_builder();
    } else if (name == "runtime-resident-agent-host-builder") {
        test_runtime_resident_agent_host_builder();
    } else if (name == "cli-runtime-host-adapter-chat") {
        test_cli_runtime_host_adapter_chat_inputs();
        test_agent_runtime_gets_turn_identity_for_project_scope();
    } else if (name == "memory-learning-profile-defaults") {
        test_memory_learning_profile_defaults();
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
        "planner-regeneration",
        "planner-resource-binding-repair",
        "planner-host-dataset-inventory",
        "planner-compact-dataset-handles",
        "planner-unfiltered-query-shape",
        "planner-required-failure-diagnostics",
        "reflection-regeneration",
        "memory-regeneration",
        "selection-metadata",
        "runtime-failure",
        "selection-failure",
        "agent-runtime-smoke",
        "runtime-request-builder",
        "runtime-execution-builder",
        "chat-runtime-driver-smoke",
        "runtime-host-chat-smoke",
        "runtime-host-agent-smoke",
        "runtime-host-input-builders",
        "runtime-host-turn-completion",
        "runtime-resident-host-multi-turn",
        "runtime-resident-chat-host-builder",
        "runtime-resident-request-builders",
        "runtime-server-context-host-invalid-model-paths",
        "runtime-resident-runtime-builder",
        "runtime-resident-agent-host-builder",
        "cli-runtime-host-adapter-chat",
        "memory-learning-profile-defaults",
        "runtime-session-reuse",
    };
    for (const char * name : tests) {
        if (!run_named_test(name)) {
            return 2;
        }
    }
    return 0;
}
