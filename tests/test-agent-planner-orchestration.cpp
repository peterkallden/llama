#include "tools/agent/cli/agent-cli-runtime.h"
#include "tools/agent/runtime/agent-runtime-assembly.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"
#include "plan/plan-in-memory.h"
#include "test-tool-runtime-registry-adapter.h"

#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {
using json = nlohmann::ordered_json;

// These checks deliberately remain active with NDEBUG.
void require(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

common_agent_generation_result reply(const std::string & text) {
    common_agent_generation_result result;
    result.content = text;
    result.status = common_agent_generation_status::completed;
    result.stop_reason = common_agent_generation_stop_reason::eos;
    return result;
}

const std::string inspect_plan = "plan goal=\"Inspect authorized repository\"\nstep | open! code.inspect";

struct scripted_inference final : common_agent_inference {
    std::vector<common_agent_generation_result> replies;
    std::vector<common_agent_generation_request> requests;

    bool generate(const common_agent_generation_request & request, common_agent_generation_result & result) override {
        require(request.purpose == common_agent_generation_purpose::planner,
            "planner recovery escaped to another inference purpose");
        require(requests.size() < replies.size(), "planner exceeded the scripted regeneration bound");
        result = replies[requests.size()];
        requests.push_back(request);
        return result.status == common_agent_generation_status::completed;
    }
};

struct evidence_executor final : common_action_executor {
    explicit evidence_executor(const std::vector<common_agent_tool_call> & calls) : calls(calls) {}
    const std::vector<common_agent_tool_call> & calls;
    size_t drafts = 0;

    std::string generate_draft(const common_agent_request &, const common_plan_state & plan,
            const std::vector<std::string> &, std::string & error) override {
        ++drafts;
        require(!calls.empty(), "final synthesis ran before required tool execution");
        require(std::any_of(plan.observations.begin(), plan.observations.end(),
                [](const common_plan_observation & observation) {
                    return observation.summary.find("verified-tool-result") != std::string::npos;
                }), "final synthesis did not receive the recorded tool result");
        error.clear();
        return "verified-tool-result";
    }
};

struct accepting_reflector final : common_reflection_engine {
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &,
            const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.ready_to_answer = true;
        return result;
    }
};

struct fixture {
    common_tool_catalog catalog;
    common_tool_registry registry;
    common_plan_in_memory_store plans;
    scripted_inference inference;
    std::vector<common_agent_tool_call> calls;
    evidence_executor executor{calls};
    accepting_reflector reflector;
    std::vector<common_chat_tool> tools;
    common_agent_request request;
    bool fail_tool = false;

    fixture() {
        std::string error;
        common_tool_bootstrap_result bootstrap;
        require(catalog.bootstrap("code-analysis", bootstrap, error), error);
        require(plans.open("", error), error);
        for (const auto * name : {"code.inspect", "code.symbols"}) {
            const auto * definition = catalog.find_definition(name);
            require(definition != nullptr, "missing production code tool contract");
            tools.push_back({definition->name, definition->description,
                common_tool_model_input_schema(*definition), common_tool_model_result_schema(*definition)});
            common_registered_tool registered;
            registered.name = definition->name;
            registered.arguments_schema = definition->input_schema_json;
            registered.handler = [this, name](const std::string & arguments) {
                calls.push_back({name, arguments});
                if (fail_tool) {
                    return common_tool_execution_result::failure("tool.fixture.unavailable",
                        common_tool_failure_class::network, false, "Fixture tool unavailable.", "fixture failure");
                }
                common_tool_execution_result result;
                result.ok = true;
                result.output = R"({"marker":"verified-tool-result"})";
                return result;
            };
            require(registry.register_tool(std::move(registered), error), error);
        }
        // Registration alone must not make this tool available to the planner.
        common_registered_tool hidden;
        hidden.name = "private.inspect";
        hidden.arguments_schema = R"({"type":"object","additionalProperties":false})";
        hidden.handler = [this](const std::string & arguments) {
            calls.push_back({"private.inspect", arguments});
            common_tool_execution_result result;
            result.ok = true;
            return result;
        };
        require(registry.register_tool(std::move(hidden), error), error);
        request.namespace_id = "orchestration-contract";
        request.project_id = "project-a";
        request.session_id = "session-a";
        request.turn_id = "turn-a";
        request.prompt = "HOST_CONTEXT_MARKER: inspect the selected repository in the current project.";
        request.require_tool_execution = true;
        request.tool_output_format = common_agent_tool_output_format::compact_dsl;
        request.max_tool_batches = 2;
    }

    common_agent_result run() {
        common_agent_generation_config config;
        config.n_predict = 64;
        config.planner_n_predict = 96;
        auto planner = make_llama_cli_planner(inference, config, tools);
        test_tool_runtime_registry_adapter tool_runtime(registry);
        common_agent_runtime runtime(plans, *planner, executor, reflector, &tool_runtime);
        return runtime.run(request);
    }

    void check_generation_context(size_t expected_requests) const {
        require(inference.requests.size() == expected_requests, "unexpected planner regeneration count");
        for (const auto & generated : inference.requests) {
            require(generated.json_schema.empty(), "DSL planner silently switched to JSON");
            require(!generated.grammar.empty(), "DSL planner or repair lost sampling-time grammar");
            require(generated.scope && generated.scope->namespace_id == request.namespace_id &&
                generated.scope->project_id == request.project_id &&
                generated.scope->session_id == request.session_id && generated.scope->turn_id == request.turn_id,
                "planner or repair lost the host scope");
            require(generated.options.n_predict == 96, "planner repair changed the response reservation");
            size_t occurrences = 0;
            for (const auto & message : generated.messages) {
                for (size_t at = 0; (at = message.content.find(request.prompt, at)) != std::string::npos;
                        at += request.prompt.size()) ++occurrences;
            }
            require(occurrences == 1, "planner lost or duplicated the host-composed turn context");
        }
    }

    void check_rejected(const common_agent_result & result, size_t generations) {
        require(!result.error.empty() && result.response.empty(), "invalid plan produced a successful answer");
        require(calls.empty() && executor.drafts == 0, "invalid plan reached tool execution or final synthesis");
        std::string error;
        require(plans.list(error).empty() && error.empty(), "rejected candidate published a partial plan");
        check_generation_context(generations);
    }

    void check_completed(const common_agent_result & result, const std::string & expected_tool) {
        require(result.error.empty() && result.response == "verified-tool-result", result.error);
        require(calls.size() == 1 && calls.front().name == expected_tool,
            "rejected or unrelated steps leaked into execution");
        require(executor.drafts == 1 && result.plan_id.has_value(), "valid plan did not reach final synthesis once");
        std::string error;
        const auto plan = plans.get(*result.plan_id, error);
        require(error.empty() && plan && plan->status == common_plan_status::completed,
            "executed plan was not persisted as completed");
        require(plan->namespace_id == request.namespace_id && plan->project_id == request.project_id &&
            plan->session_id == request.session_id, "published plan lost its host scope");
        require(plan->observations.size() == 1 &&
            plan->observations.front().summary.find("verified-tool-result") != std::string::npos,
            "tool evidence was lost or duplicated in the published plan");
    }
};

void rejection_contracts() {
    // Tool routing and a mandatory-tool binding are separate decisions.
    // Invalid planning must fail closed even without a forced tool binding.
    fixture unbound;
    unbound.request.require_tool_execution = false;
    unbound.inference.replies = {reply("Here are the steps to inspect the repository."),
        reply("I recommend opening the repository first.")};
    unbound.check_rejected(unbound.run(), 2);

    std::string too_many_steps = "plan goal=\"Inspect\"";
    for (int step = 0; step < 7; ++step) too_many_steps += "\nstep | open! code.inspect";
    const std::vector<std::string> invalid = {
        inspect_plan + "\nfield: result",
        inspect_plan + "\nfield: backend\nfield: listed_operations",
        "plan goal=\"Inspect\"\nstep | open! code.inspect.",
        "plan goal=\"Inspect\"\nstep | open! private.inspect",
        inspect_plan + "\nstep | open! code.inspect\nfield: result",
        too_many_steps,
        R"({"goal":"Inspect","steps":[{"tool":"code.inspect","args":{}}]})",
    };
    for (const auto & candidate : invalid) {
        fixture test;
        test.inference.replies = {reply(candidate), reply(candidate)};
        test.check_rejected(test.run(), 2);
    }
}

void recovery_contracts() {
    // A valid prefix from a rejected candidate must never be persisted or dispatched.
    fixture test;
    test.inference.replies = {
        reply("plan goal=\"Inspect\"\nstep | open! code.symbols\nsymbol: stale_candidate\n"
            "step | open! code.inspect\nfield: result"),
        reply(inspect_plan),
    };
    const auto result = test.run();
    test.check_completed(result, "code.inspect");
    test.check_generation_context(2);
    require(json::parse(test.calls.front().arguments_json).empty(), "repaired no-argument call acquired arguments");

    fixture missing_argument;
    missing_argument.inference.replies = {
        reply("plan goal=\"Locate definition\"\nstep | open! code.symbols"),
        reply("symbol: selected_symbol"),
    };
    missing_argument.check_completed(missing_argument.run(), "code.symbols");
    missing_argument.check_generation_context(2);
    require(json::parse(missing_argument.calls.front().arguments_json).value("symbol", "") == "selected_symbol",
        "DSL argument repair was not consumed as host-canonical plan data");

    fixture bad_repair;
    bad_repair.inference.replies = {
        reply("plan goal=\"Locate definition\"\nstep | open! code.symbols"),
        reply("symbol: 42"),
    };
    bad_repair.check_rejected(bad_repair.run(), 2);
}

void dispatch_contracts() {
    const std::vector<std::string> prompts = {
        "Inspect repository capabilities.",
        "Fortsätt undersökningen i valt projekt.",
        "Inspect the current repository; compare and verify the latest local symbols without new research.",
        "The source literally contains current, latest, verify, compare and uncertain.",
    };
    for (size_t index = 0; index < prompts.size(); ++index) {
        fixture test;
        if (index % 2) std::reverse(test.tools.begin(), test.tools.end());
        test.request.prompt = prompts[index];
        test.inference.replies = {reply(inspect_plan)};
        test.check_completed(test.run(), "code.inspect");
        test.check_generation_context(1);
        require(json::parse(test.calls.front().arguments_json).empty(), "no-argument call acquired invented fields");
    }
    fixture bound;
    bound.request.tool_argument_bindings.push_back({"code.symbols", R"({"symbol":"host_bound_symbol"})", "host-selection", true});
    bound.inference.replies = {reply("plan goal=\"Inspect selected symbol\"\nstep | open! code.symbols")};
    bound.check_completed(bound.run(), "code.symbols");
    require(json::parse(bound.calls.front().arguments_json).value("symbol", "") == "host_bound_symbol",
        "dispatch lost the host-owned argument binding");

    fixture conflict;
    conflict.request.tool_argument_bindings = bound.request.tool_argument_bindings;
    const auto conflicting_plan = reply("plan goal=\"Inspect\"\nstep | open! code.symbols\nsymbol: model_override");
    conflict.inference.replies = {conflicting_plan, conflicting_plan};
    conflict.check_rejected(conflict.run(), 2);

    fixture failed_tool;
    failed_tool.fail_tool = true;
    failed_tool.inference.replies = {reply(inspect_plan)};
    const auto result = failed_tool.run();
    require(!result.error.empty() && result.response.empty() && failed_tool.calls.size() == 1 &&
        failed_tool.executor.drafts == 0, "mandatory tool failure allowed unsupported final synthesis");
}

void interruption_contracts() {
    for (const auto status : {common_agent_generation_status::cancelled, common_agent_generation_status::errored}) {
        fixture test;
        auto interrupted = reply(inspect_plan);
        interrupted.status = status;
        interrupted.stop_reason = status == common_agent_generation_status::cancelled
            ? common_agent_generation_stop_reason::cancelled : common_agent_generation_stop_reason::error;
        interrupted.error_message = "fixture backend interruption";
        // Even a syntactically valid partial payload must not override backend cancellation/error.
        test.inference.replies = {interrupted, reply(inspect_plan)};
        const auto result = test.run();
        test.check_rejected(result, 1);
        require(result.error.find(interrupted.error_message) != std::string::npos,
            "planner replaced a backend interruption with a schema-selection failure");
    }
    fixture truncated;
    auto partial = reply("plan goal=\"Inspect\"\nstep | open! code.inspect\nfield:");
    partial.stop_reason = common_agent_generation_stop_reason::limit;
    truncated.inference.replies = {partial, reply(inspect_plan)};
    truncated.check_completed(truncated.run(), "code.inspect");
    truncated.check_generation_context(2);
}
void blueprint_planning_contracts() {
    fixture test;
    common_plan_state method;
    method.id = "unbound-method"; method.kind = common_plan_kind::blueprint;
    method.goal = "UNDERLAY_METHOD_GOAL"; method.success_criteria = "Retained evidence before conclusions";
    method.steps = {{"inspect", "Inspect", "Inspect the authorized repository"}, {"assess", "Assess", "Assess observed capabilities"}};
    method.steps.back().depends_on = {"inspect"};
    method.constraints.push_back({"scope", "Keep the authorized project scope", true});
    test.request.blueprint_plan_id = method.id;
    test.request.blueprint_definition = method;
    test.inference.replies = {reply(inspect_plan)};
    const auto result = test.run();
    test.check_completed(result, "code.inspect");
    test.check_generation_context(1);
    const auto & prompt = test.inference.requests.front().messages;
    require(std::any_of(prompt.begin(), prompt.end(), [](const auto & message) {
        return message.content.find("UNDERLAY_METHOD_GOAL") != std::string::npos &&
            message.content.find("depends_on") != std::string::npos;
    }), "ordinary planner lost the unbound blueprint method");
    std::string error;
    const auto plan = test.plans.get(*result.plan_id, error);
    require(plan && plan->derived_from_plan_id == method.id && !plan->workflow_definition,
        "unbound method invented a persistent workflow definition");
    require(std::any_of(plan->observations.begin(), plan->observations.end(), [](const auto & item) {
        return item.source == "blueprint_outline" && item.confidence == 0.0f;
    }), "blueprint outline was lost or counted as execution evidence");
}
} // namespace

int main(int argc, char ** argv) {
    require(argc == 2, "select rejection, recovery, dispatch or interruptions");
    const std::string group = argv[1];
    if (group == "rejection") rejection_contracts();
    else if (group == "recovery") recovery_contracts();
    else if (group == "dispatch") dispatch_contracts();
    else if (group == "interruptions") interruption_contracts();
    else if (group == "blueprint") blueprint_planning_contracts();
    else require(false, "unknown orchestration contract group");
    return 0;
}
