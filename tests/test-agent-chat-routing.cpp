#include "tools/agent/runtime/agent-runtime-chat-driver.h"
#include "tools/agent/runtime/agent-runtime-host.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "json-schema-to-grammar.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include <cstdlib>
#include <iostream>

static void require(bool ok, const char * message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
struct routing_inference final : common_agent_inference {
    std::vector<common_agent_generation_request> requests;
    std::string selection = "NO_TOOLS";
    bool fail_conversation = false;
    bool generate(const common_agent_generation_request & request, common_agent_generation_result & result) override {
        requests.push_back(request);
        result = {};
        if (request.purpose == common_agent_generation_purpose::tool_family_selection) {
            require(request.tools.empty(), "family routing leaked tool schemas");
            result.content = selection;
            result.status = common_agent_generation_status::completed;
            result.stop_reason = common_agent_generation_stop_reason::eos;
            return true;
        }
        if (!request.tools.empty()) { result.error_message = "stop before dispatch"; return false; }
        if (fail_conversation) {
            result.content = "partial";
            result.decoded_tokens = 7;
            result.error_message = "synthetic generation failure";
            return false;
        }
        result.content = "OK";
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::eos;
        return true;
    }
};
int main() {
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    std::string bootstrap_error;
    require(catalog.bootstrap("memory", bootstrap, bootstrap_error), "catalog bootstrap failed");
    for (const auto * name : {"dataset.inspect", "dataset.schema", "dataset.sample", "dataset.validate"}) {
        const auto * definition = catalog.find_definition(name);
        require(definition != nullptr, "missing dataset contract");
        for (const auto & text : {definition->input_schema_json, definition->model_input_schema_json}) {
            const auto schema = common_json::parse(text);
            const auto document = common_chat_schema_from_json(schema);
            require(document.root->kind() == common_chat_schema::KIND_ANY_OF, "missing source alternatives");
            const auto & alternatives = static_cast<const common_chat_schema_any_of &>(*document.root);
            require(alternatives.children.size() == 3, "wrong source alternative count");
            for (const auto & child : alternatives.children) {
                require(child->kind() == common_chat_schema::KIND_OBJECT, "source grammar fell back to arbitrary JSON");
                const auto & object = static_cast<const common_chat_schema_object &>(*child);
                require(!object.additional_properties, "source branch allows arbitrary fields");
                size_t sources = 0;
                bool rules_required = false;
                for (const auto & property : object.properties) {
                    if (property.name == "dataset" || property.name == "resource" || property.name == "path") {
                        ++sources;
                        require(property.required, "source branch has optional source");
                    }
                    if (property.name == "rules") rules_required = property.required;
                }
                require(sources == 1, "source grammar permits competing source fields");
                if (std::string(name) == "dataset.validate") require(rules_required, "source union lost required rules");
            }
            require(!json_schema_to_grammar(document).empty(), "source grammar compilation failed");
            if (std::string(name) != "dataset.validate") {
                std::string normalized, error;
                require(common_schema_normalize_and_validate_object(R"({"dataset":"dataset://test"})", text, normalized, error), "valid source rejected");
                require(!common_schema_normalize_and_validate_object(R"({})", text, normalized, error), "missing source accepted");
                require(!common_schema_normalize_and_validate_object(R"({"dataset":"dataset://test","resource":"resource://test"})", text, normalized, error), "competing sources accepted");
            }
        }
    }
    {
        std::string normalized, error;
        require(!common_schema_normalize_and_validate_object(R"({"a":1,"b":2})",
            R"({"type":"object","oneOf":[{"required":["a"]},{"required":["b"]}]})", normalized, error), "overlapping alternatives accepted");
        require(error.find("got 2") != std::string::npos && error.find("missing") == std::string::npos,
            "overlap diagnostic incorrectly reports a missing field");
    }

    routing_inference inference;
    common_agent_runtime_tooling tooling;
    tooling.tools = {{"dataset.schema", "schema", R"({"type":"object"})", "{}"},
        {"web.search", "search", R"({"type":"object"})", "{}"}};
    common_agent_request request;
    request.prompt = "Reply exactly OK.";
    request.tool_output_format = common_agent_tool_output_format::native;
    common_agent_chat_runtime_execution execution{inference, request, {}, {16, 2, true}, tooling, {}};
    common_agent_result result;
    std::string error;
    require(run_agent_chat_runtime(execution, result, error), error.c_str());
    require(result.response == "OK", "ordinary answer lost");
    require(inference.requests.size() == 2 && inference.requests.back().tools.empty(), "ordinary chat exposed schemas");
    inference.requests.clear();
    execution.request = request;
    execution.request.prompt.clear();
    common_chat_msg history_question;
    history_question.role = "user";
    history_question.content = "Reply exactly OK from message history.";
    execution.request.messages = {history_question};
    require(run_agent_chat_runtime(execution, result, error), error.c_str());
    require(inference.requests.front().messages.back().content.find("from message history") != std::string::npos,
        "message-only chat lost the routing question");
    inference.requests.clear();
    inference.selection = "TOOLS: dataset";
    execution.request = request;
    require(!run_agent_chat_runtime(execution, result, error), "mock dispatch stop ignored");
    require(inference.requests.size() == 2 && inference.requests.back().tools.size() == 1 &&
        inference.requests.back().tools.front().name == "dataset.schema", "selected family was not bounded");
    require(inference.requests.back().tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, "selected tool requirement lost");
    inference.requests.clear();
    inference.selection = "NO_TOOLS";
    execution.request = request;
    execution.request.require_tool_execution = true;
    require(!run_agent_chat_runtime(execution, result, error), "explicit required tools were downgraded");
    for (const auto & generated : inference.requests) require(generated.purpose == common_agent_generation_purpose::tool_family_selection,
        "required request reached ungrounded conversation");
    inference.requests.clear();
    inference.selection = "TOOLS: unknown";
    execution.request = request;
    require(!run_agent_chat_runtime(execution, result, error), "unknown family accepted");
    // A failed family-routed conversation must retain partial decode metadata
    // and the driver's already-completed preflight trace.
    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    require(memories.open("", error), error.c_str());
    require(plans.open("", error), error.c_str());
    common_agent_runtime_turn_request turn;
    turn.request = request;
    turn.runtime_config.generation_config.enable_tool_family_routing = true;
    turn.runtime_config.generation_config.n_predict = 16;
    turn.orchestration_config.agent_plan = "auto";
    turn.orchestration_config.agent_blueprint = "off";
    turn.orchestration_config.prompt = request.prompt;
    turn.scope.namespace_id = "test";
    turn.scope.session_id = "session";
    turn.scope.turn_id = "turn";
    std::string current_plan;
    std::vector<common_blueprint_candidate> blueprints;
    std::vector<common_memory_hit> memories_found;
    common_agent_runtime_host_build_context context{memories, &plans, turn, &current_plan,
        &blueprints, memories_found, tooling};
    auto inputs = make_agent_runtime_host_agent_inputs(context, turn.orchestration_config);
    auto host_execution = make_agent_runtime_host_execution(inputs, inference);
    inference.selection = "NO_TOOLS";
    inference.fail_conversation = true;
    require(!run_agent_runtime_host(host_execution, result, error), "failed conversation was reported as success");
    require(result.total_decoded_tokens == 7 && !result.trace.empty(), "early failure lost partial tokens or trace");
    require(result.error.find("synthetic generation failure") != std::string::npos, "early failure lost provider reason");
    return 0;
}
