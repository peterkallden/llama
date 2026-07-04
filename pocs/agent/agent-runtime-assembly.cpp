#include "agent-runtime-assembly.h"

#include "agent-cli-runtime.h"

namespace {

class provider_agent_tool_runtime final : public common_agent_tool_runtime {
public:
    explicit provider_agent_tool_runtime(agent_tool_view & tool_view)
        : tool_view(tool_view) {}

    bool is_read_only(const std::string & tool_name) const override {
        return tool_view.is_read_only(tool_name);
    }

    bool is_policy_gated(const std::string &) const override {
        return false;
    }

    bool validate(const common_registered_tool_call & call, std::string & error) const override {
        return tool_view.validate({"", call.name, call.arguments_json}, error);
    }

    common_tool_execution_result execute(const common_registered_tool_call & call) const override {
        std::string error;
        const auto result = tool_view.call({"", call.name, call.arguments_json}, error);
        if (result.ok) {
            return common_tool_execution_result::success(result.content_json);
        }
        return common_tool_execution_result::failure(
            result.failure_code.empty() ? "tool.execution_failed" : result.failure_code,
            result.failure_class,
            result.retryable,
            result.safe_summary.empty() ? "The tool failed." : result.safe_summary,
            result.raw_diagnostic);
    }

private:
    agent_tool_view & tool_view;
};

} // namespace

common_agent_inference_options make_agent_inference_options(common_agent_inference_options config) {
    return config;
}

common_agent_runtime_config make_agent_runtime_config(common_agent_runtime_build_config build_config) {
    common_agent_runtime_config config;
    config.generation_config = std::move(build_config.generation_config);
    config.enable_memory_learning = build_config.enable_memory_learning;
    config.memory_learning_config = std::move(build_config.memory_learning_config);
    config.embed_memory = std::move(build_config.embed_memory);
    return config;
}

bool parse_agent_inference_backend(const std::string & value, agent_inference_backend & backend) {
    if (value == "cli") {
        backend = agent_inference_backend::cli;
        return true;
    }
    if (value == "server-context") {
        backend = agent_inference_backend::server_context;
        return true;
    }
    return false;
}

common_agent_runtime_assembly make_agent_runtime_assembly(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    common_agent_inference & inference,
    const common_agent_runtime_config & runtime_config,
    const std::vector<common_chat_tool> & tools,
    agent_tool_view * tool_view) {
    common_agent_runtime_assembly assembly;
    assembly.planner = make_llama_cli_planner(inference, runtime_config.generation_config, tools);
    assembly.executor = make_llama_cli_action_executor(inference, runtime_config.generation_config);
    assembly.reflector = make_llama_cli_reflection_engine(inference, runtime_config.generation_config);

    if (runtime_config.enable_memory_learning) {
        assembly.candidate_extractor = make_llama_cli_memory_candidate_extractor(inference, runtime_config.generation_config);
        assembly.memory_learner = std::make_unique<common_memory_post_turn_learner>(
            memory_store,
            *assembly.candidate_extractor,
            runtime_config.embed_memory,
            runtime_config.memory_learning_config);
    }

    if (tool_view != nullptr) {
        assembly.tool_runtime = std::make_unique<provider_agent_tool_runtime>(*tool_view);
    }

    assembly.runtime = std::make_unique<common_agent_runtime>(
        plan_store,
        *assembly.planner,
        *assembly.executor,
        *assembly.reflector,
        assembly.tool_runtime.get(),
        assembly.memory_learner.get());
    return assembly;
}
