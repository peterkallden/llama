#include "agent-cli-generation-utils.h"

std::string make_agent_cli_generation_trace_id(
        const common_agent_request & request,
        common_agent_generation_purpose purpose) {
    const std::string base = !request.turn_id.empty() ? request.turn_id : request.session_id;
    return base + ":" + common_agent_generation_purpose_name(purpose);
}

std::string describe_agent_cli_generation_failure(
        const char * label,
        const common_agent_generation_result & result) {
    std::string text = std::string(label) + " failed";
    text += " (status=" + std::string(common_agent_generation_status_name(result.status));
    text += ", stop=" + std::string(common_agent_generation_stop_reason_name(result.stop_reason)) + ")";
    if (!result.error_message.empty()) {
        text += ": " + result.error_message;
    }
    return text;
}

common_agent_generation_options make_agent_cli_generation_options(
        const common_agent_generation_config & generation_config,
        int n_predict) {
    common_agent_generation_options options;
    options.n_predict = generation_config.n_predict;
    options.n_threads = generation_config.n_threads;
    options.generation_trace = generation_config.generation_trace;
    return common_agent_generation_options_with_n_predict(
        options,
        n_predict);
}

common_agent_generation_request make_agent_cli_generation_request(
        const common_agent_request & request,
        common_agent_generation_purpose purpose,
        std::vector<common_chat_msg> messages,
        common_agent_generation_options options,
        std::string json_schema,
        std::vector<common_chat_tool> tools,
        common_chat_tool_choice tool_choice) {
    auto generation = common_agent_make_generation_request(
        purpose,
        make_agent_cli_generation_trace_id(request, purpose),
        common_agent_scope_from_request(request),
        std::move(messages),
        std::move(options),
        std::move(json_schema),
        std::move(tools),
        tool_choice);
    generation.input_resources.reserve(request.input_resources.size());
    for (const auto & input : request.input_resources) {
        generation.input_resources.push_back({input.resource, input.role, input.required});
    }
    return generation;
}
