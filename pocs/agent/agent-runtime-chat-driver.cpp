#include "agent-runtime-chat-driver.h"

#include "agent-tool-provider.h"
#include "agent/tool-chat-bridge.h"

#include <cstdio>

namespace {

std::string make_generation_trace_id(
        const common_agent_request & request,
        common_agent_generation_purpose purpose) {
    const std::string base = !request.turn_id.empty() ? request.turn_id : request.session_id;
    return base + ":" + common_agent_generation_purpose_name(purpose);
}

std::string describe_generation_failure(
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

common_agent_generation_request make_generation_request(
        const common_agent_request & request,
        common_agent_generation_purpose purpose,
        const std::vector<common_chat_msg> & messages,
        const common_agent_generation_options & options,
        const std::vector<common_chat_tool> & tools,
        common_chat_tool_choice tool_choice) {
    return common_agent_make_generation_request(
        purpose,
        make_generation_trace_id(request, purpose),
        common_agent_scope_from_request(request),
        messages,
        options,
        {},
        tools,
        tool_choice);
}

bool parse_assistant_message(
        const common_agent_generation_result & generation_result,
        bool parse_tool_calls,
        common_chat_msg & assistant_message,
        std::string & error) {
    assistant_message = {};
    assistant_message.role = "assistant";
    if (!parse_tool_calls && !generation_result.chat_params) {
        assistant_message.content = generation_result.content;
        error.clear();
        return true;
    }
    if (!generation_result.chat_params) {
        error = "chat generation did not return parser metadata";
        return false;
    }
    common_chat_parser_params parser_params(*generation_result.chat_params);
    parser_params.parse_tool_calls = parse_tool_calls;
    if (!generation_result.chat_params->parser.empty()) {
        parser_params.parser.load(generation_result.chat_params->parser);
    }
    assistant_message = common_chat_parse(generation_result.content, false, parser_params);
    if (assistant_message.role.empty()) {
        assistant_message.role = "assistant";
    }
    error.clear();
    return true;
}

bool append_dispatched_tool_messages(
        common_agent_chat_runtime_execution & execution,
        common_chat_msg & assistant_message,
        std::vector<common_chat_msg> & messages,
        std::string & error) {
    if (execution.profile_tools_active) {
        if (execution.tool_view == nullptr) {
            error = "profile tool chat dispatch requires a resolved tool view";
            return false;
        }
        common_tool_chat_dispatch_result dispatched;
        if (!agent_dispatch_chat_tool_calls(assistant_message, *execution.tool_view, 1, dispatched, error)) {
            return false;
        }
        messages.push_back(std::move(assistant_message));
        for (auto & tool_message : dispatched.tool_messages) {
            messages.push_back(std::move(tool_message));
        }
        error.clear();
        return true;
    }

    if (!execution.tool_handler) {
        error = "legacy chat tool dispatch requires a tool handler";
        return false;
    }

    common_chat_msg tool_message;
    tool_message.role = "tool";
    if (assistant_message.tool_calls.size() != 1) {
        tool_message.content = R"({"ok":false,"error":"only one memory tool call is allowed per chat turn"})";
        std::fprintf(stderr, "warning: rejected unsupported memory tool call\n");
    } else {
        const auto & call = assistant_message.tool_calls.front();
        tool_message.tool_name = call.name;
        tool_message.tool_call_id = call.id.empty() ? "memory-tool-1" : call.id;
        assistant_message.tool_calls.front().id = tool_message.tool_call_id;
        tool_message.content = execution.tool_handler(call);
    }
    messages.push_back(std::move(assistant_message));
    messages.push_back(std::move(tool_message));
    error.clear();
    return true;
}

} // namespace

bool run_agent_chat_runtime(
        common_agent_chat_runtime_execution & execution,
        common_agent_result & result,
        std::string & error) {
    result = {};
    std::vector<common_chat_msg> messages = execution.request.messages;
    auto available_tools = execution.tools;
    const auto initial_tool_choice = available_tools.empty() ? COMMON_CHAT_TOOL_CHOICE_NONE : COMMON_CHAT_TOOL_CHOICE_AUTO;
    auto generation_result = execution.inference.generate_result(make_generation_request(
        execution.request,
        common_agent_generation_purpose::conversation,
        messages,
        execution.generation_options,
        available_tools,
        initial_tool_choice));
    result.total_decoded_tokens = generation_result.decoded_tokens;
    result.response_decoded_tokens = generation_result.decoded_tokens;
    result.response_generation_status = generation_result.status;
    result.response_stop_reason = generation_result.stop_reason;
    if (!common_agent_generation_succeeded(generation_result)) {
        error = describe_generation_failure("chat generation", generation_result);
        return false;
    }

    common_chat_msg assistant_message;
    if (!parse_assistant_message(generation_result, !execution.tools.empty(), assistant_message, error)) {
        return false;
    }

    size_t tool_rounds = 0;
    while (!assistant_message.tool_calls.empty()) {
        if (tool_rounds >= execution.policy.max_tool_rounds) {
            error = "tool call round limit reached";
            result.limit_reached = true;
            return false;
        }
        if (!append_dispatched_tool_messages(execution, assistant_message, messages, error)) {
            return false;
        }

        ++tool_rounds;
        const bool allow_another_tool_round = tool_rounds < execution.policy.max_tool_rounds;
        const std::vector<common_chat_tool> next_tools = allow_another_tool_round ? execution.tools : std::vector<common_chat_tool>{};
        generation_result = execution.inference.generate_result(make_generation_request(
            execution.request,
            common_agent_generation_purpose::tool_followup,
            messages,
            execution.generation_options,
            next_tools,
            allow_another_tool_round && !next_tools.empty() ? COMMON_CHAT_TOOL_CHOICE_AUTO : COMMON_CHAT_TOOL_CHOICE_NONE));
        result.total_decoded_tokens += generation_result.decoded_tokens;
        result.response_decoded_tokens = generation_result.decoded_tokens;
        result.response_generation_status = generation_result.status;
        result.response_stop_reason = generation_result.stop_reason;
        if (!common_agent_generation_succeeded(generation_result)) {
            error = describe_generation_failure("chat generation", generation_result);
            return false;
        }
        if (!parse_assistant_message(generation_result, allow_another_tool_round && !execution.tools.empty(), assistant_message, error)) {
            return false;
        }
    }

    result.response = assistant_message.content.empty() ? generation_result.content : assistant_message.content;
    error.clear();
    return true;
}
