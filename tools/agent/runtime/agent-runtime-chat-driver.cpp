#include "agent-runtime-chat-driver.h"

#include "../tooling/agent-tool-provider.h"
#include "agent/tool-chat-bridge.h"

#include <ctime>

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
        common_chat_tool_choice tool_choice,
        const common_agent_runtime_tooling & execution_tooling) {
    auto generation = common_agent_make_generation_request(
        purpose,
        make_generation_trace_id(request, purpose),
        common_agent_scope_from_request(request),
        messages,
        options,
        {},
        tools,
        tool_choice);
    generation.input_resources.reserve(request.input_resources.size());
    for (const auto & input : request.input_resources) {
        common_agent_generation_resource resource{input.resource, input.role, input.required};
        if (execution_tooling.resource_runtime.store != nullptr) {
            auto * store = execution_tooling.resource_runtime.store;
            const auto authority = make_agent_resource_read_authority(
                execution_tooling.resource_runtime, static_cast<int64_t>(std::time(nullptr)));
            const auto uri = input.resource.uri;
            resource.read_bytes = [store, authority, uri](
                    size_t max_bytes, std::string & out, std::string & error) {
                return store->read_bytes(uri, authority, max_bytes, out, error);
            };
        }
        generation.input_resources.push_back(std::move(resource));
    }
    return generation;
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
    if (execution.tooling.tool_view == nullptr) {
        error = "chat tool dispatch requires a resolved tool view";
        return false;
    }
    common_tool_chat_dispatch_result dispatched;
    if (!agent_dispatch_chat_tool_calls(assistant_message, *execution.tooling.tool_view, 1, dispatched, error)) {
        return false;
    }
    messages.push_back(std::move(assistant_message));
    for (auto & tool_message : dispatched.tool_messages) {
        messages.push_back(std::move(tool_message));
    }
    error.clear();
    return true;
}

void record_truncated_chat_generation(
        common_agent_result & result,
        const common_agent_generation_result & generation_result) {
    if (generation_result.stop_reason != common_agent_generation_stop_reason::limit) {
        return;
    }
    result.limit_reached = true;
    result.trace.push_back({
        common_runtime_trace_stage::turn,
        common_runtime_trace_kind::decided,
        "chat output was truncated at the generation limit; continuation is required",
    });
}

bool reject_truncated_chat_generation(
        common_agent_result & result,
        const common_agent_generation_result & generation_result,
        std::string & error) {
    if (generation_result.stop_reason != common_agent_generation_stop_reason::limit) {
        return false;
    }
    record_truncated_chat_generation(result, generation_result);
    result.response = generation_result.content;
    error = "chat output was truncated; a continuation checkpoint is required before parsing or dispatch";
    return true;
}

} // namespace

bool run_agent_chat_runtime(
        common_agent_chat_runtime_execution & execution,
        common_agent_result & result,
        std::string & error) {
    result = {};
    std::vector<common_chat_msg> messages = execution.request.messages;
    auto available_tools = execution.tooling.tools;
    const auto initial_tool_choice = available_tools.empty() ? COMMON_CHAT_TOOL_CHOICE_NONE : COMMON_CHAT_TOOL_CHOICE_AUTO;
    auto generation_result = execution.inference.generate_result(make_generation_request(
        execution.request,
        common_agent_generation_purpose::conversation,
        messages,
        execution.generation_options,
        available_tools,
        initial_tool_choice,
        execution.tooling));
    result.total_decoded_tokens = generation_result.decoded_tokens;
    result.response_decoded_tokens = generation_result.decoded_tokens;
    result.response_generation_status = generation_result.status;
    result.response_stop_reason = generation_result.stop_reason;
    std::string response_prefix;
    size_t continuation_count = 0;
    while (generation_result.stop_reason == common_agent_generation_stop_reason::limit) {
        // Text-only chat can resume at a message boundary. Tool-enabled chat
        // remains conservative: a truncated envelope must be regenerated
        // structurally instead of being parsed or dispatched.
        if (!available_tools.empty() ||
                continuation_count >= execution.policy.max_continuations) {
            result.response = response_prefix + generation_result.content;
            if (reject_truncated_chat_generation(result, generation_result, error)) {
                return false;
            }
        }
        record_truncated_chat_generation(result, generation_result);
        response_prefix += generation_result.content;
        messages.push_back({"assistant", generation_result.content});
        messages.push_back({
            "user",
            "Continue the same response from the preceding text. Do not repeat it; produce only the next bounded text segment.",
        });
        ++continuation_count;
        generation_result = execution.inference.generate_result(make_generation_request(
            execution.request,
            common_agent_generation_purpose::conversation,
            messages,
            execution.generation_options,
            available_tools,
            initial_tool_choice,
            execution.tooling));
        result.total_decoded_tokens += generation_result.decoded_tokens;
        result.response_decoded_tokens = generation_result.decoded_tokens;
        result.response_generation_status = generation_result.status;
        result.response_stop_reason = generation_result.stop_reason;
    }
    if (!common_agent_generation_succeeded(generation_result)) {
        result.response = response_prefix + generation_result.content;
        error = describe_generation_failure("chat generation", generation_result);
        return false;
    }

    common_chat_msg assistant_message;
    if (!parse_assistant_message(generation_result, !execution.tooling.tools.empty(), assistant_message, error)) {
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
        const std::vector<common_chat_tool> next_tools = allow_another_tool_round ? execution.tooling.tools : std::vector<common_chat_tool>{};
        generation_result = execution.inference.generate_result(make_generation_request(
            execution.request,
            common_agent_generation_purpose::tool_followup,
            messages,
            execution.generation_options,
            next_tools,
            allow_another_tool_round && !next_tools.empty() ? COMMON_CHAT_TOOL_CHOICE_AUTO : COMMON_CHAT_TOOL_CHOICE_NONE,
            execution.tooling));
        result.total_decoded_tokens += generation_result.decoded_tokens;
        result.response_decoded_tokens = generation_result.decoded_tokens;
        result.response_generation_status = generation_result.status;
        result.response_stop_reason = generation_result.stop_reason;
        if (reject_truncated_chat_generation(result, generation_result, error)) {
            return false;
        }
        if (!common_agent_generation_succeeded(generation_result)) {
            error = describe_generation_failure("chat generation", generation_result);
            return false;
        }
        if (!parse_assistant_message(generation_result, allow_another_tool_round && !execution.tooling.tools.empty(), assistant_message, error)) {
            return false;
        }
    }

    result.response = response_prefix +
        (assistant_message.content.empty() ? generation_result.content : assistant_message.content);
    result.limit_reached = false;
    if (result.limit_reached) {
        error = "chat output was truncated; a continuation checkpoint is required before completion";
        return false;
    }
    error.clear();
    return true;
}
