#include "agent-runtime-chat-driver.h"

#include "../tooling/agent-tool-provider.h"
#include "tools/agent/resource/agent-resource-processing-service.h"
#include "agent/tooling/bridge/tool-chat-bridge.h"
#include "agent/tooling/schema/tool-output-codec.h"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include "agent/input-resources.h"

namespace {

bool is_textual_tool_output_format(common_agent_tool_output_format format) {
    return format == common_agent_tool_output_format::jsonl ||
        format == common_agent_tool_output_format::compact_dsl;
}

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
    const bool textual_tool_output = is_textual_tool_output_format(request.tool_output_format) &&
        !tools.empty();
    std::vector<common_chat_msg> model_messages = messages;
    std::vector<common_chat_tool> model_tools = tools;
    common_chat_tool_choice model_tool_choice = tool_choice;
    if (textual_tool_output) {
        std::vector<std::string> unsupported;
        std::string format_error;
        const auto instructions = common_render_model_tool_output_instructions(
            request.tool_output_format, tools, &unsupported, format_error);
        if (!instructions.empty()) {
            model_messages.insert(model_messages.begin(), {"system", instructions});
        }
        // Textual output is parsed by the shared model-output codec below;
        // passing native tools here would activate the chat-template parser
        // and make the two model-facing dialects impossible to distinguish.
        model_tools.clear();
        model_tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    }
    auto generation = common_agent_make_generation_request(
        purpose,
        make_generation_trace_id(request, purpose),
        common_agent_scope_from_request(request),
        model_messages,
        options,
        {},
        model_tools,
        model_tool_choice,
        request.flydelta_activation,
        request.flydelta_capture);
    generation.input_resources.reserve(request.input_resources.size());
    for (const auto & input : request.input_resources) {
        common_agent_generation_resource resource{input.resource, input.role, input.required};
        if (execution_tooling.resource_runtime.store != nullptr) {
            auto * store = execution_tooling.resource_runtime.store;
            const auto authority = make_agent_resource_read_authority(
                execution_tooling.resource_runtime, static_cast<int64_t>(std::time(nullptr)));
            const auto uri = input.resource.uri;
            const auto mime_type = input.resource.mime_type;
            const auto processing_factory = execution_tooling.resource_runtime.processing_provider_factory;
            resource.read_bytes = [store, authority, uri](
                    size_t max_bytes, std::string & out, std::string & error) {
                return store->read_bytes(uri, authority, max_bytes, out, error);
            };
            if (processing_factory) {
                resource.read_text_fallback = [store, authority, uri, mime_type, processing_factory](
                        size_t max_bytes, std::string & out, std::string & error) {
                    agent_resource_processing_binding_request binding;
                    binding.source_uri = uri;
                    binding.operation_id = "multimodal-fallback/" + uri;
                    binding.authority = authority;
                    binding.media_type.declared_type = mime_type;
                    binding.target_representation = "text";
                    binding.target_media_type = "text/plain";
                    const auto provider = processing_factory(binding);
                    if (!provider) {
                        error = "no configured text fallback processor";
                        return false;
                    }
                    const auto processed = provider->process(binding);
                    if (!processed.success || processed.resources.empty()) {
                        error = processed.safe_summary.empty()
                            ? "text fallback processor failed"
                            : processed.safe_summary;
                        return false;
                    }
                    return store->read_text(
                        processed.resources.front().uri,
                        authority,
                        max_bytes,
                        out,
                        error);
                };
            }
        }
        generation.input_resources.push_back(std::move(resource));
    }
    return generation;
}

bool parse_assistant_message(
        const common_agent_generation_result & generation_result,
        bool parse_tool_calls,
        common_agent_tool_output_format output_format,
        const std::vector<common_chat_tool> & tools,
        common_chat_msg & assistant_message,
        std::string & error) {
    assistant_message = {};
    assistant_message.role = "assistant";
    if (parse_tool_calls && is_textual_tool_output_format(output_format)) {
        assistant_message.content = generation_result.content;
        common_agent_tool_call call;
        if (!common_parse_model_tool_call(output_format, generation_result.content, tools, call, error)) {
            return false;
        }
        assistant_message.tool_calls.push_back({call.name, call.arguments_json, "textual-tool-1"});
        error.clear();
        return true;
    }
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
    if (!generation_result.chat_params->parser.empty()) {
        parser_params.parser = generation_result.chat_params->parser;
    }
    assistant_message = common_chat_parse(common_chat_input(generation_result.content), false, parser_params);
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

bool stop_chat_runtime_if_requested(
        common_agent_chat_runtime_execution & execution,
        common_agent_result & result,
        std::string & error) {
    if (!execution.execution_control.should_stop()) {
        return false;
    }
    result.response_generation_status = common_agent_generation_status::cancelled;
    result.response_stop_reason = common_agent_generation_stop_reason::cancelled;
    error = execution.execution_control.stop_reason();
    if (error.empty()) {
        error = "chat runtime execution stopped";
    }
    result.error = error;
    return true;
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

bool common_agent_select_tool_families(
        common_agent_inference & inference,
        const common_agent_request & request,
        const common_agent_generation_options & generation_options,
        const std::vector<common_tool_family_index> & families,
        bool required_family_selection,
        common_tool_family_selection & selection,
        std::string & error,
        bool * invalid_selection) {
    selection = {};
    if (invalid_selection) *invalid_selection = false;
    const std::string family_view = common_render_tool_family_index(families, 2048);
    common_chat_msg system{
        "system",
        required_family_selection
            ? common_tool_family_required_selection_prompt(families)
            : "Decide whether the user request needs external tools. Reply with exactly one line: "
              "NO_TOOLS for ordinary conversation, or TOOLS: family_id[, family_id...] when tools are needed. "
              "If the user asks about text already included in the prompt, including "
              "host-validated document context, answer from that text and return NO_TOOLS. Do not "
              "select resource or document only because the user mentions an attached file. Select "
              "those families only when more content must be fetched or extracted; select web when "
              "external research is requested, and document tools when editing is requested. "
              "For other requests, attached files are optional inputs: choose resource or dataset "
              "when their contents must be read or analyzed, and keep NO_TOOLS for unrelated conversation. "
              "Return family ids, not tool names; for example, "
              "dataset.validate belongs to the dataset family. Do not explain, use JSON, or invent family ids. Available "
              "families:\n" + family_view,
    };
    std::string routing_prompt = request.prompt;
    if (routing_prompt.empty()) {
        const auto user = std::find_if(request.messages.rbegin(), request.messages.rend(),
            [](const auto & message) { return message.role == "user"; });
        if (user != request.messages.rend()) routing_prompt = user->content;
    }
    common_chat_msg user{"user", common_tool_family_selection_user_prompt(routing_prompt) +
        common_agent_render_input_resource_context(
        request.input_resources, 2048, request.available_resources)};
    auto options = generation_options;
    // This is a one-line classifier, not a planning turn. A small hard cap
    // prevents a weak model from spending the whole turn elaborating before
    // the host can route it.
    options.n_predict = 64;
    options.n_threads = generation_options.n_threads;
    options.generation_trace = generation_options.generation_trace;
    const auto generate_selection = [&](const std::string & request_suffix,
                                        const std::vector<common_chat_msg> & messages,
                                        const std::string & json_schema = std::string{}) {
        return inference.generate_result(
            common_agent_make_generation_request(
                common_agent_generation_purpose::tool_family_selection,
                request.turn_id + request_suffix,
                common_agent_scope_from_request(request),
                messages,
                options,
                json_schema));
    };
    const std::string required_schema = required_family_selection
        ? common_tool_family_required_selection_schema(families)
        : std::string{};
    auto generation = generate_selection(
        ":tool-family-selection", {system, user}, required_schema);
    if (!common_agent_generation_succeeded(generation)) {
        error = "tool family selection failed: " + generation.error_message;
        if (generation.error_message.empty()) error += " model did not return a completed selection";
        return false;
    }

    const bool parsed = required_family_selection
        ? common_parse_tool_family_required_selection(
            generation.content, families, selection, error)
        : common_parse_tool_family_selection_text(
            generation.content, families, selection, error);
    if (!parsed) {
        const char * trace = std::getenv("LLAMA_AGENT_RESIDENT_TRACE");
        if (trace && *trace && *trace != '0') {
            // Diagnose format mismatches without logging model/user content.
            const auto boxed = nlohmann::ordered_json::parse(generation.content, nullptr, false);
            const bool quoted_none = boxed.is_string() && boxed.get<std::string>() == "NO_TOOLS";
            const bool leading_blank = !generation.content.empty() &&
                (generation.content.front() == '\n' || generation.content.front() == '\r');
            std::fprintf(stderr, "agent family trace: event=selection-format-rejected bytes=%zu json_string=%s quoted_no_tools=%s leading_blank=%s\n",
                generation.content.size(), boxed.is_string() ? "yes" : "no", quoted_none ? "yes" : "no", leading_blank ? "yes" : "no");
            std::fflush(stderr);
        }
        // Keep the parser strict: a malformed first classifier response must
        // never grant a family that it did not name exactly.  Small models do
        // occasionally append prompt text after an otherwise valid `TOOLS:`
        // answer, though.  Give that one-line classifier one bounded repair
        // turn with a structured response instead of weakening the
        // authorization grammar itself.
        common_chat_msg repair{
            "system",
            required_family_selection
                ? common_tool_family_required_selection_repair_prompt()
                : "Your previous tool-family selection was invalid. Return the required JSON object only. "
                  "Families may contain only ids from the available list; do not include tool arguments, "
                  "resource ids, explanations, or any other text.",
        };
        const auto repaired = generate_selection(
            ":tool-family-selection-repair", {system, repair, user},
            required_family_selection
                ? common_tool_family_required_selection_schema(families)
                : common_tool_family_selection_schema(families));
        const bool repair_generation_succeeded = common_agent_generation_succeeded(repaired);
        const bool repaired_ok = repair_generation_succeeded &&
            (required_family_selection
                ? common_parse_tool_family_required_selection(
                    repaired.content, families, selection, error)
                : common_parse_tool_family_selection(
                    repaired.content, selection, error));
        if (repaired_ok) {
            // Repaired selection is subject to the same policy filtering as
            // the original response.
        } else {
            if (repair_generation_succeeded && invalid_selection) *invalid_selection = true;
            error = repair_generation_succeeded
                ? "tool family selection returned an invalid response after bounded repair"
                : "tool family selection repair failed: " + repaired.error_message;
            return false;
        }
    }
    std::vector<std::string> required_attachment_families;
    const bool bound_resource_read = request.resource_read_binding &&
        request.resource_read_binding->require_content_read;
    for (const auto & input : request.input_resources) {
        if (!bound_resource_read && input.role != "attachment_question" && input.role != "document_context") continue;
        required_attachment_families.push_back("resource");
        const auto & mime = input.resource.mime_type;
        if (mime == "text/csv" || mime == "text/tab-separated-values") {
            required_attachment_families.push_back("dataset");
        }
        if (mime == "text/html" ||
                mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
                mime == "application/epub+zip") {
            required_attachment_families.push_back("document");
        }
    }
    std::sort(required_attachment_families.begin(), required_attachment_families.end());
    required_attachment_families.erase(
        std::unique(required_attachment_families.begin(), required_attachment_families.end()),
        required_attachment_families.end());
    if (bound_resource_read) {
        for (const auto & family : families) {
            if ((family.id == "resource" || family.id == "dataset" || family.id == "document") &&
                    std::find(required_attachment_families.begin(), required_attachment_families.end(), family.id) ==
                        required_attachment_families.end()) {
                required_attachment_families.push_back(family.id);
            }
        }
        std::sort(required_attachment_families.begin(), required_attachment_families.end());
    }
    // A caller-bound attachment question must actually inspect its resource;
    // a model-selected web or unrelated tool alone is not grounded. Add the
    // host-required families after model classification so the model may also
    // select web/research or other capabilities requested by the user.
    for (const auto & required_family : required_attachment_families) {
        const bool family_available = std::any_of(
            families.begin(), families.end(), [&required_family](const auto & family) {
                return family.id == required_family;
            });
        if (!family_available) {
            error = "attached resource requires unavailable tool family: " + required_family;
            return false;
        }
        if (std::find(selection.family_ids.begin(), selection.family_ids.end(), required_family) ==
                selection.family_ids.end()) {
            selection.family_ids.push_back(required_family);
        }
        selection.needs_tools = true;
    }
    // Validate repaired responses against the same host-filtered family view.
    if ((!selection.needs_tools && !selection.family_ids.empty()) ||
            (required_family_selection && !selection.needs_tools)) {
        if (invalid_selection) *invalid_selection = true;
        error = "tool family selection declined required tools or has inconsistent families";
        return false;
    }
    for (const auto & id : selection.family_ids) {
        if (std::none_of(families.begin(), families.end(), [&](const auto & family) { return family.id == id; })) {
            if (invalid_selection) *invalid_selection = true;
            error = "tool family selection returned unknown family: " + id;
            return false;
        }
    }
    return true;
}

bool run_agent_chat_runtime(
        common_agent_chat_runtime_execution & execution,
        common_agent_result & result,
        std::string & error) {
    result = {};
    if (stop_chat_runtime_if_requested(execution, result, error)) {
        return false;
    }
    std::vector<common_chat_msg> messages = execution.request.messages;
    // Driver callers may provide a standalone prompt without constructing a
    // chat history (notably the tool-family singleton fast path). Preserve
    // that user turn in the ordinary chat/tool lifecycle so the model sees
    // the request and the follow-up retains a complete conversation.
    if (messages.empty() && !execution.request.prompt.empty()) {
        messages.push_back({"user", execution.request.prompt});
    }
    auto available_tools = execution.tooling.tools;
    bool require_tools = execution.request.require_tool_execution;
    if (execution.policy.enable_tool_family_routing && !available_tools.empty()) {
        const auto families = common_generate_tool_family_index(available_tools, execution.tooling.family_descriptions);
        common_tool_family_selection selection;
        bool invalid_family_selection = false;
        if (!common_agent_select_tool_families(execution.inference, execution.request,
                execution.generation_options, families,
                require_tools, selection, error, &invalid_family_selection)) {
            if (!invalid_family_selection) return false;
            available_tools = execution.tooling.tools;
            selection.needs_tools = true;
            require_tools = true;
            result.trace.push_back({common_runtime_trace_stage::plan,
                common_runtime_trace_kind::decided,
                "tool family selection outcome=invalid_fallback_normal_tool_route; using host-authorized catalog"});
            error.clear();
        } else {
            available_tools = selection.needs_tools
                ? common_expand_tool_family_prerequisites(available_tools, selection.family_ids)
                : std::vector<common_chat_tool>{};
            require_tools = require_tools || selection.needs_tools;
        }
        if (selection.needs_tools && available_tools.empty()) {
            error = "tool family selection did not resolve any registered tools";
            return false;
        }
        execution.request.require_tool_execution = require_tools;
    }
    const auto initial_tool_choice = available_tools.empty()
        ? COMMON_CHAT_TOOL_CHOICE_NONE
        : (require_tools
            ? COMMON_CHAT_TOOL_CHOICE_REQUIRED
            : COMMON_CHAT_TOOL_CHOICE_AUTO);
    if (!available_tools.empty() && is_textual_tool_output_format(execution.request.tool_output_format)) {
        std::vector<std::string> unsupported;
        std::string format_error;
        const auto instructions = common_render_model_tool_output_instructions(
            execution.request.tool_output_format,
            available_tools,
            &unsupported,
            format_error);
        if (instructions.empty() || !format_error.empty()) {
            error = format_error.empty()
                ? "model-facing tool output format has no representable tools"
                : "model-facing tool output contract is invalid: " + format_error;
            result.error = error;
            return false;
        }
    }
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
    if (stop_chat_runtime_if_requested(execution, result, error)) {
        return false;
    }
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
    if (!parse_assistant_message(
            generation_result,
            !available_tools.empty(),
            execution.request.tool_output_format,
            available_tools,
            assistant_message,
            error)) {
        return false;
    }

    if (assistant_message.tool_calls.empty() && execution.request.require_tool_execution) {
        error = "required tool execution did not produce a tool call";
        return false;
    }

    size_t tool_rounds = 0;
    while (!assistant_message.tool_calls.empty()) {
        if (stop_chat_runtime_if_requested(execution, result, error)) {
            return false;
        }
        if (tool_rounds >= execution.policy.max_tool_rounds) {
            error = "tool call round limit reached";
            result.limit_reached = true;
            return false;
        }
        if (!append_dispatched_tool_messages(execution, assistant_message, messages, error)) {
            return false;
        }

        if (stop_chat_runtime_if_requested(execution, result, error)) {
            return false;
        }

        ++tool_rounds;
        const bool allow_another_tool_round = tool_rounds < execution.policy.max_tool_rounds;
        const std::vector<common_chat_tool> next_tools = allow_another_tool_round ? available_tools : std::vector<common_chat_tool>{};
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
        if (!parse_assistant_message(
                generation_result,
                allow_another_tool_round && !available_tools.empty(),
                execution.request.tool_output_format,
                next_tools,
                assistant_message,
                error)) {
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
