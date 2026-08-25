#pragma once

#include "agent/agent-generation.h"
#include "agent/agent-scope.h"
#include "chat.h"
#include "resource/resource-contract.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

enum class common_agent_generation_purpose {
    planner,
    tool_family_selection,
    plan_selection,
    blueprint_selection,
    blueprint_binding,
    conversation,
    tool_followup,
    draft,
    reasoning,
    reflection,
    memory_learning,
};

inline const char * common_agent_generation_purpose_name(common_agent_generation_purpose purpose) {
    switch (purpose) {
        case common_agent_generation_purpose::planner:             return "planner";
        case common_agent_generation_purpose::tool_family_selection: return "tool_family_selection";
        case common_agent_generation_purpose::plan_selection:      return "plan_selection";
        case common_agent_generation_purpose::blueprint_selection: return "blueprint_selection";
        case common_agent_generation_purpose::blueprint_binding:   return "blueprint_binding";
        case common_agent_generation_purpose::conversation:        return "conversation";
        case common_agent_generation_purpose::tool_followup:       return "tool_followup";
        case common_agent_generation_purpose::draft:               return "draft";
        case common_agent_generation_purpose::reasoning:           return "reasoning";
        case common_agent_generation_purpose::reflection:          return "reflection";
        case common_agent_generation_purpose::memory_learning:     return "memory_learning";
    }
    return "draft";
}

struct common_agent_generation_options {
    int n_predict = 0;
    // Zero means that the host/runtime must supply the configured thread
    // count.  Keeping this unset prevents a library default from silently
    // forcing model-backed agent turns onto a single CPU thread.
    int n_threads = 0;
    // Emit bounded progress diagnostics for model generation. This never
    // includes the generated content itself.
    bool generation_trace = false;
    std::optional<int64_t> t_max_prompt_ms;
    std::optional<int64_t> t_max_predict_ms;
};

// Resource identity crossing the orchestration/inference boundary. Bytes and
// derived representations remain host-owned; inference receives references
// until a backend explicitly resolves them.
struct common_agent_generation_resource {
    common_runtime_resource_ref resource;
    std::string role = "reference";
    bool required = false;
    std::function<bool(size_t max_bytes, std::string & out, std::string & error)> read_bytes;
    // Optional host-owned fallback for image text extraction when native
    // multimodal inference is unavailable.
    std::function<bool(size_t max_bytes, std::string & out, std::string & error)> read_text_fallback;
};

inline common_agent_generation_options common_agent_generation_options_with_n_predict(
        common_agent_generation_options options,
        int n_predict) {
    options.n_predict = n_predict;
    return options;
}

struct common_agent_generation_request {
    common_agent_generation_purpose purpose = common_agent_generation_purpose::draft;
    std::optional<std::string> trace_id;
    std::optional<common_agent_scope> scope;
    std::vector<common_chat_msg> messages;
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    common_agent_generation_options options;
    std::string json_schema;
    std::vector<common_agent_generation_resource> input_resources;
};

inline common_agent_generation_request common_agent_make_generation_request(
        common_agent_generation_purpose purpose,
        std::optional<std::string> trace_id,
        std::optional<common_agent_scope> scope,
        std::vector<common_chat_msg> messages,
        common_agent_generation_options options,
        std::string json_schema = {},
        std::vector<common_chat_tool> tools = {},
        common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE) {
    common_agent_generation_request request;
    request.purpose = purpose;
    request.trace_id = std::move(trace_id);
    request.scope = std::move(scope);
    request.messages = std::move(messages);
    request.tools = std::move(tools);
    request.tool_choice = tool_choice;
    request.options = std::move(options);
    request.json_schema = std::move(json_schema);
    return request;
}

struct common_agent_generation_result {
    std::string content;
    int decoded_tokens = 0;
    common_agent_generation_status status = common_agent_generation_status::errored;
    common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::error;
    std::string error_message;
    std::optional<common_chat_params> chat_params;
};

inline bool common_agent_generation_succeeded(const common_agent_generation_result & result) {
    return result.status == common_agent_generation_status::completed;
}

inline common_agent_generated_text_result common_agent_generated_text_result_from_generation_result(
        const common_agent_generation_result & result) {
    return {
        result.content,
        result.decoded_tokens,
        result.status,
        result.stop_reason,
        result.error_message,
    };
}

class common_agent_inference {
public:
    virtual ~common_agent_inference() = default;
    virtual bool generate(
        const common_agent_generation_request & request,
        common_agent_generation_result & result) = 0;
    common_agent_generation_result generate_result(const common_agent_generation_request & request) {
        common_agent_generation_result result;
        generate(request, result);
        return result;
    }
};
