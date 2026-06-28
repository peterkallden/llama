#pragma once

#include "chat.h"
#include "common/cli-config.h"

#include <optional>
#include <string>
#include <vector>

enum class common_agent_generation_purpose {
    planner,
    plan_selection,
    blueprint_selection,
    blueprint_binding,
    draft,
    reasoning,
    reflection,
    memory_learning,
};

inline const char * common_agent_generation_purpose_name(common_agent_generation_purpose purpose) {
    switch (purpose) {
        case common_agent_generation_purpose::planner:             return "planner";
        case common_agent_generation_purpose::plan_selection:      return "plan_selection";
        case common_agent_generation_purpose::blueprint_selection: return "blueprint_selection";
        case common_agent_generation_purpose::blueprint_binding:   return "blueprint_binding";
        case common_agent_generation_purpose::draft:               return "draft";
        case common_agent_generation_purpose::reasoning:           return "reasoning";
        case common_agent_generation_purpose::reflection:          return "reflection";
        case common_agent_generation_purpose::memory_learning:     return "memory_learning";
    }
    return "draft";
}

enum class common_agent_generation_status {
    completed,
    cancelled,
    errored,
};

inline const char * common_agent_generation_status_name(common_agent_generation_status status) {
    switch (status) {
        case common_agent_generation_status::completed: return "completed";
        case common_agent_generation_status::cancelled: return "cancelled";
        case common_agent_generation_status::errored:   return "errored";
    }
    return "errored";
}

enum class common_agent_generation_stop_reason {
    none,
    eos,
    limit,
    json_schema,
    cancelled,
    error,
};

inline const char * common_agent_generation_stop_reason_name(common_agent_generation_stop_reason reason) {
    switch (reason) {
        case common_agent_generation_stop_reason::none:        return "none";
        case common_agent_generation_stop_reason::eos:         return "eos";
        case common_agent_generation_stop_reason::limit:       return "limit";
        case common_agent_generation_stop_reason::json_schema: return "json_schema";
        case common_agent_generation_stop_reason::cancelled:   return "cancelled";
        case common_agent_generation_stop_reason::error:       return "error";
    }
    return "error";
}

struct common_agent_generation_request {
    common_agent_generation_purpose purpose = common_agent_generation_purpose::draft;
    std::optional<std::string> trace_id;
    std::vector<common_chat_msg> messages;
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    args options;
    std::string json_schema;
};

struct common_agent_generation_result {
    std::string content;
    common_chat_params chat_params;
    int decoded_tokens = 0;
    common_agent_generation_status status = common_agent_generation_status::errored;
    common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::error;
    std::string error_message;
};

class common_agent_inference {
public:
    virtual ~common_agent_inference() = default;
    virtual bool generate(
        const common_agent_generation_request & request,
        common_agent_generation_result & result) = 0;
};
