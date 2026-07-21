#pragma once

#include <string>

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

struct common_agent_generated_text_result {
    std::string content;
    int decoded_tokens = 0;
    common_agent_generation_status status = common_agent_generation_status::errored;
    common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::error;
    std::string error_message;
};

enum class common_agent_generation_stage {
    planner,
    plan_selection,
    blueprint_selection,
    blueprint_binding,
    reasoning,
    draft,
    reflection,
    memory_learning,
};

inline const char * common_agent_generation_stage_name(common_agent_generation_stage stage) {
    switch (stage) {
        case common_agent_generation_stage::planner:             return "planner";
        case common_agent_generation_stage::plan_selection:      return "plan_selection";
        case common_agent_generation_stage::blueprint_selection: return "blueprint_selection";
        case common_agent_generation_stage::blueprint_binding:   return "blueprint_binding";
        case common_agent_generation_stage::reasoning:           return "reasoning";
        case common_agent_generation_stage::draft:               return "draft";
        case common_agent_generation_stage::reflection:          return "reflection";
        case common_agent_generation_stage::memory_learning:     return "memory_learning";
    }
    return "draft";
}

struct common_agent_generation_record {
    common_agent_generation_stage stage = common_agent_generation_stage::draft;
    int decoded_tokens = 0;
    common_agent_generation_status status = common_agent_generation_status::errored;
    common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::error;
    std::string error_message;
};

inline common_agent_generation_record common_agent_generation_record_from_result(
        common_agent_generation_stage stage,
        const common_agent_generated_text_result & result) {
    return {
        stage,
        result.decoded_tokens,
        result.status,
        result.stop_reason,
        result.error_message,
    };
}
