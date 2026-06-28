#pragma once

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
