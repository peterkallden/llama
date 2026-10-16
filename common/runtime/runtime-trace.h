#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

inline std::string common_runtime_trace_safe_url(const std::string & url) {
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) return url.empty() ? std::string{} : "[redacted-url]";
    const auto authority_begin = scheme + 3;
    const auto authority_end = url.find_first_of("/?#", authority_begin);
    const auto path_begin = authority_end != std::string::npos && url[authority_end] == '/'
        ? authority_end : std::string::npos;
    auto authority = url.substr(authority_begin,
        authority_end == std::string::npos ? std::string::npos : authority_end - authority_begin);
    const auto user = authority.rfind('@');
    if (user != std::string::npos) authority.erase(0, user + 1);
    auto path = path_begin == std::string::npos ? std::string("/") : url.substr(path_begin);
    const auto suffix = path.find_first_of("?#");
    if (suffix != std::string::npos) path.resize(suffix);
    return url.substr(0, scheme + 3) + authority + path;
}

enum class common_runtime_trace_stage {
    turn,
    plan,
    step,
    tool,
    observation,
    reflection,
    research,
    memory_learning,
    response,
};

inline const char * common_runtime_trace_stage_name(common_runtime_trace_stage stage) {
    switch (stage) {
        case common_runtime_trace_stage::turn:            return "turn";
        case common_runtime_trace_stage::plan:            return "plan";
        case common_runtime_trace_stage::step:            return "step";
        case common_runtime_trace_stage::tool:            return "tool";
        case common_runtime_trace_stage::observation:     return "observation";
        case common_runtime_trace_stage::reflection:      return "reflection";
        case common_runtime_trace_stage::research:        return "research";
        case common_runtime_trace_stage::memory_learning: return "memory_learning";
        case common_runtime_trace_stage::response:        return "response";
    }
    return "turn";
}

enum class common_runtime_trace_kind {
    started,
    completed,
    updated,
    recorded,
    succeeded,
    failed,
    decided,
    skipped,
    summary,
};

inline const char * common_runtime_trace_kind_name(common_runtime_trace_kind kind) {
    switch (kind) {
        case common_runtime_trace_kind::started:   return "started";
        case common_runtime_trace_kind::completed: return "completed";
        case common_runtime_trace_kind::updated:   return "updated";
        case common_runtime_trace_kind::recorded:  return "recorded";
        case common_runtime_trace_kind::succeeded: return "succeeded";
        case common_runtime_trace_kind::failed:    return "failed";
        case common_runtime_trace_kind::decided:   return "decided";
        case common_runtime_trace_kind::skipped:   return "skipped";
        case common_runtime_trace_kind::summary:   return "summary";
    }
    return "summary";
}

struct common_runtime_trace_entry {
    common_runtime_trace_stage stage = common_runtime_trace_stage::turn;
    common_runtime_trace_kind kind = common_runtime_trace_kind::summary;
    std::string detail;
    std::string plan_id;
    std::string step_id;
    std::string tool_name;
    std::string observation_id;
    std::string related_id;
    std::string http_provider;
    std::string http_operation;
    std::string http_method;
    std::string http_url;
    std::string http_final_url;
    std::string http_resource_ref;
    int http_status = 0;
    std::string http_content_type;
    size_t http_response_bytes = 0;
    int64_t http_duration_ms = 0;
    int http_retry_after_seconds = -1;
    // Bounded MCP App invocation context, present only for app-visible tools.
    // Oversized JSON values are omitted rather than truncated.
    bool app_context_available = false;
    std::string app_provider_id;
    std::string app_server_tool_name;
    std::string app_tool_input_json;
    std::string app_tool_result_json;
};
