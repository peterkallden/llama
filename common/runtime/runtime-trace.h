#pragma once

#include <string>

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
};
