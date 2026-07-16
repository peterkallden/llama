#pragma once

#include <chrono>
#include <string>

enum class common_runtime_operation_kind {
    inference,
    tool,
};

inline const char * common_runtime_operation_kind_name(
        common_runtime_operation_kind kind) {
    switch (kind) {
        case common_runtime_operation_kind::inference: return "inference";
        case common_runtime_operation_kind::tool:      return "tool";
    }
    return "tool";
}

struct common_runtime_operation {
    std::string operation_id;
    common_runtime_operation_kind kind = common_runtime_operation_kind::tool;
    std::string detail;
    std::chrono::steady_clock::time_point deadline{};
};

struct common_runtime_operation_ref {
    std::string operation_id;
    common_runtime_operation_kind kind = common_runtime_operation_kind::tool;
    std::string subject_name;
};
