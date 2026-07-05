#pragma once

#include <cstddef>
#include <string>

enum class common_runtime_resource_scope {
    turn,
    session,
    project,
};

inline const char * common_runtime_resource_scope_name(common_runtime_resource_scope scope) {
    switch (scope) {
        case common_runtime_resource_scope::turn:    return "turn";
        case common_runtime_resource_scope::session: return "session";
        case common_runtime_resource_scope::project: return "project";
    }
    return "turn";
}

struct common_runtime_resource_ref {
    std::string uri;
    std::string name;
    std::string description;
    std::string mime_type;
    size_t size_bytes = 0;
    common_runtime_resource_scope scope = common_runtime_resource_scope::turn;
};
