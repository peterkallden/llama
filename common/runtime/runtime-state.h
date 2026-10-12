#pragma once

#include <string>

enum class common_agent_state_class {
    durable_domain,
    resident_runtime,
    turn_workspace,
    event_projection,
};

enum class common_agent_state_lifetime {
    operation,
    turn,
    session,
    project,
    process,
    durable,
};

enum class common_agent_state_persistence {
    none,
    checkpointable,
    persistent,
};

struct common_agent_state_identity {
    std::string namespace_id;
    std::string project_id;
    std::string session_id;
    std::string turn_id;
    std::string operation_id;
};

struct common_agent_state_descriptor {
    std::string state_id;
    std::string state_type;
    common_agent_state_class state_class = common_agent_state_class::resident_runtime;
    common_agent_state_lifetime lifetime = common_agent_state_lifetime::process;
    common_agent_state_persistence persistence = common_agent_state_persistence::none;
    common_agent_state_identity identity;
    std::string owner;
    std::string source_of_truth;
};

inline const char * common_agent_state_class_name(common_agent_state_class value) {
    switch (value) {
        case common_agent_state_class::durable_domain: return "durable_domain";
        case common_agent_state_class::resident_runtime: return "resident_runtime";
        case common_agent_state_class::turn_workspace: return "turn_workspace";
        case common_agent_state_class::event_projection: return "event_projection";
    }
    return "resident_runtime";
}

inline const char * common_agent_state_lifetime_name(common_agent_state_lifetime value) {
    switch (value) {
        case common_agent_state_lifetime::operation: return "operation";
        case common_agent_state_lifetime::turn: return "turn";
        case common_agent_state_lifetime::session: return "session";
        case common_agent_state_lifetime::project: return "project";
        case common_agent_state_lifetime::process: return "process";
        case common_agent_state_lifetime::durable: return "durable";
    }
    return "process";
}

inline const char * common_agent_state_persistence_name(common_agent_state_persistence value) {
    switch (value) {
        case common_agent_state_persistence::none: return "none";
        case common_agent_state_persistence::checkpointable: return "checkpointable";
        case common_agent_state_persistence::persistent: return "persistent";
    }
    return "none";
}
