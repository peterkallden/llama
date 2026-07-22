#pragma once

#include <string>

struct common_agent_data_store_config {
    std::string backend = "auto";
    std::string path;
};

// Host-owned backend seam for structured data tools. Tool arguments remain
// semantic JSON; the selected backend owns storage and query translation.
class common_agent_data_store {
public:
    virtual ~common_agent_data_store() = default;

    virtual bool execute(
        const std::string & operation,
        const std::string & request_json,
        std::string & result_json,
        std::string & error) = 0;
};
