#pragma once

#include "agent-clangd-session.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"

#include <filesystem>
#include <memory>
#include <mutex>

class agent_clangd_code_provider {
public:
    explicit agent_clangd_code_provider(agent_clangd_session_config config);

    common_tool_execution_result symbol(const std::string & arguments_json);
    common_tool_execution_result references(const std::string & arguments_json);
    common_tool_execution_result call_hierarchy(const std::string & arguments_json);

private:
    struct source_location {
        std::filesystem::path path;
        std::string uri;
        std::string text;
        int line = 0;   // Zero-based UTF-16 LSP position in text.
        int column = 0;
    };
    common_tool_execution_result workspace_symbol(const agent_clangd_json & arguments);
    void start_project_index();
    bool resolve_location(const agent_clangd_json & arguments, source_location & location, std::string & error);
    bool open_location(const source_location & location, std::string & error);
    common_tool_execution_result location_query(const agent_clangd_json & arguments, const std::string & method);

    // Serialize open/update/query sequences, not only individual LSP messages.
    std::mutex mutex_;
    std::unique_ptr<agent_clangd_session> session_;
    std::string repository_root_;
    std::string compile_commands_dir_;
    bool project_index_started_ = false;
};
