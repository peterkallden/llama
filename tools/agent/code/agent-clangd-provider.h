#pragma once

#include "agent-clangd-session.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

class agent_clangd_code_provider {
public:
    explicit agent_clangd_code_provider(agent_clangd_session_config config);
    ~agent_clangd_code_provider();

    common_tool_execution_result symbol(const std::string & arguments_json);
    common_tool_execution_result references(const std::string & arguments_json);
    common_tool_execution_result call_hierarchy(const std::string & arguments_json);
    agent_clangd_json status_snapshot();
    common_tool_execution_result refresh();

private:
    struct source_location {
        std::filesystem::path path;
        std::string uri;
        std::string text;
        int line = 0;   // Zero-based UTF-16 LSP position in text.
        int column = 0;
    };
    common_tool_execution_result workspace_symbol(const agent_clangd_json & arguments);
    bool start_project_index(std::string & error);
    bool ensure_watcher(std::string & error);
    bool on_file_changes(const std::vector<agent_project_file_change> & changes);
    bool update_compile_commands(bool & changed, std::string & error);
    bool synchronize_compile_commands(std::string & error);
    bool wait_for_changes(std::unique_lock<std::mutex> & lock);
    bool resolve_location(const agent_clangd_json & arguments, source_location & location, std::string & error);
    bool open_location(const source_location & location, std::string & error);
    common_tool_execution_result location_query(const agent_clangd_json & arguments, const std::string & method);

    // Serialize open/update/query sequences, not only individual LSP messages.
    mutable std::mutex mutex_;
    std::unique_ptr<agent_clangd_session> session_;
    std::shared_ptr<agent_project_file_watch_service> file_watch_service_;
    uint64_t watch_subscription_id_ = 0;
    std::string repository_root_;
    std::string compile_commands_dir_;
    std::string compile_commands_source_;
    std::string compile_commands_destination_;
    std::string watcher_error_;
    std::string compile_commands_error_;
    std::atomic<uint64_t> latest_change_generation_{0};
    std::atomic<uint64_t> delivered_change_generation_{0};
    std::atomic<bool> watcher_reconciliation_required_{false};
    std::atomic<bool> compile_commands_may_be_stale_{false};
    std::atomic<bool> project_index_started_{false};
};
