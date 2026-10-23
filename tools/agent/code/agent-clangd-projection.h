#pragma once

#include "agent-clangd-protocol.h"

#include <filesystem>

// LSP wire data is normalized here before entering the code tool contract.
// Only local file locations within the host repository are exposed.
bool agent_clangd_repository_path(const std::string & repository_root, const std::filesystem::path & input,
    std::filesystem::path & output, std::string & error);
bool agent_clangd_project_location(const agent_clangd_json & input,
    const std::string & repository_root, agent_clangd_json & output, std::string & error);
bool agent_clangd_project_call(const agent_clangd_json & input,
    const agent_clangd_json & selected, bool incoming, const std::string & repository_root,
    agent_clangd_json & output, std::string & error);
