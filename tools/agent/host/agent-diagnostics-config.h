#pragma once

#include <string>

struct agent_host_diagnostics_config {
    std::string semantic_backend = "auto";
    std::string clang_executable = "clang";
    std::string clangd_executable = "clangd";
    std::string compile_commands = "auto";
};
