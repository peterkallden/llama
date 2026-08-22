#pragma once

#include <string>

struct agent_host_diagnostics_config {
    std::string semantic_backend = "auto";
    std::string clang_executable = "clang";
    std::string clangd_executable = "clangd";
    std::string compile_commands = "auto";
    std::string native_crash_backend = "auto";
    std::string gdb_executable = "gdb";
    std::string cdb_executable = "cdb.exe";
    uint32_t native_crash_timeout_ms = 30000;
    uint32_t native_crash_max_frames = 64;
    uint32_t native_crash_max_threads = 16;
    uint32_t native_crash_max_output_bytes = 262144;
};
