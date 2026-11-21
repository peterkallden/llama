#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "agent/tooling/contracts/tool-runtime-contract.h"

enum class agent_native_crash_status {
    analyzed,
    unavailable,
    invalid_input,
    invalid_dump,
    timeout,
    process_failed,
    parse_failed,
};

struct agent_native_crash_frame {
    size_t index = 0;
    std::string address;
    std::string function;
    std::string module;
    std::string file;
    uint32_t line = 0;
};

struct agent_native_crash_result {
    agent_native_crash_status status = agent_native_crash_status::parse_failed;
    std::string backend;
    std::string signal_or_exception;
    std::string signal_description;
    std::string fault_address;
    std::string faulting_function;
    std::string faulting_module;
    std::string failure_bucket;
    std::string symbol_quality = "none";
    std::vector<agent_native_crash_frame> stack;
    std::vector<std::vector<agent_native_crash_frame>> thread_stacks;
    std::string summary;
    std::string raw_output;
    std::string error;
};

struct agent_native_crash_limits {
    uint32_t timeout_ms = 30000;
    size_t max_frames = 64;
    size_t max_threads = 16;
    size_t max_output_bytes = 262144;
};

// Arguments are returned as individual argv entries. Callers must pass them
// to a process API directly; they must never be joined into a shell command.
std::vector<std::string> agent_gdb_mi_argv(
        const std::string & executable,
        const std::string & core_file);

std::vector<std::string> agent_cdb_argv(
        const std::string & executable,
        const std::string & dump_file);

agent_native_crash_result agent_parse_gdb_mi(
        const std::string & output,
        const agent_native_crash_limits & limits = {});

agent_native_crash_result agent_parse_cdb_output(
        const std::string & output,
        const agent_native_crash_limits & limits = {});

const char * agent_native_crash_status_name(agent_native_crash_status status);

common_tool_execution_result agent_execute_native_crash(
        const std::string & arguments_json,
        const std::string & backend,
        const std::string & gdb_executable,
        const std::string & cdb_executable,
        const std::string & repository_root,
        const agent_native_crash_limits & limits = {});
