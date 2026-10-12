#include "agent-native-crash.h"

#include <cassert>
#include <string>

int main() {
    const auto gdb = agent_gdb_mi_argv("build/llama-agent", "artifacts/core.123");
    assert(gdb.size() == 15);
    assert(gdb[0] == "gdb");
    assert(gdb[6] == "set pagination off");
    assert(gdb[10] == "interpreter-exec mi \"-stack-list-frames\"");
    assert(gdb[13] == "build/llama-agent");
    assert(gdb[14] == "artifacts/core.123");

    const auto cdb = agent_cdb_argv("llama-agent.exe", "artifacts/crash.dmp");
    assert(cdb.size() == 6);
    assert(cdb[0] == "cdb.exe");
    assert(cdb[2] == "artifacts/crash.dmp");
    assert(cdb[3] == "-c");
    assert(cdb[4] == "!analyze -v; kv; q");
    assert(cdb[5] == "llama-agent.exe");

    const auto gdb_result = agent_parse_gdb_mi(
        R"(^done,signal-name="SIGSEGV",signal-meaning="Segmentation fault",stack=[frame={level="0",addr="0x1234",func="crash_here",file="crash.cpp",line="42"},frame={level="1",addr="0x5678",func="main",file="main.cpp",line="17"}])");
    assert(gdb_result.status == agent_native_crash_status::analyzed);
    assert(gdb_result.signal_or_exception == "SIGSEGV");
    assert(gdb_result.faulting_function == "crash_here");
    assert(gdb_result.symbol_quality == "full");
    assert(gdb_result.stack.size() == 2);

    const auto cdb_result = agent_parse_cdb_output(
        "ExceptionCode:    c0000005 (Access violation)\n"
        "MODULE_NAME: llama_agent\n"
        "FAULTING_IP:\n"
        "llama_agent!crash_here+0x23\n"
        "FAILURE_BUCKET_ID: NULL_POINTER_READ_c0000005\n"
        "STACK_TEXT:\n"
        "00000000 00000000 00000000 llama_agent!crash_here+0x23\n"
        "00000000 00000000 00000000 llama_agent!main+0x10\n");
    assert(cdb_result.status == agent_native_crash_status::analyzed);
    assert(cdb_result.signal_or_exception == "c0000005 (Access violation)");
    assert(cdb_result.faulting_module == "llama_agent");
    assert(cdb_result.faulting_function == "crash_here+0x23");
    assert(cdb_result.fault_address == "llama_agent!crash_here+0x23");
    assert(cdb_result.failure_bucket == "NULL_POINTER_READ_c0000005");
    assert(cdb_result.stack.size() == 2);

    const auto malformed = agent_parse_gdb_mi("^done,thread-id=\"1\"");
    assert(malformed.status == agent_native_crash_status::parse_failed);
    return 0;
}
