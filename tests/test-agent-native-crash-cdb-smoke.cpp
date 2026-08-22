#include "agent-native-crash.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <string>

#include "nlohmann/json.hpp"

int main() {
#ifndef _WIN32
    return 77;
#else
    const char * root = std::getenv("LLAMA_AGENT_NATIVE_CRASH_SMOKE_ROOT");
    const char * executable = std::getenv("LLAMA_AGENT_NATIVE_CRASH_SMOKE_EXECUTABLE");
    const char * dump = std::getenv("LLAMA_AGENT_NATIVE_CRASH_SMOKE_DUMP");
    const char * cdb = std::getenv("LLAMA_AGENT_CDB_EXECUTABLE");
    if (root == nullptr || executable == nullptr || dump == nullptr ||
            std::string(root).empty() || std::string(executable).empty() || std::string(dump).empty()) return 77;

    const auto result = agent_execute_native_crash(
        nlohmann::json{{"executable", executable}, {"dump", dump}}.dump(),
        "cdb", "gdb", cdb == nullptr || std::string(cdb).empty() ? "cdb.exe" : cdb, root);
    if (!result.ok) {
        std::fprintf(stderr, "native CDB smoke failed: %s\n%s\n",
                result.failure_code.c_str(), result.raw_diagnostic.c_str());
    }
    assert(result.ok);
    assert(result.output.find("native_crash") != std::string::npos);
    assert(result.output.find("analyzed") != std::string::npos);
    return 0;
#endif
}
