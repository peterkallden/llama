#include "agent/sandbox/sandbox-local-runtime.h"

#include <cassert>
#include <filesystem>

int main() {
    common_agent_sandbox_local_runtime runtime;
    common_agent_sandbox_request request;
    request.operation_id = "local-runtime-smoke";
#if defined(_WIN32)
    request.command.program = "cmd.exe";
    request.command.arguments = {"/c", "echo", "local-runtime-ok"};
#else
    request.command.program = "sh";
    request.command.arguments = {"-c", "echo local-runtime-ok"};
#endif
    request.command.working_directory = "/workspace/source";
    request.workspace.source_path = std::filesystem::temp_directory_path().string();
    request.limits.timeout_ms = 5000;
    request.limits.max_output_bytes = 4096;

    common_agent_sandbox_result result;
    std::string error;
    assert(runtime.execute(request, result, error));
    assert(error.empty());
    assert(result.status == common_agent_sandbox_status::completed);
    assert(result.stdout_excerpt.find("local-runtime-ok") != std::string::npos);
    assert(result.backend_execution_id == "local/local-runtime-smoke");
    return 0;
}
