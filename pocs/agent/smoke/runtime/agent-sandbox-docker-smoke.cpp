#include "agent/sandbox-docker-runtime.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

int main() {
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-docker-smoke";
    std::filesystem::create_directories(root / "source");
    std::filesystem::create_directories(root / "writable");
    std::filesystem::create_directories(root / "artifacts");

    const char * configured_executable = std::getenv("LLAMA_AGENT_SANDBOX_EXECUTABLE");
    const std::string executable = configured_executable != nullptr && *configured_executable != '\0'
        ? configured_executable
        : "docker";
    common_agent_sandbox_docker_runtime runtime({executable, "alpine:3.20"});
    common_agent_sandbox_request request;
    request.operation_id = "docker-smoke-1";
    request.execution_class = "readonly-analysis";
    request.command.program = "sh";
    request.command.arguments = {"-c", "printf docker-ok > /workspace/artifacts/result.txt; printf docker-ok"};
    request.workspace.source_path = (root / "source").string();
    request.workspace.writable_path = (root / "writable").string();
    request.workspace.artifact_path = (root / "artifacts").string();
    request.limits.timeout_ms = 30000;
    request.limits.cpu_count = 1;
    request.limits.max_output_bytes = 4096;
    request.network = common_agent_sandbox_network_scope::none;
    request.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;

    common_agent_sandbox_result result;
    std::string error;
    if (!runtime.execute(request, result, error) ||
            result.status != common_agent_sandbox_status::completed ||
            result.exit_code != 0 || result.stdout_excerpt.find("docker-ok") == std::string::npos ||
            result.artifacts.empty()) {
        const std::string diagnostic = error + "\n" + result.error;
        if (diagnostic == "unable to start Docker process" ||
                diagnostic.find("permission denied while trying to connect") != std::string::npos ||
                diagnostic.find("Cannot connect to the Docker daemon") != std::string::npos ||
                diagnostic.find("Access is denied") != std::string::npos ||
                diagnostic.find("access is denied") != std::string::npos) {
            std::printf("docker_sandbox_smoke=skipped\n");
            return 77;
        }
        std::fprintf(stderr, "Docker sandbox smoke failed: %s\n", (error.empty() ? result.error : error).c_str());
        return 1;
    }

    std::printf("sandbox_status=%s\n", common_agent_sandbox_status_name(result.status));
    std::printf("sandbox_backend=%s\n", result.backend_execution_id.c_str());
    std::printf("sandbox_artifacts=%zu\n", result.artifacts.size());
    return 0;
}
