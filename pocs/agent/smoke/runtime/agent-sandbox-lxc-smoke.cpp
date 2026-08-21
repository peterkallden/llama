#include "agent/sandbox/sandbox-lxc-runtime.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>

int main() {
    const char * enabled = std::getenv("LLAMA_AGENT_LXC_SMOKE");
    if (enabled == nullptr || *enabled != '1') {
        std::printf("lxc_sandbox_smoke=skipped\n");
        return 77;
    }
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-lxc-smoke";
    std::filesystem::create_directories(root / "source");
    std::filesystem::create_directories(root / "writable");
    std::filesystem::create_directories(root / "artifacts");
    const char * executable = std::getenv("LLAMA_AGENT_LXC_EXECUTABLE");
    const char * image = std::getenv("LLAMA_AGENT_LXC_IMAGE");
    common_agent_sandbox_lxc_runtime runtime({
        executable != nullptr && *executable != '\0' ? executable : "lxc",
        image != nullptr && *image != '\0' ? image : "ubuntu:24.04",
        "none", {}, true,
    });
    common_agent_sandbox_request request;
    request.operation_id = "lxc-smoke-1";
    request.execution_class = "readonly-analysis";
    request.command.program = "sh";
    request.command.arguments = {"-c", "printf lxc-ok > /workspace/artifacts/result.txt; printf lxc-ok"};
    request.command.working_directory = "/workspace/source";
    request.workspace.source_path = (root / "source").string();
    request.workspace.writable_path = (root / "writable").string();
    request.workspace.artifact_path = (root / "artifacts").string();
    request.limits.timeout_ms = 120000;
    request.limits.cpu_count = 1;
    request.limits.max_output_bytes = 4096;
    request.network = common_agent_sandbox_network_scope::none;
    request.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
    common_agent_sandbox_result result;
    std::string error;
    if (!runtime.execute(request, result, error) ||
            result.status != common_agent_sandbox_status::completed ||
            result.stdout_excerpt.find("lxc-ok") == std::string::npos || result.artifacts.empty()) {
        const std::string diagnostic = error + "\n" + result.error;
        if (diagnostic.find("could not be started") != std::string::npos ||
                diagnostic.find("permission") != std::string::npos ||
                diagnostic.find("not found") != std::string::npos ||
                diagnostic.find("LXC command failed") != std::string::npos) {
            std::printf("lxc_sandbox_smoke=skipped\n");
            return 77;
        }
        std::fprintf(stderr, "LXC sandbox smoke failed: %s\n",
            (error.empty() ? result.error : error).c_str());
        return 1;
    }
    std::printf("sandbox_status=%s\n", common_agent_sandbox_status_name(result.status));
    std::printf("sandbox_backend=%s\n", result.backend_execution_id.c_str());
    std::printf("sandbox_artifacts=%zu\n", result.artifacts.size());
    return 0;
}
