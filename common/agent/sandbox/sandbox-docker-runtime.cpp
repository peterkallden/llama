#include "sandbox-docker-runtime.h"

#include "resource/resource-contract.h"

#include <sheredom/subprocess.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

void append_limited(std::string & output, const char * data, size_t size, size_t limit) {
    if (output.size() >= limit) return;
    output.append(data, std::min(size, limit - output.size()));
}

bool add_mount(
        std::vector<std::string> & argv,
        const std::string & source,
        const char * target,
        bool readonly,
        std::string & error) {
    if (source.empty()) return true;
    std::error_code fs_error;
    if (!fs::is_directory(source, fs_error) || fs_error) {
        error = "sandbox mount source is not an accessible directory: " + source;
        return false;
    }
    argv.emplace_back("--mount");
    argv.emplace_back(
        "type=bind,source=" + source + ",target=" + target +
        (readonly ? ",readonly" : ""));
    return true;
}

bool valid_working_directory(const std::string & path) {
    return path.empty() || path == "/workspace/source" ||
        path == "/workspace/writable" || path == "/workspace/artifacts";
}

void collect_artifacts(
        const common_agent_sandbox_request & request,
        common_agent_sandbox_result & result) {
    if (!request.artifacts.collect || request.workspace.artifact_path.empty()) return;

    std::error_code fs_error;
    const fs::path root(request.workspace.artifact_path);
    if (!fs::is_directory(root, fs_error) || fs_error) return;

    size_t collected = 0;
    for (const auto & entry : fs::recursive_directory_iterator(root, fs_error)) {
        if (fs_error || collected >= request.artifacts.max_bytes) break;
        if (!entry.is_regular_file(fs_error) || fs_error) continue;
        const auto size = static_cast<size_t>(entry.file_size(fs_error));
        if (fs_error || collected + size > request.artifacts.max_bytes) continue;
        const auto relative = fs::relative(entry.path(), root, fs_error);
        if (fs_error) continue;
        common_runtime_resource_ref artifact;
        artifact.uri = "artifact://" + request.operation_id + "/" + relative.generic_string();
        artifact.name = relative.generic_string();
        artifact.size_bytes = size;
        artifact.scope = common_runtime_resource_scope::turn;
        result.artifacts.push_back(std::move(artifact));
        collected += size;
    }
}

std::string network_name(common_agent_sandbox_network_scope network) {
    return network == common_agent_sandbox_network_scope::none ? "none" : "";
}

} // namespace

bool common_agent_sandbox_docker_runtime::execute(
        const common_agent_sandbox_request & request,
        common_agent_sandbox_result & result,
        std::string & error) {
    result = {};
    result.backend_execution_id = "docker/" + request.operation_id;

    if (request.operation_id.empty() || request.command.program.empty()) {
        error = "Docker sandbox requires an operation id and program";
        result.error = error;
        return false;
    }
    if (request.network != common_agent_sandbox_network_scope::none) {
        error = "Docker sandbox currently supports only network=none";
        result.error = error;
        return false;
    }
    const std::string image = request.image.empty() ? config.default_image : request.image;
    if (image.empty()) {
        error = "Docker sandbox requires an image or a configured default image";
        result.error = error;
        return false;
    }
    if (!valid_working_directory(request.command.working_directory)) {
        error = "Docker sandbox working directory must use a sandbox workspace path";
        result.error = error;
        return false;
    }

    std::vector<std::string> args = {
        config.executable.empty() ? "docker" : config.executable,
        "run", "--rm", "--init", "--network", network_name(request.network),
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--cpus", std::to_string(request.limits.cpu_count),
    };
    if (request.limits.memory_bytes != 0) {
        args.push_back("--memory");
        args.push_back(std::to_string(request.limits.memory_bytes));
    }
    if (request.limits.process_count != 0) {
        args.push_back("--pids-limit");
        args.push_back(std::to_string(request.limits.process_count));
    }

    if (!add_mount(args, request.workspace.source_path, "/workspace/source", true, error)) {
        result.error = error;
        return false;
    }
    const bool writable = request.filesystem == common_agent_sandbox_filesystem_scope::workspace_write;
    if (!add_mount(args, request.workspace.writable_path, "/workspace/writable", !writable, error)) {
        result.error = error;
        return false;
    }
    const bool artifacts_writable = request.filesystem == common_agent_sandbox_filesystem_scope::artifact_write || writable;
    if (!add_mount(args, request.workspace.artifact_path, "/workspace/artifacts", !artifacts_writable, error)) {
        result.error = error;
        return false;
    }

    args.emplace_back("--workdir");
    args.emplace_back(request.command.working_directory.empty() ? "/workspace/source" : request.command.working_directory);
    args.push_back(image);
    args.push_back(request.command.program);
    args.insert(args.end(), request.command.arguments.begin(), request.command.arguments.end());

    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (auto & arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);

    subprocess_s process{};
    const int options = subprocess_option_combined_stdout_stderr |
        subprocess_option_enable_async | subprocess_option_no_window |
        subprocess_option_inherit_environment | subprocess_option_search_user_path;
    if (subprocess_create(argv.data(), options, &process) != 0) {
        error = "unable to start Docker process";
        result.error = error;
        return false;
    }

    const auto started = std::chrono::steady_clock::now();
    std::string output;
    bool timed_out = false;
    char buffer[4096];
    while (subprocess_alive(&process)) {
        const unsigned count = subprocess_read_stdout(&process, buffer, sizeof(buffer));
        append_limited(output, buffer, count, request.limits.max_output_bytes);
        if (std::chrono::steady_clock::now() - started >=
                std::chrono::milliseconds(request.limits.timeout_ms)) {
            timed_out = true;
            subprocess_terminate(&process);
            break;
        }
        if (count == 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    while (const unsigned count = subprocess_read_stdout(&process, buffer, sizeof(buffer))) {
        append_limited(output, buffer, count, request.limits.max_output_bytes);
    }

    int exit_code = -1;
    subprocess_join(&process, &exit_code);
    subprocess_destroy(&process);
    result.exit_code = exit_code;
    result.stdout_excerpt = output;
    result.usage.wall_time_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count());
    result.usage.output_bytes = output.size();

    if (timed_out) {
        result.status = common_agent_sandbox_status::timed_out;
        result.error = "Docker sandbox timed out";
    } else if (exit_code == 0) {
        result.status = common_agent_sandbox_status::completed;
        collect_artifacts(request, result);
    } else {
        result.status = common_agent_sandbox_status::failed;
        result.error = output.empty() ? "Docker command failed" : output;
    }
    error.clear();
    return true;
}
