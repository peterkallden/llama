#include "sandbox-lxc-runtime.h"

#include "subproc.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>

namespace {

std::string safe_name(const std::string & operation_id) {
    std::string result = "llama-agent-";
    for (const char value : operation_id) {
        const bool safe = (value >= 'a' && value <= 'z') ||
            (value >= 'A' && value <= 'Z') ||
            (value >= '0' && value <= '9') || value == '-';
        result.push_back(safe ? value : '-');
    }
    if (result.size() > 60) result.resize(60);
    return result;
}

std::string read_bounded(FILE * stream, size_t max_bytes) {
    if (stream == nullptr) return {};
    std::string result;
    char buffer[4096];
    while (true) {
        const size_t count = std::fread(buffer, 1, sizeof(buffer), stream);
        if (count == 0) break;
        if (result.size() < max_bytes) {
            result.append(buffer, std::min(count, max_bytes - result.size()));
        }
    }
    return result;
}

bool run_command(
        const std::vector<std::string> & args,
        uint32_t timeout_ms,
        size_t max_output_bytes,
        std::string & output,
        std::string & error) {
    common_subproc process;
    const int options = subprocess_option_no_window |
        subprocess_option_combined_stdout_stderr |
        subprocess_option_inherit_environment |
        subprocess_option_search_user_path;
    if (!process.create(args, options)) {
        error = "LXC command could not be started";
        return false;
    }
    std::thread reader([&process, &output, max_output_bytes]() {
        output = read_bounded(process.stdout_file(), max_output_bytes);
    });
    const auto started = std::chrono::steady_clock::now();
    while (process.alive()) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        if (elapsed >= timeout_ms) {
            process.terminate();
            reader.join();
            error = "LXC command timed out";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const int exit_code = process.join();
    reader.join();
    if (exit_code != 0) {
        error = output.empty() ? "LXC command failed" : output;
        return false;
    }
    return true;
}

bool add_mount(
        const std::string & executable,
        const std::string & container,
        const std::string & source,
        const char * device_name,
        const char * target,
        bool readonly,
        uint32_t timeout_ms,
        std::string & error) {
    if (source.empty()) return true;
    std::error_code fs_error;
    if (!std::filesystem::is_directory(source, fs_error) || fs_error) {
        error = "LXC mount source is not an accessible directory: " + source;
        return false;
    }
    std::string output;
    std::vector<std::string> args = {
        executable, "config", "device", "add", container,
        std::string("llama-") + device_name + "-disk", "disk",
        "source=" + source, "path=" + std::string(target),
        "readonly=" + std::string(readonly ? "true" : "false"),
    };
    return run_command(args, timeout_ms, 4096, output, error);
}

bool set_limit(
        const std::string & executable,
        const std::string & container,
        const char * key,
        const std::string & value,
        uint32_t timeout_ms,
        std::string & error) {
    std::string output;
    return run_command({executable, "config", "set", container, key, value},
        timeout_ms, 4096, output, error);
}

void cleanup_container(
        const std::string & executable,
        const std::string & container) {
    std::string output;
    std::string ignored_error;
    run_command({executable, "stop", container, "--force"}, 30000, 4096, output, ignored_error);
    run_command({executable, "delete", container}, 30000, 4096, output, ignored_error);
}

void collect_artifacts(
        const common_agent_sandbox_request & request,
        common_agent_sandbox_result & result) {
    if (!request.artifacts.collect || request.workspace.artifact_path.empty()) return;
    std::error_code fs_error;
    const std::filesystem::path root(request.workspace.artifact_path);
    if (!std::filesystem::is_directory(root, fs_error) || fs_error) return;
    size_t collected = 0;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(root, fs_error)) {
        if (fs_error || collected >= request.artifacts.max_bytes) break;
        if (!entry.is_regular_file(fs_error) || fs_error) continue;
        const auto size = static_cast<size_t>(entry.file_size(fs_error));
        if (fs_error || collected + size > request.artifacts.max_bytes) continue;
        const auto relative = std::filesystem::relative(entry.path(), root, fs_error);
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

} // namespace

common_agent_sandbox_capabilities common_agent_sandbox_lxc_runtime::capabilities() const {
    common_agent_sandbox_capabilities result;
    result.process_isolation = true;
    result.filesystem_readonly = true;
    result.filesystem_workspace_write = true;
    result.filesystem_artifact_write = true;
    // A profile name alone says nothing about what it enforces. Only the
    // operator-declared scope of a configured profile becomes a capability.
    if (!config.network_profile.empty()) {
        if (config.network_profile_scope == "none") {
            result.network_none = true;
        } else if (config.network_profile_scope == "dns_only") {
            result.network_dns_only = true;
        } else if (config.network_profile_scope == "allowlisted") {
            result.network_allowlisted = true;
        } else if (config.network_profile_scope == "package_registry") {
            result.network_package_registry = true;
        } else if (config.network_profile_scope == "research_web") {
            result.network_research_web = true;
        }
    }
    result.cpu_limit = true;
    result.memory_limit = true;
    result.process_limit = true;
    result.artifact_collection = true;
    return result;
}

bool common_agent_sandbox_lxc_runtime::execute(
        const common_agent_sandbox_request & request,
        common_agent_sandbox_result & result,
        std::string & error) {
    result = {};
    const std::string container = safe_name(request.operation_id);
    result.backend_execution_id = "lxc/" + container;
    if (request.operation_id.empty() || request.command.program.empty()) {
        error = "LXC sandbox requires an operation id and program";
        result.error = error;
        return false;
    }
    if (!common_agent_sandbox_validate_capabilities(request, capabilities(), error)) {
        result.error = error;
        return false;
    }
    const std::string executable = config.executable.empty() ? "lxc" : config.executable;
    const std::string image = request.image.empty() ? config.default_image : request.image;
    if (image.empty()) {
        error = "LXC sandbox requires an image or a configured default image";
        result.error = error;
        return false;
    }
    if (config.network_profile.empty()) {
        error = "LXC sandbox requires an explicit operator-managed network profile";
        result.error = error;
        return false;
    }

    std::vector<std::string> launch{executable, "launch", image, container};
    if (!config.network_profile.empty()) {
        launch.push_back("--profile");
        launch.push_back(config.network_profile);
    }
    std::string output;
    if (!run_command(launch, request.limits.timeout_ms, request.limits.max_output_bytes, output, error)) {
        result.error = error;
        return false;
    }

    // LXC/Incus resource limits are instance configuration, not properties of
    // the host request. Apply every requested limit and fail closed if the
    // runtime cannot enforce one.
    if (!set_limit(executable, container, "limits.cpu",
            std::to_string(request.limits.cpu_count), request.limits.timeout_ms, error) ||
            (request.limits.memory_bytes != 0 && !set_limit(
                executable, container, "limits.memory",
                std::to_string(request.limits.memory_bytes) + "B",
                request.limits.timeout_ms, error)) ||
            (request.limits.process_count != 0 && !set_limit(
                executable, container, "limits.processes",
                std::to_string(request.limits.process_count),
                request.limits.timeout_ms, error))) {
        result.error = error;
        if (config.cleanup) cleanup_container(executable, container);
        return false;
    }

    if (!add_mount(executable, container, request.workspace.source_path, "source",
            "/workspace/source", true, request.limits.timeout_ms, error) ||
            !add_mount(executable, container, request.workspace.writable_path, "writable",
            "/workspace/writable",
            request.filesystem != common_agent_sandbox_filesystem_scope::workspace_write,
            request.limits.timeout_ms, error) ||
            !add_mount(executable, container, request.workspace.artifact_path, "artifacts",
            "/workspace/artifacts",
            request.filesystem != common_agent_sandbox_filesystem_scope::artifact_write,
            request.limits.timeout_ms, error)) {
        result.error = error;
        if (config.cleanup) {
            cleanup_container(executable, container);
        }
        return false;
    }

    std::vector<std::string> exec{executable, "exec", container};
    if (!request.command.working_directory.empty()) {
        exec.push_back("--cwd");
        exec.push_back(request.command.working_directory);
    }
    exec.push_back("--");
    exec.push_back(request.command.program);
    exec.insert(exec.end(), request.command.arguments.begin(), request.command.arguments.end());
    const bool ok = run_command(exec, request.limits.timeout_ms, request.limits.max_output_bytes,
        result.stdout_excerpt, error);
    if (config.cleanup) {
        cleanup_container(executable, container);
    }
    if (!ok) {
        result.error = error;
        result.status = error == "LXC command timed out"
            ? common_agent_sandbox_status::timed_out
            : common_agent_sandbox_status::failed;
        return false;
    }
    collect_artifacts(request, result);
    result.status = common_agent_sandbox_status::completed;
    result.exit_code = 0;
    return true;
}
