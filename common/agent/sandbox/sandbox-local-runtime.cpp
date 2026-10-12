#include "sandbox-local-runtime.h"

#include "subproc.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

namespace {

std::string map_workspace_path(
        std::string value,
        const common_agent_sandbox_workspace & workspace) {
    const auto replace_prefix = [&value](const std::string & prefix, const std::string & replacement) {
        if (prefix.empty() || replacement.empty() || value.rfind(prefix, 0) != 0) return false;
        value = replacement + value.substr(prefix.size());
        return true;
    };
    if (replace_prefix("/workspace/source", workspace.source_path)) return value;
    if (replace_prefix("/workspace/writable", workspace.writable_path)) return value;
    if (replace_prefix("/workspace/artifacts", workspace.artifact_path)) return value;
    return value;
}

std::string read_bounded(FILE * stream, size_t max_bytes) {
    if (stream == nullptr) return {};
    std::string result;
    char buffer[4096];
    while (true) {
        const size_t count = std::fread(buffer, 1, sizeof(buffer), stream);
        if (count == 0) break;
        if (result.size() < max_bytes) {
            const size_t take = std::min(count, max_bytes - result.size());
            result.append(buffer, take);
        }
    }
    return result;
}

} // namespace

bool common_agent_sandbox_local_runtime::execute(
        const common_agent_sandbox_request & request,
        common_agent_sandbox_result & result,
        std::string & error) {
    result = {};
    if (request.command.program.empty()) {
        error = "local sandbox requires a declared program";
        return false;
    }
    if (request.limits.timeout_ms == 0 || request.limits.max_output_bytes == 0) {
        error = "local sandbox requires non-zero timeout and output limits";
        return false;
    }

    std::vector<std::string> args;
    args.push_back(request.command.program);
    for (const auto & argument : request.command.arguments) {
        args.push_back(map_workspace_path(argument, request.workspace));
    }
    const std::string working_directory = map_workspace_path(
        request.command.working_directory, request.workspace);

    common_subproc process;
    const int options = subprocess_option_no_window |
        subprocess_option_combined_stdout_stderr |
        subprocess_option_inherit_environment |
        subprocess_option_search_user_path;
    if (!process.create(args, options, {}, working_directory.empty() ? nullptr : working_directory.c_str())) {
        error = "local sandbox process could not be started";
        result.status = common_agent_sandbox_status::failed;
        return false;
    }

    std::string output;
    std::thread output_reader([&process, &output, max_bytes = request.limits.max_output_bytes]() {
        output = read_bounded(process.stdout_file(), max_bytes);
    });
    const auto started = std::chrono::steady_clock::now();
    bool timed_out = false;
    while (process.alive()) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        if (elapsed >= request.limits.timeout_ms) {
            timed_out = true;
            process.terminate();
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (output_reader.joinable()) output_reader.join();
    const int exit_code = process.join();

    result.exit_code = exit_code;
    result.stdout_excerpt = std::move(output);
    result.backend_execution_id = "local/" + request.operation_id;
    if (timed_out) {
        result.status = common_agent_sandbox_status::timed_out;
        result.error = "local sandbox process exceeded its timeout";
    } else if (exit_code == 0) {
        result.status = common_agent_sandbox_status::completed;
    } else {
        result.status = common_agent_sandbox_status::failed;
        result.error = "local sandbox process returned a non-zero exit code";
    }
    error.clear();
    return true;
}
