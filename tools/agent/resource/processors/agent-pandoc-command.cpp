#include "agent-pandoc-command.h"

#include <utility>

namespace {

std::string safe_file_name(const std::string & value) {
    std::string result;
    for (const char character : value) {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '-' || character == '_' || character == '.';
        result += safe ? character : '_';
    }
    while (!result.empty() && result.front() == '.') result.front() = '_';
    return result;
}

} // namespace

bool validate_agent_pandoc_options(
        const agent_pandoc_options & options,
        std::string & error) {
    if (options.input_format != "docx" && options.input_format != "markdown" &&
            options.input_format != "odt" && options.input_format != "html") {
        error = "Pandoc input format must be docx, markdown, odt or html";
        return false;
    }
    if (options.output_format != "plain" && options.output_format != "docx" &&
            options.output_format != "markdown" && options.output_format != "json") {
        error = "Pandoc output format must be plain, docx, markdown or json";
        return false;
    }
    const bool valid_direction =
        (options.input_format == "docx" && options.output_format == "plain") ||
        (options.input_format == "docx" && options.output_format == "json") ||
        (options.input_format == "markdown" && options.output_format == "docx") ||
        (options.input_format == "odt" && options.output_format == "markdown") ||
        (options.input_format == "html" && options.output_format == "markdown") ||
        (options.input_format == "html" && options.output_format == "json");
    if (!valid_direction) {
        error = "Pandoc format direction is not enabled for the agent resource processor";
        return false;
    }
    if (options.output_extension.empty() || options.output_extension.size() > 8 ||
            options.output_extension.find_first_not_of("abcdefghijklmnopqrstuvwxyz") != std::string::npos) {
        error = "Pandoc output extension is invalid";
        return false;
    }
    if (options.max_output_bytes == 0 || options.max_output_bytes > 64 * 1024 * 1024) {
        error = "Pandoc output must be between 1 byte and 64 MiB";
        return false;
    }
    error.clear();
    return true;
}

bool make_agent_pandoc_request(
        agent_resource_backend_kind backend,
        const std::string & executable,
        const std::string & operation_id,
        const std::string & project_id,
        const std::string & workspace_id,
        const common_runtime_resource_ref & source,
        const agent_pandoc_options & options,
        common_agent_sandbox_request & request,
        std::string & error) {
    if (backend != agent_resource_backend_kind::local_pandoc &&
            backend != agent_resource_backend_kind::docker &&
            backend != agent_resource_backend_kind::kubernetes) {
        error = "Pandoc backend must be local, Docker or Kubernetes";
        return false;
    }
    if (executable.empty() || operation_id.empty() || workspace_id.empty() ||
            source.uri.empty() || source.name.empty() ||
            source.mime_type.empty()) {
        error = "Pandoc request requires executable, operation, workspace and source identity";
        return false;
    }
    if (!validate_agent_pandoc_options(options, error)) return false;

    const std::string input_name = safe_file_name(source.name);
    if (input_name.empty()) {
        error = "Pandoc source name cannot be represented as a safe staged filename";
        return false;
    }
    const std::string output_name = "pandoc-output-" + input_name + "." + options.output_extension;
    request = {};
    request.operation_id = operation_id;
    request.project_id = project_id;
    request.workspace_id = workspace_id;
    request.execution_class = "resource.processor.pandoc";
    request.workspace.input_resources.push_back(source);
    request.limits.timeout_ms = 120000;
    request.limits.max_output_bytes = options.max_output_bytes;
    request.network = common_agent_sandbox_network_scope::none;
    request.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
    request.artifacts.collect = true;
    request.artifacts.max_bytes = options.max_output_bytes;
    request.artifacts.paths.push_back(output_name);
    request.command.program = executable;
    request.command.working_directory = "/workspace/source";
    request.command.arguments = {
        "/workspace/source/" + input_name,
        "--from=" + options.input_format,
        "--to=" + options.output_format,
        "--wrap=none",
        "--output",
        "/workspace/artifacts/" + output_name,
    };
    error.clear();
    return true;
}
