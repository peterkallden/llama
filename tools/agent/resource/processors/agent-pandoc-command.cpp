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

bool validate_agent_pandoc_docx_options(
        const agent_pandoc_docx_options & options,
        std::string & error) {
    if (options.output_format != "text") {
        error = "Pandoc DOCX output format must be text";
        return false;
    }
    if (options.max_output_bytes == 0 || options.max_output_bytes > 64 * 1024 * 1024) {
        error = "Pandoc DOCX output must be between 1 byte and 64 MiB";
        return false;
    }
    error.clear();
    return true;
}

bool make_agent_pandoc_docx_request(
        agent_resource_backend_kind backend,
        const std::string & executable,
        const std::string & operation_id,
        const std::string & project_id,
        const std::string & workspace_id,
        const common_runtime_resource_ref & source,
        const agent_pandoc_docx_options & options,
        common_agent_sandbox_request & request,
        std::string & error) {
    if (backend != agent_resource_backend_kind::local_pandoc) {
        error = "sandbox backend must be resolved to a concrete Pandoc executable before command construction";
        return false;
    }
    if (executable.empty() || operation_id.empty() || workspace_id.empty() ||
            source.uri.empty() || source.name.empty() ||
            source.mime_type != "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
        error = "Pandoc DOCX request requires executable, operation, workspace and DOCX source identity";
        return false;
    }
    if (!validate_agent_pandoc_docx_options(options, error)) return false;

    const std::string input_name = safe_file_name(source.name);
    if (input_name.empty()) {
        error = "DOCX source name cannot be represented as a safe staged filename";
        return false;
    }
    const std::string output_name = "docx-text-" + input_name + ".txt";
    request = {};
    request.operation_id = operation_id;
    request.project_id = project_id;
    request.workspace_id = workspace_id;
    request.execution_class = "resource.processor.docx.text";
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
        "--from=docx",
        "--to=plain",
        "--wrap=none",
        "--output",
        "/workspace/artifacts/" + output_name,
    };
    error.clear();
    return true;
}
