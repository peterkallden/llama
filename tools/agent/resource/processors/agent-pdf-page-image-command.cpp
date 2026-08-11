#include "agent-pdf-page-image-command.h"

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

bool validate_agent_pdf_page_image_options(
        const agent_pdf_page_image_options & options,
        std::string & error) {
    if (options.page == 0) {
        error = "PDF page selection must be one-based";
        return false;
    }
    if (options.dpi < 72 || options.dpi > 600) {
        error = "PDF rendering DPI must be between 72 and 600";
        return false;
    }
    if (options.format != "png") {
        error = "the bounded PDF page-image processor currently supports only png";
        return false;
    }
    if (options.colorspace != "rgb" && options.colorspace != "gray") {
        error = "PDF rendering colorspace must be rgb or gray";
        return false;
    }
    if (options.max_width > 16384 || options.max_height > 16384) {
        error = "PDF rendering dimensions must not exceed 16384 pixels";
        return false;
    }
    if (options.max_output_bytes == 0 || options.max_output_bytes > 64 * 1024 * 1024) {
        error = "PDF rendering output must be between 1 byte and 64 MiB";
        return false;
    }
    error.clear();
    return true;
}

bool make_agent_pdf_page_image_request(
        agent_resource_backend_kind backend,
        const std::string & executable,
        const std::string & operation_id,
        const std::string & project_id,
        const std::string & workspace_id,
        const common_runtime_resource_ref & source,
        const agent_pdf_page_image_options & options,
        common_agent_sandbox_request & request,
        std::string & error) {
    if (executable.empty() || operation_id.empty() || workspace_id.empty() ||
            source.uri.empty() || source.name.empty() ||
            source.mime_type != "application/pdf") {
        error = "PDF page-image request requires executable, operation, workspace and PDF source identity";
        return false;
    }
    if (!validate_agent_pdf_page_image_options(options, error)) return false;

    const std::string input_name = safe_file_name(source.name);
    if (input_name.empty()) {
        error = "PDF source name cannot be represented as a safe staged filename";
        return false;
    }
    const std::string output_name = "page-" + std::to_string(options.page) + ".png";
    request = {};
    request.operation_id = operation_id;
    request.project_id = project_id;
    request.workspace_id = workspace_id;
    request.execution_class = "resource.processor.pdf.page_image";
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

    switch (backend) {
        case agent_resource_backend_kind::local_mupdf:
            request.command.arguments = {
                "draw", "-F", "png", "-r", std::to_string(options.dpi),
                "-o", "/workspace/artifacts/" + output_name,
                "/workspace/source/" + input_name,
                std::to_string(options.page),
            };
            if (options.colorspace == "gray") {
                request.command.arguments.insert(request.command.arguments.begin() + 2, {"-c", "gray"});
            }
            if (options.max_width > 0) {
                request.command.arguments.insert(request.command.arguments.begin() + 4, {"-w", std::to_string(options.max_width)});
            }
            if (options.max_height > 0) {
                request.command.arguments.insert(request.command.arguments.begin() + 4, {"-h", std::to_string(options.max_height)});
            }
            break;
        case agent_resource_backend_kind::local_ghostscript:
            request.command.arguments = {
                "-dSAFER", "-dBATCH", "-dNOPAUSE",
                options.colorspace == "gray" ? "-sDEVICE=pnggray" : "-sDEVICE=png16m",
                "-r" + std::to_string(options.dpi),
                "-dFirstPage=" + std::to_string(options.page),
                "-dLastPage=" + std::to_string(options.page),
                "-sOutputFile=/workspace/artifacts/" + output_name,
                "/workspace/source/" + input_name,
            };
            break;
        case agent_resource_backend_kind::docker:
        case agent_resource_backend_kind::kubernetes:
        case agent_resource_backend_kind::local_pandoc:
            error = "sandbox backend must be resolved to a concrete PDF renderer before command construction";
            return false;
    }
    error.clear();
    return true;
}
