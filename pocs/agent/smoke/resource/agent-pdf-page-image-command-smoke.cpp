#include "agent-pdf-page-image-command.h"

#include <cstdio>
#include <string>

namespace {

bool contains_argument(
        const common_agent_sandbox_request & request,
        const std::string & value) {
    for (const auto & argument : request.command.arguments) {
        if (argument == value) return true;
    }
    return false;
}

} // namespace

int main() {
    common_runtime_resource_ref source;
    source.uri = "agent-resource://document/report.pdf";
    source.name = "report final.pdf";
    source.mime_type = "application/pdf";

    agent_pdf_page_image_options options;
    options.page = 42;
    options.dpi = 200;
    options.colorspace = "gray";

    common_agent_sandbox_request request;
    std::string error;
    if (!make_agent_pdf_page_image_request(
            agent_resource_backend_kind::local_mupdf,
            "mutool",
            "operation-1",
            "project-1",
            "workspace-1",
            source,
            options,
            request,
            error)) {
        std::fprintf(stderr, "MuPDF command construction failed: %s\n", error.c_str());
        return 1;
    }
    if (request.command.program != "mutool" ||
            request.command.working_directory != "/workspace/source" ||
            !contains_argument(request, "draw") ||
            !contains_argument(request, "-F") ||
            !contains_argument(request, "png") ||
            !contains_argument(request, "-c") ||
            !contains_argument(request, "gray") ||
            !contains_argument(request, "200") ||
            !contains_argument(request, "/workspace/source/report_final.pdf") ||
            !contains_argument(request, "42") ||
            !contains_argument(request, "/workspace/artifacts/page-42.png") ||
            request.network != common_agent_sandbox_network_scope::none ||
            request.filesystem != common_agent_sandbox_filesystem_scope::artifact_write ||
            request.workspace.input_resources.size() != 1) {
        std::fprintf(stderr, "MuPDF command contract was unexpected\n");
        return 1;
    }

    if (!make_agent_pdf_page_image_request(
            agent_resource_backend_kind::local_ghostscript,
            "gswin64c.exe",
            "operation-2",
            "project-1",
            "workspace-1",
            source,
            options,
            request,
            error) ||
            !contains_argument(request, "-dSAFER") ||
            !contains_argument(request, "-dFirstPage=42") ||
            !contains_argument(request, "-dLastPage=42") ||
            !contains_argument(request, "-sDEVICE=pnggray")) {
        std::fprintf(stderr, "Ghostscript command contract was unexpected: %s\n", error.c_str());
        return 1;
    }

    options.page = 0;
    if (validate_agent_pdf_page_image_options(options, error)) {
        std::fprintf(stderr, "zero-based page was accepted\n");
        return 1;
    }
    options.page = 42;
    options.dpi = 601;
    if (validate_agent_pdf_page_image_options(options, error)) {
        std::fprintf(stderr, "unbounded DPI was accepted\n");
        return 1;
    }
    if (make_agent_pdf_page_image_request(
            agent_resource_backend_kind::docker,
            "mutool",
            "operation-3",
            "project-1",
            "workspace-1",
            source,
            agent_pdf_page_image_options{42, 150, "png", "rgb", 0, 0, 1024},
            request,
            error)) {
        std::fprintf(stderr, "unresolved sandbox backend was accepted\n");
        return 1;
    }

    std::printf("pdf_page_image_command=passed\n");
    return 0;
}
