#include "agent-tesseract-ocr-command.h"

#include <cstdio>
#include <string>

namespace {

bool contains_argument(const common_agent_sandbox_request & request, const std::string & value) {
    for (const auto & argument : request.command.arguments) {
        if (argument == value) return true;
    }
    return false;
}

} // namespace

int main() {
    common_runtime_resource_ref source;
    source.uri = "agent-resource://document/page-1.png";
    source.name = "page 1.png";
    source.mime_type = "image/png";

    agent_tesseract_ocr_options options;
    options.language = "eng+swe";
    options.oem = 3;
    options.psm = 6;
    options.output_format = "hocr";

    common_agent_sandbox_request request;
    std::string error;
    if (!make_agent_tesseract_ocr_request(
            agent_resource_backend_kind::local_tesseract,
            "tesseract",
            "operation-1",
            "project-1",
            "workspace-1",
            source,
            options,
            request,
            error)) {
        std::fprintf(stderr, "Tesseract command construction failed: %s\n", error.c_str());
        return 1;
    }
    if (request.command.program != "tesseract" ||
            !contains_argument(request, "/workspace/source/page_1.png") ||
            !contains_argument(request, "/workspace/artifacts/ocr-page_1.png") ||
            request.command.arguments.size() < 2 ||
            request.command.arguments[2] != "-l" ||
            !contains_argument(request, "-l") || !contains_argument(request, "eng+swe") ||
            !contains_argument(request, "--oem") || !contains_argument(request, "3") ||
            !contains_argument(request, "--psm") || !contains_argument(request, "6") ||
            !contains_argument(request, "hocr") ||
            request.network != common_agent_sandbox_network_scope::none ||
            request.filesystem != common_agent_sandbox_filesystem_scope::artifact_write ||
            request.artifacts.paths.size() != 1 ||
            request.artifacts.paths.front() != "ocr-page_1.png.hocr") {
        std::fprintf(stderr, "Tesseract command contract was unexpected\n");
        return 1;
    }

    options.language = "eng;rm -rf";
    if (validate_agent_tesseract_ocr_options(options, error)) {
        std::fprintf(stderr, "unsafe Tesseract language was accepted\n");
        return 1;
    }
    options.language = "eng";
    options.output_format = "pdf";
    if (validate_agent_tesseract_ocr_options(options, error)) {
        std::fprintf(stderr, "unsupported Tesseract output format was accepted\n");
        return 1;
    }
    options.output_format = "text";
    options.language = "auto";
    if (!validate_agent_tesseract_ocr_options(options, error)) {
        std::fprintf(stderr, "automatic Tesseract language was rejected: %s\n", error.c_str());
        return 1;
    }
    options.fallback_language = "swe";
    if (!validate_agent_tesseract_ocr_options(options, error)) {
        std::fprintf(stderr, "Tesseract fallback language was rejected: %s\n", error.c_str());
        return 1;
    }
    if (make_agent_tesseract_ocr_request(
            agent_resource_backend_kind::docker,
            "tesseract",
            "operation-2",
            "project-1",
            "workspace-1",
            source,
            agent_tesseract_ocr_options{},
            request,
            error)) {
        std::fprintf(stderr, "unresolved Tesseract backend was accepted\n");
        return 1;
    }

    std::printf("tesseract_ocr_command=passed\n");
    return 0;
}
