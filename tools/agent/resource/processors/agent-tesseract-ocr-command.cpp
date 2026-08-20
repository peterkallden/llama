#include "agent-tesseract-ocr-command.h"

#include <cctype>
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

bool valid_language(const std::string & language) {
    if (language.empty() || language.size() > 64) return false;
    for (const char character : language) {
        if (!std::isalnum(static_cast<unsigned char>(character)) &&
                character != '_' && character != '+' && character != '-' &&
                character != '.') {
            return false;
        }
    }
    return true;
}

bool valid_language_or_auto(const std::string & language) {
    return language == "auto" || valid_language(language);
}

const char * output_extension(const std::string & format) {
    if (format == "text") return "txt";
    if (format == "hocr") return "hocr";
    if (format == "tsv") return "tsv";
    return nullptr;
}

} // namespace

bool validate_agent_tesseract_ocr_options(
        const agent_tesseract_ocr_options & options,
        std::string & error) {
    if (!valid_language_or_auto(options.language)) {
        error = "Tesseract language must be a bounded identifier list";
        return false;
    }
    if (!options.fallback_language.empty() && !valid_language(options.fallback_language)) {
        error = "Tesseract fallback language must be a bounded identifier list";
        return false;
    }
    if (options.oem > 3) {
        error = "Tesseract OEM must be between 0 and 3";
        return false;
    }
    if (options.psm > 13) {
        error = "Tesseract PSM must be between 0 and 13";
        return false;
    }
    if (output_extension(options.output_format) == nullptr) {
        error = "Tesseract output format must be text, hocr or tsv";
        return false;
    }
    if (options.max_output_bytes == 0 || options.max_output_bytes > 64 * 1024 * 1024) {
        error = "Tesseract output must be between 1 byte and 64 MiB";
        return false;
    }
    error.clear();
    return true;
}

bool make_agent_tesseract_ocr_request(
        agent_resource_backend_kind backend,
        const std::string & executable,
        const std::string & operation_id,
        const std::string & project_id,
        const std::string & workspace_id,
        const common_runtime_resource_ref & source,
        const agent_tesseract_ocr_options & options,
        common_agent_sandbox_request & request,
        std::string & error) {
    if (backend != agent_resource_backend_kind::local_tesseract) {
        error = "sandbox backend must be resolved to a concrete Tesseract executable before command construction";
        return false;
    }
    if (executable.empty() || operation_id.empty() || workspace_id.empty() ||
            source.uri.empty() || source.name.empty() ||
            source.mime_type.rfind("image/", 0) != 0) {
        error = "Tesseract request requires executable, operation, workspace and image source identity";
        return false;
    }
    if (!validate_agent_tesseract_ocr_options(options, error)) return false;

    const std::string input_name = safe_file_name(source.name);
    if (input_name.empty()) {
        error = "OCR source name cannot be represented as a safe staged filename";
        return false;
    }
    const std::string output_name = "ocr-" + input_name;
    const std::string output_base = "/workspace/artifacts/" + output_name;
    const std::string artifact_name = output_name + "." + output_extension(options.output_format);

    request = {};
    request.operation_id = operation_id;
    request.project_id = project_id;
    request.workspace_id = workspace_id;
    request.execution_class = "resource.processor.ocr.tesseract";
    request.workspace.input_resources.push_back(source);
    request.limits.timeout_ms = 120000;
    request.limits.max_output_bytes = options.max_output_bytes;
    request.network = common_agent_sandbox_network_scope::none;
    request.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
    request.artifacts.collect = true;
    request.artifacts.max_bytes = options.max_output_bytes;
    request.artifacts.paths.push_back(artifact_name);
    request.command.program = executable;
    request.command.working_directory = "/workspace/source";
    request.command.arguments = {
        "/workspace/source/" + input_name,
        output_base,
        "-l", options.language,
        "--oem", std::to_string(options.oem),
        "--psm", std::to_string(options.psm),
    };
    if (options.output_format != "text") {
        request.command.arguments.push_back(options.output_format);
    }
    error.clear();
    return true;
}
