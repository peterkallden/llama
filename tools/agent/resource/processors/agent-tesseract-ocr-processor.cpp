#include "agent-tesseract-ocr-processor.h"

#include <utility>

namespace {

agent_resource_processing_result failure(
        const std::string & processor_id,
        std::string code,
        std::string summary) {
    agent_resource_processing_result result;
    result.processor_id = processor_id;
    result.failure_code = std::move(code);
    result.safe_summary = std::move(summary);
    return result;
}

std::string output_mime_type(const std::string & representation) {
    if (representation == "hocr") return "application/xhtml+xml";
    if (representation == "tsv") return "text/tab-separated-values";
    return "text/plain";
}

std::string tesseract_language_code(std::string value) {
    if (value == "sv") return "swe";
    if (value == "en") return "eng";
    if (value == "de") return "deu";
    if (value == "fr") return "fra";
    return value;
}

} // namespace

std::string agent_tesseract_ocr_processor::id() const {
    return "tesseract-ocr-v1";
}

std::string agent_tesseract_ocr_processor::cache_key() const {
    return id() + ";language=" + options_.language +
        ";fallback=" + options_.fallback_language +
        ";oem=" + std::to_string(options_.oem) +
        ";psm=" + std::to_string(options_.psm);
}

bool agent_tesseract_ocr_processor::supports(
        const std::string & mime_type,
        const std::string & target_representation) const {
    return mime_type.rfind("image/", 0) == 0 &&
        (target_representation == "text" ||
         target_representation == "hocr" ||
         target_representation == "tsv");
}

agent_resource_processing_result agent_tesseract_ocr_processor::process(
        const agent_resource_processing_request & request) const {
    if (!supports(request.source.mime_type, request.target_representation)) {
        return failure(id(), "resource.unsupported_media_type", "Tesseract OCR requires an image source and a supported text representation.");
    }
    if (executable_.empty()) {
        return failure(id(), "resource.processor_unavailable", "The configured Tesseract executable is unavailable.");
    }

    agent_tesseract_ocr_options options = options_;
    options.output_format = request.target_representation;
    std::string language_source = "processor.config";
    if (options.language == "auto") {
        if (!request.source.metadata.resolved_language.empty()) {
            options.language = tesseract_language_code(request.source.metadata.resolved_language);
            language_source = "resource.metadata.resolved";
        } else if (!request.source.metadata.declared_language.empty()) {
            options.language = tesseract_language_code(request.source.metadata.declared_language);
            language_source = "resource.metadata.declared";
        } else if (!options.fallback_language.empty()) {
            options.language = options.fallback_language;
            language_source = "processor.fallback";
        } else {
            return failure(id(), "resource.ocr_language_uncertain", "OCR language is automatic but the resource has no accepted language metadata or fallback.");
        }
    }
    if (request.limits.max_output_bytes > 0) {
        options.max_output_bytes = request.limits.max_output_bytes;
    }
    common_agent_sandbox_request sandbox_request;
    std::string error;
    if (!make_agent_tesseract_ocr_request(
            backend_,
            executable_,
            context_.operation_id,
            context_.workspace.project_id,
            context_.workspace.workspace_id,
            request.source,
            options,
            sandbox_request,
            error)) {
        return failure(id(), "resource.invalid_request", std::move(error));
    }

    common_agent_sandbox_result sandbox_result;
    std::vector<agent_resource_processing_host_artifact> artifacts;
    if (!host_.execute(context_, std::move(sandbox_request), sandbox_result, artifacts, error)) {
        return failure(id(), "resource.processor_execution_failed", std::move(error));
    }
    if (sandbox_result.status != common_agent_sandbox_status::completed) {
        return failure(
            id(),
            sandbox_result.status == common_agent_sandbox_status::timed_out
                ? "resource.processing_timeout"
                : "resource.processor_failed",
            sandbox_result.error.empty()
                ? "Tesseract did not complete successfully."
                : sandbox_result.error);
    }
    if (artifacts.size() != 1 || artifacts.front().bytes.empty()) {
        return failure(id(), "resource.output_invalid", "Tesseract did not return one bounded OCR result.");
    }

    const auto & artifact = artifacts.front();
    agent_resource_processing_result result;
    result.success = true;
    result.processor_id = id();
    result.safe_summary = "Image OCR completed through the host resource processor.";
    agent_resource_processing_output output;
    output.name = artifact.name.empty() ? "ocr-result" : artifact.name;
    output.description = "Derived OCR representation";
    output.mime_type = output_mime_type(request.target_representation);
    output.bytes = artifact.bytes;
    output.metadata.purpose = "OCR representation of an image resource";
    output.metadata.content_summary = "Bounded text extracted from an image";
    output.metadata.usage_hint = "Use this derived text representation for bounded reads and chunking.";
    output.metadata.limitations = "OCR is derived evidence; the original image or document remains authoritative.";
    output.metadata.declared_language = request.source.metadata.declared_language;
    output.metadata.resolved_language = options.language;
    output.metadata.language_confidence = language_source == "processor.config" ? 1.0 :
        (request.source.metadata.language_confidence > 0.0 ? request.source.metadata.language_confidence : 0.5);
    output.metadata.language_source = language_source;
    output.lineage.parent_uri = request.source.uri;
    output.lineage.chunk_index = 0;
    output.lineage.chunk_count = 1;
    output.lineage.byte_offset = 0;
    output.lineage.byte_length = request.source.size_bytes;
    output.lineage.derivation = "resource.process:" + id();
    result.outputs.push_back(std::move(output));
    return result;
}
