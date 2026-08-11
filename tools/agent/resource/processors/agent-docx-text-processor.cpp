#include "agent-docx-text-processor.h"

#include <utility>

namespace {

constexpr const char * docx_mime =
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
constexpr const char * markdown_mime = "text/markdown";

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

bool has_docx_package_signature(const std::string & bytes) {
    return bytes.size() >= 4 && bytes.compare(0, 2, "PK") == 0 &&
        bytes.find("[Content_Types].xml") != std::string::npos &&
        bytes.find("word/document.xml") != std::string::npos;
}

} // namespace

std::string agent_pandoc_processor::id() const {
    return options_.input_format == "markdown"
        ? "pandoc-markdown-docx-v1"
        : "pandoc-docx-text-v1";
}

std::string agent_pandoc_processor::cache_key() const {
    return id() + ";input=" + options_.input_format + ";output=" + options_.output_format;
}

bool agent_pandoc_processor::supports(
        const std::string & mime_type,
        const std::string & target_representation) const {
    const auto normalized = common_normalize_resource_media_type(mime_type);
    return (options_.input_format == "docx" && normalized == docx_mime &&
            target_representation == "text") ||
        (options_.input_format == "markdown" && normalized == markdown_mime &&
            target_representation == "docx");
}

agent_resource_processor::support agent_pandoc_processor::supports(
        const agent_resource_processing_request & request) const {
    const auto source_type = request.media_type.resolved_type.empty()
        ? request.source.mime_type
        : request.media_type.resolved_type;
    const bool forward = options_.input_format == "docx" &&
        common_normalize_resource_media_type(source_type) == docx_mime &&
        request.target_representation == "text" &&
        (request.target_media_type.empty() ||
         common_normalize_resource_media_type(request.target_media_type) == "text/plain");
    const bool reverse = options_.input_format == "markdown" &&
        common_normalize_resource_media_type(source_type) == markdown_mime &&
        request.target_representation == "docx" &&
        common_normalize_resource_media_type(request.target_media_type) == docx_mime &&
        request.purpose == agent_resource_processing_purpose::artifact_generation;
    return {forward || reverse, reverse ? 100 : 50, false, false};
}

agent_resource_processing_result agent_pandoc_processor::process(
        const agent_resource_processing_request & request) const {
    const auto selected = supports(request);
    if (!selected.supported) {
        return failure(id(), "resource.unsupported_media_type", "Pandoc does not support the requested source and target format.");
    }
    const bool forward = options_.input_format == "docx";
    if (forward && !has_docx_package_signature(request.source_bytes)) {
        return failure(id(), "resource.invalid_document", "The source is not a recognizable DOCX package.");
    }
    if (executable_.empty()) {
        return failure(id(), "resource.processor_unavailable", "The configured Pandoc executable is unavailable.");
    }

    agent_pandoc_options options = options_;
    if (request.limits.max_output_bytes > 0) options.max_output_bytes = request.limits.max_output_bytes;
    common_agent_sandbox_request sandbox_request;
    std::string error;
    if (!make_agent_pandoc_request(
            backend_, executable_, context_.operation_id, context_.workspace.project_id,
            context_.workspace.workspace_id, request.source, options, sandbox_request, error)) {
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
                ? "resource.processing_timeout" : "resource.processor_failed",
            sandbox_result.error.empty() ? "Pandoc did not complete successfully." : sandbox_result.error);
    }
    if (artifacts.size() != 1 || artifacts.front().bytes.empty()) {
        return failure(id(), "resource.output_invalid", "Pandoc did not return one bounded output artifact.");
    }

    const auto & artifact = artifacts.front();
    agent_resource_processing_result result;
    result.success = true;
    result.processor_id = id();
    result.safe_summary = "DOCX text extracted through the host resource processor.";
    agent_resource_processing_output output;
    output.name = artifact.name.empty()
        ? request.source.name + (forward ? ".txt" : ".docx")
        : artifact.name;
    output.description = forward
        ? "Normalized text representation extracted from a DOCX document."
        : "DOCX artifact generated from a Markdown resource.";
    output.mime_type = forward ? "text/plain" : docx_mime;
    output.bytes = artifact.bytes;
    output.metadata.purpose = forward
        ? "normalized DOCX text representation"
        : "DOCX artifact generated from a Markdown resource";
    output.metadata.content_summary = forward
        ? "Text extracted from the authoritative DOCX resource."
        : "DOCX artifact generated from the authoritative Markdown resource.";
    output.metadata.usage_hint = forward
        ? "Read and chunk this derived text resource."
        : "Download or export this derived artifact resource.";
    output.metadata.limitations = forward
        ? "Pandoc plain-text conversion may not preserve complete document layout or complex tables."
        : "Pandoc DOCX generation does not guarantee pixel-identical Word layout.";
    output.lineage.parent_uri = request.source.uri;
    output.lineage.chunk_index = 0;
    output.lineage.chunk_count = 1;
    output.lineage.byte_offset = 0;
    output.lineage.byte_length = request.source.size_bytes;
    output.lineage.derivation = "resource.process:" + id();
    result.outputs.push_back(std::move(output));
    return result;
}
