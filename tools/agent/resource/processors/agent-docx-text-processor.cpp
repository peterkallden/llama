#include "agent-docx-text-processor.h"

#include <utility>

namespace {

constexpr const char * docx_mime =
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
constexpr const char * odt_mime = "application/vnd.oasis.opendocument.text";
constexpr const char * markdown_mime = "text/markdown";
constexpr const char * html_mime = "text/html";

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

bool has_odt_package_signature(const std::string & bytes) {
    return bytes.size() >= 4 && bytes.compare(0, 2, "PK") == 0 &&
        bytes.find("content.xml") != std::string::npos;
}

bool is_markdown_normalization(const agent_pandoc_options & options) {
    return (options.input_format == "odt" || options.input_format == "html") &&
        options.output_format == "markdown";
}

} // namespace

std::string agent_pandoc_processor::id() const {
    if (options_.input_format == "markdown") return "pandoc-markdown-docx-v1";
    if (options_.input_format == "odt") return "pandoc-odt-markdown-v1";
    if (options_.input_format == "docx" && options_.output_format == "json") {
        return "pandoc-docx-document-json-v1";
    }
    if (options_.input_format == "html" && options_.output_format == "json") {
        return "pandoc-html-document-json-v1";
    }
    if (options_.input_format == "html") return "pandoc-html-markdown-v1";
    return "pandoc-docx-text-v1";
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
        (options_.input_format == "docx" && normalized == docx_mime &&
            target_representation == "document-json") ||
        (options_.input_format == "markdown" && normalized == markdown_mime &&
            target_representation == "docx") ||
        (is_markdown_normalization(options_) &&
            ((options_.input_format == "odt" && normalized == odt_mime) ||
             (options_.input_format == "html" && normalized == html_mime)) &&
            target_representation == "text") ||
        (options_.input_format == "html" && options_.output_format == "json" &&
            normalized == html_mime && target_representation == "document-json");
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
    const bool document_json = options_.output_format == "json" &&
        request.target_representation == "document-json" &&
        (options_.input_format == "docx" || options_.input_format == "html") &&
        ((options_.input_format == "docx" &&
          common_normalize_resource_media_type(source_type) == docx_mime) ||
         (options_.input_format == "html" &&
          common_normalize_resource_media_type(source_type) == html_mime)) &&
        (request.target_media_type.empty() ||
         common_normalize_resource_media_type(request.target_media_type) == "application/json");
    const bool reverse = options_.input_format == "markdown" &&
        common_normalize_resource_media_type(source_type) == markdown_mime &&
        request.target_representation == "docx" &&
        common_normalize_resource_media_type(request.target_media_type) == docx_mime &&
        request.purpose == agent_resource_processing_purpose::artifact_generation;
    const bool normalized_markdown = is_markdown_normalization(options_) &&
        ((options_.input_format == "odt" && common_normalize_resource_media_type(source_type) == odt_mime) ||
         (options_.input_format == "html" && common_normalize_resource_media_type(source_type) == html_mime)) &&
        request.target_representation == "text" &&
        (request.target_media_type.empty() ||
         common_normalize_resource_media_type(request.target_media_type) == markdown_mime);
    return {forward || document_json || reverse || normalized_markdown,
        reverse ? 100 : 50, normalized_markdown || document_json, false};
}

agent_resource_processing_result agent_pandoc_processor::process(
        const agent_resource_processing_request & request) const {
    const auto selected = supports(request);
    if (!selected.supported) {
        return failure(id(), "resource.unsupported_media_type", "Pandoc does not support the requested source and target format.");
    }
    const bool forward = options_.input_format == "docx";
    const bool odt_forward = options_.input_format == "odt";
    const bool document_json = options_.output_format == "json" &&
        request.target_representation == "document-json";
    const bool markdown_normalization = is_markdown_normalization(options_);
    if (forward && !has_docx_package_signature(request.source_bytes)) {
        return failure(id(), "resource.invalid_document", "The source is not a recognizable DOCX package.");
    }
    if (odt_forward && !has_odt_package_signature(request.source_bytes)) {
        return failure(id(), "resource.invalid_document", "The source is not a recognizable ODT package.");
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
    const bool text_forward = (forward || odt_forward) && !document_json;
    agent_resource_processing_result result;
    result.success = true;
    result.processor_id = id();
    result.safe_summary = document_json
        ? "Structured Pandoc document representation created through the host resource processor."
        : text_forward
        ? "Document text extracted through the host resource processor."
        : markdown_normalization
            ? "Markdown normalized through the host Pandoc resource processor."
            : "DOCX artifact generated through the host resource processor.";
    agent_resource_processing_output output;
    output.name = artifact.name.empty()
        ? request.source.name + (document_json ? ".json" : text_forward ? ".txt" : markdown_normalization ? ".md" : ".docx")
        : artifact.name;
    output.description = document_json
        ? "Structured document representation derived from an authoritative document resource."
        : text_forward
        ? "Normalized text representation extracted from a DOCX document."
        : markdown_normalization
            ? "Normalized Markdown representation derived from a document resource."
            : "DOCX artifact generated from a Markdown resource.";
    output.mime_type = document_json ? "application/json" :
        (text_forward || markdown_normalization)
            ? (odt_forward || markdown_normalization ? markdown_mime : "text/plain")
            : docx_mime;
    output.bytes = artifact.bytes;
    output.metadata.purpose = document_json
        ? "structured document representation"
        : text_forward
        ? "normalized DOCX text representation"
        : markdown_normalization
            ? "normalized Markdown representation"
            : "DOCX artifact generated from a Markdown resource";
    output.metadata.content_summary = document_json
        ? "Structured intermediate representation derived from the authoritative document resource."
        : text_forward
        ? "Text extracted from the authoritative DOCX resource."
        : markdown_normalization
            ? "Markdown derived from the authoritative source resource."
            : "DOCX artifact generated from the authoritative Markdown resource.";
    output.metadata.usage_hint = document_json
        ? "Pass to the host-owned document/table normalizer; do not treat this intermediate JSON as a dataset itself."
        : text_forward
        ? "Read and chunk this derived text resource."
        : markdown_normalization
            ? "Read and chunk this derived Markdown resource."
            : "Download or export this derived artifact resource.";
    output.metadata.limitations = document_json
        ? "Pandoc JSON preserves document structure but does not by itself infer table semantics or create datasets."
        : text_forward
        ? "Pandoc plain-text conversion may not preserve complete document layout or complex tables."
        : markdown_normalization
            ? "Pandoc Markdown conversion may not preserve complete document layout or complex tables."
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
