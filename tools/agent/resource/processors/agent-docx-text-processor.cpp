#include "agent-docx-text-processor.h"

#include <utility>

namespace {

constexpr const char * docx_mime =
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

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

std::string agent_docx_text_processor::id() const {
    return "pandoc-docx-text-v1";
}

std::string agent_docx_text_processor::cache_key() const {
    return id() + ";format=" + options_.output_format;
}

bool agent_docx_text_processor::supports(
        const std::string & mime_type,
        const std::string & target_representation) const {
    return common_normalize_resource_media_type(mime_type) == docx_mime &&
        target_representation == "text";
}

agent_resource_processing_result agent_docx_text_processor::process(
        const agent_resource_processing_request & request) const {
    if (!supports(request.source.mime_type, request.target_representation)) {
        return failure(id(), "resource.unsupported_media_type", "Pandoc DOCX processing requires a DOCX source and text representation.");
    }
    if (!has_docx_package_signature(request.source_bytes)) {
        return failure(id(), "resource.invalid_document", "The source is not a recognizable DOCX package.");
    }
    if (executable_.empty()) {
        return failure(id(), "resource.processor_unavailable", "The configured Pandoc executable is unavailable.");
    }

    agent_pandoc_docx_options options = options_;
    if (request.limits.max_output_bytes > 0) options.max_output_bytes = request.limits.max_output_bytes;
    common_agent_sandbox_request sandbox_request;
    std::string error;
    if (!make_agent_pandoc_docx_request(
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
        return failure(id(), "resource.output_invalid", "Pandoc did not return one bounded DOCX text result.");
    }

    const auto & artifact = artifacts.front();
    agent_resource_processing_result result;
    result.success = true;
    result.processor_id = id();
    result.safe_summary = "DOCX text extracted through the host resource processor.";
    agent_resource_processing_output output;
    output.name = artifact.name.empty() ? request.source.name + ".txt" : artifact.name;
    output.description = "Normalized text representation extracted from a DOCX document.";
    output.mime_type = "text/plain";
    output.bytes = artifact.bytes;
    output.metadata.purpose = "normalized DOCX text representation";
    output.metadata.content_summary = "Text extracted from the authoritative DOCX resource.";
    output.metadata.usage_hint = "Read and chunk this derived text resource.";
    output.metadata.limitations = "Pandoc plain-text conversion may not preserve complete document layout or complex tables.";
    output.lineage.parent_uri = request.source.uri;
    output.lineage.chunk_index = 0;
    output.lineage.chunk_count = 1;
    output.lineage.byte_offset = 0;
    output.lineage.byte_length = request.source.size_bytes;
    output.lineage.derivation = "resource.process:" + id();
    result.outputs.push_back(std::move(output));
    return result;
}
