#include "agent-pdf-page-image-processor.h"

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

} // namespace

std::string agent_pdf_page_image_processor::id() const {
    return "pdf-page-image-v1";
}

bool agent_pdf_page_image_processor::supports(
        const std::string & mime_type,
        const std::string & target_representation) const {
    return mime_type == "application/pdf" && target_representation == "page-image";
}

agent_resource_processing_result agent_pdf_page_image_processor::process(
        const agent_resource_processing_request & request) const {
    if (!request.page || *request.page == 0) {
        return failure(id(), "resource.invalid_request", "PDF page-image processing requires a one-based page.");
    }
    if (executable_.empty()) {
        return failure(id(), "resource.processor_unavailable", "The configured PDF renderer is unavailable.");
    }

    agent_pdf_page_image_options options;
    options.page = *request.page;
    if (request.limits.max_output_bytes > 0) {
        options.max_output_bytes = request.limits.max_output_bytes;
    }
    common_agent_sandbox_request sandbox_request;
    std::string error;
    if (!make_agent_pdf_page_image_request(
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
                ? "The PDF renderer did not complete successfully."
                : sandbox_result.error);
    }
    if (artifacts.size() != 1 || artifacts.front().bytes.empty()) {
        return failure(id(), "resource.output_invalid", "The PDF renderer did not return one bounded page image.");
    }

    const auto & artifact = artifacts.front();
    agent_resource_processing_result result;
    result.success = true;
    result.processor_id = id();
    result.safe_summary = "PDF page image rendered through the host resource processor.";
    agent_resource_processing_output output;
    output.name = artifact.name.empty()
        ? "page-" + std::to_string(*request.page) + ".png"
        : artifact.name;
    output.description = "Derived PDF page image";
    output.mime_type = artifact.mime_type.empty() ? "image/png" : artifact.mime_type;
    output.bytes = artifact.bytes;
    output.metadata.purpose = "derived PDF page image";
    output.metadata.content_summary = "Bounded rendered page image";
    output.metadata.usage_hint = "Inspect this image as a visual representation of the source PDF page.";
    output.metadata.limitations = "Derived representation; the original PDF remains authoritative.";
    output.lineage.parent_uri = request.source.uri;
    output.lineage.chunk_index = 0;
    output.lineage.chunk_count = 1;
    output.lineage.byte_offset = 0;
    output.lineage.byte_length = request.source.size_bytes;
    output.lineage.derivation = "resource.process:" + id();
    result.outputs.push_back(std::move(output));
    return result;
}
