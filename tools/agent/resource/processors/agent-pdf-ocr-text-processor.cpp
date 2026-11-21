#include "agent-pdf-ocr-text-processor.h"

#include "agent-pdf-page-image-command.h"
#include "../agent-resource-text-normalize.h"

#include <algorithm>
#include <chrono>
#include <string>

namespace {

agent_resource_processing_result fail(std::string code, std::string summary) {
    agent_resource_processing_result result;
    result.failure_code = std::move(code);
    result.safe_summary = std::move(summary);
    result.processor_id = "pdf-ocr-text-v1";
    return result;
}

} // namespace

agent_pdf_ocr_text_processor::agent_pdf_ocr_text_processor(
        agent_resource_store & store,
        agent_resource_processing_host & page_host,
        agent_resource_processing_host & ocr_host,
        agent_resource_processing_execution_context context,
        std::string renderer_executable,
        std::string ocr_executable,
        size_t max_pages)
    : store_(store),
      page_host_(page_host),
      ocr_host_(ocr_host),
      context_(std::move(context)),
      renderer_executable_(std::move(renderer_executable)),
      ocr_executable_(std::move(ocr_executable)),
      max_pages_(max_pages) {}

std::string agent_pdf_ocr_text_processor::id() const {
    return "pdf-ocr-text-v1";
}

std::string agent_pdf_ocr_text_processor::cache_key() const {
    return id() + ";renderer=" + renderer_executable_ +
        ";ocr=" + ocr_executable_ +
        ";max_pages=" + std::to_string(max_pages_);
}

bool agent_pdf_ocr_text_processor::supports(
        const std::string & mime_type,
        const std::string & target_representation) const {
    return common_normalize_resource_media_type(mime_type) == "application/pdf" &&
        target_representation == "text";
}

agent_resource_processing_result agent_pdf_ocr_text_processor::process(
        const agent_resource_processing_request & request) const {
    agent_pdf_text_processor direct;
    auto direct_result = direct.process(request);
    if (direct_result.success || direct_result.failure_code != "resource.text_representation_unavailable") {
        direct_result.processor_id = id();
        return direct_result;
    }
    if (max_pages_ == 0 || renderer_executable_.empty() || ocr_executable_.empty()) {
        return fail("resource.processor_unavailable",
            "Direct PDF text is unavailable and the host has no complete PDF OCR policy.");
    }

    auto count_context = context_;
    count_context.operation_id += "/pdf-page-count";
    common_agent_sandbox_request count_request;
    std::string error;
    if (!make_agent_pdf_page_count_request(
            renderer_executable_, count_context.operation_id,
            count_context.workspace.project_id, count_context.workspace.workspace_id,
            request.source, count_request, error)) {
        return fail("resource.page_count_unavailable", std::move(error));
    }
    common_agent_sandbox_result count_result;
    std::vector<agent_resource_processing_host_artifact> no_artifacts;
    if (!page_host_.execute(count_context, std::move(count_request), count_result, no_artifacts, error) ||
            count_result.status != common_agent_sandbox_status::completed) {
        return fail("resource.page_count_unavailable",
            error.empty() ? count_result.error : std::move(error));
    }
    size_t page_count = 0;
    if (!parse_agent_pdf_page_count(count_result.stdout_excerpt, page_count, error)) {
        return fail("resource.page_count_unavailable", std::move(error));
    }

    const size_t pages_to_process = std::min(page_count, max_pages_);
    const size_t max_output_bytes = request.limits.max_output_bytes > 0
        ? std::min<size_t>(request.limits.max_output_bytes, 4 * 1024 * 1024)
        : 1024 * 1024;
    std::string text;
    size_t processed_pages = 0;
    bool partial = page_count > pages_to_process;
    std::string partial_reason = partial ? "configured page limit reached" : "";

    for (size_t page = 1; page <= pages_to_process; ++page) {
        auto page_context = context_;
        page_context.operation_id += "/pdf-page-" + std::to_string(page);
        agent_pdf_page_image_processor renderer(
            page_host_, page_context, agent_resource_backend_kind::local_mupdf,
            renderer_executable_);
        agent_resource_processing_request render_request = request;
        render_request.target_representation = "page-image";
        render_request.page = page;
        render_request.limits.max_output_bytes = 4 * 1024 * 1024;
        const auto rendered = renderer.process(render_request);
        if (!rendered.success || rendered.outputs.size() != 1) {
            partial = true;
            partial_reason = rendered.safe_summary.empty()
                ? "PDF page rendering failed at page " + std::to_string(page)
                : "PDF page rendering failed at page " + std::to_string(page) + ": " + rendered.safe_summary;
            break;
        }

        const auto & image_output = rendered.outputs.front();
        agent_resource_put_request image_put;
        image_put.name = image_output.name;
        image_put.description = image_output.description;
        image_put.mime_type = image_output.mime_type;
        image_put.bytes = image_output.bytes;
        image_put.scope = request.source.scope;
        image_put.namespace_id = request.authority.namespace_id;
        image_put.session_id = request.authority.session_id;
        image_put.project_id = request.authority.project_id;
        image_put.turn_id = request.authority.turn_id;
        image_put.source_provider = "resource_processor";
        image_put.source_tool = "pdf-page-image-v1";
        image_put.metadata = image_output.metadata;
        image_put.lineage = image_output.lineage;
        agent_resource_descriptor image_descriptor;
        if (!store_.put_bytes(image_put, image_descriptor, error)) {
            partial = true;
            partial_reason = "rendered page could not be registered at page " + std::to_string(page);
            break;
        }

        auto ocr_context = context_;
        ocr_context.operation_id += "/ocr-page-" + std::to_string(page);
        agent_tesseract_ocr_processor ocr(
            ocr_host_, ocr_context, agent_resource_backend_kind::local_tesseract,
            ocr_executable_);
        agent_resource_processing_request ocr_request;
        ocr_request.source = image_descriptor;
        ocr_request.authority = request.authority;
        ocr_request.media_type.declared_type = image_descriptor.mime_type;
        ocr_request.media_type.resolved_type = image_descriptor.mime_type;
        ocr_request.media_type.content_verified = true;
        ocr_request.target_representation = "text";
        ocr_request.limits.max_output_bytes = max_output_bytes;
        const auto recognized = ocr.process(ocr_request);
        if (!recognized.success || recognized.outputs.size() != 1) {
            partial = true;
            partial_reason = "OCR failed at page " + std::to_string(page);
            break;
        }

        const auto page_text = agent_resource_normalize_utf8(recognized.outputs.front().bytes);
        const std::string page_heading = "\n--- PDF page " + std::to_string(page) + " ---\n";
        if (text.size() + page_heading.size() + page_text.size() > max_output_bytes) {
            partial = true;
            partial_reason = "configured text output limit reached at page " + std::to_string(page);
            break;
        }
        text += page_heading;
        text += page_text;
        ++processed_pages;
    }

    if (text.empty()) {
        return fail("resource.text_representation_unavailable",
            partial_reason.empty() ? "PDF OCR produced no readable text." : partial_reason);
    }

    agent_resource_processing_result result;
    result.success = true;
    result.processor_id = id();
    result.safe_summary = partial
        ? "PDF OCR returned partial text; " + partial_reason
        : "PDF OCR extracted text from all pages.";
    agent_resource_processing_output output;
    output.name = request.source.name + ".ocr.txt";
    output.description = "Text extracted from rendered PDF pages using OCR.";
    output.mime_type = "text/plain";
    output.bytes = std::move(text);
    output.metadata.purpose = "OCR text representation of a PDF";
    output.metadata.content_summary = partial
        ? "Partial OCR text; page coverage is recorded in limitations."
        : "OCR text for all pages in the PDF.";
    output.metadata.usage_hint = "OCR is derived evidence; verify claims against the source PDF when possible.";
    output.metadata.limitations = partial
        ? "Partial OCR coverage: processed " + std::to_string(processed_pages) +
            " of " + std::to_string(page_count) + " pages; " + partial_reason + "."
        : "OCR-derived text; processed " + std::to_string(processed_pages) +
            " of " + std::to_string(page_count) + " pages.";
    output.lineage.parent_uri = request.source.uri;
    output.lineage.chunk_index = 0;
    output.lineage.chunk_count = 1;
    output.lineage.byte_offset = 0;
    output.lineage.byte_length = request.source.size_bytes;
    output.lineage.derivation = "resource.process:pdf-page-image-v1>tesseract-ocr-v1";
    result.outputs.push_back(std::move(output));
    return result;
}
