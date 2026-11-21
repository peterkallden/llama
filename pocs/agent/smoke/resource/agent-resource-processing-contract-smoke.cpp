#include "resource/resource-contract.h"

#include <cstdio>
#include <string>
#include <utility>

namespace {

class fake_processor : public agent_resource_processor {
public:
    fake_processor(
            std::string processor_id,
            std::string mime_type,
            std::string representation)
        : processor_id_(std::move(processor_id)),
          mime_type_(std::move(mime_type)),
          representation_(std::move(representation)) {}

    std::string id() const override {
        return processor_id_;
    }

    bool supports(
            const std::string & mime_type,
            const std::string & target_representation) const override {
        return mime_type == mime_type_ && target_representation == representation_;
    }

    agent_resource_processing_result process(
            const agent_resource_processing_request & request) const override {
        agent_resource_processing_result result;
        result.processor_id = processor_id_;
        if (!supports(request.media_type.resolved_type, request.target_representation)) {
            result.failure_code = "resource.unsupported_media_type";
            result.safe_summary = "Resource representation is unsupported.";
            return result;
        }

        agent_resource_processing_output derived;
        derived.name = request.source.name + " (" + representation_ + ")";
        derived.mime_type = representation_ == "text" ? "text/plain" : "application/octet-stream";
        derived.bytes = "derived:" + representation_;
        derived.lineage.parent_uri = request.source.uri;
        derived.lineage.chunk_index = 0;
        derived.lineage.chunk_count = 1;
        derived.lineage.byte_offset = request.range ? request.range->offset : 0;
        derived.lineage.byte_length = request.range ? request.range->max_bytes : request.source.size_bytes;
        derived.lineage.derivation = "resource.process:" + processor_id_;

        result.success = true;
        result.safe_summary = "Derived resource representation created.";
        result.outputs.push_back(std::move(derived));
        return result;
    }

private:
    std::string processor_id_;
    std::string mime_type_;
    std::string representation_;
};

} // namespace

int main() {
    std::string error;
    agent_resource_processor_registry registry;

    fake_processor pdf_text("pdf-text-v1", "application/pdf", "text");
    fake_processor pdf_page_image("pdf-page-image-v1", "application/pdf", "page-image");
    fake_processor fallback_text("fallback-text-v1", "text/plain", "text");
    fake_processor duplicate_pdf_text("pdf-text-v1", "application/pdf", "text");

    if (!registry.add(pdf_text, error) ||
            !registry.add(pdf_page_image, error) ||
            !registry.add(fallback_text, error)) {
        std::fprintf(stderr, "processor registration failed: %s\n", error.c_str());
        return 1;
    }
    if (registry.add(duplicate_pdf_text, error) || error.empty()) {
        std::fprintf(stderr, "duplicate processor id was accepted\n");
        return 1;
    }

    const auto * resolved_pdf_text = registry.resolve("application/pdf", "text");
    const auto * resolved_pdf_page = registry.resolve("application/pdf", "page-image");
    const auto * unresolved_ocr = registry.resolve("image/png", "text");
    if (resolved_pdf_text == nullptr || resolved_pdf_text->id() != "pdf-text-v1" ||
            resolved_pdf_page == nullptr || resolved_pdf_page->id() != "pdf-page-image-v1" ||
            unresolved_ocr != nullptr) {
        std::fprintf(stderr, "processor registry did not resolve deterministically\n");
        return 1;
    }

    agent_resource_processing_request request;
    request.source.uri = "agent-resource://resource/report";
    request.source.name = "report.pdf";
    request.source.mime_type = "application/pdf";
    request.source.size_bytes = 4096;
    request.source.scope = common_runtime_resource_scope::session;
    request.authority.namespace_id = "tenant-a";
    request.authority.session_id = "session-1";
    request.media_type.declared_type = "application/octet-stream";
    request.media_type.resolved_type = "application/pdf";
    request.media_type.content_verified = true;
    request.target_representation = "text";
    request.range = agent_resource_byte_range{128, 512};
    request.limits.max_source_bytes = 1024 * 1024;
    request.limits.max_output_bytes = 64 * 1024;
    request.limits.max_generated_resources = 4;
    request.limits.max_duration_ms = 30000;

    auto result = resolved_pdf_text->process(request);
    if (!result.success ||
            result.processor_id != "pdf-text-v1" ||
            result.outputs.size() != 1 ||
            result.outputs[0].lineage.parent_uri != request.source.uri ||
            result.outputs[0].lineage.byte_offset != 128 ||
            result.outputs[0].lineage.byte_length != 512 ||
            result.outputs[0].lineage.derivation != "resource.process:pdf-text-v1") {
        std::fprintf(stderr, "processor result did not preserve staged output lineage\n");
        return 1;
    }

    request.target_representation = "page-image";
    auto page_result = resolved_pdf_page->process(request);
    if (!page_result.success ||
            page_result.outputs.empty() ||
            page_result.outputs[0].mime_type != "application/octet-stream") {
        std::fprintf(stderr, "page-image processor result was unexpected\n");
        return 1;
    }

    std::printf("processors=%zu\n", registry.size());
    std::printf("derived_output=%s\n", result.outputs[0].name.c_str());
    std::printf("derived_lineage=%s\n", result.outputs[0].lineage.derivation.c_str());
    return 0;
}
