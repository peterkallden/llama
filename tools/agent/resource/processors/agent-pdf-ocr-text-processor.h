#pragma once

#include "agent-pdf-page-image-processor.h"
#include "agent-pdf-text-processor.h"
#include "agent-tesseract-ocr-processor.h"
#include "tools/agent/resource/agent-resource-store.h"

// Host-owned lazy fallback for PDF text reads. Direct text extraction is
// attempted first; rendering and OCR are used only when direct text is absent.
class agent_pdf_ocr_text_processor final : public agent_resource_processor {
public:
    agent_pdf_ocr_text_processor(
            agent_resource_store & store,
            agent_resource_processing_host & page_host,
            agent_resource_processing_host & ocr_host,
            agent_resource_processing_execution_context context,
            std::string renderer_executable,
            std::string ocr_executable,
            size_t max_pages);

    std::string id() const override;
    std::string cache_key() const override;
    bool supports(const std::string & mime_type, const std::string & target_representation) const override;
    agent_resource_processing_result process(
        const agent_resource_processing_request & request) const override;

private:
    agent_resource_store & store_;
    agent_resource_processing_host & page_host_;
    agent_resource_processing_host & ocr_host_;
    agent_resource_processing_execution_context context_;
    std::string renderer_executable_;
    std::string ocr_executable_;
    size_t max_pages_ = 0;
};
