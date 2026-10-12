#pragma once

#include "resource/resource-contract.h"

// Bounded local processor for PDFs that contain a directly extractable text
// layer. It deliberately does not render pages or perform OCR; those are
// separate representations/providers behind the same processor contract.
class agent_pdf_text_processor final : public agent_resource_processor {
public:
    std::string id() const override;

    bool supports(
            const std::string & mime_type,
            const std::string & target_representation) const override;

    agent_resource_processing_result process(
            const agent_resource_processing_request & request) const override;
};
