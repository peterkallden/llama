#pragma once

#include "agent-pdf-page-image-command.h"
#include "tools/agent/resource/agent-resource-processing-host.h"

// Operation-bound PDF page-image processor. The processor owns PDF-specific
// typed options and output lineage; the host owns execution placement and
// artifact collection.
class agent_pdf_page_image_processor final : public agent_resource_processor {
public:
    agent_pdf_page_image_processor(
            agent_resource_processing_host & host,
            agent_resource_processing_execution_context context,
            agent_resource_backend_kind backend,
            std::string executable)
        : host_(host),
          context_(std::move(context)),
          backend_(backend),
          executable_(std::move(executable)) {}

    std::string id() const override;

    bool supports(
            const std::string & mime_type,
            const std::string & target_representation) const override;

    agent_resource_processing_result process(
            const agent_resource_processing_request & request) const override;

private:
    agent_resource_processing_host & host_;
    agent_resource_processing_execution_context context_;
    agent_resource_backend_kind backend_;
    std::string executable_;
};
