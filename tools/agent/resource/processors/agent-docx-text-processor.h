#pragma once

#include "agent-pandoc-command.h"
#include "tools/agent/resource/agent-resource-processing-host.h"

// Operation-bound DOCX text processor. Pandoc is an external implementation;
// the host owns execution placement and artifact collection.
class agent_docx_text_processor final : public agent_resource_processor {
public:
    agent_docx_text_processor(
            agent_resource_processing_host & host,
            agent_resource_processing_execution_context context,
            agent_resource_backend_kind backend,
            std::string executable,
            agent_pandoc_docx_options options = {})
        : host_(host),
          context_(std::move(context)),
          backend_(backend),
          executable_(std::move(executable)),
          options_(std::move(options)) {}

    std::string id() const override;
    std::string cache_key() const override;

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
    agent_pandoc_docx_options options_;
};
