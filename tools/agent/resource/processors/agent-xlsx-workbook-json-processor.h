#pragma once

#include "agent-resource-backend.h"
#include "resource/resource-contract.h"
#include "tools/agent/resource/agent-resource-processing-host.h"

// Host-owned XLSX ingestion. The implementation executable is an external
// processor; the agent sees only a bounded workbook envelope and derived
// resource provenance. Pandoc is intentionally not used for XLSX input.
class agent_xlsx_workbook_json_processor final : public agent_resource_processor {
public:
    agent_xlsx_workbook_json_processor(
            agent_resource_processing_host & host,
            agent_resource_processing_execution_context context,
            agent_resource_backend_kind backend,
            std::string executable,
            std::string script)
        : host_(host), context_(std::move(context)), backend_(backend),
          executable_(std::move(executable)), script_(std::move(script)) {}

    std::string id() const override { return "xlsx-workbook-json-v1"; }
    std::string cache_key() const override { return id() + ";script=" + script_; }

    bool supports(const std::string & mime_type,
                  const std::string & target_representation) const override;
    support supports(const agent_resource_processing_request & request) const override;
    agent_resource_processing_result process(
            const agent_resource_processing_request & request) const override;

private:
    agent_resource_processing_host & host_;
    agent_resource_processing_execution_context context_;
    agent_resource_backend_kind backend_;
    std::string executable_;
    std::string script_;
};
