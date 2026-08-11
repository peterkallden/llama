#pragma once

#include "agent-resource-backend.h"
#include "resource/resource-contract.h"
#include "agent/sandbox-contract.h"

#include <string>

struct agent_pandoc_docx_options {
    std::string output_format = "text";
    size_t max_output_bytes = 4 * 1024 * 1024;
};

bool validate_agent_pandoc_docx_options(
        const agent_pandoc_docx_options & options,
        std::string & error);

bool make_agent_pandoc_docx_request(
        agent_resource_backend_kind backend,
        const std::string & executable,
        const std::string & operation_id,
        const std::string & project_id,
        const std::string & workspace_id,
        const common_runtime_resource_ref & source,
        const agent_pandoc_docx_options & options,
        common_agent_sandbox_request & request,
        std::string & error);
