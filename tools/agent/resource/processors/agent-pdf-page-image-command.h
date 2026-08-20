#pragma once

#include "agent-resource-backend.h"
#include "resource/resource-contract.h"
#include "agent/sandbox/sandbox-contract.h"

#include <cstdint>
#include <string>

struct agent_pdf_page_image_options {
    size_t page = 0; // PDF page numbers are one-based.
    uint32_t dpi = 150;
    std::string format = "png";
    std::string colorspace = "rgb";
    size_t max_width = 0;
    size_t max_height = 0;
    size_t max_output_bytes = 4 * 1024 * 1024;
};

bool validate_agent_pdf_page_image_options(
        const agent_pdf_page_image_options & options,
        std::string & error);

bool make_agent_pdf_page_image_request(
        agent_resource_backend_kind backend,
        const std::string & executable,
        const std::string & operation_id,
        const std::string & project_id,
        const std::string & workspace_id,
        const common_runtime_resource_ref & source,
        const agent_pdf_page_image_options & options,
        common_agent_sandbox_request & request,
        std::string & error);
