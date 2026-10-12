#pragma once

#include "agent-resource-backend.h"
#include "resource/resource-contract.h"
#include "agent/sandbox/sandbox-contract.h"

#include <cstdint>
#include <string>

struct agent_tesseract_ocr_options {
    std::string language = "eng";
    std::string fallback_language;
    uint32_t oem = 3;
    uint32_t psm = 3;
    std::string output_format = "text";
    size_t max_output_bytes = 4 * 1024 * 1024;
};

bool validate_agent_tesseract_ocr_options(
        const agent_tesseract_ocr_options & options,
        std::string & error);

bool make_agent_tesseract_ocr_request(
        agent_resource_backend_kind backend,
        const std::string & executable,
        const std::string & operation_id,
        const std::string & project_id,
        const std::string & workspace_id,
        const common_runtime_resource_ref & source,
        const agent_tesseract_ocr_options & options,
        common_agent_sandbox_request & request,
        std::string & error);
