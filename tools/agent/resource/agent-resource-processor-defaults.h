#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Built-in document processors use bounded host-owned artifact workspaces.
// Keep elevated limits scoped by execution class; unrelated sandbox commands
// retain the generic limits.
inline constexpr size_t AGENT_RESOURCE_PROCESSOR_DEFAULT_OUTPUT_BYTES = 4 * 1024 * 1024;
inline constexpr uint32_t AGENT_RESOURCE_PROCESSOR_DEFAULT_TIMEOUT_MS = 180000;
inline constexpr size_t AGENT_RESOURCE_PROCESSOR_PANDOC_OUTPUT_BYTES = 8 * 1024 * 1024;

inline size_t agent_resource_processor_default_sandbox_output_bytes(const std::string & execution_class) {
    if (execution_class == "resource.processor.pdf.page_image" ||
            execution_class == "resource.processor.ocr.tesseract") {
        return AGENT_RESOURCE_PROCESSOR_DEFAULT_OUTPUT_BYTES;
    }
    if (execution_class == "resource.processor.pandoc") {
        return AGENT_RESOURCE_PROCESSOR_PANDOC_OUTPUT_BYTES;
    }
    return 0;
}

inline uint32_t agent_resource_processor_default_sandbox_timeout_ms(const std::string & execution_class) {
    if (execution_class == "resource.processor.pdf.page_image" ||
            execution_class == "resource.processor.ocr.tesseract" ||
            execution_class == "resource.processor.pandoc") {
        return AGENT_RESOURCE_PROCESSOR_DEFAULT_TIMEOUT_MS;
    }
    return 0;
}

inline bool agent_resource_processor_default_sandbox_artifact_write(const std::string & execution_class) {
    return execution_class == "resource.processor.pandoc";
}
