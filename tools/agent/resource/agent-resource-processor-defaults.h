#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// PDF rendering and OCR are bounded host-owned processors that need more
// room than generic sandbox commands. Keep request defaults scoped by class;
// host policy must allow these bounds but does not change the request itself.
inline constexpr size_t AGENT_RESOURCE_PROCESSOR_DEFAULT_OUTPUT_BYTES = 4 * 1024 * 1024;
inline constexpr uint32_t AGENT_RESOURCE_PROCESSOR_DEFAULT_TIMEOUT_MS = 180000;

inline size_t agent_resource_processor_default_sandbox_output_bytes(const std::string & execution_class) {
    if (execution_class == "resource.processor.pdf.page_image" ||
            execution_class == "resource.processor.ocr.tesseract") {
        return AGENT_RESOURCE_PROCESSOR_DEFAULT_OUTPUT_BYTES;
    }
    return 0;
}

inline uint32_t agent_resource_processor_default_sandbox_timeout_ms(const std::string & execution_class) {
    if (execution_class == "resource.processor.pdf.page_image" ||
            execution_class == "resource.processor.ocr.tesseract") {
        return AGENT_RESOURCE_PROCESSOR_DEFAULT_TIMEOUT_MS;
    }
    return 0;
}
