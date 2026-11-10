#pragma once

#include <algorithm>
#include <cstddef>
#include <string_view>

// Return the largest prefix within max_bytes that ends before a UTF-8
// continuation byte. Callers supply valid UTF-8 from model-facing text.
inline size_t common_utf8_prefix_bytes(std::string_view text, size_t max_bytes) {
    size_t end = std::min(text.size(), max_bytes);
    while (end < text.size() && end > 0 &&
            (static_cast<unsigned char>(text[end]) & 0xc0u) == 0x80u) --end;
    return end;
}

// JSON Schema minLength/maxLength count Unicode code points, not UTF-8 bytes.
// JSON parsing has already validated the string before this is called.
inline size_t common_utf8_codepoint_count(std::string_view text) {
    return std::count_if(text.begin(), text.end(), [](char byte) {
        return (static_cast<unsigned char>(byte) & 0xc0u) != 0x80u;
    });
}
