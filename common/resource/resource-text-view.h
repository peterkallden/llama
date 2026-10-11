#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

// A structural, host-owned view of a text representation.  It deliberately
// does not rank or select evidence: it only gives bounded consumers stable
// paragraph/sentence units and maps them back to the original bytes.
struct common_runtime_resource_text_segment {
    std::string ref;
    std::string text;
    size_t byte_offset = 0;
    size_t byte_length = 0;
};

struct common_runtime_resource_text_view {
    std::string text;
    std::vector<size_t> source_offsets;
    std::vector<common_runtime_resource_text_segment> segments;
};

inline std::string common_runtime_resource_text_normalize(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t index = 0; index < text.size(); ++index) {
        if (text[index] == '\r') {
            if (index + 1 < text.size() && text[index + 1] == '\n') ++index;
            out.push_back('\n');
        } else {
            out.push_back(text[index]);
        }
    }
    return out;
}

inline bool common_runtime_resource_make_text_view(
        const std::string & source,
        common_runtime_resource_text_view & out,
        std::string & error) {
    out = {};
    out.text.reserve(source.size());
    out.source_offsets.reserve(source.size() + 1);
    for (size_t index = 0; index < source.size(); ++index) {
        out.source_offsets.push_back(index);
        if (source[index] == '\r' && index + 1 < source.size() && source[index + 1] == '\n') {
            out.text.push_back('\n');
            ++index;
        } else {
            out.text.push_back(source[index]);
        }
    }
    out.source_offsets.push_back(source.size());
    size_t begin = 0;
    size_t paragraph = 0;
    while (begin < out.text.size()) {
        size_t end = out.text.find("\n\n", begin);
        if (end == std::string::npos) end = out.text.size();
        const auto first = out.text.find_first_not_of(" \t\r\n", begin);
        const auto last = out.text.find_last_not_of(" \t\r\n", end == 0 ? 0 : end - 1);
        if (first != std::string::npos && first < end && first <= last) {
            const size_t length = last - first + 1;
            const size_t source_begin = out.source_offsets[first];
            const size_t source_end = out.source_offsets[last + 1];
            out.segments.push_back({
                "P" + std::to_string(++paragraph),
                out.text.substr(first, length),
                source_begin,
                source_end - source_begin});
        }
        if (end == out.text.size()) break;
        begin = end + 2;
    }
    if (out.text.empty()) {
        error.clear();
        return true;
    }
    if (out.segments.empty()) {
        out.segments.push_back({"P1", out.text, 0, source.size()});
    }
    error.clear();
    return true;
}

inline std::string common_runtime_resource_render_text_segments(
        const common_runtime_resource_text_view & view,
        size_t byte_offset,
        size_t byte_length,
        std::vector<common_runtime_resource_text_segment> & selected) {
    selected.clear();
    const size_t end = byte_offset + byte_length;
    std::string rendered;
    for (const auto & segment : view.segments) {
        const size_t segment_end = segment.byte_offset + segment.byte_length;
        if (segment_end <= byte_offset || segment.byte_offset >= end) continue;
        selected.push_back(segment);
        if (!rendered.empty()) rendered += "\n";
        rendered += segment.text;
    }
    return rendered;
}
