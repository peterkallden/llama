#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

struct common_runtime_resource_chunk {
    std::string text;
    size_t byte_offset = 0;
    size_t byte_length = 0;
    size_t overlap_bytes = 0;
    std::string boundary;
};

struct common_runtime_resource_chunk_policy {
    size_t max_bytes = 4096;
    size_t overlap_bytes = 256;
};

namespace common_runtime_resource_chunk_detail {

inline bool is_utf8_continuation(unsigned char byte) {
    return (byte & 0xc0u) == 0x80u;
}

inline size_t safe_boundary(const std::string & text, size_t position) {
    position = std::min(position, text.size());
    while (position > 0 && position < text.size() &&
            is_utf8_continuation(static_cast<unsigned char>(text[position]))) {
        --position;
    }
    return position;
}

inline size_t line_start(const std::string & text, size_t position) {
    const size_t newline = text.rfind('\n', position == 0 ? 0 : position - 1);
    return newline == std::string::npos ? 0 : newline + 1;
}

inline bool looks_like_table_row(const std::string & text, size_t position) {
    const size_t start = line_start(text, position);
    const size_t end = text.find('\n', start);
    const std::string row = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
    return row.find('|') != std::string::npos &&
        (row.find_first_not_of(" \t") != std::string::npos &&
         row[row.find_first_not_of(" \t")] == '|');
}

inline size_t last_boundary(
        const std::string & text,
        size_t begin,
        size_t limit,
        const char * & kind) {
    size_t paragraph = std::string::npos;
    size_t table_row = std::string::npos;
    size_t sentence = std::string::npos;
    size_t line = std::string::npos;
    for (size_t position = begin; position < limit; ++position) {
        if (text[position] != '\n') {
            if (text[position] == '.' || text[position] == '!' || text[position] == '?') {
                size_t next = position + 1;
                while (next < limit && (text[next] == ' ' || text[next] == '\t')) ++next;
                if (next < limit && text[next] == '\n') ++next;
                if (next <= limit) sentence = next;
            }
            continue;
        }
        line = position + 1;
        if (looks_like_table_row(text, position + 1)) {
            table_row = position + 1;
        }
        if (position > begin && text[position - 1] == '\n') {
            paragraph = position + 1;
        }
    }
    if (paragraph != std::string::npos) {
        kind = "paragraph";
        return paragraph;
    }
    if (table_row != std::string::npos) {
        kind = "table_row";
        return table_row;
    }
    if (sentence != std::string::npos) {
        kind = "sentence";
        return sentence;
    }
    if (line != std::string::npos) {
        kind = "line";
        return line;
    }
    kind = "hard_limit";
    return safe_boundary(text, limit);
}

} // namespace common_runtime_resource_chunk_detail

inline bool common_runtime_resource_chunk_text(
        const std::string & text,
        const common_runtime_resource_chunk_policy & policy,
        std::vector<common_runtime_resource_chunk> & chunks,
        std::string & error) {
    chunks.clear();
    if (policy.max_bytes == 0) {
        error = "resource chunk max_bytes must be non-zero";
        return false;
    }
    if (policy.overlap_bytes >= policy.max_bytes) {
        error = "resource chunk overlap_bytes must be smaller than max_bytes";
        return false;
    }
    if (text.empty()) {
        error.clear();
        return true;
    }

    size_t start = 0;
    while (start < text.size()) {
        const size_t limit = std::min(start + policy.max_bytes, text.size());
        const char * boundary_kind = "document";
        size_t end = limit;
        if (limit < text.size()) {
            end = common_runtime_resource_chunk_detail::last_boundary(
                text, start, limit, boundary_kind);
            if (end <= start) {
                end = common_runtime_resource_chunk_detail::safe_boundary(text, limit);
                boundary_kind = "hard_limit";
            }
        }
        if (end <= start) {
            error = "resource chunker could not make progress";
            chunks.clear();
            return false;
        }

        const size_t overlap = chunks.empty()
            ? 0 : std::min(policy.overlap_bytes, end - start);
        const size_t payload_start = start - overlap;
        common_runtime_resource_chunk chunk;
        chunk.text = text.substr(payload_start, end - payload_start);
        chunk.byte_offset = payload_start;
        chunk.byte_length = chunk.text.size();
        chunk.overlap_bytes = overlap;
        chunk.boundary = boundary_kind;
        chunks.push_back(std::move(chunk));

        start = end;
    }

    error.clear();
    return true;
}
