#include "common/resource/resource-chunker.h"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    const std::string text =
        "Heading\n\n"
        "First paragraph contains the stable project decision.\n\n"
        "| Name | Status |\n"
        "|------|--------|\n"
        "| CPU  | ready  |\n"
        "| CUDA | ready  |\n\n"
        "The final paragraph explains the verification result.";

    std::vector<common_runtime_resource_chunk> chunks;
    std::string error;
    if (!common_runtime_resource_chunk_text(text, {96, 8}, chunks, error)) {
        std::fprintf(stderr, "chunking failed: %s\n", error.c_str());
        return 1;
    }
    if (chunks.size() < 2 || chunks.front().byte_offset != 0 ||
            chunks.front().boundary != "paragraph") {
        std::fprintf(stderr, "paragraph boundary was not selected\n");
        return 1;
    }
    bool saw_table_row = false;
    for (size_t index = 0; index < chunks.size(); ++index) {
        const auto & chunk = chunks[index];
        if (chunk.boundary == "table_row") saw_table_row = true;
        if (chunk.byte_length == 0 || chunk.byte_offset + chunk.byte_length > text.size() ||
                (index > 0 && chunk.overlap_bytes == 0)) {
            std::fprintf(stderr, "chunk metadata is inconsistent\n");
            return 1;
        }
    }
    const std::string table_text =
        "| Name | Status |\n"
        "|------|--------|\n"
        "| CPU  | ready  |\n"
        "| CUDA | ready  |\n";
    if (!common_runtime_resource_chunk_text(table_text, {40, 4}, chunks, error)) {
        std::fprintf(stderr, "table chunking failed: %s\n", error.c_str());
        return 1;
    }
    for (const auto & chunk : chunks) {
        saw_table_row = saw_table_row || chunk.boundary == "table_row";
    }
    if (!saw_table_row) {
        std::fprintf(stderr, "table row boundary was not selected\n");
        return 1;
    }
    const size_t table_chunk_count = chunks.size();
    if (common_runtime_resource_chunk_text(text, {0, 8}, chunks, error) ||
            error != "resource chunk max_bytes must be non-zero") {
        std::fprintf(stderr, "invalid max_bytes was not rejected\n");
        return 1;
    }
    if (common_runtime_resource_chunk_text(text, {16, 16}, chunks, error) ||
            error != "resource chunk overlap_bytes must be smaller than max_bytes") {
        std::fprintf(stderr, "invalid overlap was not rejected\n");
        return 1;
    }
    std::printf("resource_chunks=%zu\n", table_chunk_count);
    return 0;
}
