#include "agent-pdf-text-processor.h"

#include <cctype>

namespace {

bool has_pdf_signature(const std::string & bytes) {
    return bytes.size() >= 5 && bytes.compare(0, 5, "%PDF-") == 0;
}

std::string decode_pdf_literal(const std::string & literal) {
    std::string decoded;
    decoded.reserve(literal.size());
    for (size_t i = 0; i < literal.size(); ++i) {
        if (literal[i] != '\\' || i + 1 >= literal.size()) {
            decoded.push_back(literal[i]);
            continue;
        }
        const char escaped = literal[++i];
        switch (escaped) {
            case 'n': decoded.push_back('\n'); break;
            case 'r': decoded.push_back('\r'); break;
            case 't': decoded.push_back('\t'); break;
            case 'b': decoded.push_back('\b'); break;
            case 'f': decoded.push_back('\f'); break;
            case '\\': decoded.push_back('\\'); break;
            case '(' : decoded.push_back('('); break;
            case ')' : decoded.push_back(')'); break;
            default: decoded.push_back(escaped); break;
        }
    }
    return decoded;
}

bool extract_text_layer(
        const std::string & bytes,
        size_t max_output_bytes,
        std::string & text) {
    text.clear();
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (bytes[i] != '(') continue;
        std::string literal;
        bool escaped = false;
        bool closed = false;
        for (++i; i < bytes.size(); ++i) {
            const char ch = bytes[i];
            if (!escaped && ch == ')') {
                closed = true;
                break;
            }
            literal.push_back(ch);
            escaped = !escaped && ch == '\\';
            if (escaped && i + 1 < bytes.size()) {
                literal.push_back(bytes[++i]);
                escaped = false;
            }
        }
        if (!closed) break;
        const std::string decoded = decode_pdf_literal(literal);
        if (decoded.empty()) continue;
        if (!text.empty()) text.push_back('\n');
        if (text.size() + decoded.size() > max_output_bytes) return false;
        text += decoded;
    }
    return !text.empty();
}

size_t count_pdf_pages(const std::string & bytes) {
    constexpr char marker[] = "/Type /Page";
    size_t count = 0;
    for (size_t pos = 0; (pos = bytes.find(marker, pos)) != std::string::npos; pos += sizeof(marker) - 1) {
        ++count;
    }
    return count;
}

agent_resource_processing_result failure(
        const std::string & code,
        const std::string & summary) {
    agent_resource_processing_result result;
    result.failure_code = code;
    result.safe_summary = summary;
    result.processor_id = "pdf-text-local-v1";
    return result;
}

} // namespace

std::string agent_pdf_text_processor::id() const {
    return "pdf-text-local-v1";
}

bool agent_pdf_text_processor::supports(
        const std::string & mime_type,
        const std::string & target_representation) const {
    return common_normalize_resource_media_type(mime_type) == "application/pdf" &&
        target_representation == "text";
}

agent_resource_processing_result agent_pdf_text_processor::process(
        const agent_resource_processing_request & request) const {
    if (!has_pdf_signature(request.source_bytes)) {
        return failure("resource.invalid_document", "The source is not a valid PDF signature.");
    }
    if (request.page.has_value()) {
        return failure("resource.page_selection_unsupported", "The local text processor does not select individual pages.");
    }
    const size_t page_count = count_pdf_pages(request.source_bytes);
    if (request.limits.max_pages > 0 && page_count > request.limits.max_pages) {
        return failure("resource.processing_limit", "The PDF page count exceeded the configured limit.");
    }
    const size_t max_output_bytes = request.limits.max_output_bytes > 0
        ? request.limits.max_output_bytes
        : 1024 * 1024;
    std::string text;
    if (!extract_text_layer(request.source_bytes, max_output_bytes, text)) {
        return failure("resource.text_representation_unavailable", "The PDF has no bounded extractable text layer or the output limit was exceeded.");
    }

    agent_resource_processing_result result;
    result.success = true;
    result.processor_id = id();
    result.safe_summary = "PDF text layer extracted.";
    agent_resource_processing_output output;
    output.name = request.source.name + ".txt";
    output.description = "Normalized text representation extracted from a PDF text layer.";
    output.mime_type = "text/plain";
    output.bytes = std::move(text);
    output.metadata.purpose = "normalized PDF text representation";
    output.metadata.content_summary = "Text extracted from the authoritative PDF resource.";
    output.metadata.usage_hint = "Read and chunk this derived text resource.";
    output.metadata.limitations = "Direct text layer only; scanned pages require a separate OCR processor.";
    output.lineage = {
        request.source.uri,
        0,
        1,
        request.range ? request.range->offset : 0,
        output.bytes.size(),
        0,
        "resource.process:" + id(),
    };
    result.outputs.push_back(std::move(output));
    return result;
}
