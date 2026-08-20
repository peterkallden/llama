#include "agent-resource-media-type.h"

#include <algorithm>
#include <initializer_list>

namespace {

bool starts_with_bytes(const std::string & value, const char * prefix, size_t size) {
    return value.size() >= size &&
        std::equal(prefix, prefix + size, value.begin());
}

std::string make_signature(std::initializer_list<unsigned char> values) {
    std::string out;
    out.reserve(values.size());
    for (const auto value : values) {
        out.push_back(static_cast<char>(value));
    }
    return out;
}

bool has_nul(const std::string & value) {
    return value.find('\0') != std::string::npos;
}

std::string detect_signature_media_type(const std::string & bytes) {
    if (starts_with_bytes(bytes, "%PDF-", 5)) return "application/pdf";
    const auto png = make_signature({0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'});
    const auto jpeg = make_signature({0xff, 0xd8, 0xff});
    if (starts_with_bytes(bytes, png.data(), png.size())) return "image/png";
    if (starts_with_bytes(bytes, jpeg.data(), jpeg.size())) return "image/jpeg";
    if (starts_with_bytes(bytes, "GIF87a", 6) || starts_with_bytes(bytes, "GIF89a", 6)) return "image/gif";
    if (starts_with_bytes(bytes, "PK\x03\x04", 4) ||
            starts_with_bytes(bytes, "PK\x05\x06", 4) ||
            starts_with_bytes(bytes, "PK\x07\x08", 4)) {
        return "application/zip";
    }
    return {};
}

bool has_zip_signature(const std::string & bytes) {
    return starts_with_bytes(bytes, "PK\x03\x04", 4) ||
        starts_with_bytes(bytes, "PK\x05\x06", 4) ||
        starts_with_bytes(bytes, "PK\x07\x08", 4);
}

} // namespace

common_runtime_resource_media_type resolve_agent_resource_media_type(
        const agent_resource_media_type_resolution_request & request) {
    common_runtime_resource_media_type media_type;
    media_type.declared_type = common_normalize_resource_media_type(request.declared_type);

    // Office Open XML is a ZIP container, but a DOCX declaration plus a ZIP
    // signature is enough for the bounded media-type decision. The processor
    // performs the stronger full-package validation before execution.
    if (media_type.declared_type ==
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" &&
            has_zip_signature(request.sample_bytes)) {
        media_type.resolved_type = media_type.declared_type;
        media_type.content_verified = true;
        return media_type;
    }

    const auto signature_type = detect_signature_media_type(request.sample_bytes);
    if (!signature_type.empty()) {
        media_type.resolved_type = signature_type;
        media_type.content_verified = true;
        return media_type;
    }

    if (request.allow_text_heuristic &&
            common_resource_media_type_is_text_like(media_type.declared_type) &&
            !has_nul(request.sample_bytes)) {
        media_type.resolved_type = media_type.declared_type;
        media_type.content_verified = true;
        return media_type;
    }

    if (!media_type.declared_type.empty()) {
        media_type.resolved_type = media_type.declared_type;
        media_type.content_verified = false;
        return media_type;
    }

    media_type.resolved_type = has_nul(request.sample_bytes)
        ? "application/octet-stream"
        : "text/plain";
    media_type.content_verified = !request.sample_bytes.empty();
    return media_type;
}
