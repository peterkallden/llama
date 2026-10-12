#include "tools/agent/resource/agent-resource-media-type.h"

#include <cstdio>
#include <initializer_list>
#include <string>

namespace {

std::string bytes(std::initializer_list<unsigned char> values) {
    std::string out;
    out.reserve(values.size());
    for (const auto value : values) out.push_back(static_cast<char>(value));
    return out;
}

bool expect_media(
        const common_runtime_resource_media_type & media_type,
        const std::string & declared,
        const std::string & resolved,
        bool verified,
        const char * label) {
    if (media_type.declared_type != declared ||
            media_type.resolved_type != resolved ||
            media_type.content_verified != verified) {
        std::fprintf(stderr,
            "%s mismatch: declared=%s resolved=%s verified=%d\n",
            label,
            media_type.declared_type.c_str(),
            media_type.resolved_type.c_str(),
            media_type.content_verified ? 1 : 0);
        return false;
    }
    return true;
}

} // namespace

int main() {
    auto pdf = resolve_agent_resource_media_type({
        "application/octet-stream",
        "%PDF-1.7\n",
        true,
    });
    if (!expect_media(pdf, "application/octet-stream", "application/pdf", true, "pdf")) return 1;

    auto png = resolve_agent_resource_media_type({
        "image/unknown",
        bytes({0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0x00}),
        true,
    });
    if (!expect_media(png, "image/unknown", "image/png", true, "png")) return 1;

    auto json = resolve_agent_resource_media_type({
        " Application/JSON; charset=utf-8 ",
        R"({"ok":true})",
        true,
    });
    if (!expect_media(json, "application/json", "application/json", true, "json")) return 1;

    auto docx = resolve_agent_resource_media_type({
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        bytes({'P', 'K', 0x03, 0x04, 'x'}),
        true,
    });
    if (!expect_media(docx,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            true, "docx")) return 1;

    auto binary_declared = resolve_agent_resource_media_type({
        "application/vnd.example.binary",
        bytes({'a', 0x00, 'b'}),
        true,
    });
    if (!expect_media(binary_declared, "application/vnd.example.binary", "application/vnd.example.binary", false, "binary_declared")) return 1;

    auto empty_binary = resolve_agent_resource_media_type({
        "",
        bytes({'a', 0x00, 'b'}),
        true,
    });
    if (!expect_media(empty_binary, "", "application/octet-stream", true, "empty_binary")) return 1;

    auto empty_text = resolve_agent_resource_media_type({
        "",
        "plain text",
        true,
    });
    if (!expect_media(empty_text, "", "text/plain", true, "empty_text")) return 1;

    std::printf("pdf=%s\n", pdf.resolved_type.c_str());
    std::printf("png=%s\n", png.resolved_type.c_str());
    std::printf("json_verified=%d\n", json.content_verified ? 1 : 0);
    std::printf("docx=%s\n", docx.resolved_type.c_str());
    return 0;
}
