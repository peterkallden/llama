#include "agent-resource-processing-cache.h"

#include <string>

namespace {

void append_field(std::string & out, const char * name, const std::string & value) {
    out += name;
    out += std::to_string(value.size());
    out += ':';
    out += value;
    out += ';';
}

void append_field(std::string & out, const char * name, size_t value) {
    append_field(out, name, std::to_string(value));
}

} // namespace

std::string make_agent_resource_processing_cache_key(
        const agent_resource_descriptor & source,
        const common_runtime_resource_media_type & media_type,
        const agent_resource_processor & processor,
        const std::string & target_representation,
        const std::string & target_media_type,
        agent_resource_processing_purpose purpose,
        const std::optional<size_t> & page,
        const std::optional<agent_resource_byte_range> & range,
        const agent_resource_processing_limits & limits) {
    std::string key = "resource-processing-cache-v1;";
    append_field(key, "source_uri=", source.uri);
    append_field(key, "source_sha256=", source.sha256);
    append_field(key, "source_size=", source.size_bytes);
    append_field(key, "resolved_mime=", common_normalize_resource_media_type(media_type.resolved_type));
    append_field(key, "processor=", processor.cache_key());
    append_field(key, "representation=", target_representation);
    append_field(key, "target_mime=", common_normalize_resource_media_type(target_media_type));
    append_field(key, "purpose=", std::to_string(static_cast<int>(purpose)));
    append_field(key, "declared_language=", source.metadata.declared_language);
    append_field(key, "resolved_language=", source.metadata.resolved_language);
    append_field(key, "page=", page ? std::to_string(*page) : "none");
    append_field(key, "range_offset=", range ? range->offset : 0);
    append_field(key, "range_max_bytes=", range ? range->max_bytes : 0);
    append_field(key, "max_source_bytes=", limits.max_source_bytes);
    append_field(key, "max_output_bytes=", limits.max_output_bytes);
    append_field(key, "max_generated_resources=", limits.max_generated_resources);
    append_field(key, "max_duration_ms=", limits.max_duration_ms);
    append_field(key, "max_pages=", limits.max_pages);
    append_field(key, "max_page_bytes=", limits.max_page_bytes);
    return key;
}
