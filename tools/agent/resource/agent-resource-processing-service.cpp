#include "agent-resource-processing-service.h"

#include <utility>

namespace {

agent_resource_processing_result processing_failure(
        std::string code,
        std::string summary,
        std::string processor_id = {}) {
    agent_resource_processing_result result;
    result.success = false;
    result.failure_code = std::move(code);
    result.safe_summary = std::move(summary);
    result.processor_id = std::move(processor_id);
    return result;
}

std::string resolved_media_type(
        const common_runtime_resource_media_type & media_type,
        const agent_resource_descriptor & source) {
    if (!media_type.resolved_type.empty()) return media_type.resolved_type;
    if (!media_type.declared_type.empty()) return media_type.declared_type;
    return source.mime_type;
}

} // namespace

agent_resource_processing_service::agent_resource_processing_service(
        agent_resource_store & store,
        const agent_resource_processor_registry & registry)
    : store_(store),
      registry_(registry) {}

agent_resource_processing_result agent_resource_processing_service::process(
        const agent_resource_processing_service_request & request) const {
    if (request.source_uri.empty()) {
        return processing_failure(
            "resource.invalid_request",
            "Resource processing requires a source URI.");
    }
    if (request.target_representation.empty()) {
        return processing_failure(
            "resource.invalid_request",
            "Resource processing requires a target representation.");
    }

    std::string error;
    agent_resource_descriptor source;
    if (!store_.stat(request.source_uri, request.authority, source, error)) {
        return processing_failure(
            "resource.unavailable",
            error.empty() ? "Source resource is unavailable." : std::move(error));
    }

    const std::string media_type = resolved_media_type(request.media_type, source);
    const agent_resource_processor * processor =
        registry_.resolve(media_type, request.target_representation);
    if (processor == nullptr) {
        return processing_failure(
            "resource.unsupported_media_type",
            "No resource processor supports the requested representation.");
    }

    agent_resource_processing_request processing_request;
    processing_request.source = source;
    processing_request.authority = request.authority;
    processing_request.media_type = request.media_type;
    processing_request.media_type.resolved_type = media_type;
    if (processing_request.media_type.declared_type.empty()) {
        processing_request.media_type.declared_type = source.mime_type;
    }
    processing_request.target_representation = request.target_representation;
    processing_request.page = request.page;
    processing_request.range = request.range;
    processing_request.limits = request.limits;

    agent_resource_processing_result result = processor->process(processing_request);
    if (!result.processor_id.empty() && result.processor_id != processor->id()) {
        return processing_failure(
            "resource.processor_failed",
            "Resource processor returned an unexpected processor identity.",
            processor->id());
    }
    result.processor_id = processor->id();
    if (!result.success) {
        if (result.failure_code.empty()) result.failure_code = "resource.processor_failed";
        if (result.safe_summary.empty()) result.safe_summary = "Resource processor failed.";
        return result;
    }

    const size_t max_outputs = request.limits.max_generated_resources > 0
        ? request.limits.max_generated_resources
        : 16;
    const size_t max_output_bytes = request.limits.max_output_bytes > 0
        ? request.limits.max_output_bytes
        : 1024 * 1024;
    if (result.outputs.size() > max_outputs) {
        return processing_failure(
            "resource.processing_limit",
            "Resource processor produced too many derived resources.",
            processor->id());
    }

    result.resources.clear();
    for (const auto & output : result.outputs) {
        if (output.name.empty() || output.mime_type.empty()) {
            return processing_failure(
                "resource.output_invalid",
                "Resource processor produced an invalid derived resource descriptor.",
                processor->id());
        }
        if (output.bytes.size() > max_output_bytes) {
            return processing_failure(
                "resource.output_too_large",
                "Resource processor output exceeded the configured byte limit.",
                processor->id());
        }

        agent_resource_put_request put;
        put.name = output.name;
        put.description = output.description;
        put.mime_type = output.mime_type;
        put.bytes = output.bytes;
        put.scope = source.scope;
        put.namespace_id = source.namespace_id;
        put.session_id = source.session_id;
        put.project_id = source.project_id;
        put.turn_id = source.turn_id;
        put.source_provider = "resource_processor";
        put.source_tool = processor->id();
        put.metadata = output.metadata;
        put.lineage = output.lineage;
        if (put.lineage.parent_uri.empty()) {
            put.lineage.parent_uri = source.uri;
            put.lineage.chunk_index = 0;
            put.lineage.chunk_count = 1;
            put.lineage.byte_offset = request.range ? request.range->offset : 0;
            put.lineage.byte_length = request.range ? request.range->max_bytes : source.size_bytes;
            put.lineage.derivation = "resource.process:" + processor->id();
        }

        agent_resource_descriptor descriptor;
        if (!store_.put_bytes(put, descriptor, error)) {
            return processing_failure(
                "resource.store_failed",
                error.empty() ? "Derived resource could not be persisted." : std::move(error),
                processor->id());
        }
        result.resources.push_back(descriptor);
    }
    result.outputs.clear();

    if (result.safe_summary.empty()) {
        result.safe_summary = "Resource processing completed.";
    }
    return result;
}
