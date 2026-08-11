#include "agent-resource-processing-service.h"

#include "agent-resource-media-type.h"
#include "agent-resource-processing-cache.h"

#include <chrono>
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

common_runtime_resource_media_type resolve_processing_media_type(
        agent_resource_store & store,
        const agent_resource_processing_service_request & request,
        const agent_resource_descriptor & source,
        std::string & error) {
    common_runtime_resource_media_type media_type = request.media_type;
    if (media_type.declared_type.empty()) {
        media_type.declared_type = source.mime_type;
    }
    if (!media_type.resolved_type.empty() && media_type.content_verified) {
        media_type.declared_type = common_normalize_resource_media_type(media_type.declared_type);
        media_type.resolved_type = common_normalize_resource_media_type(media_type.resolved_type);
        error.clear();
        return media_type;
    }

    std::string sample;
    if (!store.read_bytes_range(source.uri, request.authority, 0, 512, sample, error)) {
        return media_type;
    }
    media_type = resolve_agent_resource_media_type({
        media_type.declared_type,
        std::move(sample),
        true,
    });
    error.clear();
    return media_type;
}

} // namespace

agent_resource_processing_service::agent_resource_processing_service(
        agent_resource_store & store,
        const agent_resource_processor_registry & registry)
    : store_(store),
      registry_(registry) {}

agent_resource_processing_result agent_resource_processing_service::process(
        const agent_resource_processing_binding_request & request) const {
    agent_resource_processing_service_request service_request;
    service_request.source_uri = request.source_uri;
    service_request.operation_id = request.operation_id;
    service_request.authority = request.authority;
    service_request.media_type = request.media_type;
    service_request.target_representation = request.target_representation;
    service_request.target_media_type = request.target_media_type;
    service_request.purpose = request.purpose;
    service_request.page = request.page;
    service_request.range = request.range;
    service_request.limits = request.limits;
    return process(service_request);
}

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
    common_runtime_resource_media_type media_type =
        resolve_processing_media_type(store_, request, source, error);
    if (!error.empty()) {
        return processing_failure(
            "resource.media_type_resolution_failed",
            std::move(error));
    }
    agent_resource_processing_request selection_request;
    selection_request.source = source;
    selection_request.authority = request.authority;
    selection_request.media_type = media_type;
    selection_request.target_representation = request.target_representation;
    selection_request.target_media_type = request.target_media_type;
    selection_request.purpose = request.purpose;
    selection_request.page = request.page;
    selection_request.range = request.range;
    selection_request.limits = request.limits;
    const agent_resource_processor * processor = registry_.resolve(selection_request);
    if (processor == nullptr) {
        return processing_failure(
            "resource.unsupported_media_type",
            "No resource processor supports the requested representation.");
    }
    if (request.event_sink) {
        request.event_sink({
            common_agent_event_type::resource_processing_started,
            "resource processing started",
            {},
            std::nullopt,
            {},
            processor->id(),
            source.uri,
        });
    }

    const size_t max_source_bytes = request.limits.max_source_bytes > 0
        ? request.limits.max_source_bytes
        : 16 * 1024 * 1024;
    if (!request.range && source.size_bytes > max_source_bytes) {
        return processing_failure(
            "resource.processing_limit",
            "Resource processor source exceeded the configured byte limit.",
            processor->id());
    }
    size_t source_offset = 0;
    size_t source_limit = max_source_bytes;
    if (request.range) {
        source_offset = request.range->offset;
        source_limit = request.range->max_bytes;
        if (source_limit == 0 || source_limit > max_source_bytes) {
            return processing_failure(
                "resource.processing_limit",
                "Resource processor range exceeded the configured byte limit.",
                processor->id());
        }
    }

    const std::string processing_cache_key = make_agent_resource_processing_cache_key(
        source,
        media_type,
        *processor,
        request.target_representation,
        request.target_media_type,
        request.purpose,
        request.page,
        request.range,
        request.limits);
    std::vector<agent_resource_descriptor> existing_resources;
    if (!store_.list(request.authority, existing_resources, error)) {
        return processing_failure(
            "resource.cache_lookup_failed",
            error.empty() ? "The derived resource cache could not be inspected." : std::move(error),
            processor->id());
    }
    for (const auto & existing : existing_resources) {
        if (existing.metadata.processing_cache_key != processing_cache_key ||
                existing.source_provider != "resource_processor" ||
                existing.source_tool != processor->id() ||
                existing.lineage.parent_uri != source.uri) {
            continue;
        }
        agent_resource_processing_result cached;
        cached.success = true;
        cached.processor_id = processor->id();
        cached.resources.push_back(existing);
        cached.safe_summary = "Reused a cached derived resource representation.";
        if (request.event_sink) {
            request.event_sink({
                common_agent_event_type::resource_processing_completed,
                cached.safe_summary,
                {},
                std::nullopt,
                {},
                processor->id(),
                existing.uri,
            });
        }
        return cached;
    }

    agent_resource_processing_request processing_request;
    processing_request.source = source;
    processing_request.authority = request.authority;
    processing_request.media_type = media_type;
    processing_request.target_representation = request.target_representation;
    processing_request.target_media_type = request.target_media_type;
    processing_request.purpose = request.purpose;
    processing_request.page = request.page;
    processing_request.range = request.range;
    processing_request.limits = request.limits;
    if (!store_.read_bytes_range(
            source.uri,
            request.authority,
            source_offset,
            source_limit,
            processing_request.source_bytes,
            error)) {
        return processing_failure(
            "resource.source_read_failed",
            error.empty() ? "Resource processor could not read the source." : std::move(error),
            processor->id());
    }

    const auto started_at = std::chrono::steady_clock::now();
    agent_resource_processing_result result = processor->process(processing_request);
    const auto elapsed_ms = static_cast<size_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started_at).count());
    if (request.limits.max_duration_ms > 0 && elapsed_ms > request.limits.max_duration_ms) {
        if (request.event_sink) {
            request.event_sink({
                common_agent_event_type::resource_processing_failed,
                "resource processing duration limit reached",
                {},
                std::nullopt,
                {},
                processor->id(),
                source.uri,
            });
        }
        return processing_failure(
            "resource.processing_timeout",
            "Resource processor exceeded the configured duration limit.",
            processor->id());
    }
    if (!result.processor_id.empty() && result.processor_id != processor->id()) {
        return processing_failure(
            "resource.processor_failed",
            "Resource processor returned an unexpected processor identity.",
            processor->id());
    }
    result.processor_id = processor->id();
    if (!result.success) {
        if (request.event_sink) {
            request.event_sink({
                common_agent_event_type::resource_processing_failed,
                result.safe_summary,
                {},
                std::nullopt,
                {},
                processor->id(),
                source.uri,
            });
        }
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
        put.metadata.processing_cache_key = processing_cache_key;
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
        if (request.event_sink) {
            request.event_sink({
                common_agent_event_type::resource_created,
                "derived resource created by processor",
                {},
                std::nullopt,
                {},
                processor->id(),
                descriptor.uri,
            });
        }
    }
    result.outputs.clear();

    if (result.safe_summary.empty()) {
        result.safe_summary = "Resource processing completed.";
    }
    if (request.event_sink) {
        request.event_sink({
            common_agent_event_type::resource_processing_completed,
            result.safe_summary,
            {},
            std::nullopt,
            {},
            processor->id(),
            source.uri,
        });
    }
    return result;
}
