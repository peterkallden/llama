#include "tools/agent/resource/agent-resource-processing-service.h"
#include "tools/agent/resource/agent-resource-chunker.h"
#include "tools/agent/resource/agent-resource-store.h"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

class fake_pdf_text_processor : public agent_resource_processor {
public:
    std::string id() const override {
        return "fake-pdf-text-v1";
    }

    bool supports(
            const std::string & mime_type,
            const std::string & target_representation) const override {
        return mime_type == "application/pdf" && target_representation == "text";
    }

    agent_resource_processing_result process(
            const agent_resource_processing_request & request) const override {
        ++calls_;
        agent_resource_processing_result result;
        result.success = true;
        result.processor_id = id();
        result.safe_summary = "Fake PDF text extracted.";

        agent_resource_processing_output output;
        output.name = request.source.name + ".txt";
        output.description = "Derived text representation";
        output.mime_type = "text/plain";
        if (request.source_bytes != "%PDF-smoke") {
            result.success = false;
            result.failure_code = "resource.source_bytes_missing";
            result.safe_summary = "The processor did not receive the bounded source bytes.";
            return result;
        }
        output.bytes = "First extracted paragraph.\n\nSecond extracted paragraph.\n\n";
        output.metadata.purpose = "derived text representation";
        output.metadata.content_summary = "Bounded extracted text fixture";
        output.metadata.usage_hint = "Read this derived resource as text.";
        output.metadata.limitations = "Model-free smoke fixture.";
        output.lineage.parent_uri = request.source.uri;
        output.lineage.chunk_index = 0;
        output.lineage.chunk_count = 1;
        output.lineage.byte_offset = request.range ? request.range->offset : 0;
        output.lineage.byte_length = output.bytes.size();
        output.lineage.derivation = "resource.process:" + id();
        result.outputs.push_back(std::move(output));
        return result;
    }

    size_t calls() const { return calls_; }

private:
    mutable size_t calls_ = 0;
};

class oversized_output_processor : public agent_resource_processor {
public:
    std::string id() const override {
        return "oversized-output-v1";
    }

    bool supports(
            const std::string & mime_type,
            const std::string & target_representation) const override {
        return mime_type == "application/pdf" && target_representation == "oversized-text";
    }

    agent_resource_processing_result process(
            const agent_resource_processing_request & request) const override {
        agent_resource_processing_result result;
        result.success = true;
        result.processor_id = id();
        agent_resource_processing_output output;
        output.name = request.source.name + ".large.txt";
        output.mime_type = "text/plain";
        output.bytes.assign(32, 'x');
        output.lineage.parent_uri = request.source.uri;
        output.lineage.chunk_index = 0;
        output.lineage.chunk_count = 1;
        output.lineage.byte_length = request.source.size_bytes;
        output.lineage.derivation = "resource.process:" + id();
        result.outputs.push_back(std::move(output));
        return result;
    }
};

} // namespace

int main() {
    std::string error;
    agent_catalogued_resource_store store(
        std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());

    agent_resource_put_request source_request;
    source_request.name = "report.pdf";
    source_request.description = "Authoritative PDF source";
    source_request.mime_type = "application/pdf";
    source_request.bytes = "%PDF-smoke";
    source_request.scope = common_runtime_resource_scope::session;
    source_request.namespace_id = "tenant-a";
    source_request.session_id = "session-1";
    source_request.project_id = "project-x";
    source_request.turn_id = "turn-1";

    agent_resource_descriptor source;
    if (!store.put_bytes(source_request, source, error)) {
        std::fprintf(stderr, "source put_bytes failed: %s\n", error.c_str());
        return 1;
    }

    agent_resource_processor_registry registry;
    fake_pdf_text_processor pdf_text;
    oversized_output_processor oversized;
    if (!registry.add(pdf_text, error) || !registry.add(oversized, error)) {
        std::fprintf(stderr, "processor registration failed: %s\n", error.c_str());
        return 1;
    }

    agent_resource_processing_service service(store, registry);
    agent_resource_processing_service_request request;
    request.source_uri = source.uri;
    request.authority.namespace_id = "tenant-a";
    request.authority.session_id = "session-1";
    request.authority.project_id = "project-x";
    request.media_type.declared_type = "application/octet-stream";
    request.target_representation = "text";
    request.range = agent_resource_byte_range{0, source.size_bytes};
    request.limits.max_output_bytes = 1024;
    request.limits.max_generated_resources = 2;
    std::vector<common_agent_event_type> processing_events;
    request.event_sink = [&processing_events](const common_agent_event & event) {
        processing_events.push_back(event.type);
    };

    auto result = service.process(request);
    if (!result.success ||
            result.resources.size() != 1 ||
            !result.outputs.empty() ||
            result.resources[0].lineage.parent_uri != source.uri ||
            result.resources[0].lineage.derivation != "resource.process:fake-pdf-text-v1") {
        std::fprintf(stderr, "processing service did not persist a derived resource with lineage\n");
        return 1;
    }
    if (processing_events.size() != 3 ||
            processing_events[0] != common_agent_event_type::resource_processing_started ||
            processing_events[1] != common_agent_event_type::resource_created ||
            processing_events[2] != common_agent_event_type::resource_processing_completed) {
        std::fprintf(stderr, "resource processing did not emit the expected host events\n");
        return 1;
    }

    const auto first_derived_uri = result.resources[0].uri;
    auto cached = service.process(request);
    if (!cached.success ||
            cached.resources.size() != 1 ||
            cached.resources[0].uri != first_derived_uri ||
            pdf_text.calls() != 1 ||
            processing_events.size() != 5 ||
            processing_events.back() != common_agent_event_type::resource_processing_completed) {
        std::fprintf(stderr, "identical processing request did not reuse the derived resource cache\n");
        return 1;
    }

    agent_resource_descriptor derived_descriptor;
    if (!store.stat(result.resources[0].uri, request.authority, derived_descriptor, error) ||
            derived_descriptor.source_provider != "resource_processor" ||
            derived_descriptor.source_tool != "fake-pdf-text-v1" ||
            derived_descriptor.metadata.processing_cache_key.empty()) {
        std::fprintf(stderr, "derived descriptor did not preserve processor provenance\n");
        return 1;
    }

    std::string derived_text;
    if (!store.read_text(result.resources[0].uri, request.authority, 1024, derived_text, error) ||
            derived_text.find("First extracted paragraph.") == std::string::npos) {
        std::fprintf(stderr, "derived resource was not readable through the existing store: %s\n", error.c_str());
        return 1;
    }

    common_runtime_resource_chunk_policy chunk_policy;
    chunk_policy.max_bytes = 32;
    chunk_policy.overlap_bytes = 0;
    agent_resource_chunk_plan chunk_plan;
    if (!plan_agent_resource_text_chunks(
            store,
            result.resources[0].uri,
            request.authority,
            chunk_policy,
            chunk_plan,
            error) || chunk_plan.ranges.size() < 2) {
        std::fprintf(stderr, "derived text did not enter the existing chunk path: %s\n", error.c_str());
        return 1;
    }
    std::string first_chunk;
    if (!read_agent_resource_chunk(
            store, request.authority, chunk_plan, 0, first_chunk, error) ||
            first_chunk.empty()) {
        std::fprintf(stderr, "derived text chunk could not be read: %s\n", error.c_str());
        return 1;
    }

    request.media_type.resolved_type = "image/png";
    request.media_type.content_verified = true;
    auto unsupported = service.process(request);
    if (unsupported.success ||
            unsupported.failure_code != "resource.unsupported_media_type") {
        std::fprintf(stderr, "unsupported media type was not rejected deterministically\n");
        return 1;
    }

    request.media_type.resolved_type = "application/pdf";
    request.media_type.content_verified = true;
    request.target_representation = "oversized-text";
    request.limits.max_output_bytes = 8;
    auto limited = service.process(request);
    if (limited.success ||
            limited.failure_code != "resource.output_too_large") {
        std::fprintf(stderr, "oversized processor output was not rejected\n");
        return 1;
    }

    std::printf("source_uri=%s\n", source.uri.c_str());
    std::printf("derived_uri=%s\n", result.resources[0].uri.c_str());
    std::printf("processor=%s\n", result.processor_id.c_str());
    return 0;
}
