#include "tools/agent/resource/processors/agent-pdf-text-processor.h"
#include "tools/agent/resource/agent-resource-processing-service.h"
#include "tools/agent/resource/agent-resource-store.h"

#include <cstdio>
#include <memory>

int main() {
    std::string error;
    agent_catalogued_resource_store store(
        std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());

    agent_resource_put_request request;
    request.name = "text-layer.pdf";
    request.mime_type = "application/octet-stream";
    request.bytes = "%PDF-1.7\n1 0 obj\n<< /Type /Page >>\nstream\nBT (Hello PDF) Tj (Second line) Tj ET\nendstream\nendobj\n";
    request.scope = common_runtime_resource_scope::session;
    request.namespace_id = "tenant-a";
    request.session_id = "session-1";
    request.project_id = "project-x";

    agent_resource_descriptor source;
    if (!store.put_bytes(request, source, error)) {
        std::fprintf(stderr, "source put failed: %s\n", error.c_str());
        return 1;
    }

    agent_pdf_text_processor processor;
    agent_resource_processor_registry registry;
    if (!registry.add(processor, error)) {
        std::fprintf(stderr, "processor registration failed: %s\n", error.c_str());
        return 1;
    }
    agent_resource_processing_service service(store, registry);
    agent_resource_processing_service_request processing;
    processing.source_uri = source.uri;
    processing.authority.namespace_id = "tenant-a";
    processing.authority.session_id = "session-1";
    processing.authority.project_id = "project-x";
    processing.media_type.declared_type = "application/octet-stream";
    processing.target_representation = "text";
    processing.limits.max_source_bytes = 4096;
    processing.limits.max_output_bytes = 128;
    processing.limits.max_pages = 2;

    const auto result = service.process(processing);
    if (!result.success || result.resources.size() != 1 ||
            result.processor_id != processor.id()) {
        std::fprintf(stderr, "PDF text processing failed: %s\n", result.safe_summary.c_str());
        return 1;
    }

    std::string text;
    if (!store.read_text(
            result.resources[0].uri,
            processing.authority,
            128,
            text,
            error) ||
            text != "Hello PDF\nSecond line") {
        std::fprintf(stderr, "unexpected PDF text output: %s (%s)\n", text.c_str(), error.c_str());
        return 1;
    }

    const auto unsupported_page = [&]() {
        auto request_with_page = processing;
        request_with_page.page = 0;
        return service.process(request_with_page);
    }();
    if (unsupported_page.success ||
            unsupported_page.failure_code != "resource.page_selection_unsupported") {
        std::fprintf(stderr, "page selection was not rejected by the local text processor\n");
        return 1;
    }

    std::printf("processor=%s\nderived_uri=%s\ntext=%s\n",
        result.processor_id.c_str(), result.resources[0].uri.c_str(), text.c_str());
    return 0;
}
