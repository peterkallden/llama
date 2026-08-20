#include "tools/agent/resource/processors/agent-pdf-page-image-processor.h"

#include <cassert>

class fixture_processing_host final : public agent_resource_processing_host {
public:
    bool execute(
            const agent_resource_processing_execution_context & context,
            common_agent_sandbox_request request,
            common_agent_sandbox_result & result,
            std::vector<agent_resource_processing_host_artifact> & artifacts,
            std::string & error) override {
        last_context = context;
        last_request = std::move(request);
        result = {};
        result.status = common_agent_sandbox_status::completed;
        result.backend_execution_id = "fixture/" + context.operation_id;
        artifacts = {{"page-42.png", "image/png", "PNG-fixture", {}}};
        error.clear();
        return true;
    }

    agent_resource_processing_execution_context last_context;
    common_agent_sandbox_request last_request;
};

int main() {
    fixture_processing_host host;
    agent_resource_processing_execution_context context;
    context.operation_id = "pdf-page-image-processor-smoke";
    context.workspace.project_id = "project-1";
    context.workspace.workspace_id = "workspace-1";

    agent_pdf_page_image_processor processor(
        host, context, agent_resource_backend_kind::local_mupdf, "mutool");
    agent_resource_processing_request request;
    request.source.uri = "agent-resource://document/report.pdf";
    request.source.name = "report.pdf";
    request.source.mime_type = "application/pdf";
    request.source.size_bytes = 512;
    request.target_representation = "page-image";
    request.page = 42;
    request.limits.max_output_bytes = 1024;

    const auto result = processor.process(request);
    assert(result.success);
    assert(result.processor_id == "pdf-page-image-v1");
    assert(result.outputs.size() == 1);
    assert(result.outputs[0].mime_type == "image/png");
    assert(result.outputs[0].bytes == "PNG-fixture");
    assert(result.outputs[0].lineage.parent_uri == request.source.uri);
    assert(result.outputs[0].lineage.derivation == "resource.process:pdf-page-image-v1");
    assert(host.last_request.command.program == "mutool");
    assert(host.last_request.command.arguments.back() == "42");
    return 0;
}
