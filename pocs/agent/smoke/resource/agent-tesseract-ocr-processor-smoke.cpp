#include "tools/agent/resource/processors/agent-tesseract-ocr-processor.h"

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
        artifacts = {{"ocr-page.png.txt", "text/plain", "recognized text", {}}};
        error.clear();
        return true;
    }

    agent_resource_processing_execution_context last_context;
    common_agent_sandbox_request last_request;
};

int main() {
    fixture_processing_host host;
    agent_resource_processing_execution_context context;
    context.operation_id = "tesseract-ocr-processor-smoke";
    context.workspace.project_id = "project-1";
    context.workspace.workspace_id = "workspace-1";

    agent_tesseract_ocr_processor processor(
        host, context, agent_resource_backend_kind::local_tesseract, "tesseract");
    agent_resource_processing_request request;
    request.source.uri = "agent-resource://document/page-1.png";
    request.source.name = "page-1.png";
    request.source.mime_type = "image/png";
    request.source.size_bytes = 512;
    request.source.metadata.declared_language = "sv";
    request.source.metadata.resolved_language = "swe";
    request.source.metadata.language_confidence = 0.97;
    request.source.metadata.language_source = "user";
    request.target_representation = "text";
    request.limits.max_output_bytes = 1024;

    const auto result = processor.process(request);
    assert(result.success);
    assert(result.processor_id == "tesseract-ocr-v1");
    assert(result.outputs.size() == 1);
    assert(result.outputs[0].mime_type == "text/plain");
    assert(result.outputs[0].bytes == "recognized text");
    assert(result.outputs[0].lineage.parent_uri == request.source.uri);
    assert(result.outputs[0].lineage.derivation == "resource.process:tesseract-ocr-v1");
    assert(host.last_request.command.program == "tesseract");
    assert(host.last_request.command.arguments.back() == "3");

    agent_tesseract_ocr_options automatic_options;
    automatic_options.language = "auto";
    agent_tesseract_ocr_processor automatic_processor(
        host, context, agent_resource_backend_kind::local_tesseract, "tesseract", automatic_options);
    const auto automatic_result = automatic_processor.process(request);
    assert(automatic_result.success);
    assert(automatic_result.outputs[0].metadata.resolved_language == "swe");
    assert(automatic_result.outputs[0].metadata.language_source == "resource.metadata.resolved");
    assert(automatic_result.outputs[0].metadata.declared_language == "sv");
    assert(automatic_processor.cache_key() != processor.cache_key());

    request.source.metadata.resolved_language.clear();
    request.source.metadata.declared_language.clear();
    const auto uncertain_result = automatic_processor.process(request);
    assert(!uncertain_result.success);
    assert(uncertain_result.failure_code == "resource.ocr_language_uncertain");
    return 0;
}
