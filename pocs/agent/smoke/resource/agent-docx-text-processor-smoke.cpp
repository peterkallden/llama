#include "tools/agent/resource/processors/agent-docx-text-processor.h"

#include <cassert>

class fixture_processing_host final : public agent_resource_processing_host {
public:
    bool execute(
            const agent_resource_processing_execution_context &,
            common_agent_sandbox_request request,
            common_agent_sandbox_result & result,
            std::vector<agent_resource_processing_host_artifact> & artifacts,
            std::string & error) override {
        last_request = std::move(request);
        result = {};
        result.status = common_agent_sandbox_status::completed;
        artifacts = {{"docx-text-report.docx.txt", "text/plain", "Heading\nBody text\n", {}}};
        error.clear();
        return true;
    }

    common_agent_sandbox_request last_request;
};

int main() {
    fixture_processing_host host;
    agent_resource_processing_execution_context context;
    context.operation_id = "docx-text-processor-smoke";
    context.workspace.project_id = "project-1";
    context.workspace.workspace_id = "workspace-1";

    agent_pandoc_processor processor(
        host, context, agent_resource_backend_kind::local_pandoc, "E:\\tools\\pandoc-3.10.1\\pandoc.exe");
    agent_resource_processing_request request;
    request.source.uri = "agent-resource://document/report.docx";
    request.source.name = "report.docx";
    request.source.mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
    request.source.size_bytes = 4096;
    request.source_bytes = "PK fake [Content_Types].xml word/document.xml";
    request.target_representation = "text";
    request.limits.max_output_bytes = 1024;

    const auto result = processor.process(request);
    assert(result.success);
    assert(result.processor_id == "pandoc-docx-text-v1");
    assert(result.outputs.size() == 1);
    assert(result.outputs[0].mime_type == "text/plain");
    assert(result.outputs[0].bytes == "Heading\nBody text\n");
    assert(result.outputs[0].lineage.parent_uri == request.source.uri);
    assert(result.outputs[0].lineage.derivation == "resource.process:pandoc-docx-text-v1");
    assert(host.last_request.execution_class == "resource.processor.pandoc");
    assert(host.last_request.command.program.find("pandoc.exe") != std::string::npos);
    assert(host.last_request.command.arguments[1] == "--from=docx");
    assert(host.last_request.command.arguments[2] == "--to=plain");
    assert(host.last_request.command.arguments[3] == "--wrap=none");

    agent_pandoc_options reverse_options;
    reverse_options.input_format = "markdown";
    reverse_options.output_format = "docx";
    reverse_options.output_extension = "docx";
    agent_pandoc_processor reverse_processor(
        host, context, agent_resource_backend_kind::local_pandoc,
        "E:\\tools\\pandoc-3.10.1\\pandoc.exe", reverse_options);
    agent_resource_processing_request reverse_request;
    reverse_request.source.uri = "agent-resource://document/report.md";
    reverse_request.source.name = "report.md";
    reverse_request.source.mime_type = "text/markdown";
    reverse_request.source.size_bytes = 128;
    reverse_request.source_bytes = "# Heading\n\nBody\n";
    reverse_request.target_representation = "docx";
    reverse_request.target_media_type =
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
    reverse_request.purpose = agent_resource_processing_purpose::artifact_generation;
    reverse_request.limits.max_output_bytes = 1024;
    const auto reverse_result = reverse_processor.process(reverse_request);
    assert(reverse_result.success);
    assert(reverse_result.processor_id == "pandoc-markdown-docx-v1");
    assert(reverse_result.outputs.size() == 1);
    assert(reverse_result.outputs[0].mime_type ==
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
    assert(host.last_request.command.arguments[1] == "--from=markdown");
    assert(host.last_request.command.arguments[2] == "--to=docx");

    agent_pandoc_processor docker_processor(
        host, context, agent_resource_backend_kind::docker, "pandoc");
    const auto docker_result = docker_processor.process(request);
    assert(docker_result.success);
    assert(host.last_request.command.program == "pandoc");
    assert(host.last_request.execution_class == "resource.processor.pandoc");

    agent_pandoc_processor kubernetes_processor(
        host, context, agent_resource_backend_kind::kubernetes, "pandoc");
    const auto kubernetes_result = kubernetes_processor.process(request);
    assert(kubernetes_result.success);

    request.source_bytes = "PK not a docx";
    const auto invalid = processor.process(request);
    assert(!invalid.success);
    assert(invalid.failure_code == "resource.invalid_document");
    return 0;
}
