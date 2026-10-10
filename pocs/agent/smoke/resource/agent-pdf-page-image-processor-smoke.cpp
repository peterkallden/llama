#include "tools/agent/resource/processors/agent-pdf-page-image-processor.h"
#include "tools/agent/resource/processors/agent-pdf-ocr-text-processor.h"
#include "tools/agent/resource/agent-resource-store.h"

#include <cassert>
#include <memory>

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

class composed_fixture_host final : public agent_resource_processing_host {
public:
    bool execute(
            const agent_resource_processing_execution_context & context,
            common_agent_sandbox_request request,
            common_agent_sandbox_result & result,
            std::vector<agent_resource_processing_host_artifact> & artifacts,
            std::string & error) override {
        (void) context;
        result = {};
        result.status = common_agent_sandbox_status::completed;
        artifacts.clear();
        if (request.command.program == "mutool" && !request.command.arguments.empty() &&
                request.command.arguments.front() == "info") {
            assert(!request.artifacts.collect);
            result.stdout_excerpt = "Pages: 2\n";
        } else if (request.command.program == "mutool") {
            ++render_calls;
            artifacts.push_back({"page.png", "image/png", "png-fixture", {}});
        } else {
            ++ocr_calls;
            artifacts.push_back({"page.png.txt", "text/plain", "recognized page\xe3(", {}});
        }
        error.clear();
        return true;
    }

    size_t render_calls = 0;
    size_t ocr_calls = 0;
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

    auto page_host = std::make_shared<composed_fixture_host>();
    auto ocr_host = std::make_shared<composed_fixture_host>();
    agent_catalogued_resource_store store(
        std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());
    agent_resource_processing_execution_context composed_context = context;
    composed_context.operation_id = "pdf-ocr-composed-smoke";
    agent_pdf_ocr_text_processor composed(
        store, *page_host, *ocr_host, composed_context, "mutool", "tesseract", 2);
    agent_resource_processing_request pdf_request;
    pdf_request.source.uri = "agent-resource://document/scanned.pdf";
    pdf_request.source.name = "scanned.pdf";
    pdf_request.source.mime_type = "application/pdf";
    pdf_request.source.size_bytes = 64;
    pdf_request.source.scope = common_runtime_resource_scope::session;
    pdf_request.authority.namespace_id = "local";
    pdf_request.authority.project_id = "project-1";
    pdf_request.authority.session_id = "session-1";
    pdf_request.authority.turn_id = "turn-1";
    pdf_request.source_bytes = "%PDF-1.7\\n/Type /Pages";
    pdf_request.target_representation = "text";
    pdf_request.limits.max_output_bytes = 1024;

    pdf_request.source_bytes = "%PDF-1.7\\n(direct text is available)";
    const auto direct_result = composed.process(pdf_request);
    assert(direct_result.success);
    assert(direct_result.processor_id == "pdf-ocr-text-v1");
    assert(direct_result.outputs.front().bytes == "direct text is available");
    assert(page_host->render_calls == 0);
    assert(ocr_host->ocr_calls == 0);

    pdf_request.source_bytes = "%PDF-1.7\\n/Type /Pages";
    const auto composed_result = composed.process(pdf_request);
    assert(composed_result.success);
    assert(composed_result.processor_id == "pdf-ocr-text-v1");
    assert(composed_result.outputs.size() == 1);
    assert(composed_result.outputs.front().bytes.find("recognized page") != std::string::npos);
    assert(composed_result.outputs.front().bytes.find("\xef\xbf\xbd") != std::string::npos);
    assert(composed_result.outputs.front().lineage.parent_uri == pdf_request.source.uri);
    assert(composed_result.outputs.front().lineage.derivation ==
        "resource.process:pdf-page-image-v1>tesseract-ocr-v1");
    assert(page_host->render_calls == 2);
    assert(ocr_host->ocr_calls == 2);
    assert(composed_result.outputs.front().metadata.limitations.find("2 of 2 pages") != std::string::npos);

    agent_pdf_ocr_text_processor capped(
        store, *page_host, *ocr_host, composed_context, "mutool", "tesseract", 1);
    const auto capped_result = capped.process(pdf_request);
    assert(capped_result.success);
    assert(capped_result.safe_summary.find("partial") != std::string::npos);
    assert(capped_result.outputs.front().metadata.limitations.find("1 of 2 pages") != std::string::npos);
    assert(page_host->render_calls == 3);
    assert(ocr_host->ocr_calls == 3);

    return 0;
}
