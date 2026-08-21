#include "tools/agent/resource/agent-resource-processing-host.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "tools/agent/resource/agent-resource-processing-service.h"
#include "tools/agent/resource/processors/agent-pdf-page-image-processor.h"
#include "tools/agent/resource/processors/agent-tesseract-ocr-processor.h"
#include "agent/sandbox/sandbox-local-runtime.h"
#include "agent/sandbox/sandbox-docker-runtime.h"
#include "agent/sandbox/sandbox-kubernetes-runtime.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char ** argv) {
    const std::string executable = argc > 1 ? argv[1] : "mutool";
    const std::filesystem::path source_path = argc > 2
        ? argv[2]
        : std::filesystem::path("tools/ui/tests/stories/fixtures/assets/example.pdf");
    const size_t page = argc > 3 ? std::stoull(argv[3]) : 1;
    const std::string backend = argc > 4 ? argv[4] : "local";
    const std::string image = argc > 5 ? argv[5] : "llama-agent-pdf-ocr-worker:local";
    const std::string tesseract_executable = argc > 6 ? argv[6] : "tesseract";

    std::ifstream input(source_path, std::ios::binary);
    if (!input) {
        std::cerr << "could not open PDF fixture: " << source_path.string() << "\n";
        return 2;
    }
    const std::string pdf_bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (pdf_bytes.empty()) {
        std::cerr << "PDF fixture is empty\n";
        return 2;
    }

    std::string error;
    agent_catalogued_resource_store store(
        std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());
    agent_resource_put_request put;
    put.name = source_path.filename().string();
    put.description = "Local MuPDF E2E source";
    put.mime_type = "application/pdf";
    put.bytes = pdf_bytes;
    put.scope = common_runtime_resource_scope::session;
    put.namespace_id = "local";
    put.session_id = "pdf-e2e-session";
    put.project_id = "pdf-e2e-project";
    put.turn_id = "pdf-e2e-turn";
    agent_resource_descriptor source;
    if (!store.put_bytes(put, source, error)) {
        std::cerr << "source resource creation failed: " << error << "\n";
        return 1;
    }

    std::unique_ptr<common_agent_sandbox_runtime> runtime;
    if (backend == "docker") {
        runtime = std::make_unique<common_agent_sandbox_docker_runtime>(
            common_agent_docker_sandbox_config{"docker", image});
    } else if (backend == "kubernetes") {
        common_agent_kubernetes_sandbox_config config;
        config.executable = "kubectl";
        config.insecure_skip_tls_verify = true;
        config.namespace_name = "default";
        config.staging_image = "alpine:3.20";
        config.pvc_retention = "project";
        config.cleanup = std::getenv("LLAMA_AGENT_K8S_KEEP") == nullptr;
        runtime = std::make_unique<common_agent_sandbox_kubernetes_runtime>(config);
    } else if (backend == "local") {
        runtime = std::make_unique<common_agent_sandbox_local_runtime>();
    } else {
        std::cerr << "unsupported E2E backend: " << backend << "\n";
        return 2;
    }
    common_agent_sandbox_policy policy;
    policy.execution_class = "resource.processor.pdf.page_image";
    if (backend != "local") policy.image = image;
    policy.limits.timeout_ms = 120000;
    policy.limits.max_output_bytes = 4 * 1024 * 1024;
    policy.network = common_agent_sandbox_network_scope::none;
    policy.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
    agent_sandbox_resource_processing_host host(*runtime, policy);

    common_agent_sandbox_policy ocr_policy = policy;
    ocr_policy.execution_class = "resource.processor.ocr.tesseract";
    agent_sandbox_resource_processing_host ocr_host(*runtime, ocr_policy);

    const auto workspace_root = std::filesystem::temp_directory_path() / "llama-agent-pdf-e2e-workspaces";
    const auto artifact_root = std::filesystem::temp_directory_path() / "llama-agent-pdf-e2e-artifacts";
    common_agent_workspace_manager workspace_manager({
        workspace_root.string(), artifact_root.string(),
    });
    host.set_workspace_manager(&workspace_manager);
    agent_resource_read_authority source_authority;
    source_authority.namespace_id = "local";
    source_authority.project_id = "pdf-e2e-project";
    source_authority.session_id = "pdf-e2e-session";
    source_authority.turn_id = "pdf-e2e-turn";
    host.set_resource_store(&store, source_authority);
    ocr_host.set_workspace_manager(&workspace_manager);
    ocr_host.set_resource_store(&store, source_authority);

    agent_resource_processing_execution_context execution;
    execution.operation_id = "pdf-page-image-e2e";
    execution.workspace.workspace_id = "pdf-e2e-workspace";
    execution.workspace.project_id = "pdf-e2e-project";
    execution.workspace.namespace_id = "local";
    execution.workspace.session_id = "pdf-e2e-session";
    execution.workspace.turn_id = "pdf-e2e-turn";

    agent_pdf_page_image_processor processor(
        host, execution, agent_resource_backend_kind::local_mupdf,
        backend == "docker" || backend == "kubernetes" ? "mutool" : executable);
    agent_resource_processing_execution_context ocr_execution = execution;
    ocr_execution.operation_id = "tesseract-ocr-e2e";
    agent_tesseract_ocr_processor ocr_processor(
        ocr_host, ocr_execution, agent_resource_backend_kind::local_tesseract,
        backend == "docker" || backend == "kubernetes" ? "tesseract" : tesseract_executable);
    agent_resource_processor_registry registry;
    if (!registry.add(processor, error)) {
        std::cerr << "processor registration failed: " << error << "\n";
        return 1;
    }
    if (!registry.add(ocr_processor, error)) {
        std::cerr << "OCR processor registration failed: " << error << "\n";
        return 1;
    }
    agent_resource_processing_service service(store, registry);
    agent_resource_processing_service_request request;
    request.source_uri = source.uri;
    request.authority.namespace_id = "local";
    request.authority.project_id = "pdf-e2e-project";
    request.authority.session_id = "pdf-e2e-session";
    request.authority.turn_id = "pdf-e2e-turn";
    request.target_representation = "page-image";
    request.page = page;
    request.limits.max_source_bytes = pdf_bytes.size() + 1;
    request.limits.max_output_bytes = 4 * 1024 * 1024;
    request.limits.max_generated_resources = 1;

    const auto result = service.process(request);
    if (!result.success || result.resources.size() != 1) {
        std::cerr << "PDF local E2E processing failed: " << result.failure_code
                  << " " << result.safe_summary << "\n";
        return 1;
    }
    agent_resource_descriptor derived;
    if (!store.stat(result.resources.front().uri, request.authority, derived, error) ||
            derived.mime_type != "image/png" ||
            derived.lineage.parent_uri != source.uri ||
            derived.source_tool != "pdf-page-image-v1") {
        std::cerr << "derived PDF page image lost MIME, provenance or lineage: " << error << "\n";
        return 1;
    }
    std::string image_bytes;
    if (!store.read_bytes(result.resources.front().uri, request.authority, 4 * 1024 * 1024, image_bytes, error) ||
            image_bytes.size() < 8 || image_bytes.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) != 0) {
        std::cerr << "derived PDF page image is not a PNG: " << error << "\n";
        return 1;
    }

    agent_resource_processing_service_request ocr_request;
    ocr_request.source_uri = result.resources.front().uri;
    ocr_request.authority = request.authority;
    ocr_request.target_representation = "text";
    ocr_request.limits.max_source_bytes = image_bytes.size() + 1;
    ocr_request.limits.max_output_bytes = 4 * 1024 * 1024;
    ocr_request.limits.max_generated_resources = 1;
    const auto ocr_result = service.process(ocr_request);
    if (!ocr_result.success || ocr_result.resources.size() != 1) {
        std::cerr << "Tesseract OCR E2E processing failed: " << ocr_result.failure_code
                  << " " << ocr_result.safe_summary << "\n";
        return 1;
    }
    agent_resource_descriptor ocr_derived;
    if (!store.stat(ocr_result.resources.front().uri, ocr_request.authority, ocr_derived, error) ||
            ocr_derived.mime_type != "text/plain" ||
            ocr_derived.lineage.parent_uri != result.resources.front().uri ||
            ocr_derived.source_tool != "tesseract-ocr-v1") {
        std::cerr << "derived OCR text lost MIME, provenance or lineage: " << error << "\n";
        return 1;
    }
    std::string ocr_text;
    if (!store.read_text(ocr_result.resources.front().uri, ocr_request.authority, 4 * 1024 * 1024, ocr_text, error) ||
            ocr_text.empty()) {
        std::cerr << "derived OCR text is empty or unreadable: " << error << "\n";
        return 1;
    }

    std::cout << "source_uri=" << source.uri << "\n";
    std::cout << "derived_uri=" << result.resources.front().uri << "\n";
    std::cout << "image_bytes=" << image_bytes.size() << "\n";
    std::cout << "ocr_uri=" << ocr_result.resources.front().uri << "\n";
    std::cout << "ocr_text_bytes=" << ocr_text.size() << "\n";
    return 0;
}
