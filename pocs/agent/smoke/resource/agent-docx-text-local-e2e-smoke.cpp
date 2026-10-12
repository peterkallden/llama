#include "tools/agent/resource/agent-resource-processing-service.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "tools/agent/resource/processors/agent-docx-text-processor.h"
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
    const std::string pandoc = argc > 1 ? argv[1] : "pandoc";
    const std::filesystem::path source_path = argc > 2
        ? argv[2] : std::filesystem::path("E:/tools/pandoc-3.10.1/docx-smoke.docx");
    const std::string backend = argc > 3 ? argv[3] : "local";
    const std::string image = argc > 4 ? argv[4] : "llama-agent-pdf-ocr-worker:local";
    std::ifstream input(source_path, std::ios::binary);
    if (!input) {
        std::cerr << "could not open DOCX fixture: " << source_path.string() << "\n";
        return 2;
    }
    const std::string bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (bytes.empty()) return 2;

    std::string error;
    agent_catalogued_resource_store store(
        std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());
    agent_resource_put_request put;
    put.name = source_path.filename().string();
    put.description = "DOCX Pandoc E2E source";
    put.mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
    put.bytes = bytes;
    put.scope = common_runtime_resource_scope::session;
    put.namespace_id = "local";
    put.session_id = "docx-e2e-session";
    put.project_id = "docx-e2e-project";
    put.turn_id = "docx-e2e-turn";
    agent_resource_descriptor source;
    assert(store.put_bytes(put, source, error));

    std::unique_ptr<common_agent_sandbox_runtime> runtime;
    agent_resource_backend_kind processor_backend = agent_resource_backend_kind::local_pandoc;
    if (backend == "docker") {
        runtime = std::make_unique<common_agent_sandbox_docker_runtime>(
            common_agent_docker_sandbox_config{"docker", image});
        processor_backend = agent_resource_backend_kind::docker;
    } else if (backend == "kubernetes") {
        common_agent_kubernetes_sandbox_config config;
        config.executable = "kubectl";
        config.insecure_skip_tls_verify = true;
        config.namespace_name = "default";
        config.staging_image = "alpine:3.20";
        config.pvc_retention = "project";
        config.cleanup = std::getenv("LLAMA_AGENT_K8S_KEEP") == nullptr;
        runtime = std::make_unique<common_agent_sandbox_kubernetes_runtime>(config);
        processor_backend = agent_resource_backend_kind::kubernetes;
    } else if (backend == "local") {
        runtime = std::make_unique<common_agent_sandbox_local_runtime>();
    } else {
        std::cerr << "unsupported E2E backend: " << backend << "\n";
        return 2;
    }
    common_agent_sandbox_policy policy;
    policy.execution_class = "resource.processor.pandoc";
    if (backend != "local") policy.image = image;
    policy.limits.timeout_ms = 120000;
    policy.limits.max_output_bytes = 4 * 1024 * 1024;
    policy.network = common_agent_sandbox_network_scope::none;
    policy.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
    agent_sandbox_resource_processing_host host(*runtime, policy);
    const auto workspace_root = std::filesystem::temp_directory_path() / "llama-agent-docx-e2e-workspaces";
    const auto artifact_root = std::filesystem::temp_directory_path() / "llama-agent-docx-e2e-artifacts";
    common_agent_workspace_manager workspace_manager({workspace_root.string(), artifact_root.string()});
    host.set_workspace_manager(&workspace_manager);
    agent_resource_read_authority authority;
    authority.namespace_id = "local";
    authority.project_id = "docx-e2e-project";
    authority.session_id = "docx-e2e-session";
    authority.turn_id = "docx-e2e-turn";
    host.set_resource_store(&store, authority);

    agent_resource_processing_execution_context context;
    context.operation_id = "docx-text-e2e";
    context.workspace.workspace_id = "docx-e2e-workspace";
    context.workspace.project_id = "docx-e2e-project";
    context.workspace.namespace_id = "local";
    context.workspace.session_id = "docx-e2e-session";
    context.workspace.turn_id = "docx-e2e-turn";
    agent_pandoc_processor processor(
        host, context, processor_backend, backend == "local" ? pandoc : "pandoc");
    agent_pandoc_options reverse_options;
    reverse_options.input_format = "markdown";
    reverse_options.output_format = "docx";
    reverse_options.output_extension = "docx";
    agent_pandoc_processor reverse_processor(
        host, context, processor_backend, backend == "local" ? pandoc : "pandoc", reverse_options);
    agent_resource_processor_registry registry;
    assert(registry.add(processor, error));
    assert(registry.add(reverse_processor, error));
    agent_resource_processing_service service(store, registry);
    agent_resource_processing_service_request request;
    request.source_uri = source.uri;
    request.authority = authority;
    request.target_representation = "text";
    request.limits.max_source_bytes = bytes.size() + 1;
    request.limits.max_output_bytes = 4 * 1024 * 1024;
    request.limits.max_generated_resources = 1;
    const auto result = service.process(request);
    if (!result.success || result.resources.size() != 1) {
        std::cerr << "DOCX E2E failed: " << result.failure_code << " " << result.safe_summary << "\n";
        return 1;
    }
    agent_resource_descriptor derived;
    assert(store.stat(result.resources.front().uri, authority, derived, error));
    assert(derived.mime_type == "text/plain");
    assert(derived.lineage.parent_uri == source.uri);
    std::string text;
    assert(store.read_text(result.resources.front().uri, authority, 4 * 1024 * 1024, text, error));
    assert(text.find("DOCX smoke heading") != std::string::npos);
    std::cout << "source_uri=" << source.uri << "\n";
    std::cout << "derived_uri=" << result.resources.front().uri << "\n";
    std::cout << "text_bytes=" << text.size() << "\n";

    agent_resource_put_request markdown_put;
    markdown_put.name = "generated-report.md";
    markdown_put.description = "Markdown source for Pandoc artifact generation";
    markdown_put.mime_type = "text/markdown";
    markdown_put.bytes = "# Generated report\n\nThis DOCX was generated by the resource processor.\n";
    markdown_put.scope = common_runtime_resource_scope::session;
    markdown_put.namespace_id = "local";
    markdown_put.session_id = "docx-e2e-session";
    markdown_put.project_id = "docx-e2e-project";
    markdown_put.turn_id = "docx-e2e-turn";
    agent_resource_descriptor markdown_source;
    assert(store.put_bytes(markdown_put, markdown_source, error));

    agent_resource_processing_service_request reverse_request;
    reverse_request.source_uri = markdown_source.uri;
    reverse_request.authority = authority;
    reverse_request.target_representation = "docx";
    reverse_request.target_media_type =
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
    reverse_request.purpose = agent_resource_processing_purpose::artifact_generation;
    reverse_request.limits.max_source_bytes = 1024 * 1024;
    reverse_request.limits.max_output_bytes = 4 * 1024 * 1024;
    reverse_request.limits.max_generated_resources = 1;
    const auto reverse_result = service.process(reverse_request);
    if (!reverse_result.success || reverse_result.resources.size() != 1) {
        std::cerr << "Markdown to DOCX E2E failed: " << reverse_result.failure_code
                  << " " << reverse_result.safe_summary << "\n";
        return 1;
    }
    agent_resource_descriptor generated;
    assert(store.stat(reverse_result.resources.front().uri, authority, generated, error));
    assert(generated.mime_type ==
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
    assert(generated.lineage.parent_uri == markdown_source.uri);
    std::string generated_bytes;
    assert(store.read_bytes(
        reverse_result.resources.front().uri, authority, 4 * 1024 * 1024,
        generated_bytes, error));
    assert(generated_bytes.size() >= 2 && generated_bytes.compare(0, 2, "PK") == 0);
    std::cout << "generated_docx_uri=" << reverse_result.resources.front().uri << "\n";
    std::cout << "generated_docx_bytes=" << generated_bytes.size() << "\n";
    return 0;
}
