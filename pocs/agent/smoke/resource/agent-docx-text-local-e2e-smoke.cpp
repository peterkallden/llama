#include "tools/agent/resource/agent-resource-processing-service.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "tools/agent/resource/processors/agent-docx-text-processor.h"
#include "agent/sandbox-local-runtime.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

int main(int argc, char ** argv) {
    const std::string pandoc = argc > 1 ? argv[1] : "pandoc";
    const std::filesystem::path source_path = argc > 2
        ? argv[2] : std::filesystem::path("E:/tools/pandoc-3.10.1/docx-smoke.docx");
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

    common_agent_sandbox_local_runtime runtime;
    common_agent_sandbox_policy policy;
    policy.execution_class = "resource.processor.docx.text";
    policy.limits.timeout_ms = 120000;
    policy.limits.max_output_bytes = 4 * 1024 * 1024;
    policy.network = common_agent_sandbox_network_scope::none;
    policy.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
    agent_sandbox_resource_processing_host host(runtime, policy);
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
    agent_docx_text_processor processor(
        host, context, agent_resource_backend_kind::local_pandoc, pandoc);
    agent_resource_processor_registry registry;
    assert(registry.add(processor, error));
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
    return 0;
}
