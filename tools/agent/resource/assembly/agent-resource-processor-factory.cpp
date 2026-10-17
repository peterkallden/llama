#include "agent-resource-processor-factory.h"
#include "tools/agent/resource/agent-resource-processor-defaults.h"

#include "tools/agent/resource/agent-resource-processing-host.h"
#include "tools/agent/resource/processors/agent-docx-text-processor.h"
#include "tools/agent/resource/processors/agent-pdf-page-image-processor.h"
#include "tools/agent/resource/processors/agent-pdf-ocr-text-processor.h"
#include "tools/agent/resource/processors/agent-tesseract-ocr-processor.h"
#include "tools/agent/resource/processors/agent-xlsx-workbook-json-processor.h"

#include "common/agent/workspace-contract.h"

#include <utility>
#include <vector>

namespace {

class assembled_resource_processing_provider final
        : public agent_resource_processing_provider {
public:
    assembled_resource_processing_provider(
            std::vector<std::shared_ptr<agent_resource_processing_host>> hosts,
            std::vector<std::shared_ptr<agent_resource_processor>> processors,
            std::shared_ptr<agent_resource_processor_registry> registry,
            std::shared_ptr<agent_resource_processing_service> service)
        : hosts_(std::move(hosts)),
          processors_(std::move(processors)),
          registry_(std::move(registry)),
          service_(std::move(service)) {}

    agent_resource_processing_result process(
            const agent_resource_processing_binding_request & request) const override {
        return service_->process(request);
    }

private:
    // The registry stores non-owning processor pointers. Keep all assembled
    // objects alive for the complete operation-scoped provider lifetime.
    std::vector<std::shared_ptr<agent_resource_processing_host>> hosts_;
    std::vector<std::shared_ptr<agent_resource_processor>> processors_;
    std::shared_ptr<agent_resource_processor_registry> registry_;
    std::shared_ptr<agent_resource_processing_service> service_;
};

common_agent_sandbox_policy make_sandbox_policy(
        const agent_resource_processing_assembly_request & assembly,
        const std::string & execution_class,
        const std::string & image) {
    auto policy = assembly.sandbox_defaults;
    const auto it = assembly.sandbox_classes.find(execution_class);
    if (it != assembly.sandbox_classes.end()) {
        policy = it->second;
    } else if (const size_t class_default = agent_resource_processor_default_sandbox_output_bytes(execution_class);
            class_default != 0) {
        policy.limits.max_output_bytes = class_default;
    }
    if (it == assembly.sandbox_classes.end()) {
        if (const uint32_t class_default = agent_resource_processor_default_sandbox_timeout_ms(execution_class);
                class_default != 0) {
            policy.limits.timeout_ms = class_default;
        }
        if (agent_resource_processor_default_sandbox_artifact_write(execution_class)) {
            policy.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
        }
    }
    policy.execution_class = execution_class;
    if (!image.empty()) policy.image = image;
    return policy;
}

std::shared_ptr<agent_resource_processing_host> make_processing_host(
        const agent_resource_processing_assembly_request & assembly,
        const std::string & backend,
        const common_agent_sandbox_policy & policy,
        const agent_resource_processing_binding_request & binding) {
    std::shared_ptr<common_agent_sandbox_runtime> runtime;
    if (backend == "docker") runtime = assembly.docker_runtime;
    else if (backend == "kubernetes") runtime = assembly.kubernetes_runtime;
    else if (backend == "lxc") runtime = assembly.lxc_runtime;
    else if (backend == "local") runtime = assembly.local_runtime;
    else return {};
    if (!runtime) return {};

    auto host = std::make_shared<agent_sandbox_resource_processing_host>(*runtime, policy);
    host->set_workspace_manager(assembly.workspace_manager.get());
    host->set_resource_store(assembly.resource_store, binding.authority);
    return host;
}

agent_pandoc_options pandoc_options_for_source(const std::string & source_type, const std::string & target_representation) {
    agent_pandoc_options options;
    if (source_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" && target_representation == "markdown") {
        options.input_format = "docx";options.output_format = "gfm";options.output_extension = "md";
    } else if (source_type == "text/markdown") {
        options.input_format = "markdown";
        options.output_format = "docx";
        options.output_extension = "docx";
    } else if (source_type == "application/vnd.oasis.opendocument.text") {
        options.input_format = "odt";
        options.output_format = "markdown";
        options.output_extension = "md";
    } else if (source_type == "text/html") {
        options.input_format = "html";
        options.output_format = "markdown";
        options.output_extension = "md";
    }
    return options;
}

std::shared_ptr<agent_resource_processing_provider> assemble_provider(
        const agent_resource_processing_assembly_request & assembly,
        const agent_resource_processing_binding_request & binding) {
    const auto source_type = common_normalize_resource_media_type(
            !binding.media_type.resolved_type.empty()
                ? binding.media_type.resolved_type
                : binding.media_type.declared_type);
    const auto dispatch = resolve_agent_resource_processor_dispatch({
        assembly.policies,
        source_type,
        assembly.sandbox_backend,
        assembly.local_runtime != nullptr,
        assembly.sandbox_backend != "none"});
    if (!dispatch.selected || assembly.resource_store == nullptr) return {};

    const auto operation_id = binding.operation_id.empty()
        ? "resource-read/turn"
        : binding.operation_id;
    agent_resource_processing_execution_context execution;
    execution.workspace.namespace_id = binding.authority.namespace_id;
    execution.workspace.session_id = binding.authority.session_id;
    execution.workspace.project_id = binding.authority.project_id;
    execution.workspace.turn_id = binding.authority.turn_id;
    execution.workspace.workspace_id = execution.workspace.project_id.empty()
        ? "session:" + execution.workspace.session_id
        : "project:" + execution.workspace.project_id;
    execution.operation_id = operation_id;

    const auto policy = make_sandbox_policy(
        assembly, dispatch.execution_class, dispatch.policy.image);
    auto host = make_processing_host(assembly, dispatch.execution_backend, policy, binding);
    if (!host) return {};
    std::vector<std::shared_ptr<agent_resource_processing_host>> hosts{host};

    const auto resolve_processor_host = [&](agent_resource_processor_kind kind,
            const char * execution_class) -> std::shared_ptr<agent_resource_processing_host> {
        const auto & selected = dispatch.policy_for(kind);
        if (!selected) return {};
        const agent_resource_processor_dispatch_request processor_request{
            assembly.policies,
            source_type,
            assembly.sandbox_backend,
            assembly.local_runtime != nullptr,
            assembly.sandbox_backend != "none"};
        const auto backend = resolve_agent_resource_processor_backend(*selected, processor_request);
        if (backend.empty()) return {};
        auto processor_policy = make_sandbox_policy(assembly, execution_class, selected->image);
        auto processor_host = make_processing_host(assembly, backend, processor_policy, binding);
        if (processor_host) hosts.push_back(processor_host);
        return processor_host;
    };

    auto page_host = resolve_processor_host(
        agent_resource_processor_kind::page_image, "resource.processor.pdf.page_image");
    auto ocr_host = resolve_processor_host(
        agent_resource_processor_kind::ocr, "resource.processor.ocr.tesseract");

    auto registry = std::make_shared<agent_resource_processor_registry>();
    std::vector<std::shared_ptr<agent_resource_processor>> processors;
    std::string error;
    const auto add = [&](std::shared_ptr<agent_resource_processor> processor) {
        if (!registry->add(*processor, error)) return false;
        processors.push_back(std::move(processor));
        return true;
    };

    if (source_type == "application/pdf" && page_host && ocr_host &&
            dispatch.policy_for(agent_resource_processor_kind::page_image)->max_pages > 0) {
        const auto & page_policy = *dispatch.policy_for(agent_resource_processor_kind::page_image);
        const auto & ocr_policy = *dispatch.policy_for(agent_resource_processor_kind::ocr);
        if (!add(std::make_shared<agent_pdf_ocr_text_processor>(
                *assembly.resource_store,
                *page_host,
                *ocr_host,
                execution,
                page_policy.executable.empty() ? "mutool" : page_policy.executable,
                ocr_policy.executable.empty() ? "tesseract" : ocr_policy.executable,
                page_policy.max_pages))) return {};
    }

    const auto page = agent_resource_processor_kind::page_image;
    if (page_host && dispatch.has_policy(page)) {
        const auto & selected = *dispatch.policy_for(page);
        if (!add(std::make_shared<agent_pdf_page_image_processor>(
                *page_host, execution, agent_resource_backend_kind::local_mupdf,
                selected.executable.empty() ? "mutool" : selected.executable))) return {};
    }
    const auto ocr = agent_resource_processor_kind::ocr;
    if (ocr_host && dispatch.has_policy(ocr)) {
        const auto & selected = *dispatch.policy_for(ocr);
        if (!add(std::make_shared<agent_tesseract_ocr_processor>(
                *ocr_host, execution, agent_resource_backend_kind::local_tesseract,
                selected.executable.empty() ? "tesseract" : selected.executable))) return {};
    }
    const auto pandoc = agent_resource_processor_kind::pandoc;
    if (dispatch.has_policy(pandoc) && dispatch.wants_local_for(pandoc)) {
        const auto & selected = *dispatch.policy_for(pandoc);
        if (!add(std::make_shared<agent_pandoc_processor>(
                *host, execution, agent_resource_backend_kind::local_pandoc,
                selected.executable.empty() ? "pandoc" : selected.executable,
                pandoc_options_for_source(source_type, binding.target_representation)))) return {};
    }
    if (dispatch.has_policy(pandoc) && dispatch.wants_sandbox_for(pandoc)) {
        const auto & selected = *dispatch.policy_for(pandoc);
        const auto backend = selected.backend == "kubernetes"
            ? agent_resource_backend_kind::kubernetes
            : (selected.backend == "lxc"
                ? agent_resource_backend_kind::lxc
                : agent_resource_backend_kind::docker);
        if (!add(std::make_shared<agent_pandoc_processor>(
                *host, execution, backend,
                selected.executable.empty() ? "pandoc" : selected.executable,
                pandoc_options_for_source(source_type, binding.target_representation)))) return {};
    }
    const auto xlsx = agent_resource_processor_kind::xlsx;
    if (dispatch.has_policy(xlsx) &&
            (dispatch.wants_local_for(xlsx) || dispatch.wants_sandbox_for(xlsx))) {
        const auto & selected = *dispatch.policy_for(xlsx);
        const auto backend = dispatch.wants_local_for(xlsx)
            ? agent_resource_backend_kind::local_process
            : (selected.backend == "kubernetes"
                ? agent_resource_backend_kind::kubernetes
                : (selected.backend == "lxc"
                    ? agent_resource_backend_kind::lxc
                    : agent_resource_backend_kind::docker));
        if (!add(std::make_shared<agent_xlsx_workbook_json_processor>(
                *host, execution, backend,
                selected.executable.empty() ? "python" : selected.executable,
                selected.script.empty() ? "scripts/agent-xlsx-to-json.py" : selected.script))) return {};
    }

    auto service = std::make_shared<agent_resource_processing_service>(
        *assembly.resource_store, *registry);
    return std::make_shared<assembled_resource_processing_provider>(
        std::move(hosts), std::move(processors), std::move(registry), std::move(service));
}

} // namespace

agent_resource_processing_provider_factory make_agent_resource_processing_provider_factory(
        agent_resource_processing_assembly_request request) {
    return [request = std::move(request)](
            const agent_resource_processing_binding_request & binding) {
        return assemble_provider(request, binding);
    };
}
