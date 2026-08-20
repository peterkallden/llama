#include "tools/agent/resource/dispatch/agent-resource-processor-dispatch.h"

#include <cstdio>
#include <map>

namespace {

agent_resource_processor_execution_policy policy(
        std::string execution, std::string backend = "auto") {
    agent_resource_processor_execution_policy result;
    result.execution = std::move(execution);
    result.backend = std::move(backend);
    return result;
}

} // namespace

int main() {
    const std::map<std::string, agent_resource_processor_execution_policy> sandbox_policies{
        {"pdf.page_image", policy("sandbox_preferred")},
        {"ocr.tesseract", policy("sandbox_preferred")},
    };
    const auto sandbox = resolve_agent_resource_processor_dispatch({
        sandbox_policies, "application/pdf", "docker"});
    if (!sandbox.selected ||
            sandbox.kind != agent_resource_processor_kind::page_image ||
            sandbox.execution_backend != "docker" ||
            sandbox.execution_class != "resource.processor.pdf.page_image") {
        std::fprintf(stderr, "sandbox page-image dispatch was not selected deterministically\n");
        return 1;
    }

    const std::map<std::string, agent_resource_processor_execution_policy> local_policies{
        {"docx.text", policy("local_required", "local")},
    };
    const auto local = resolve_agent_resource_processor_dispatch({
        local_policies,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ""});
    if (!local.selected ||
            local.kind != agent_resource_processor_kind::pandoc ||
            local.execution_backend != "local") {
        std::fprintf(stderr, "local Pandoc dispatch was not resolved\n");
        return 1;
    }

    const std::map<std::string, agent_resource_processor_execution_policy> xlsx_policies{
        {"xlsx.workbook", policy("local_required", "local")},
    };
    const auto xlsx = resolve_agent_resource_processor_dispatch({
        xlsx_policies,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ""});
    if (!xlsx.selected || xlsx.kind != agent_resource_processor_kind::xlsx) {
        std::fprintf(stderr, "local XLSX dispatch was not resolved\n");
        return 1;
    }

    const auto unavailable = resolve_agent_resource_processor_dispatch({
        {}, "application/pdf", ""});
    if (unavailable.selected) {
        std::fprintf(stderr, "empty processor policy unexpectedly selected a processor\n");
        return 1;
    }

    std::printf("sandbox=%s local=%s xlsx=%s\n",
        sandbox.execution_backend.c_str(),
        local.execution_backend.c_str(),
        xlsx.execution_class.c_str());
    return 0;
}
