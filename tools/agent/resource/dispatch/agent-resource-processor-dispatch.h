#pragma once

#include "common/resource/resource-contract.h"

#include <map>
#include <optional>
#include <string>

// Host-owned processor selection. This is deliberately independent of model
// input: the resolver only combines configured processor policies with the
// source media type and the host's available sandbox backend.
enum class agent_resource_processor_kind {
    page_image,
    ocr,
    pandoc,
    xlsx,
};

struct agent_resource_processor_dispatch_request {
    const std::map<std::string, agent_resource_processor_execution_policy> & policies;
    std::string source_type;
    std::string sandbox_backend;
};

struct agent_resource_processor_dispatch_selection {
    bool has_page_policy = false;
    bool has_ocr_policy = false;
    bool has_selected_pandoc_policy = false;
    bool has_selected_xlsx_policy = false;

    bool wants_page_local = false;
    bool wants_page_sandbox = false;
    bool wants_ocr_local = false;
    bool wants_ocr_sandbox = false;
    bool wants_pandoc_local = false;
    bool wants_pandoc_sandbox = false;
    bool wants_xlsx_local = false;
    bool wants_xlsx_sandbox = false;

    std::optional<agent_resource_processor_execution_policy> page_policy;
    std::optional<agent_resource_processor_execution_policy> ocr_policy;
    std::optional<agent_resource_processor_execution_policy> selected_pandoc_policy;
    std::optional<agent_resource_processor_execution_policy> selected_xlsx_policy;

    bool selected = false;
    agent_resource_processor_kind kind = agent_resource_processor_kind::page_image;
    std::string execution_backend;
    std::string execution_class;
    agent_resource_processor_execution_policy policy;
};

inline bool agent_resource_processor_policy_allows_local(
        const agent_resource_processor_execution_policy & policy) {
    return (policy.execution == "local_preferred" ||
            policy.execution == "local_required") &&
        (policy.backend == "auto" || policy.backend == "local");
}

inline bool agent_resource_processor_policy_allows_sandbox(
        const agent_resource_processor_execution_policy & policy,
        const std::string & sandbox_backend) {
    return (policy.execution == "sandbox_preferred" ||
            policy.execution == "sandbox_required") &&
        (policy.backend == "auto" || policy.backend == sandbox_backend);
}

inline agent_resource_processor_dispatch_selection resolve_agent_resource_processor_dispatch(
        const agent_resource_processor_dispatch_request & request) {
    agent_resource_processor_dispatch_selection result;
    const auto find_policy = [&request](const char * id)
            -> std::optional<agent_resource_processor_execution_policy> {
        const auto it = request.policies.find(id);
        return it == request.policies.end()
            ? std::nullopt
            : std::optional<agent_resource_processor_execution_policy>(it->second);
    };

    result.page_policy = find_policy("pdf.page_image");
    result.ocr_policy = find_policy("ocr.tesseract");
    const auto docx_policy = find_policy("docx.text");
    const auto odt_policy = find_policy("odt.text");
    const auto html_policy = find_policy("html.text");
    const auto xlsx_policy = find_policy("xlsx.workbook");
    result.has_page_policy = result.page_policy.has_value();
    result.has_ocr_policy = result.ocr_policy.has_value();

    if (request.source_type == "application/vnd.oasis.opendocument.text") {
        result.selected_pandoc_policy = odt_policy;
    } else if (request.source_type == "text/html") {
        result.selected_pandoc_policy = html_policy;
    } else if (request.source_type ==
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
            request.source_type.empty()) {
        result.selected_pandoc_policy = docx_policy;
    }
    result.selected_xlsx_policy = request.source_type ==
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ? xlsx_policy
        : std::nullopt;
    result.has_selected_pandoc_policy = result.selected_pandoc_policy.has_value();
    result.has_selected_xlsx_policy = result.selected_xlsx_policy.has_value();

    if (result.page_policy) {
        result.wants_page_local = agent_resource_processor_policy_allows_local(*result.page_policy);
        result.wants_page_sandbox = agent_resource_processor_policy_allows_sandbox(
            *result.page_policy, request.sandbox_backend);
    }
    if (result.ocr_policy) {
        result.wants_ocr_local = agent_resource_processor_policy_allows_local(*result.ocr_policy);
        result.wants_ocr_sandbox = agent_resource_processor_policy_allows_sandbox(
            *result.ocr_policy, request.sandbox_backend);
    }
    if (result.selected_pandoc_policy) {
        result.wants_pandoc_local = agent_resource_processor_policy_allows_local(
            *result.selected_pandoc_policy);
        result.wants_pandoc_sandbox = agent_resource_processor_policy_allows_sandbox(
            *result.selected_pandoc_policy, request.sandbox_backend);
    }
    if (result.selected_xlsx_policy) {
        result.wants_xlsx_local = agent_resource_processor_policy_allows_local(
            *result.selected_xlsx_policy);
        result.wants_xlsx_sandbox = agent_resource_processor_policy_allows_sandbox(
            *result.selected_xlsx_policy, request.sandbox_backend);
    }

    const auto select = [&result, &request](agent_resource_processor_kind kind,
            bool local, bool sandbox, const auto & policy, const char * execution_class) {
        if (!local && !sandbox) return;
        result.selected = true;
        result.kind = kind;
        result.policy = *policy;
        result.execution_class = execution_class;
        result.execution_backend = local
            ? "local"
            : (policy->backend == "kubernetes" ? "kubernetes" : request.sandbox_backend);
    };

    // Preserve the existing deterministic priority: page image, OCR, Pandoc,
    // then XLSX. A future registry can replace this ordering with explicit
    // priorities without changing the processor implementations.
    if (result.page_policy) {
        select(agent_resource_processor_kind::page_image,
            result.wants_page_local, result.wants_page_sandbox,
            result.page_policy, "resource.processor.pdf.page_image");
    }
    if (!result.selected && result.ocr_policy) {
        select(agent_resource_processor_kind::ocr,
            result.wants_ocr_local, result.wants_ocr_sandbox,
            result.ocr_policy, "resource.processor.ocr.tesseract");
    }
    if (!result.selected && result.selected_pandoc_policy) {
        select(agent_resource_processor_kind::pandoc,
            result.wants_pandoc_local, result.wants_pandoc_sandbox,
            result.selected_pandoc_policy, "resource.processor.pandoc");
    }
    if (!result.selected && result.selected_xlsx_policy) {
        select(agent_resource_processor_kind::xlsx,
            result.wants_xlsx_local, result.wants_xlsx_sandbox,
            result.selected_xlsx_policy, "resource.processor.xlsx");
    }
    return result;
}
