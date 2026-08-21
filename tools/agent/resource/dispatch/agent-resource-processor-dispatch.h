#pragma once

#include "common/resource/resource-contract.h"

#include <array>
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
    std::array<std::optional<agent_resource_processor_execution_policy>, 4> policies;
    std::array<bool, 4> wants_local{};
    std::array<bool, 4> wants_sandbox{};

    bool selected = false;
    agent_resource_processor_kind kind = agent_resource_processor_kind::page_image;
    std::string execution_backend;
    std::string execution_class;
    agent_resource_processor_execution_policy policy;

    static constexpr size_t index(agent_resource_processor_kind kind) {
        return static_cast<size_t>(kind);
    }
    const auto & policy_for(agent_resource_processor_kind kind) const {
        return policies[index(kind)];
    }
    bool has_policy(agent_resource_processor_kind kind) const {
        return policy_for(kind).has_value();
    }
    bool wants_local_for(agent_resource_processor_kind kind) const {
        return wants_local[index(kind)];
    }
    bool wants_sandbox_for(agent_resource_processor_kind kind) const {
        return wants_sandbox[index(kind)];
    }
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

    result.policies[result.index(agent_resource_processor_kind::page_image)] = find_policy("pdf.page_image");
    result.policies[result.index(agent_resource_processor_kind::ocr)] = find_policy("ocr.tesseract");
    const auto docx_policy = find_policy("docx.text");
    const auto odt_policy = find_policy("odt.text");
    const auto html_policy = find_policy("html.text");
    const auto xlsx_policy = find_policy("xlsx.workbook");
    if (request.source_type == "application/vnd.oasis.opendocument.text") {
        result.policies[result.index(agent_resource_processor_kind::pandoc)] = odt_policy;
    } else if (request.source_type == "text/html") {
        result.policies[result.index(agent_resource_processor_kind::pandoc)] = html_policy;
    } else if (request.source_type ==
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
            request.source_type.empty()) {
        result.policies[result.index(agent_resource_processor_kind::pandoc)] = docx_policy;
    }
    result.policies[result.index(agent_resource_processor_kind::xlsx)] = request.source_type ==
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ? xlsx_policy
        : std::nullopt;
    for (size_t i = 0; i < result.policies.size(); ++i) {
        if (!result.policies[i]) continue;
        result.wants_local[i] = agent_resource_processor_policy_allows_local(*result.policies[i]);
        result.wants_sandbox[i] = agent_resource_processor_policy_allows_sandbox(
            *result.policies[i], request.sandbox_backend);
    }

    const auto select = [&result, &request](agent_resource_processor_kind kind,
            const char * execution_class) {
        const auto & policy = result.policy_for(kind);
        const bool local = result.wants_local_for(kind);
        const bool sandbox = result.wants_sandbox_for(kind);
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
    if (result.has_policy(agent_resource_processor_kind::page_image)) {
        select(agent_resource_processor_kind::page_image, "resource.processor.pdf.page_image");
    }
    if (!result.selected && result.has_policy(agent_resource_processor_kind::ocr)) {
        select(agent_resource_processor_kind::ocr, "resource.processor.ocr.tesseract");
    }
    if (!result.selected && result.has_policy(agent_resource_processor_kind::pandoc)) {
        select(agent_resource_processor_kind::pandoc, "resource.processor.pandoc");
    }
    if (!result.selected && result.has_policy(agent_resource_processor_kind::xlsx)) {
        select(agent_resource_processor_kind::xlsx, "resource.processor.xlsx");
    }
    return result;
}
