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
    // These are host-resolved availability facts. They are deliberately not
    // inferred from the model-facing policy; preferred modes use them to
    // choose a deterministic fallback without weakening required modes.
    bool local_available = true;
    bool sandbox_available = false;
};

struct agent_resource_processor_dispatch_selection {
    std::array<std::optional<agent_resource_processor_execution_policy>, 4> policies;
    // These flags describe the selected execution placement, not every
    // candidate that could have been used. This prevents the factory from
    // registering duplicate local and sandbox processor variants after a
    // preferred-mode fallback has been resolved.
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

inline std::string resolve_agent_resource_processor_backend(
        const agent_resource_processor_execution_policy & policy,
        const agent_resource_processor_dispatch_request & request) {
    const bool local_mode = policy.execution == "local_preferred" ||
        policy.execution == "local_required";
    const bool sandbox_mode = policy.execution == "sandbox_preferred" ||
        policy.execution == "sandbox_required";
    if (policy.backend == "local") {
        return local_mode && request.local_available ? "local" : std::string();
    }
    if (policy.backend != "auto") {
        return sandbox_mode && request.sandbox_available && !request.sandbox_backend.empty() &&
            policy.backend == request.sandbox_backend ? request.sandbox_backend : std::string();
    }
    const bool local_candidate = local_mode || policy.execution == "sandbox_preferred";
    const bool sandbox_candidate = sandbox_mode || policy.execution == "local_preferred";
    const bool has_local = request.local_available && local_candidate;
    const bool has_sandbox = request.sandbox_available && !request.sandbox_backend.empty() &&
        sandbox_candidate;

    if (policy.execution == "local_required") {
        return has_local ? "local" : std::string();
    }
    if (policy.execution == "sandbox_required") {
        return has_sandbox ? request.sandbox_backend : std::string();
    }
    if (policy.execution == "local_preferred") {
        if (has_local) return "local";
        return has_sandbox ? request.sandbox_backend : std::string();
    }
    if (policy.execution == "sandbox_preferred") {
        if (has_sandbox) return request.sandbox_backend;
        return has_local ? "local" : std::string();
    }
    return {};
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
    const auto select = [&result, &request](agent_resource_processor_kind kind,
            const char * execution_class) {
        const auto & policy = result.policy_for(kind);
        const auto backend = resolve_agent_resource_processor_backend(*policy, request);
        if (backend.empty()) return;
        result.selected = true;
        result.kind = kind;
        result.policy = *policy;
        result.execution_class = execution_class;
        result.execution_backend = backend;
        result.wants_local[result.index(kind)] = backend == "local";
        result.wants_sandbox[result.index(kind)] = backend != "local";
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
