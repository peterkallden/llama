#include "agent/runtime-json-contracts.h"

#include <cctype>
#include <regex>

using json = nlohmann::ordered_json;

namespace {

bool infer_calculator_expression(const std::string & text, std::string & expression) {
    static const std::regex arithmetic(R"((\(?\s*\d+(?:\.\d+)?(?:\s*[-+*/]\s*\d+(?:\.\d+)?)+\s*\)?))");
    std::smatch match;
    if (!std::regex_search(text, match, arithmetic) || match.size() < 2) {
        return false;
    }
    expression = match[1].str();
    return true;
}

bool infer_memory_search_query(const std::string & prompt, std::string & query) {
    const auto first = prompt.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return false;
    }
    const auto last = prompt.find_last_not_of(" \t\r\n");
    query = prompt.substr(first, last - first + 1);
    return query.size() <= 1024;
}

} // namespace

nlohmann::ordered_json common_agent_runtime_user_correction_to_json(
        const std::string & source_turn_id,
        const std::string & statement) {
    return {
        {"source_turn_id", source_turn_id},
        {"statement", statement},
    };
}

std::string common_agent_runtime_user_correction_json(
        const std::string & source_turn_id,
        const std::string & statement) {
    return common_agent_runtime_user_correction_to_json(
        source_turn_id,
        statement).dump();
}

nlohmann::ordered_json common_agent_runtime_failure_observation_to_json(
        const common_agent_failure & failure) {
    return {
        {"failure", {
            {"code", failure.code},
            {"class", common_agent_failure_class_name(failure.classification)},
            {"stage", failure.stage},
            {"tool", failure.tool_name},
            {"step_id", failure.step_id},
            {"retryable", failure.retryable},
            {"safe_summary", failure.safe_summary},
            {"evidence_id", failure.evidence_id},
        }},
    };
}

std::string common_agent_runtime_failure_observation_json(
        const common_agent_failure & failure) {
    return common_agent_runtime_failure_observation_to_json(failure).dump();
}

nlohmann::ordered_json common_agent_runtime_reflection_learning_hint_to_json(
        const common_reflection_learning_hint & hint) {
    return {
        {"category", hint.category},
        {"statement", hint.statement},
        {"expected_reuse", hint.expected_reuse},
    };
}

std::string common_agent_runtime_reflection_learning_hint_json(
        const common_reflection_learning_hint & hint) {
    return common_agent_runtime_reflection_learning_hint_to_json(hint).dump();
}

nlohmann::ordered_json common_agent_runtime_reasoning_observation_to_json(
        const std::string & reasoning_text) {
    const auto parsed = json::parse(reasoning_text, nullptr, false);
    if (parsed.is_object()) {
        return parsed;
    }
    return {
        {"summary", reasoning_text},
        {"format", "unstructured"},
    };
}

std::string common_agent_runtime_normalize_reasoning_observation_json(
        const std::string & reasoning_text) {
    return common_agent_runtime_reasoning_observation_to_json(reasoning_text).dump();
}

bool common_agent_runtime_apply_safe_tool_defaults_to_json(
        const common_agent_request & request,
        const std::string & tool_name,
        const json & arguments,
        json & normalized_arguments,
        bool & changed,
        std::string & error) {
    error.clear();
    changed = false;
    if (!arguments.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }

    normalized_arguments = arguments;
    const auto set_prompt_query = [&](size_t max_length) {
        if (normalized_arguments.contains("query")) {
            return;
        }
        std::string query;
        if (infer_memory_search_query(request.prompt, query) && query.size() <= max_length) {
            normalized_arguments["query"] = std::move(query);
            changed = true;
        }
    };

    if (tool_name == "calculator" && !normalized_arguments.contains("expression")) {
        std::string expression;
        if (infer_calculator_expression(request.prompt, expression)) {
            normalized_arguments["expression"] = std::move(expression);
            changed = true;
        }
    } else if (tool_name == "memory_search") {
        set_prompt_query(1024);
    } else if (tool_name == "repository_search" || tool_name == "workspace_search") {
        set_prompt_query(256);
        if (!normalized_arguments.contains("path")) { normalized_arguments["path"] = ""; changed = true; }
        if (!normalized_arguments.contains("max_results")) { normalized_arguments["max_results"] = 16; changed = true; }
    } else if (tool_name == "web_search") {
        set_prompt_query(256);
        if (!normalized_arguments.contains("limit")) { normalized_arguments["limit"] = 5; changed = true; }
    } else if (tool_name == "repository_read" || tool_name == "workspace_read") {
        if (!normalized_arguments.contains("start_line")) { normalized_arguments["start_line"] = 1; changed = true; }
        if (!normalized_arguments.contains("end_line")) { normalized_arguments["end_line"] = 200; changed = true; }
    } else if (tool_name == "resource_read") {
        if (!normalized_arguments.contains("max_bytes")) { normalized_arguments["max_bytes"] = 8192; changed = true; }
    } else if (tool_name == "repository_list" || tool_name == "workspace_list") {
        if (!normalized_arguments.contains("path")) { normalized_arguments["path"] = ""; changed = true; }
        if (!normalized_arguments.contains("depth")) { normalized_arguments["depth"] = 1; changed = true; }
    }

    return true;
}

bool common_agent_runtime_apply_safe_tool_defaults(
        const common_agent_request & request,
        common_agent_tool_call & call) {
    const auto arguments = json::parse(call.arguments_json, nullptr, false);
    json normalized_arguments;
    bool changed = false;
    std::string error;
    if (!common_agent_runtime_apply_safe_tool_defaults_to_json(
            request,
            call.name,
            arguments,
            normalized_arguments,
            changed,
            error)) {
        return false;
    }

    if (changed) {
        call.arguments_json = normalized_arguments.dump();
    }
    return changed;
}
