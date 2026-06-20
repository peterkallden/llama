#include "plan/plan-json.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

static json parse_tool_arguments_json(const std::string & text) {
    auto parsed = json::parse(text, nullptr, false);
    if (!parsed.is_discarded()) {
        return parsed;
    }

    // Some small instruct models emit Python-like single-quoted objects inside
    // an otherwise valid JSON string. Accept that bounded variant and
    // canonicalize it back to strict JSON for tool execution.
    std::string normalized = text;
    bool changed = false;
    for (char & ch : normalized) {
        if (ch == '\'') {
            ch = '"';
            changed = true;
        }
    }
    if (!changed) {
        return json();
    }
    return json::parse(normalized, nullptr, false);
}

static bool calculator_expression_from_value(const json & value, std::string & expression) {
    if (value.is_number_integer()) {
        expression = std::to_string(value.get<long long>());
        return true;
    }
    if (value.is_number_unsigned()) {
        expression = std::to_string(value.get<unsigned long long>());
        return true;
    }
    if (value.is_number_float()) {
        expression = std::to_string(value.get<double>());
        while (expression.size() > 2 && expression.find('.') != std::string::npos && expression.back() == '0') {
            expression.pop_back();
        }
        if (!expression.empty() && expression.back() == '.') {
            expression.pop_back();
        }
        return true;
    }
    if (value.is_object() && value.contains("value")) {
        return calculator_expression_from_value(value["value"], expression);
    }
    return false;
}

static json normalize_tool_arguments(const std::string & tool_name, json arguments) {
    if (tool_name != "calculator" || !arguments.is_object() || arguments.contains("expression")) {
        return arguments;
    }
    if (!arguments.contains("operation") || !arguments["operation"].is_string() || !arguments.contains("operands") || !arguments["operands"].is_array()) {
        return arguments;
    }
    const auto & operands = arguments["operands"];
    if (operands.size() != 2) {
        return arguments;
    }
    std::string lhs;
    std::string rhs;
    if (!calculator_expression_from_value(operands[0], lhs) || !calculator_expression_from_value(operands[1], rhs)) {
        return arguments;
    }
    const auto operation = arguments["operation"].get<std::string>();
    std::string symbol;
    if (operation == "add") symbol = "+";
    else if (operation == "subtract") symbol = "-";
    else if (operation == "multiply") symbol = "*";
    else if (operation == "divide") symbol = "/";
    else return arguments;
    return json{
        {"expression", lhs + " " + symbol + " " + rhs}
    };
}

std::string common_plan_proposal_json_schema() {
    const json schema = {
        {"type", "object"},
        {"additionalProperties", false},
        {"required", {"goal", "success_criteria", "operations", "next_action"}},
        {"properties", {
            // Keep grammar repetition bounds below llama-grammar's defensive
            // threshold. These are control-plane fields, not user documents.
            {"goal", {{"type", "string"}, {"maxLength", 256}}},
            {"success_criteria", {{"type", "string"}, {"maxLength", 256}}},
            {"next_action", {{"type", "string"}, {"maxLength", 256}}},
            {"operations", {{"type", "array"}, {"minItems", 1}, {"maxItems", 1}, {"items", {{"type", "object"}, {"additionalProperties", false}, {"required", {"kind", "step", "reason_summary", "evidence_ids"}}, {"properties", {
                {"kind", {{"const", "add_step"}}},
                {"reason_summary", {{"type", "string"}, {"maxLength", 256}}},
                {"evidence_ids", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"step", {{"type", "object"}, {"additionalProperties", false}, {"required", {"id", "title", "objective", "depends_on", "required_evidence"}}, {"properties", {
                    {"id", {{"type", "string"}, {"maxLength", 64}}}, {"title", {{"type", "string"}, {"maxLength", 128}}}, {"objective", {{"type", "string"}, {"maxLength", 256}}},
                    {"depends_on", {{"type", "array"}, {"items", {{"type", "string"}}}}}, {"required_evidence", {{"type", "array"}, {"items", {{"type", "string"}}}}}, {"source_memory_ids", {{"type", "array"}, {"maxItems", 4}, {"items", {{"type", "string"}, {"maxLength", 256}}}}},
                    {"tool", {{"type", "object"}, {"additionalProperties", false}, {"required", {"name", "arguments_json"}}, {"properties", {{"name", {{"type", "string"}, {"maxLength", 256}}}, {"arguments_json", {{"type", "string"}, {"maxLength", 512}}}}}}}
                }}}}
            }}}}}}
        }}
    };
    return schema.dump();
}

bool common_plan_parse_proposal_json(const std::string & text, common_plan_state & plan, std::vector<common_plan_operation> & operations, std::string & error, size_t max_operations) {
    try {
        const auto input = json::parse(text);
        if (!input.is_object() || !input.contains("goal") || !input.contains("success_criteria") || !input.contains("next_action") || !input.contains("operations")) { error = "plan proposal is missing required fields"; return false; }
        if (!input["goal"].is_string() || !input["success_criteria"].is_string() || !input["next_action"].is_string() || !input["operations"].is_array()) { error = "plan proposal has invalid field types"; return false; }
        if (input["operations"].empty() || input["operations"].size() > max_operations) { error = "invalid number of plan operations"; return false; }
        common_plan_state parsed = plan;
        parsed.goal = input["goal"].get<std::string>(); parsed.success_criteria = input["success_criteria"].get<std::string>(); parsed.next_action = input["next_action"].get<std::string>();
        std::vector<common_plan_operation> parsed_operations;
        for (const auto & item : input["operations"]) {
            if (!item.is_object() || item.value("kind", std::string()) != "add_step" || !item.contains("step") || !item["step"].is_object()) { error = "unsupported plan operation"; return false; }
            const auto & source = item["step"];
            if (!source.contains("id") || !source.contains("title") || !source.contains("objective") || !source["id"].is_string() || !source["title"].is_string() || !source["objective"].is_string()) { error = "invalid add_step payload"; return false; }
            common_plan_step step; step.id = source["id"].get<std::string>(); step.title = source["title"].get<std::string>(); step.objective = source["objective"].get<std::string>();
            if (source.contains("depends_on") && source["depends_on"].is_array()) step.depends_on = source["depends_on"].get<std::vector<std::string>>();
            if (source.contains("required_evidence") && source["required_evidence"].is_array()) step.required_evidence = source["required_evidence"].get<std::vector<std::string>>();
            if (source.contains("source_memory_ids") && source["source_memory_ids"].is_array()) step.source_memory_ids = source["source_memory_ids"].get<std::vector<std::string>>();
            if (source.contains("tool")) { if (!source["tool"].is_object() || !source["tool"].contains("name") || !source["tool"].contains("arguments_json") || !source["tool"]["name"].is_string() || !source["tool"]["arguments_json"].is_string()) { error = "invalid step tool payload"; return false; } const auto tool_name = source["tool"]["name"].get<std::string>(); const auto arguments_json = source["tool"]["arguments_json"].get<std::string>(); auto arguments = parse_tool_arguments_json(arguments_json); if (!arguments.is_object()) { error = "tool arguments_json must encode an object"; return false; } arguments = normalize_tool_arguments(tool_name, std::move(arguments)); step.tool_call = common_plan_tool_call{tool_name, arguments.dump()}; step.selected_tool = step.tool_call->name; }
            common_plan_operation operation; operation.kind = common_plan_operation_kind::add_step; operation.step = std::move(step); operation.reason_summary = item.value("reason_summary", std::string()); if (item.contains("evidence_ids") && item["evidence_ids"].is_array()) operation.evidence_ids = item["evidence_ids"].get<std::vector<std::string>>(); parsed_operations.push_back(std::move(operation));
        }
        plan = std::move(parsed); operations = std::move(parsed_operations); error.clear(); return true;
    } catch (const json::exception &) { error = "malformed plan proposal JSON"; return false; }
}
