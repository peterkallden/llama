#include "plan/plan-json.h"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <set>

using json = nlohmann::ordered_json;

namespace {

json parse_tool_arguments_json(const std::string & text) {
    auto parsed = json::parse(text, nullptr, false);
    if (!parsed.is_discarded()) return parsed;

    // Compatibility for older small-model output that placed Python-like JSON
    // in arguments_json. New model output uses an ordinary JSON object.
    std::string normalized = text;
    bool changed = false;
    for (char & ch : normalized) if (ch == '\'') { ch = '"'; changed = true; }
    return changed ? json::parse(normalized, nullptr, false) : json();
}

bool calculator_expression_from_value(const json & value, std::string & expression) {
    if (value.is_number_integer()) { expression = std::to_string(value.get<long long>()); return true; }
    if (value.is_number_unsigned()) { expression = std::to_string(value.get<unsigned long long>()); return true; }
    if (value.is_number_float()) { expression = std::to_string(value.get<double>()); return true; }
    return value.is_object() && value.contains("value") && calculator_expression_from_value(value["value"], expression);
}

json normalize_tool_arguments(const std::string & tool_name, json arguments) {
    if (tool_name != "calculator" || !arguments.is_object() || arguments.contains("expression") ||
        !arguments.contains("operation") || !arguments["operation"].is_string() ||
        !arguments.contains("operands") || !arguments["operands"].is_array() || arguments["operands"].size() != 2) return arguments;
    std::string lhs, rhs;
    if (!calculator_expression_from_value(arguments["operands"][0], lhs) || !calculator_expression_from_value(arguments["operands"][1], rhs)) return arguments;
    const auto operation = arguments["operation"].get<std::string>();
    const char * symbol = operation == "add" ? "+" : operation == "subtract" ? "-" : operation == "multiply" ? "*" : operation == "divide" ? "/" : nullptr;
    return symbol ? json{{"expression", lhs + " " + symbol + " " + rhs}} : arguments;
}

json normalize_safe_integer_arguments(json arguments) {
    if (!arguments.is_object()) return arguments;
    // These are bounded, read-only control fields. Convert only a canonical
    // decimal string; paths, IDs, queries and all mutation content stay exact.
    for (const char * key : {"max_results", "limit", "depth", "start_line", "end_line"}) {
        if (!arguments.contains(key) || !arguments[key].is_string()) continue;
        const auto & text = arguments[key].get_ref<const std::string &>();
        if (text.empty() || text.size() > 9 || !std::all_of(text.begin(), text.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; })) continue;
        try { arguments[key] = std::stoll(text); } catch (const std::exception &) {}
    }
    return arguments;
}

bool string_array(const json & value, std::vector<std::string> & output) {
    if (!value.is_array()) return false;
    output.clear();
    for (const auto & item : value) { if (!item.is_string()) return false; output.push_back(item.get<std::string>()); }
    return true;
}

bool normalize_dependencies(const json & source, std::vector<std::string> & dependencies, std::string & error) {
    const auto it = source.contains("after") ? source.find("after") : source.find("depends_on");
    if (it == source.end()) { dependencies.clear(); return true; }
    if (it->is_string()) { dependencies = {it->get<std::string>()}; return true; }
    if (string_array(*it, dependencies)) return true;
    error = "step dependencies must be a string or string array";
    return false;
}

bool has_explicit_dependencies(const json & source) {
    return source.contains("after") || source.contains("depends_on");
}

bool parse_mode(const json & source, bool has_tool, common_plan_step_mode & mode, std::string & error) {
    const auto value = source.value("mode", has_tool ? std::string("tool") : std::string("final_response"));
    if (value == "tool") mode = common_plan_step_mode::tool;
    else if (value == "reasoning") mode = common_plan_step_mode::reasoning;
    else if (value == "final" || value == "final_response") mode = common_plan_step_mode::final_response;
    else { error = "invalid step mode"; return false; }
    if ((has_tool && mode != common_plan_step_mode::tool) || (!has_tool && mode == common_plan_step_mode::tool)) { error = "step mode does not match tool payload"; return false; }
    return true;
}

bool parse_tool(const json & source, common_plan_step & step, std::string & error) {
    if (!source.contains("tool")) return true;
    const auto & tool = source["tool"];
    std::string name;
    json arguments = json::object();
    if (tool.is_string()) name = tool.get<std::string>();
    else if (tool.is_object() && tool.contains("name") && tool["name"].is_string()) {
        name = tool["name"].get<std::string>();
        if (tool.contains("arguments")) arguments = tool["arguments"];
        else if (tool.contains("args")) arguments = tool["args"];
        else if (tool.contains("arguments_json") && tool["arguments_json"].is_string()) arguments = parse_tool_arguments_json(tool["arguments_json"].get<std::string>());
    } else { error = "invalid step tool payload"; return false; }
    if (source.contains("args")) arguments = source["args"];
    if (!arguments.is_object()) { error = "tool arguments must be a JSON object"; return false; }
    arguments = normalize_safe_integer_arguments(std::move(arguments));
    step.tool_call = common_plan_tool_call{name, normalize_tool_arguments(name, std::move(arguments)).dump()};
    step.selected_tool = name;
    return true;
}

bool parse_step(const json & source, const std::string & goal, const std::string & fallback_id, common_plan_step & step, std::string & error) {
    if (!source.is_object()) { error = "step must be an object"; return false; }
    if (source.contains("id")) {
        if (!source["id"].is_string() || source["id"].get<std::string>().empty()) { error = "step requires a non-empty id"; return false; }
        step.id = source["id"].get<std::string>();
    } else if (!fallback_id.empty()) {
        step.id = fallback_id;
    } else {
        error = "step requires a non-empty id";
        return false;
    }
    if (source.contains("title") && !source["title"].is_string()) { error = "step title must be a string"; return false; }
    if (source.contains("objective") && !source["objective"].is_string()) { error = "step objective must be a string"; return false; }
    step.title = source.value("title", step.id);
    step.objective = source.value("objective", goal);
    if (source.contains("contribution") && !source["contribution"].is_string()) { error = "step contribution must be a string"; return false; }
    step.intended_contribution = source.value("contribution", step.objective);
    if (!normalize_dependencies(source, step.depends_on, error)) return false;
    if (source.contains("required_evidence") && !string_array(source["required_evidence"], step.required_evidence)) { error = "required_evidence must be a string array"; return false; }
    if (source.contains("source_memory_ids") && !string_array(source["source_memory_ids"], step.source_memory_ids)) { error = "source_memory_ids must be a string array"; return false; }
    const bool has_tool = source.contains("tool");
    if (!parse_mode(source, has_tool, step.mode, error) || !parse_tool(source, step, error)) return false;
    if (step.tool_call) step.required_evidence.clear();
    return true;
}

std::string next_generated_step_id(size_t index, const std::set<std::string> & seen_step_ids) {
    std::string id = "step_" + std::to_string(index);
    while (seen_step_ids.count(id)) {
        ++index;
        id = "step_" + std::to_string(index);
    }
    return id;
}

bool parse_compact(const json & input, common_plan_state & plan, std::vector<common_plan_operation> & operations, std::string & error, size_t max_operations) {
    if (!input.contains("goal") || !input["goal"].is_string() || !input.contains("steps") || !input["steps"].is_array()) { error = "compact plan proposal requires goal and steps"; return false; }
    if (input["steps"].empty() || input["steps"].size() > max_operations) { error = "invalid number of compact plan steps"; return false; }
    plan.goal = input["goal"].get<std::string>();
    plan.purpose = input.value("purpose", plan.goal);
    plan.success_criteria = input.value("success_criteria", "Complete the requested task safely.");
    plan.next_action = input.value("next_action", "execute plan");
    bool has_final = false;
    std::set<std::string> seen_step_ids;
    size_t generated_index = 1;
    for (const auto & source : input["steps"]) {
        const std::string fallback_id = source.is_object() && source.contains("id") ? std::string() : next_generated_step_id(generated_index, seen_step_ids);
        common_plan_step step;
        if (!parse_step(source, plan.goal, fallback_id, step, error)) return false;
        if (!seen_step_ids.insert(step.id).second) { error = "duplicate step id"; return false; }
        if (step.id == fallback_id) ++generated_index;
        if (step.depends_on.empty() && !operations.empty() && !has_explicit_dependencies(source)) {
            const auto & previous = operations.back();
            if (previous.step) step.depends_on = {previous.step->id};
        }
        has_final = has_final || common_plan_step_effective_mode(step) == common_plan_step_mode::final_response;
        common_plan_operation operation;
        operation.kind = common_plan_operation_kind::add_step;
        operation.reason_summary = source.value("reason_summary", std::string());
        operation.step = std::move(step);
        operations.push_back(std::move(operation));
    }
    if (!has_final && operations.size() < max_operations) {
        common_plan_step final_step;
        final_step.id = "answer";
        if (seen_step_ids.count(final_step.id)) { error = "native final step id conflicts with proposed step"; return false; }
        final_step.title = "Answer";
        final_step.objective = "Answer the user using the completed plan.";
        for (const auto & operation : operations) if (operation.step) final_step.depends_on.push_back(operation.step->id);
        common_plan_operation final_operation;
        final_operation.kind = common_plan_operation_kind::add_step;
        final_operation.reason_summary = "native final synthesis";
        final_operation.step = std::move(final_step);
        operations.push_back(std::move(final_operation));
    }
    return true;
}

bool parse_legacy(const json & input, common_plan_state & plan, std::vector<common_plan_operation> & operations, std::string & error, size_t max_operations) {
    if (!input.contains("goal") || !input.contains("success_criteria") || !input.contains("next_action") || !input.contains("operations") ||
        !input["goal"].is_string() || !input["success_criteria"].is_string() || !input["next_action"].is_string() || !input["operations"].is_array()) { error = "plan proposal is missing required fields"; return false; }
    if (input["operations"].empty() || input["operations"].size() > max_operations) { error = "invalid number of plan operations"; return false; }
    plan.goal = input["goal"].get<std::string>();
    plan.purpose = input.value("purpose", plan.goal);
    plan.success_criteria = input["success_criteria"].get<std::string>();
    plan.next_action = input["next_action"].get<std::string>();
    std::set<std::string> seen_step_ids;
    for (const auto & item : input["operations"]) {
        if (!item.is_object() || item.value("kind", std::string()) != "add_step" || !item.contains("step")) { error = "unsupported plan operation"; return false; }
        common_plan_step step;
        if (!parse_step(item["step"], plan.goal, std::string(), step, error)) return false;
        if (!seen_step_ids.insert(step.id).second) { error = "duplicate step id"; return false; }
        common_plan_operation operation;
        operation.kind = common_plan_operation_kind::add_step;
        operation.reason_summary = item.value("reason_summary", std::string());
        if (item.contains("evidence_ids") && !string_array(item["evidence_ids"], operation.evidence_ids)) { error = "evidence_ids must be a string array"; return false; }
        operation.step = std::move(step);
        operations.push_back(std::move(operation));
    }
    return true;
}

} // namespace

std::string common_plan_proposal_json_schema() {
    const json schema = {
        {"type", "object"}, {"additionalProperties", false}, {"required", {"goal", "steps"}},
        {"properties", {
            {"purpose", {{"type", "string"}, {"maxLength", 256}}}, {"goal", {{"type", "string"}, {"maxLength", 256}}},
            {"success_criteria", {{"type", "string"}, {"maxLength", 256}}}, {"next_action", {{"type", "string"}, {"maxLength", 256}}},
            {"steps", {{"type", "array"}, {"minItems", 1}, {"maxItems", 5}, {"items", {{"type", "object"}, {"additionalProperties", false}, {"properties", {
                {"id", {{"type", "string"}, {"maxLength", 64}}}, {"title", {{"type", "string"}, {"maxLength", 128}}}, {"objective", {{"type", "string"}, {"maxLength", 256}}}, {"contribution", {{"type", "string"}, {"maxLength", 256}}},
                {"mode", {{"type", "string"}, {"enum", {"tool", "reasoning", "final", "final_response"}}}},
                {"after", {{"oneOf", json::array({json{{"type", "string"}}, json{{"type", "array"}, {"items", {{"type", "string"}}}}})}}},
                {"tool", {{"oneOf", json::array({
                    json{{"type", "string"}, {"maxLength", 256}},
                    json{{"type", "object"}, {"additionalProperties", false}, {"required", {"name"}}, {"properties", {{"name", {{"type", "string"}, {"maxLength", 256}}}, {"arguments", {{"type", "object"}}}, {"args", {{"type", "object"}}}}}}
                })}}},
                {"args", {{"type", "object"}}}
            }}}}}}
        }}
    };
    return schema.dump();
}

bool common_plan_parse_proposal_json(const std::string & text, common_plan_state & plan, std::vector<common_plan_operation> & operations, std::string & error, size_t max_operations) {
    try {
        const auto input = json::parse(text);
        if (!input.is_object()) { error = "plan proposal must be a JSON object"; return false; }
        common_plan_state parsed = plan;
        std::vector<common_plan_operation> parsed_operations;
        const bool compact = input.contains("steps");
        if (!(compact ? parse_compact(input, parsed, parsed_operations, error, max_operations) : parse_legacy(input, parsed, parsed_operations, error, max_operations))) return false;
        plan = std::move(parsed);
        operations = std::move(parsed_operations);
        error.clear();
        return true;
    } catch (const json::exception &) { error = "malformed plan proposal JSON"; return false; }
}
