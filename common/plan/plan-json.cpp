#include "plan/plan-json.h"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <set>
#include <map>

using json = nlohmann::ordered_json;

namespace {

json normalize_tool_arguments(const std::string & tool_name, json arguments);
json normalize_safe_integer_arguments(json arguments);

enum class model_binding_parse_result { literal, binding, invalid };

model_binding_parse_result model_output_binding(const std::string & value, json & binding) {
    if (value.empty() || value.front() != '$') return model_binding_parse_result::literal;
    if (value.size() >= 2 && value.find('.', 1) == std::string::npos) {
        const std::string step_id = value.substr(1);
        const auto valid_identifier = [](const std::string & part) {
            return !part.empty() && std::all_of(part.begin(), part.end(), [](unsigned char ch) {
                return std::isalnum(ch) != 0 || ch == '_' || ch == '-';
            });
        };
        // A bare alias is a typed shorthand. The binding resolver supplies
        // the output field later, once the target input contract is known.
        if (valid_identifier(step_id)) {
            binding = json{{"$from_step", step_id}, {"$json_pointer", ""}};
            return model_binding_parse_result::binding;
        }
        return model_binding_parse_result::invalid;
    }
    if (value.size() < 5) return model_binding_parse_result::invalid;
    const auto separator = value.find('.', 1);
    if (separator == std::string::npos || separator == 1 || separator + 1 >= value.size()) return model_binding_parse_result::invalid;
    const auto valid_identifier = [](const std::string & part) {
        return !part.empty() && std::all_of(part.begin(), part.end(), [](unsigned char ch) {
            return std::isalnum(ch) != 0 || ch == '_' || ch == '-';
        });
    };
    const std::string step_id = value.substr(1, separator - 1);
    const std::string field_path = value.substr(separator + 1);
    std::string pointer = "/";
    size_t cursor = 0;
    while (cursor < field_path.size()) {
        const size_t field_start = cursor;
        while (cursor < field_path.size() && (std::isalnum(static_cast<unsigned char>(field_path[cursor])) != 0 || field_path[cursor] == '_' || field_path[cursor] == '-')) ++cursor;
        if (cursor == field_start) return model_binding_parse_result::invalid;
        const std::string field = field_path.substr(field_start, cursor - field_start);
        if (!valid_identifier(field)) return model_binding_parse_result::invalid;
        if (pointer.size() > 256 - field.size() - 1) return model_binding_parse_result::invalid;
        if (pointer.size() > 1) pointer += '/';
        pointer += field;
        while (cursor < field_path.size() && field_path[cursor] == '[') {
            ++cursor;
            const size_t index_start = cursor;
            while (cursor < field_path.size() && std::isdigit(static_cast<unsigned char>(field_path[cursor])) != 0) ++cursor;
            if (cursor == index_start || cursor >= field_path.size() || field_path[cursor] != ']') return model_binding_parse_result::invalid;
            const std::string index = field_path.substr(index_start, cursor - index_start);
            if (pointer.size() > 256 - index.size() - 1) return model_binding_parse_result::invalid;
            pointer += '/' + index;
            ++cursor;
        }
        if (cursor == field_path.size()) break;
        if (field_path[cursor] != '.') return model_binding_parse_result::invalid;
        ++cursor;
    }
    if (!valid_identifier(step_id) || pointer.size() <= 1) return model_binding_parse_result::invalid;
    binding = json{{"$from_step", step_id}, {"$json_pointer", pointer}};
    return model_binding_parse_result::binding;
}

bool normalize_model_output_bindings(
        json & value,
        const std::map<std::string, std::string> * aliases,
        std::string & error,
        bool strict_model_references) {
    if (value.is_array()) {
        for (auto & item : value) {
            if (!normalize_model_output_bindings(item, aliases, error, strict_model_references)) return false;
        }
        return true;
    }
    if (!value.is_object()) return true;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (it.value().is_string()) {
            json binding;
            const auto & shorthand = it.value().get_ref<const std::string &>();
            const auto parsed = model_output_binding(shorthand, binding);
            if (parsed == model_binding_parse_result::invalid && strict_model_references) {
                error = "plan.binding.invalid_syntax: '" + shorthand +
                    "' is not a valid model reference; expected '$alias', '$previous.field', '$alias.field' or '$alias.field[index]'";
                return false;
            }
            if (parsed == model_binding_parse_result::literal && strict_model_references && aliases != nullptr) {
                for (const auto & alias : *aliases) {
                    const std::string prefix = alias.first + ".";
                    if (shorthand.rfind(prefix, 0) == 0 && shorthand.size() > prefix.size()) {
                        error = "plan.binding.alias_used_as_literal: '" + shorthand +
                            "' looks like a reference to alias '" + alias.first +
                            "'; use '$" + shorthand + "'";
                        return false;
                    }
                }
            }
            if (parsed == model_binding_parse_result::binding) {
                if (aliases != nullptr) {
                    const auto from_step = binding.value("$from_step", std::string());
                    const auto alias = aliases->find(from_step);
                    if (alias == aliases->end()) {
                        error = "plan.binding.unknown_alias: reference '$" + from_step +
                            "' uses an alias that has not been declared; use $previous.field or declare as: '" + from_step + "'";
                        return false;
                    }
                    binding["$from_step"] = alias->second;
                }
                it.value() = std::move(binding);
            }
        } else {
            if (!normalize_model_output_bindings(it.value(), aliases, error, strict_model_references)) return false;
        }
    }
    return true;
}

std::string compact_schema_type(const json & schema) {
    if (schema.contains("enum") && schema["enum"].is_array()) {
        std::string result;
        for (const auto & value : schema["enum"]) {
            if (!result.empty()) result += '|';
            result += value.is_string() ? value.get<std::string>() : value.dump();
        }
        return result;
    }
    const auto type = schema.value("type", std::string("value"));
    if (type == "array") {
        return compact_schema_type(schema.value("items", json::object())) + "[]";
    }
    if (type == "object") return "object";
    return type;
}

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

bool parse_tool_arguments_contract(
        const std::string & tool_name,
        json arguments,
        common_plan_tool_arguments_contract & contract,
        std::string & error) {
    const bool memory_id_tool = tool_name == "memory_get" ||
        tool_name == "memory_propose_update" || tool_name == "memory_propose_forget";
    if (memory_id_tool && arguments.is_string()) {
        arguments = json{{"id", arguments.get<std::string>()}};
    }
    if (!arguments.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    contract.value = normalize_safe_integer_arguments(
        normalize_tool_arguments(tool_name, std::move(arguments)));
    if (!contract.value.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    error.clear();
    return true;
}

bool calculator_expression_from_value(const json & value, std::string & expression) {
    if (value.is_number_integer()) { expression = std::to_string(value.get<long long>()); return true; }
    if (value.is_number_unsigned()) { expression = std::to_string(value.get<unsigned long long>()); return true; }
    if (value.is_number_float()) { expression = std::to_string(value.get<double>()); return true; }
    return value.is_object() && value.contains("value") && calculator_expression_from_value(value["value"], expression);
}

json normalize_tool_arguments(const std::string & tool_name, json arguments) {
    if (!arguments.is_object()) return arguments;
    const bool document_table_tool = tool_name == "document.tables" || tool_name == "document.table";
    if (document_table_tool && !arguments.contains("resource")) {
        for (const char * alias : {"resource_uri", "uri"}) {
            if (arguments.contains(alias) && arguments[alias].is_string()) {
                arguments["resource"] = arguments[alias];
                arguments.erase(alias);
                break;
            }
        }
    }
    const bool memory_id_tool = tool_name == "memory_get" ||
        tool_name == "memory_propose_update" || tool_name == "memory_propose_forget";
    if (memory_id_tool && arguments.contains("memory_id") && !arguments.contains("id") &&
            arguments["memory_id"].is_string()) {
        arguments["id"] = arguments["memory_id"];
        arguments.erase("memory_id");
    }
    const auto unwrap_named_call = [&](const json & call, json & unwrapped) {
        if (!call.is_object() || !call.contains("name") || !call["name"].is_string() || call["name"].get<std::string>() != tool_name) return false;
        if (call.contains("arguments") && call["arguments"].is_object()) { unwrapped = call["arguments"]; return true; }
        if (call.contains("args") && call["args"].is_object()) { unwrapped = call["args"]; return true; }
        return false;
    };
    const auto unwrap_tool_and_arguments = [&](const json & tool, const json & payload, json & unwrapped) {
        if (!payload.is_object()) return false;
        if (tool.is_string() && tool.get<std::string>() == tool_name) { unwrapped = payload; return true; }
        return unwrap_named_call(tool, unwrapped);
    };
    json unwrapped;
    if (arguments.size() == 1 && arguments.contains("tool") && unwrap_named_call(arguments["tool"], unwrapped)) arguments = std::move(unwrapped);
    else if (arguments.size() <= 2 && arguments.contains("tool") && arguments.contains("arguments") &&
            unwrap_tool_and_arguments(arguments["tool"], arguments["arguments"], unwrapped)) arguments = std::move(unwrapped);
    else if (arguments.size() <= 2 && arguments.contains("tool") && arguments.contains("args") &&
            unwrap_tool_and_arguments(arguments["tool"], arguments["args"], unwrapped)) arguments = std::move(unwrapped);
    else if (arguments.size() <= 2 && arguments.contains("name") && unwrap_named_call(arguments, unwrapped)) arguments = std::move(unwrapped);
    else if (arguments.contains("tool") && arguments["tool"].is_string() && arguments["tool"].get<std::string>() == tool_name) {
        arguments.erase("tool");
    }
    if (tool_name == "document.table" && arguments.contains("table_name") &&
            !arguments.contains("table") && arguments["table_name"].is_string()) {
        arguments["table"] = arguments["table_name"];
        arguments.erase("table_name");
    }
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

bool parse_mode(const json & source, bool has_tool, common_plan_step_mode & mode, std::string & error) {
    const auto value = source.value("mode", has_tool ? std::string("tool") : std::string("final_response"));
    if (value == "tool") mode = common_plan_step_mode::tool;
    else if (value == "reasoning") mode = common_plan_step_mode::reasoning;
    else if (value == "final" || value == "final_response") mode = common_plan_step_mode::final_response;
    else { error = "invalid step mode"; return false; }
    if ((has_tool && mode != common_plan_step_mode::tool) || (!has_tool && mode == common_plan_step_mode::tool)) { error = "step mode does not match tool payload"; return false; }
    return true;
}

bool parse_tool(const json & source, common_plan_step & step, std::string & error,
        const std::map<std::string, std::string> * aliases = nullptr,
        bool strict_model_references = false) {
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

    // The model-facing shorthand $step.output is only syntax sugar. Store
    // the existing strict binding object in the plan IR so execution keeps
    // one binding path and one materialization policy.
    if (!normalize_model_output_bindings(arguments, aliases, error, strict_model_references)) return false;

    common_plan_tool_arguments_contract contract;
    if (!parse_tool_arguments_contract(name, std::move(arguments), contract, error)) {
        return false;
    }

    std::string normalized_arguments_json;
    if (!common_plan_serialize_tool_arguments_contract_json(
            name,
            contract,
            normalized_arguments_json,
            error)) {
        return false;
    }

    step.tool_call = common_plan_tool_call{name, std::move(normalized_arguments_json)};
    step.selected_tool = name;
    return true;
}

bool parse_step(const json & source, const std::string & goal, const std::string & fallback_id, common_plan_step & step, std::string & error,
        const std::map<std::string, std::string> * aliases = nullptr,
        bool host_owned_dependencies = false,
        bool strict_model_references = false) {
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
    if (host_owned_dependencies) step.depends_on.clear();
    else if (!normalize_dependencies(source, step.depends_on, error)) return false;
    if (source.contains("required_evidence") && !string_array(source["required_evidence"], step.required_evidence)) { error = "required_evidence must be a string array"; return false; }
    if (source.contains("source_memory_ids") && !string_array(source["source_memory_ids"], step.source_memory_ids)) { error = "source_memory_ids must be a string array"; return false; }
    const bool has_tool = source.contains("tool");
    if (!parse_mode(source, has_tool, step.mode, error) ||
            !parse_tool(source, step, error, aliases, strict_model_references)) return false;
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
    std::map<std::string, std::string> aliases;
    bool host_owned_dependencies = true;
    for (const auto & source : input["steps"]) {
        if (source.is_object() && source.contains("id")) {
            host_owned_dependencies = false;
            break;
        }
    }
    size_t generated_index = 1;
    for (const auto & source : input["steps"]) {
        // The normal model-facing form describes work for the host to run.
        // Final synthesis is host-owned, so an empty/default-final step is
        // never a useful model proposal. Reasoning remains an explicit
        // model-facing step when it is requested with mode: reasoning.
        if (host_owned_dependencies && source.is_object() && !source.contains("tool") &&
                source.value("mode", std::string("final_response")) != "reasoning") {
            error = "model plan steps require a tool or explicit mode: reasoning; final synthesis is host-owned";
            return false;
        }
        const std::string fallback_id = host_owned_dependencies || (source.is_object() && !source.contains("id"))
            ? next_generated_step_id(generated_index, seen_step_ids) : std::string();
        std::map<std::string, std::string> binding_aliases = aliases;
        if (!operations.empty() && operations.back().step) binding_aliases["previous"] = operations.back().step->id;
        if (!host_owned_dependencies) {
            for (const auto & known_id : seen_step_ids) binding_aliases.emplace(known_id, known_id);
        }
        common_plan_step step;
        if (!parse_step(source, plan.goal, fallback_id, step, error, &binding_aliases,
                host_owned_dependencies, host_owned_dependencies)) return false;
        if (!seen_step_ids.insert(step.id).second) { error = "duplicate step id"; return false; }
        if (step.id == fallback_id) ++generated_index;
        if ((host_owned_dependencies || step.depends_on.empty()) && !operations.empty()) {
            const auto & previous = operations.back();
            if (previous.step) step.depends_on = {previous.step->id};
        }
        if (source.is_object() && source.contains("as")) {
            if (!source["as"].is_string() || source["as"].get<std::string>().empty() || source["as"].get<std::string>().size() > 64) {
                error = "step alias must be a non-empty string of at most 64 characters";
                return false;
            }
            const auto alias = source["as"].get<std::string>();
            if (alias == "previous" || aliases.count(alias) || seen_step_ids.count(alias)) {
                error = "duplicate or reserved step alias";
                return false;
            }
            aliases[alias] = step.id;
            step.semantic_alias = alias;
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
                {"id", {{"type", "string"}, {"maxLength", 64}}}, {"as", {{"type", "string"}, {"maxLength", 64}}}, {"title", {{"type", "string"}, {"maxLength", 128}}}, {"objective", {{"type", "string"}, {"maxLength", 256}}}, {"contribution", {{"type", "string"}, {"maxLength", 256}}},
                {"mode", {{"type", "string"}, {"enum", {"tool", "reasoning", "final", "final_response"}}}},
                {"after", {{"type", "array"}, {"items", {{"type", "string"}}}}}, {"depends_on", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"tool", {{"type", "string"}, {"maxLength", 256}}},
                {"args", {{"type", "object"}}}
            }}}}}}
        }}
    };
    return schema.dump();
}

std::string common_plan_model_facing_json_schema(
        const std::vector<std::string> & allowed_tools) {
    json tool_schema = { {"type", "string"}, {"maxLength", 256} };
    if (!allowed_tools.empty()) tool_schema["enum"] = allowed_tools;
    const json schema = {
        {"type", "object"}, {"additionalProperties", false}, {"required", {"goal", "steps"}},
        {"properties", {
            {"goal", {{"type", "string"}, {"maxLength", 256}}},
            {"steps", {{"type", "array"}, {"minItems", 1}, {"maxItems", 5}, {"items", {
                {"type", "object"}, {"additionalProperties", false}, {"properties", {
                    {"tool", tool_schema},
                    {"args", {{"type", "object"}}},
                    {"as", {{"type", "string"}, {"minLength", 1}, {"maxLength", 64}}},
                    {"mode", {{"type", "string"}, {"enum", {"tool", "reasoning"}}}}
                }}
            }}}}
        }}
    };
    return schema.dump();
}

bool common_plan_normalize_tool_arguments_json(
        const std::string & tool_name,
        const std::string & arguments_json,
        std::string & normalized_json,
        std::string & error) {
    common_plan_tool_arguments_contract contract;
    if (!common_plan_parse_tool_arguments_contract_json(
            tool_name,
            arguments_json,
            contract,
            error)) {
        return false;
    }
    if (!common_plan_serialize_tool_arguments_contract_json(
            tool_name,
            contract,
            normalized_json,
            error)) {
        return false;
    }
    error.clear();
    return true;
}

std::string common_render_compact_plan_schema(
        const std::string & schema_json,
        std::string & error) {
    error.clear();
    const auto schema = json::parse(schema_json, nullptr, false);
    if (schema.is_discarded() || !schema.is_object() ||
            schema.value("type", std::string()) != "object") {
        error = "plan schema is not a JSON object schema";
        return {};
    }
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object() || !properties.contains("steps") ||
            !properties["steps"].is_object()) {
        error = "plan schema is missing steps";
        return {};
    }
    const auto step_schema = properties["steps"].value("items", json::object());
    if (!step_schema.is_object()) {
        error = "plan schema steps are missing an item schema";
        return {};
    }
    std::string required;
    for (const auto & value : schema.value("required", json::array())) {
        if (!value.is_string()) continue;
        if (!required.empty()) required += "; ";
        required += value.get<std::string>() + ':' +
            compact_schema_type(properties.value(value.get<std::string>(), json::object()));
    }
    std::string optional;
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (it.key() == "steps") continue;
        bool is_required = false;
        for (const auto & value : schema.value("required", json::array())) {
            if (value.is_string() && value.get<std::string>() == it.key()) {
                is_required = true;
                break;
            }
        }
        if (is_required) continue;
        if (!optional.empty()) optional += "; ";
        optional += it.key() + '?' + ':' + compact_schema_type(it.value());
    }
    std::string result = "plan\nrequired: " + required +
        "\noptional: " + (optional.empty() ? "none" : optional) +
        "\nsteps: step[]" +
        "\nstep fields: tool?:string; args?:object; as?:string; mode?:string" +
        "\nstep order: host assigns step IDs and sequential dependencies when id/after/depends_on are omitted" +
        "\noutput binding: use $previous.field or $alias.field; host resolves both to typed step bindings";
    return result;
}

bool common_plan_merge_tool_arguments_json(
        const std::string & base_json,
        const std::string & patch_json,
        std::string & merged_json,
        std::string & error) {
    const auto base = json::parse(base_json, nullptr, false);
    const auto patch = json::parse(patch_json, nullptr, false);
    if (!base.is_object() || !patch.is_object()) {
        error = "tool argument repair patch requires JSON objects";
        return false;
    }
    json merged = base;
    for (const auto & item : patch.items()) merged[item.key()] = item.value();
    merged_json = merged.dump();
    error.clear();
    return true;
}

bool common_plan_parse_tool_arguments_contract_json(
        const std::string & tool_name,
        const std::string & arguments_json,
        common_plan_tool_arguments_contract & contract,
        std::string & error) {
    const auto parsed = json::parse(arguments_json, nullptr, false);
    return parse_tool_arguments_contract(tool_name, parsed, contract, error);
}

bool common_plan_parse_tool_arguments_contract_value(
        const std::string & tool_name,
        const nlohmann::ordered_json & arguments,
        common_plan_tool_arguments_contract & contract,
        std::string & error) {
    return parse_tool_arguments_contract(tool_name, arguments, contract, error);
}

bool common_plan_serialize_tool_arguments_contract_json(
        const std::string & tool_name,
        const common_plan_tool_arguments_contract & contract,
        std::string & arguments_json,
        std::string & error) {
    (void) tool_name;
    if (!contract.value.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    arguments_json = normalize_safe_integer_arguments(contract.value).dump();
    error.clear();
    return true;
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
