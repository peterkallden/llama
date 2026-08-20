#include "agent/thinking/reflection-json.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "plan/plan-json.h"

#include <set>

namespace {

bool parse_string_array(const common_json_contract_value & value, size_t max_items, size_t max_length, std::vector<std::string> & out, std::string & error) {
    if (!value.is_array() || value.size() > max_items) { error = "reflection string list is invalid"; return false; }
    out.clear();
    for (const auto & item : value) {
        if (!item.is_string()) { error = "reflection string list must contain only strings"; return false; }
        const auto parsed = item.get<std::string>();
        if (parsed.empty() || parsed.size() > max_length) { error = "reflection string list item is invalid"; return false; }
        out.push_back(parsed);
    }
    return true;
}

bool parse_optional_step_ids(const common_json_contract_value & object, const char * key, size_t max_items, std::vector<std::string> & out, std::string & error) {
    if (!object.contains(key)) { out.clear(); return true; }
    return parse_string_array(object[key], max_items, 64, out, error);
}

bool parse_dependencies(const common_json_contract_value & source, std::vector<std::string> & dependencies, std::string & error) {
    const auto it = source.contains("after") ? source.find("after") : source.find("depends_on");
    if (it == source.end()) { dependencies.clear(); return true; }
    if (it->is_string()) { dependencies = {it->get<std::string>()}; return true; }
    return parse_string_array(*it, 8, 64, dependencies, error);
}

std::string next_generated_repair_id(size_t index, const std::set<std::string> & seen_ids) {
    std::string id = "repair_" + std::to_string(index);
    while (seen_ids.count(id)) {
        ++index;
        id = "repair_" + std::to_string(index);
    }
    return id;
}

bool normalize_tool_arguments_contract(
    const std::string & tool_name,
    const common_json_contract_value & arguments,
    std::string & normalized_arguments,
    std::string & error) {
    common_plan_tool_arguments_contract contract;
    if (!common_plan_parse_tool_arguments_contract_value(
            tool_name,
            arguments,
            contract,
            error)) {
        return false;
    }
    return common_plan_serialize_tool_arguments_contract_json(
        tool_name,
        contract,
        normalized_arguments,
        error);
}

bool parse_compact_tool(
    const common_json_contract_value & item,
    common_plan_step & step,
    std::string & error) {
    if (!item.contains("tool")) return true;
    const auto & tool = item["tool"];
    std::string name;
    common_json_contract_value arguments = common_json_contract_value::object();
    if (tool.is_string()) name = tool.get<std::string>();
    else if (tool.is_object() && tool.contains("name") && tool["name"].is_string()) {
        name = tool["name"].get<std::string>();
        if (tool.contains("arguments")) arguments = tool["arguments"];
        else if (tool.contains("args")) arguments = tool["args"];
    } else {
        error = "invalid reflection add_steps tool payload";
        return false;
    }
    if (item.contains("args")) arguments = item["args"];
    const bool memory_id_tool = name == "memory_get" ||
        name == "memory_propose_update" || name == "memory_propose_forget";
    if (!arguments.is_object() && !(memory_id_tool && arguments.is_string())) {
        error = "reflection add_steps tool arguments must be an object";
        return false;
    }
    std::string normalized_arguments;
    if (!normalize_tool_arguments_contract(name, arguments, normalized_arguments, error)) return false;
    step.tool_call = common_plan_tool_call{name, normalized_arguments};
    step.selected_tool = name;
    return true;
}

bool parse_compact_replace_step(
    const common_json_contract_value & item,
    common_plan_operation & op,
    std::string & error) {
    if (!item.is_object()) { error = "invalid reflection replace_steps item"; return false; }
    std::string step_id;
    if (item.contains("step_id") && item["step_id"].is_string() && !item["step_id"].get<std::string>().empty()) step_id = item["step_id"].get<std::string>();
    else if (item.contains("id") && item["id"].is_string() && !item["id"].get<std::string>().empty()) step_id = item["id"].get<std::string>();
    else { error = "reflection replace_steps step_id must be a non-empty string"; return false; }
    common_plan_step parsed;
    parsed.id = step_id;
    if (item.contains("title") && !item["title"].is_string()) { error = "reflection replace_steps title must be a string"; return false; }
    if (item.contains("objective") && !item["objective"].is_string()) { error = "reflection replace_steps objective must be a string"; return false; }
    if (item.contains("contribution") && !item["contribution"].is_string()) { error = "reflection replace_steps contribution must be a string"; return false; }
    parsed.title = item.value("title", step_id);
    parsed.objective = item.value("objective", std::string("Repair the failed step after reflection."));
    parsed.intended_contribution = item.value("contribution", parsed.objective);
    if (!parse_dependencies(item, parsed.depends_on, error)) return false;
    if (item.contains("required_evidence") && !parse_string_array(item["required_evidence"], 8, 128, parsed.required_evidence, error)) return false;
    if (item.contains("source_memory_ids") && !parse_string_array(item["source_memory_ids"], 8, 128, parsed.source_memory_ids, error)) return false;
    if (!parse_compact_tool(item, parsed, error)) return false;
    if (item.contains("mode")) {
        if (!item["mode"].is_string()) { error = "reflection replace_steps mode must be a string"; return false; }
        const auto mode = item["mode"].get<std::string>();
        if (mode == "tool") parsed.mode = common_plan_step_mode::tool;
        else if (mode == "reasoning") parsed.mode = common_plan_step_mode::reasoning;
        else if (mode == "final" || mode == "final_response") parsed.mode = common_plan_step_mode::final_response;
        else { error = "invalid reflection replace_steps mode"; return false; }
    } else if (parsed.tool_call) {
        parsed.mode = common_plan_step_mode::tool;
    }
    op.kind = common_plan_operation_kind::replace_step;
    op.step_id = step_id;
    op.reason_summary = item.value("reason_summary", std::string{});
    op.step = parsed;
    return true;
}

bool parse_compact_add_step(
    const common_json_contract_value & item,
    size_t & generated_index,
    std::set<std::string> & seen_ids,
    std::optional<common_plan_step> & previous_step,
    common_plan_operation & op,
    std::string & error) {
    if (!item.is_object()) { error = "invalid reflection add_steps item"; return false; }
    common_plan_step parsed;
    if (item.contains("id")) {
        if (!item["id"].is_string() || item["id"].get<std::string>().empty()) { error = "reflection add_steps id must be a non-empty string"; return false; }
        parsed.id = item["id"].get<std::string>();
    } else {
        parsed.id = next_generated_repair_id(generated_index, seen_ids);
        ++generated_index;
    }
    if (!seen_ids.insert(parsed.id).second) { error = "duplicate reflection add_step id"; return false; }
    if (item.contains("title") && !item["title"].is_string()) { error = "reflection add_steps title must be a string"; return false; }
    if (item.contains("objective") && !item["objective"].is_string()) { error = "reflection add_steps objective must be a string"; return false; }
    if (item.contains("contribution") && !item["contribution"].is_string()) { error = "reflection add_steps contribution must be a string"; return false; }
    parsed.title = item.value("title", parsed.id);
    parsed.objective = item.value("objective", std::string("Repair the plan after reflection."));
    parsed.intended_contribution = item.value("contribution", parsed.objective);
    if (!parse_dependencies(item, parsed.depends_on, error)) return false;
    if (parsed.depends_on.empty() && previous_step) parsed.depends_on = {previous_step->id};
    if (item.contains("required_evidence") && !parse_string_array(item["required_evidence"], 8, 128, parsed.required_evidence, error)) return false;
    if (item.contains("source_memory_ids") && !parse_string_array(item["source_memory_ids"], 8, 128, parsed.source_memory_ids, error)) return false;
    if (!parse_compact_tool(item, parsed, error)) return false;
    if (item.contains("mode")) {
        if (!item["mode"].is_string()) { error = "reflection add_steps mode must be a string"; return false; }
        const auto mode = item["mode"].get<std::string>();
        if (mode == "tool") parsed.mode = common_plan_step_mode::tool;
        else if (mode == "reasoning") parsed.mode = common_plan_step_mode::reasoning;
        else if (mode == "final" || mode == "final_response") parsed.mode = common_plan_step_mode::final_response;
        else { error = "invalid reflection add_steps mode"; return false; }
    } else if (parsed.tool_call) {
        parsed.mode = common_plan_step_mode::tool;
    }
    op.kind = common_plan_operation_kind::add_step;
    op.reason_summary = item.value("reason_summary", std::string{});
    op.step = parsed;
    previous_step = std::move(parsed);
    return true;
}

} // namespace

bool common_reflection_parse_json(const std::string & text, common_reflection_result & result, std::string & error, size_t max_operations) {
    common_json_contract_value j;
    if (!common_json_contract_parse_object(text, j, error)) return false;
    try {
        auto generation = std::move(result.generation);
        result = {};
        result.generation = std::move(generation);
        std::string d;
        if (!common_json_contract_required_string(j, "decision", 32, d, error)) return false;
        if (d == "accept") result.decision = common_reflection_decision::accept;
        else if (d == "revise") result.decision = common_reflection_decision::revise;
        else if (d == "replan") result.decision = common_reflection_decision::replan;
        else if (d == "request_action") result.decision = common_reflection_decision::request_action;
        else if (d == "abort") result.decision = common_reflection_decision::abort;
        else { error = "unsupported reflection decision"; return false; }
        if (j.contains("ready_to_answer") && !j["ready_to_answer"].is_boolean()) { error = "ready_to_answer must be a boolean"; return false; }
        result.ready_to_answer = j.value("ready_to_answer", false);
        if (!common_json_contract_optional_unit_number(j, "confidence", 0.5f, result.confidence, error) ||
                !common_json_contract_optional_string_array(j, "revision_guidance", 4, 512, result.revision_guidance, error)) return false;
        if (j.contains("assurance_action")) {
            if (!j["assurance_action"].is_string()) { error = "assurance_action must be a string"; return false; }
            const auto action = j["assurance_action"].get<std::string>();
            if (action == "accept") result.next_action = common_agent_reflection_next_action::accept;
            else if (action == "revise_response") result.next_action = common_agent_reflection_next_action::revise_response;
            else if (action == "revise_plan") result.next_action = common_agent_reflection_next_action::revise_plan;
            else if (action == "escalate_deliberate") result.next_action = common_agent_reflection_next_action::escalate_deliberate;
            else if (action == "escalate_research") result.next_action = common_agent_reflection_next_action::escalate_research;
            else if (action == "fail_bounded") result.next_action = common_agent_reflection_next_action::fail_bounded;
            else { error = "unsupported reflection assurance action"; return false; }
        }
        if (j.contains("learning_hint")) {
            const auto & hint = j["learning_hint"];
            if (!hint.is_object()) { error = "invalid reflection learning hint"; return false; }
            common_reflection_learning_hint parsed;
            if (!common_json_contract_required_string(hint, "category", 64, parsed.category, error) ||
                    !common_json_contract_required_string(hint, "statement", 512, parsed.statement, error) ||
                    !common_json_contract_optional_unit_number(hint, "expected_reuse", 0.5f, parsed.expected_reuse, error)) return false;
            result.learning_hint = std::move(parsed);
        }
        std::vector<std::string> complete_ids;
        if (!parse_optional_step_ids(j, "complete", max_operations, complete_ids, error)) return false;
        for (const auto & step_id : complete_ids) {
            common_plan_operation op;
            op.kind = common_plan_operation_kind::complete_step;
            op.step_id = step_id;
            result.proposed_plan_operations.push_back(std::move(op));
        }
        std::vector<std::string> activate_ids;
        if (!parse_optional_step_ids(j, "activate", max_operations, activate_ids, error)) return false;
        for (const auto & step_id : activate_ids) {
            common_plan_operation op;
            op.kind = common_plan_operation_kind::activate_step;
            op.step_id = step_id;
            result.proposed_plan_operations.push_back(std::move(op));
        }
        std::vector<std::string> retry_ids;
        if (!parse_optional_step_ids(j, "retry", max_operations, retry_ids, error)) return false;
        for (const auto & step_id : retry_ids) {
            common_plan_operation op;
            op.kind = common_plan_operation_kind::activate_step;
            op.step_id = step_id;
            result.proposed_plan_operations.push_back(std::move(op));
        }
        std::vector<std::string> reset_ids;
        if (!parse_optional_step_ids(j, "reset", max_operations, reset_ids, error)) return false;
        for (const auto & step_id : reset_ids) {
            common_plan_operation op;
            op.kind = common_plan_operation_kind::reset_step;
            op.step_id = step_id;
            result.proposed_plan_operations.push_back(std::move(op));
        }
        if (j.contains("next_action")) {
            if (!j["next_action"].is_string() || j["next_action"].get<std::string>().empty()) { error = "next_action must be a non-empty string"; return false; }
            common_plan_operation op;
            op.kind = common_plan_operation_kind::set_next_action;
            op.value = j["next_action"].get<std::string>();
            result.proposed_plan_operations.push_back(std::move(op));
        }
        if (j.contains("add_steps")) {
            if (!j["add_steps"].is_array() || j["add_steps"].size() > max_operations) { error = "too many reflection add_steps"; return false; }
            size_t generated_index = 1;
            std::set<std::string> seen_ids;
            std::optional<common_plan_step> previous_step;
            for (const auto & item : j["add_steps"]) {
                common_plan_operation op;
                if (!parse_compact_add_step(item, generated_index, seen_ids, previous_step, op, error)) return false;
                result.proposed_plan_operations.push_back(std::move(op));
            }
        }
        if (j.contains("replace_steps")) {
            if (!j["replace_steps"].is_array() || j["replace_steps"].size() > max_operations) { error = "too many reflection replace_steps"; return false; }
            for (const auto & item : j["replace_steps"]) {
                common_plan_operation op;
                if (!parse_compact_replace_step(item, op, error)) return false;
                result.proposed_plan_operations.push_back(std::move(op));
            }
        }
        if (j.contains("operations")) {
            if (!j["operations"].is_array() || j["operations"].size() > max_operations) { error = "too many reflection operations"; return false; }
            for (const auto & item : j["operations"]) {
                if (!item.is_object() || !item.contains("kind") || !item["kind"].is_string()) { error = "invalid reflection operation"; return false; }
                common_plan_operation op;
                const auto kind = item["kind"].get<std::string>();
                if (kind == "complete_step") op.kind = common_plan_operation_kind::complete_step;
                else if (kind == "activate_step") op.kind = common_plan_operation_kind::activate_step;
                else if (kind == "reset_step") op.kind = common_plan_operation_kind::reset_step;
                else if (kind == "set_next_action") op.kind = common_plan_operation_kind::set_next_action;
                else if (kind == "add_constraint") op.kind = common_plan_operation_kind::add_constraint;
                else if (kind == "add_assumption") op.kind = common_plan_operation_kind::add_assumption;
                else if (kind == "invalidate_assumption") op.kind = common_plan_operation_kind::invalidate_assumption;
                else if (kind == "add_step") op.kind = common_plan_operation_kind::add_step;
                else if (kind == "replace_step") op.kind = common_plan_operation_kind::replace_step;
                else { error = "unsupported reflection operation"; return false; }
                op.reason_summary = item.value("reason_summary", std::string{});
                if (op.kind == common_plan_operation_kind::set_next_action) {
                    if (!item.contains("value") || !item["value"].is_string()) { error = "set_next_action requires value"; return false; }
                    op.value = item["value"].get<std::string>();
                } else if (op.kind == common_plan_operation_kind::add_constraint) {
                    if (!item.contains("constraint") || !item["constraint"].is_object()) { error = "add_constraint requires constraint"; return false; }
                    const auto & constraint = item["constraint"];
                    common_plan_constraint parsed;
                    if (!common_json_contract_required_string(constraint, "id", 128, parsed.id, error) ||
                            !common_json_contract_required_string(constraint, "description", 4096, parsed.description, error)) return false;
                    if (constraint.contains("hard") && !constraint["hard"].is_boolean()) { error = "constraint hard must be a boolean"; return false; }
                    parsed.hard = constraint.value("hard", true);
                    op.constraint = std::move(parsed);
                } else if (op.kind == common_plan_operation_kind::add_assumption) {
                    if (!item.contains("assumption") || !item["assumption"].is_object()) { error = "add_assumption requires assumption"; return false; }
                    const auto & assumption = item["assumption"];
                    common_plan_assumption parsed;
                    if (!common_json_contract_required_string(assumption, "id", 128, parsed.id, error) ||
                            !common_json_contract_required_string(assumption, "statement", 4096, parsed.statement, error) ||
                            !common_json_contract_optional_unit_number(assumption, "confidence", 0.5f, parsed.confidence, error)) return false;
                    if (assumption.contains("evidence_ids") && !assumption["evidence_ids"].is_array()) { error = "assumption evidence_ids must be an array"; return false; }
                    if (assumption.contains("evidence_ids")) parsed.evidence_ids = assumption["evidence_ids"].get<std::vector<std::string>>();
                    parsed.valid = true;
                    op.assumption = std::move(parsed);
                } else if (op.kind == common_plan_operation_kind::invalidate_assumption) {
                    if (!item.contains("target_id") || !item["target_id"].is_string() || item["target_id"].get<std::string>().empty()) { error = "invalidate_assumption requires target_id"; return false; }
                    op.target_id = item["target_id"].get<std::string>();
                } else if (op.kind == common_plan_operation_kind::add_step || op.kind == common_plan_operation_kind::replace_step) {
                    if (!item.contains("step") || !item["step"].is_object()) { error = op.kind == common_plan_operation_kind::add_step ? "add_step requires step" : "replace_step requires step"; return false; }
                    const auto & step = item["step"];
                    if (!step.contains("id") || !step.contains("title") || !step.contains("objective") || !step["id"].is_string() || !step["title"].is_string() || !step["objective"].is_string()) { error = op.kind == common_plan_operation_kind::add_step ? "invalid reflection add_step" : "invalid reflection replace_step"; return false; }
                    common_plan_step parsed;
                    parsed.id = step["id"].get<std::string>();
                    parsed.title = step["title"].get<std::string>();
                    parsed.objective = step["objective"].get<std::string>();
                    if (step.contains("depends_on") && step["depends_on"].is_array()) parsed.depends_on = step["depends_on"].get<std::vector<std::string>>();
                    if (step.contains("required_evidence") && step["required_evidence"].is_array()) parsed.required_evidence = step["required_evidence"].get<std::vector<std::string>>();
                    if (step.contains("source_memory_ids") && step["source_memory_ids"].is_array()) parsed.source_memory_ids = step["source_memory_ids"].get<std::vector<std::string>>();
                    if (step.contains("mode") && step["mode"].is_string()) {
                        const auto mode = step["mode"].get<std::string>();
                        if (mode == "tool") parsed.mode = common_plan_step_mode::tool;
                        else if (mode == "reasoning") parsed.mode = common_plan_step_mode::reasoning;
                        else if (mode == "final" || mode == "final_response") parsed.mode = common_plan_step_mode::final_response;
                    }
                    if (step.contains("tool_call") && step["tool_call"].is_object()) {
                        const auto & tool_call = step["tool_call"];
                        if (!tool_call.contains("name") || !tool_call["name"].is_string()) { error = "reflection tool_call requires name"; return false; }
                        const auto tool_name = tool_call["name"].get<std::string>();
                        const auto arguments = tool_call.contains("arguments_json") && tool_call["arguments_json"].is_string() ? tool_call["arguments_json"].get<std::string>() : std::string("{}");
                        std::string normalized_arguments;
                        if (!common_plan_normalize_tool_arguments_json(tool_name, arguments, normalized_arguments, error)) { return false; }
                        parsed.tool_call = common_plan_tool_call{tool_name, normalized_arguments};
                        parsed.selected_tool = tool_name;
                        parsed.mode = common_plan_step_mode::tool;
                    }
                    op.step = parsed;
                    if (op.kind == common_plan_operation_kind::replace_step) op.step_id = parsed.id;
                    if (item.contains("evidence_ids") && item["evidence_ids"].is_array()) op.evidence_ids = item["evidence_ids"].get<std::vector<std::string>>();
                } else {
                    if (!item.contains("step_id") || !item["step_id"].is_string()) { error = "step reflection operation requires step_id"; return false; }
                    op.step_id = item["step_id"].get<std::string>();
                }
                result.proposed_plan_operations.push_back(std::move(op));
            }
        }
        if (result.proposed_plan_operations.size() > max_operations) { error = "too many reflection operations"; return false; }
        error.clear();
        return true;
    } catch (const std::exception &) {
        error = "malformed reflection JSON";
        return false;
    }
}
