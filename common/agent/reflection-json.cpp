#include "agent/reflection-json.h"
#include "agent/schema-contract.h"

bool common_reflection_parse_json(const std::string & text, common_reflection_result & result, std::string & error, size_t max_operations) {
    common_json_contract_value j;
    if (!common_json_contract_parse_object(text, j, error)) return false;
    try {
        result = {};
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
        if (j.contains("learning_hint")) {
            const auto & hint = j["learning_hint"];
            if (!hint.is_object()) { error = "invalid reflection learning hint"; return false; }
            common_reflection_learning_hint parsed;
            if (!common_json_contract_required_string(hint, "category", 64, parsed.category, error) ||
                    !common_json_contract_required_string(hint, "statement", 512, parsed.statement, error) ||
                    !common_json_contract_optional_unit_number(hint, "expected_reuse", 0.5f, parsed.expected_reuse, error)) return false;
            result.learning_hint = std::move(parsed);
        }
        if (j.contains("operations")) {
            if (!j["operations"].is_array() || j["operations"].size() > max_operations) { error = "too many reflection operations"; return false; }
            for (const auto & item : j["operations"]) {
                if (!item.is_object() || !item.contains("kind") || !item["kind"].is_string()) { error = "invalid reflection operation"; return false; }
                common_plan_operation op;
                const auto kind = item["kind"].get<std::string>();
                if (kind == "complete_step") op.kind = common_plan_operation_kind::complete_step;
                else if (kind == "activate_step") op.kind = common_plan_operation_kind::activate_step;
                else if (kind == "set_next_action") op.kind = common_plan_operation_kind::set_next_action;
                else if (kind == "add_step") op.kind = common_plan_operation_kind::add_step;
                else { error = "unsupported reflection operation"; return false; }
                op.reason_summary = item.value("reason_summary", std::string{});
                if (op.kind == common_plan_operation_kind::set_next_action) {
                    if (!item.contains("value") || !item["value"].is_string()) { error = "set_next_action requires value"; return false; }
                    op.value = item["value"].get<std::string>();
                } else if (op.kind == common_plan_operation_kind::add_step) {
                    if (!item.contains("step") || !item["step"].is_object()) { error = "add_step requires step"; return false; }
                    const auto & step = item["step"];
                    if (!step.contains("id") || !step.contains("title") || !step.contains("objective") || !step["id"].is_string() || !step["title"].is_string() || !step["objective"].is_string()) { error = "invalid reflection add_step"; return false; }
                    common_plan_step parsed;
                    parsed.id = step["id"].get<std::string>();
                    parsed.title = step["title"].get<std::string>();
                    parsed.objective = step["objective"].get<std::string>();
                    if (step.contains("depends_on") && step["depends_on"].is_array()) parsed.depends_on = step["depends_on"].get<std::vector<std::string>>();
                    if (step.contains("required_evidence") && step["required_evidence"].is_array()) parsed.required_evidence = step["required_evidence"].get<std::vector<std::string>>();
                    if (step.contains("source_memory_ids") && step["source_memory_ids"].is_array()) parsed.source_memory_ids = step["source_memory_ids"].get<std::vector<std::string>>();
                    op.step = std::move(parsed);
                    if (item.contains("evidence_ids") && item["evidence_ids"].is_array()) op.evidence_ids = item["evidence_ids"].get<std::vector<std::string>>();
                } else {
                    if (!item.contains("step_id") || !item["step_id"].is_string()) { error = "step reflection operation requires step_id"; return false; }
                    op.step_id = item["step_id"].get<std::string>();
                }
                result.proposed_plan_operations.push_back(std::move(op));
            }
        }
        error.clear();
        return true;
    } catch (const std::exception &) {
        error = "malformed reflection JSON";
        return false;
    }
}
