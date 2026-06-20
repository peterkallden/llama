#include "plan/plan-json.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

std::string common_plan_proposal_json_schema() {
    const json schema = {
        {"type", "object"},
        {"additionalProperties", false},
        {"required", {"goal", "success_criteria", "operations", "next_action"}},
        {"properties", {
            {"goal", {{"type", "string"}, {"maxLength", 4096}}},
            {"success_criteria", {{"type", "string"}, {"maxLength", 4096}}},
            {"next_action", {{"type", "string"}, {"maxLength", 4096}}},
            {"operations", {{"type", "array"}, {"maxItems", 8}, {"items", {{"type", "object"}, {"additionalProperties", false}, {"required", {"kind", "step", "reason_summary", "evidence_ids"}}, {"properties", {
                {"kind", {{"const", "add_step"}}},
                {"reason_summary", {{"type", "string"}, {"maxLength", 4096}}},
                {"evidence_ids", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                {"step", {{"type", "object"}, {"additionalProperties", false}, {"required", {"id", "title", "objective", "depends_on", "required_evidence"}}, {"properties", {
                    {"id", {{"type", "string"}, {"maxLength", 256}}}, {"title", {{"type", "string"}, {"maxLength", 1024}}}, {"objective", {{"type", "string"}, {"maxLength", 4096}}},
                    {"depends_on", {{"type", "array"}, {"items", {{"type", "string"}}}}}, {"required_evidence", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                    {"tool", {{"type", "object"}, {"additionalProperties", false}, {"required", {"name", "arguments"}}, {"properties", {{"name", {{"type", "string"}, {"maxLength", 256}}}, {"arguments", {{"type", "object"}}}}}}}
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
        if (input["operations"].size() > max_operations) { error = "too many plan operations"; return false; }
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
            if (source.contains("tool")) { if (!source["tool"].is_object() || !source["tool"].contains("name") || !source["tool"].contains("arguments") || !source["tool"]["name"].is_string() || !source["tool"]["arguments"].is_object()) { error = "invalid step tool payload"; return false; } step.tool_call = common_plan_tool_call{source["tool"]["name"].get<std::string>(), source["tool"]["arguments"].dump()}; step.selected_tool = step.tool_call->name; }
            common_plan_operation operation; operation.kind = common_plan_operation_kind::add_step; operation.step = std::move(step); operation.reason_summary = item.value("reason_summary", std::string()); if (item.contains("evidence_ids") && item["evidence_ids"].is_array()) operation.evidence_ids = item["evidence_ids"].get<std::vector<std::string>>(); operation.step->generated_from_memory = !operation.evidence_ids.empty(); parsed_operations.push_back(std::move(operation));
        }
        plan = std::move(parsed); operations = std::move(parsed_operations); error.clear(); return true;
    } catch (const json::exception &) { error = "malformed plan proposal JSON"; return false; }
}
