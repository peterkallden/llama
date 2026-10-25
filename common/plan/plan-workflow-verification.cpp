#include "plan-workflow-verification.h"
#include <set>
#include <stdexcept>

using json = nlohmann::ordered_json;
namespace {
const char * phases[] = {"before_tool", "after_tool", "before_answer"};
const char * operations[] = {"required", "equal", "member", "count", "status", "dimension_membership", "coverage"};
json operand_json(const common_plan_verification_operand & operand) {
    json value = {{"source", operand.source}, {"transition", operand.transition_id},
        {"pointer", operand.pointer}, {"value", json::parse(operand.literal_json)},
        {"dataset_rows", operand.dataset_rows}};
    if (!operand.tool_name.empty()) value["tool"] = operand.tool_name;
    return value;
}
bool read_operand(const json & value, common_plan_verification_operand & operand) {
    if (!value.is_object()) return false;
    operand.source = value.value("source", std::string("literal"));
    operand.transition_id = value.value("transition", std::string{});
    operand.pointer = value.value("pointer", std::string{});
    operand.literal_json = value.value("value", json{}).dump();
    operand.dataset_rows = value.value("dataset_rows", false);
    operand.tool_name = value.value("tool", std::string{});
    if (operand.tool_name.size() > 128) return false;
    if (operand.source != "literal" && operand.source != "arguments" &&
            operand.source != "result" && operand.source != "observation" && operand.source != "handoff" &&
            operand.source != "transition_arguments") return false;
    if ((operand.source == "observation" || operand.source == "transition_arguments") && operand.transition_id.empty() && operand.tool_name.empty()) return false;
    try { (void) json::json_pointer(operand.pointer); } catch (...) { return false; }
    return true;
}
}
void common_plan_workflow_grounding_to_json(const common_plan_workflow_definition & definition, json & value) {
    if (!definition.preferred_thinking_mode.empty()) value["preferred_thinking_mode"] = definition.preferred_thinking_mode;
    if (definition.require_declared_claims) value["require_declared_claims"] = true;
    if (!definition.declared_claim_tools.empty()) value["declared_claim_tools"] = definition.declared_claim_tools;
    if (definition.verification_checks.empty()) return;
    value["verification_checks"] = json::array();
    for (const auto & check : definition.verification_checks) {
        json item = {{"id", check.id}, {"transition", check.transition_id},
            {"phase", phases[static_cast<int>(check.phase)]},
            {"operation", operations[static_cast<int>(check.operation)]},
            {"left", operand_json(check.left)}, {"right", operand_json(check.right)},
            {"required", check.required}, {"dimension_mapping", json::parse(check.dimension_mapping_json)},
            {"clarification_question", check.clarification_question}};
        if (!check.tool_name.empty()) item["tool"] = check.tool_name;
        value["verification_checks"].push_back(std::move(item));
    }
}
bool common_plan_workflow_grounding_from_json(const json & value,
        common_plan_workflow_definition & definition, std::string & error) {
    try {
        definition.preferred_thinking_mode = value.value("preferred_thinking_mode", std::string{});
        if (!definition.preferred_thinking_mode.empty() && definition.preferred_thinking_mode != "deliberate" &&
                definition.preferred_thinking_mode != "research") throw std::runtime_error("invalid preferred_thinking_mode");
        definition.require_declared_claims = value.value("require_declared_claims", false);
        definition.declared_claim_tools.clear();
        const auto claim_tools = value.value("declared_claim_tools", json::array());
        if (!claim_tools.is_array() || claim_tools.size() > 32) throw std::runtime_error("declared_claim_tools must be a bounded array");
        for (const auto & tool : claim_tools) {
            if (!tool.is_string() || tool.get<std::string>().empty() || tool.get<std::string>().size() > 128)
                throw std::runtime_error("declared claim tool must be a canonical operation name");
            definition.declared_claim_tools.push_back(tool.get<std::string>());
        }
        definition.verification_checks.clear();
        const auto checks = value.value("verification_checks", json::array());
        if (!checks.is_array() || checks.size() > 32) throw std::runtime_error("verification_checks must be a bounded array");
        std::set<std::string> ids;
        for (const auto & item : checks) {
            common_plan_verification_check check;
            check.id = item.at("id").get<std::string>();
            check.transition_id = item.value("transition", std::string{});
            if (check.id.empty() || check.id.size() > 128 || !ids.insert(check.id).second) throw std::runtime_error("invalid verification id");
            const auto phase = item.at("phase").get<std::string>();
            const auto operation = item.at("operation").get<std::string>();
            size_t p = 0, o = 0;
            while (p < 3 && phase != phases[p]) ++p;
            while (o < 7 && operation != operations[o]) ++o;
            check.tool_name = item.value("tool", std::string{});
            if (check.tool_name.size() > 128 || (p == 2 && !check.tool_name.empty()) ||
                    p == 3 || o == 7 || (p != 2 && check.transition_id.empty() && check.tool_name.empty()))
                throw std::runtime_error("invalid verification phase/operation selector");
            check.phase = static_cast<common_plan_verification_phase>(p);
            check.operation = static_cast<common_plan_verification_operation>(o);
            check.required = item.value("required", true);
            check.clarification_question = item.value("clarification_question", std::string{});
            if (check.clarification_question.size() > 512) throw std::runtime_error("clarification question exceeds text bound");
            if (!read_operand(item.at("left"), check.left) || !read_operand(item.value("right", json::object()), check.right))
                throw std::runtime_error("invalid verification operand");
            const auto mapping = item.value("dimension_mapping", json::object());
            if (!mapping.is_object()) throw std::runtime_error("invalid dimension mapping");
            if ((check.operation == common_plan_verification_operation::dimension_membership ||
                    check.operation == common_plan_verification_operation::coverage) && mapping.empty())
                throw std::runtime_error("dimension verification requires a non-empty mapping");
            for (const auto & field : mapping.items()) if (!field.value().is_string()) throw std::runtime_error("invalid dimension name");
            check.dimension_mapping_json = mapping.dump();
            definition.verification_checks.push_back(std::move(check));
        }
        error.clear(); return true;
    } catch (const std::exception & exception) {
        error = std::string("workflow grounding contract: ") + exception.what(); return false;
    }
}
