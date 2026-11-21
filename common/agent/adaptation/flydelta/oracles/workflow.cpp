#include "agent/adaptation/flydelta/oracles/workflow.h"

#include <nlohmann/json.hpp>

#include <utility>

using json = nlohmann::ordered_json;

namespace {

void initialize_result(
        const common_flydelta_workflow_contract & contract,
        common_flydelta_oracle_result & result) {
    result = {};
    result.oracle_ref = contract.workflow_ref.empty()
        ? "flydelta://oracle/workflow" : contract.workflow_ref;
    result.oracle_revision = contract.workflow_revision.empty()
        ? "v1" : contract.workflow_revision;
    result.evaluator_ref = "flydelta://evaluator/workflow";
    result.evaluator_revision = "v1";
    result.confidence = 1.0f;
}

bool parse_canonical_steps(
        const std::string & observed,
        std::vector<common_tool_workflow_step_view> & steps,
        std::string & reason) {
    steps.clear();
    const auto value = json::parse(observed, nullptr, false);
    if (value.is_discarded() || !value.is_object() ||
            !value.contains("steps") || !value["steps"].is_array()) {
        reason = "workflow observation is not a canonical step view";
        return false;
    }
    if (value["steps"].size() > 256U) {
        reason = "workflow observation exceeds the step limit";
        return false;
    }
    for (const auto & item : value["steps"]) {
        if (!item.is_object() || !item.contains("tool") ||
                !item["tool"].is_string() || !item.contains("arguments") ||
                !item["arguments"].is_string()) {
            reason = "workflow observation contains an invalid step";
            return false;
        }
        common_tool_workflow_step_view step;
        step.tool_name = item.value("tool", "");
        step.arguments_json = item.value("arguments", "");
        if (step.tool_name.empty() || step.tool_name.size() > 256U ||
                step.arguments_json.size() > 64U * 1024U) {
            reason = "workflow observation contains an unbounded step";
            return false;
        }
        steps.push_back(std::move(step));
    }
    return true;
}

bool validate_terminal_goal(
        const std::vector<common_tool_workflow_step_view> & steps,
        const std::vector<std::string> & required_terminal_tools,
        common_tool_workflow_validation_result & result) {
    if (required_terminal_tools.empty()) return true;
    if (steps.empty()) {
        result.status = common_tool_workflow_validation_status::violated;
        result.violation_code = "workflow.goal_terminal_missing";
        result.reason = "workflow goal requires a terminal tool";
        result.checks.push_back({result.violation_code, result.status, {"goal"}});
        return false;
    }
    const auto & terminal = steps.back().tool_name;
    if (std::find(required_terminal_tools.begin(), required_terminal_tools.end(), terminal) !=
            required_terminal_tools.end()) return true;
    result.status = common_tool_workflow_validation_status::violated;
    result.violation_code = "workflow.goal_terminal_mismatch";
    result.reason = "workflow terminal step does not satisfy the blueprint goal";
    result.checks.push_back({result.violation_code, result.status, {"goal", "operation"}});
    return false;
}

void copy_checks(
        const common_tool_workflow_validation_result & workflow_result,
        const std::string & evidence_ref,
        common_flydelta_oracle_result & result) {
    for (const auto & check : workflow_result.checks) {
        result.checks.push_back({
            check.code,
            check.status == common_tool_workflow_validation_status::satisfied
                ? common_flydelta_oracle_verdict::satisfied
                : check.status == common_tool_workflow_validation_status::violated
                    ? common_flydelta_oracle_verdict::violated
                    : check.status == common_tool_workflow_validation_status::not_applicable
                        ? common_flydelta_oracle_verdict::not_applicable
                        : common_flydelta_oracle_verdict::unknown,
            check.status != common_tool_workflow_validation_status::unknown,
            check.dimensions,
            evidence_ref,
        });
    }
}

} // namespace

common_flydelta_oracle_evaluator common_flydelta_make_workflow_oracle(
        common_flydelta_workflow_contract contract) {
    return [contract = std::move(contract)](
            const common_flydelta_oracle_request & request,
            const std::string & observed,
            common_flydelta_oracle_result & result,
            std::string & error) {
        error.clear();
        if (request.semantic_kind != "tool_workflow" &&
                request.semantic_kind != "workflow") {
            return false;
        }
        if (!request.expected_contract_ref.empty() &&
                request.expected_contract_ref != contract.workflow_ref) {
            return false;
        }
        if (!request.expected_contract_revision.empty() &&
                request.expected_contract_revision != contract.workflow_revision) {
            return false;
        }
        initialize_result(contract, result);
        result.policy_revision = request.policy_revision;

        if (contract.schema_version != 1 || contract.workflow_ref.empty() ||
                contract.workflow_revision.empty() || contract.workflows.empty() ||
                contract.selected_workflow_ids.empty()) {
            result.known = false;
            result.verdict = common_flydelta_oracle_verdict::unknown;
            result.reason = "workflow contract is incomplete";
            return true;
        }

        std::vector<common_tool_workflow_step_view> steps;
        std::string parse_reason;
        if (!parse_canonical_steps(observed, steps, parse_reason)) {
            result.known = false;
            result.verdict = common_flydelta_oracle_verdict::unknown;
            result.reason = parse_reason;
            return true;
        }

        common_tool_workflow_validation_result workflow_result;
        if (!common_evaluate_tool_workflow_plan(
                contract.workflows, contract.selected_workflow_ids, steps,
                workflow_result, error)) {
            result.known = false;
            result.verdict = common_flydelta_oracle_verdict::unknown;
            result.reason = error.empty() ? "workflow evaluation failed" : error;
            return true;
        }

        if (workflow_result.status == common_tool_workflow_validation_status::satisfied) {
            validate_terminal_goal(steps, contract.required_terminal_tools, workflow_result);
        }

        copy_checks(workflow_result, contract.workflow_ref, result);
        result.reason = workflow_result.reason;
        result.violation_code = workflow_result.violation_code;
        switch (workflow_result.status) {
            case common_tool_workflow_validation_status::satisfied:
                result.known = true;
                result.verdict = common_flydelta_oracle_verdict::satisfied;
                break;
            case common_tool_workflow_validation_status::violated:
                result.known = true;
                result.verdict = common_flydelta_oracle_verdict::violated;
                result.violation_kind = common_flydelta_oracle_violation_kind::contract_violation;
                if (result.violation_code.rfind("workflow.", 0) != 0) {
                    result.violation_code = "workflow." + result.violation_code;
                }
                break;
            case common_tool_workflow_validation_status::not_applicable:
                result.known = true;
                result.verdict = common_flydelta_oracle_verdict::not_applicable;
                break;
            case common_tool_workflow_validation_status::unknown:
                result.known = false;
                result.verdict = common_flydelta_oracle_verdict::unknown;
                break;
        }
        return true;
    };
}
