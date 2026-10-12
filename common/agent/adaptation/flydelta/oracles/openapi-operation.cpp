#include "agent/adaptation/flydelta/oracles/openapi-operation.h"

#include <algorithm>
#include <utility>

namespace {

void add_check(
        common_flydelta_oracle_result & result,
        const char * code,
        common_flydelta_oracle_verdict verdict,
        const char * dimension) {
    common_flydelta_oracle_check check;
    check.code = code;
    check.verdict = verdict;
    if (dimension != nullptr && *dimension != '\0') check.dimensions.emplace_back(dimension);
    result.checks.push_back(std::move(check));
}

void replace_violation(
        common_flydelta_oracle_result & result,
        const char * code,
        const char * dimension) {
    result.violation_kind = common_flydelta_oracle_violation_kind::contract_violation;
    result.violation_code = code;
    if (dimension != nullptr && *dimension != '\0' &&
            std::find(result.violation_dimensions.begin(), result.violation_dimensions.end(),
                dimension) == result.violation_dimensions.end()) {
        result.violation_dimensions.emplace_back(dimension);
    }
    add_check(result, code, common_flydelta_oracle_verdict::violated, dimension);
}

const char * openapi_code_for_tool_violation(const std::string & code) {
    if (code == "tool.missing_required_parameter" ||
            code == "tool.host_requirement_missing") {
        return "openapi.missing_required_parameter";
    }
    if (code == "tool.invalid_argument_type" ||
            code == "tool.invalid_argument_value" ||
            code == "tool.invalid_arguments") {
        return "openapi.invalid_parameter_value";
    }
    if (code == "tool.unknown_argument") return "openapi.unknown_parameter";
    if (code == "tool.confirmation_required") return "openapi.confirmation_required";
    if (code == "tool.policy_rejected") return "openapi.access_policy_rejected";
    if (code == "tool.malformed_call") return "openapi.malformed_call";
    return nullptr;
}

} // namespace

common_flydelta_oracle_evaluator common_flydelta_make_openapi_operation_oracle(
        common_flydelta_openapi_operation_contract contract) {
    return [contract = std::move(contract)](
            const common_flydelta_oracle_request & request,
            const std::string & observed,
            common_flydelta_oracle_result & result,
            std::string & error) {
        if (request.semantic_kind != "openapi_operation" &&
                request.semantic_kind != "openapi") return false;
        if (!request.expected_contract_ref.empty() &&
                request.expected_contract_ref != contract.tool.contract_ref) return false;
        if (!request.expected_contract_revision.empty() &&
                request.expected_contract_revision != contract.tool.contract_revision) return false;

        common_agent_tool_output_format format;
        if (!common_parse_agent_tool_output_format(
                request.observed_format.empty() ? "jsonl" : request.observed_format,
                format, error)) return true;
        if (!common_flydelta_validate_model_tool_text(
                contract.tool, format, observed, result, error)) return false;

        result.oracle_ref = request.oracle_ref.empty()
            ? "flydelta://oracle/openapi-operation" : request.oracle_ref;
        result.oracle_revision = request.oracle_revision.empty()
            ? contract.tool.contract_revision : request.oracle_revision;
        result.policy_revision = request.policy_revision;

        if (result.violation_code == "tool.unknown_tool") {
            common_agent_tool_call observed_call;
            std::string parse_error;
            const bool parsed = common_flydelta_parse_model_tool_text(
                format, observed, observed_call, parse_error);
            const bool same_operation_name = parsed &&
                (observed_call.name == contract.operation_id ||
                 observed_call.name == contract.tool.exposed_tool_name ||
                 (observed_call.name.size() > contract.operation_id.size() &&
                  observed_call.name.compare(
                      observed_call.name.size() - contract.operation_id.size(),
                      contract.operation_id.size(), contract.operation_id) == 0));
            replace_violation(result,
                same_operation_name ? "openapi.wrong_provider" : "openapi.wrong_operation",
                same_operation_name ? "provider" : "operation");
            result.reason = same_operation_name
                ? "observed call uses the wrong OpenAPI provider"
                : "observed call does not select the expected OpenAPI operation";
            return true;
        }

        if (result.verdict == common_flydelta_oracle_verdict::violated) {
            if (const char * code = openapi_code_for_tool_violation(result.violation_code)) {
                replace_violation(result, code, "parameters");
            }
            return true;
        }
        if (!result.known || result.verdict != common_flydelta_oracle_verdict::satisfied) {
            return true;
        }

        add_check(result, "openapi.provider", common_flydelta_oracle_verdict::satisfied, "provider");
        add_check(result, "openapi.operation", common_flydelta_oracle_verdict::satisfied, "operation");
        add_check(result, "openapi.method_path", common_flydelta_oracle_verdict::satisfied, "operation");
        result.reason = "model-facing call satisfies the selected OpenAPI operation contract";
        return true;
    };
}
