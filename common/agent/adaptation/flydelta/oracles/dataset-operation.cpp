#include "agent/adaptation/flydelta/oracles/dataset-operation.h"
#include "agent/adaptation/flydelta/oracles/tool-contract.h"
#include "agent/tooling/contracts/schema-contract.h"

#include <algorithm>
#include <utility>

namespace {

void add_dimension(common_flydelta_oracle_result & result, const char * dimension) {
    if (std::find(result.violation_dimensions.begin(), result.violation_dimensions.end(),
            dimension) == result.violation_dimensions.end()) {
        result.violation_dimensions.emplace_back(dimension);
    }
}

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

void set_violation(
        common_flydelta_oracle_result & result,
        common_flydelta_oracle_violation_kind kind,
        const char * code,
        const char * dimension) {
    result.violation_kind = kind;
    result.violation_code = code;
    if (dimension != nullptr && *dimension != '\0') add_dimension(result, dimension);
    add_check(result, code, common_flydelta_oracle_verdict::violated, dimension);
}

void classify_aggregate_difference(
        const common_flydelta_semantic_decision & expected,
        const common_flydelta_semantic_decision & actual,
        common_flydelta_oracle_result & result) {
    if (expected.operation != actual.operation) {
        set_violation(result, common_flydelta_oracle_violation_kind::wrong_tool,
            "dataset.wrong_operation", "operation");
        return;
    }
    if (expected.operation != "aggregate") {
        set_violation(result, common_flydelta_oracle_violation_kind::unattributable,
            "dataset.unattributable", "semantic_decision");
        return;
    }
    if (actual.group_by.empty()) {
        set_violation(result, common_flydelta_oracle_violation_kind::missing_required_grouping,
            "dataset.missing_required_grouping", "grouping");
    } else if (actual.group_by != expected.group_by) {
        set_violation(result, common_flydelta_oracle_violation_kind::wrong_grouping_field,
            "dataset.wrong_grouping_field", "grouping");
    } else if (actual.aggregate_field.empty()) {
        set_violation(result, common_flydelta_oracle_violation_kind::missing_measure,
            "dataset.missing_measure", "measure");
    } else if (actual.aggregate_field != expected.aggregate_field) {
        set_violation(result, common_flydelta_oracle_violation_kind::wrong_measure_column,
            "dataset.wrong_measure_column", "measure");
    } else if (actual.aggregate_function != expected.aggregate_function) {
        set_violation(result, common_flydelta_oracle_violation_kind::wrong_measure_function,
            "dataset.wrong_measure_function", "measure");
    } else {
        set_violation(result, common_flydelta_oracle_violation_kind::unattributable,
            "dataset.unattributable", "semantic_decision");
    }
}

} // namespace

bool common_flydelta_dataset_operation_oracle(
        const common_flydelta_oracle_request & request,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error) {
    error.clear();
    if (request.semantic_kind != "dataset_operation" &&
            request.semantic_kind != "dataset") return false;
    result = {};
    result.oracle_ref = request.oracle_ref.empty()
        ? "flydelta://oracle/dataset-operation" : request.oracle_ref;
    result.oracle_revision = request.oracle_revision.empty() ? "v1" : request.oracle_revision;
    result.policy_revision = request.policy_revision;
    result.evaluator_ref = "flydelta://evaluator/dataset-operation";
    result.evaluator_revision = "v1";
    if (!request.applicable) {
        result.known = true;
        result.verdict = common_flydelta_oracle_verdict::not_applicable;
        result.confidence = 1.0f;
        result.reason = "dataset operation concept does not apply to this probe";
        add_check(result, "dataset.applicability", result.verdict, "applicability");
        return true;
    }
    if (request.expected_tool_call_available) {
        const auto schema = std::find_if(request.model_tools.begin(), request.model_tools.end(),
            [&](const common_chat_tool & tool) { return tool.name == request.expected_tool_call.name; });
        if (schema == request.model_tools.end() || schema->parameters.empty()) {
            result.reason = "host tool schema is unavailable for the expected call";
            return true;
        }
        std::string expected_arguments;
        if (!common_schema_normalize_and_validate_object(request.expected_tool_call.arguments_json,
                schema->parameters, expected_arguments, error)) return false;
        common_agent_tool_call call;
        std::string parse_error;
        const bool parsed = common_flydelta_parse_model_tool_text(
                common_agent_tool_output_format::jsonl, observed, call, parse_error) ||
            common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl,
                observed, request.model_tools, call, parse_error);
        result.known = true;
        result.confidence = 1.0f;
        std::string actual_arguments;
        if (!parsed || call.name != request.expected_tool_call.name ||
                !common_schema_normalize_and_validate_object(call.arguments_json,
                    schema->parameters, actual_arguments, parse_error)) {
            result.verdict = common_flydelta_oracle_verdict::violated;
            set_violation(result, parsed && call.name != request.expected_tool_call.name
                    ? common_flydelta_oracle_violation_kind::wrong_tool
                    : common_flydelta_oracle_violation_kind::contract_violation,
                "dataset.tool_contract", "arguments");
            result.reason = parse_error.empty() ? "observed tool differs from expected tool" : parse_error;
            return true;
        }
        result.normalized_arguments_json = actual_arguments;
        const bool matches = nlohmann::json::parse(actual_arguments) ==
            nlohmann::json::parse(expected_arguments);
        result.verdict = matches ? common_flydelta_oracle_verdict::satisfied
                                 : common_flydelta_oracle_verdict::violated;
        if (!matches) set_violation(result, common_flydelta_oracle_violation_kind::contract_violation,
            "dataset.arguments_mismatch", "arguments");
        result.reason = matches ? "normalized tool call matches the host-grounded contract"
                                : "normalized arguments differ from the host-grounded contract";
        return true;
    }
    if (!request.expected_decision_available) {
        result.reason = "dataset operation request has no host-grounded expected decision";
        add_check(result, "dataset.expected_decision", result.verdict, "contract");
        return true;
    }

    std::string semantic_text = observed;
    if (observed.find("open!") == 0) {
        if (request.model_tools.empty()) {
            result.reason = "DSL verification requires the host-owned model tool schemas";
            return true;
        }
        common_agent_tool_call call;
        std::string parse_error;
        if (!common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl,
                observed, request.model_tools, call, parse_error)) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.reason = parse_error;
            set_violation(result, common_flydelta_oracle_violation_kind::contract_violation,
                "dataset.tool_contract", "arguments");
            return true;
        }
        semantic_text = nlohmann::ordered_json{{"name", call.name},
            {"arguments", nlohmann::ordered_json::parse(call.arguments_json)}}.dump();
    }
    common_flydelta_semantic_decision actual;
    common_flydelta_semantic_decision_status status;
    std::string parse_error;
    if (!common_flydelta_parse_semantic_decision(semantic_text, actual, status, parse_error)) {
        result.known = true;
        result.verdict = common_flydelta_oracle_verdict::violated;
        result.confidence = 0.95f;
        // Unsupported operation and bounded missing-field failures can still
        // be attributed to the host contract. Generic malformed output may
        // not become negative material merely because it failed parsing.
        if (status == common_flydelta_semantic_decision_status::unsupported_operation) {
            set_violation(result, common_flydelta_oracle_violation_kind::wrong_tool,
                "dataset.wrong_operation", "operation");
        } else if (status == common_flydelta_semantic_decision_status::missing_field &&
                request.expected_decision.operation == "aggregate" &&
                parse_error.find("group_by") != std::string::npos) {
            set_violation(result, common_flydelta_oracle_violation_kind::missing_required_grouping,
                "dataset.missing_required_grouping", "grouping");
        } else {
            set_violation(result, common_flydelta_oracle_violation_kind::unattributable,
                "dataset.unattributable", "contract_shape");
        }
        result.reason = "observed output is not a valid dataset semantic decision: " + parse_error;
        return true;
    }
    result.known = true;
    result.confidence = 1.0f;
    if (common_flydelta_semantic_decision_equal(request.expected_decision, actual)) {
        result.verdict = common_flydelta_oracle_verdict::satisfied;
        result.reason = "observed decision matches the host-grounded semantic decision";
        add_check(result, "dataset.semantic_decision", result.verdict, "semantic_decision");
    } else {
        result.verdict = common_flydelta_oracle_verdict::violated;
        classify_aggregate_difference(request.expected_decision, actual, result);
        result.reason = "observed decision is valid but does not match the host-grounded decision";
    }
    return true;
}
