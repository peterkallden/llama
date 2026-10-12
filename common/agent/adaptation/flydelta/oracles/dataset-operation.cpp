#include "agent/adaptation/flydelta/oracles/dataset-operation.h"

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
    if (!request.expected_decision_available) {
        result.reason = "dataset operation request has no host-grounded expected decision";
        add_check(result, "dataset.expected_decision", result.verdict, "contract");
        return true;
    }

    common_flydelta_semantic_decision actual;
    common_flydelta_semantic_decision_status status;
    std::string parse_error;
    if (!common_flydelta_parse_semantic_decision(observed, actual, status, parse_error)) {
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
