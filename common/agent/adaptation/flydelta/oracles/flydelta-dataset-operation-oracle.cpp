#include "agent/adaptation/flydelta/oracles/flydelta-dataset-operation-oracle.h"

#include <algorithm>

namespace {

void add_dimension(common_flydelta_oracle_result & result, const char * dimension) {
    if (std::find(result.violation_dimensions.begin(), result.violation_dimensions.end(),
            dimension) == result.violation_dimensions.end()) {
        result.violation_dimensions.emplace_back(dimension);
    }
}

void classify_aggregate_difference(
        const common_flydelta_semantic_decision & expected,
        const common_flydelta_semantic_decision & actual,
        common_flydelta_oracle_result & result) {
    if (expected.operation != actual.operation) {
        result.violation_kind = common_flydelta_oracle_violation_kind::wrong_tool;
        add_dimension(result, "operation");
        return;
    }
    if (expected.operation != "aggregate") {
        result.violation_kind = common_flydelta_oracle_violation_kind::unattributable;
        add_dimension(result, "semantic_decision");
        return;
    }
    if (actual.group_by.empty()) {
        result.violation_kind = common_flydelta_oracle_violation_kind::missing_required_grouping;
        add_dimension(result, "grouping");
    } else if (actual.group_by != expected.group_by) {
        result.violation_kind = common_flydelta_oracle_violation_kind::wrong_grouping_field;
        add_dimension(result, "grouping");
    } else if (actual.aggregate_field.empty()) {
        result.violation_kind = common_flydelta_oracle_violation_kind::missing_measure;
        add_dimension(result, "measure");
    } else if (actual.aggregate_field != expected.aggregate_field) {
        result.violation_kind = common_flydelta_oracle_violation_kind::wrong_measure_column;
        add_dimension(result, "measure");
    } else if (actual.aggregate_function != expected.aggregate_function) {
        result.violation_kind = common_flydelta_oracle_violation_kind::wrong_measure_function;
        add_dimension(result, "measure");
    } else {
        result.violation_kind = common_flydelta_oracle_violation_kind::unattributable;
        add_dimension(result, "semantic_decision");
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
    if (!request.applicable) {
        result.known = true;
        result.verdict = common_flydelta_oracle_verdict::not_applicable;
        result.confidence = 1.0f;
        result.reason = "dataset operation concept does not apply to this probe";
        return true;
    }
    if (!request.expected_decision_available) {
        result.reason = "dataset operation request has no host-grounded expected decision";
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
            result.violation_kind = common_flydelta_oracle_violation_kind::wrong_tool;
            add_dimension(result, "operation");
        } else if (status == common_flydelta_semantic_decision_status::missing_field &&
                request.expected_decision.operation == "aggregate" &&
                parse_error.find("group_by") != std::string::npos) {
            result.violation_kind = common_flydelta_oracle_violation_kind::missing_required_grouping;
            add_dimension(result, "grouping");
        } else {
            result.violation_kind = common_flydelta_oracle_violation_kind::unattributable;
            add_dimension(result, "contract_shape");
        }
        result.reason = "observed output is not a valid dataset semantic decision: " + parse_error;
        return true;
    }
    result.known = true;
    result.confidence = 1.0f;
    if (common_flydelta_semantic_decision_equal(request.expected_decision, actual)) {
        result.verdict = common_flydelta_oracle_verdict::satisfied;
        result.reason = "observed decision matches the host-grounded semantic decision";
    } else {
        result.verdict = common_flydelta_oracle_verdict::violated;
        classify_aggregate_difference(request.expected_decision, actual, result);
        result.reason = "observed decision is valid but does not match the host-grounded decision";
    }
    return true;
}
