#include "agent/adaptation/flydelta/oracles/flydelta-dataset-operation-oracle.h"

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
        result.reason = "observed decision is valid but does not match the host-grounded decision";
    }
    return true;
}
