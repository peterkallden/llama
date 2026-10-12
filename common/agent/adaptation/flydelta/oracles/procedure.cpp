#include "agent/adaptation/flydelta/oracles/procedure.h"

#include <nlohmann/json.hpp>

#include <utility>

using json = nlohmann::ordered_json;

namespace {

bool canonical_identity(
        const std::string & observed,
        std::string & procedure_ref,
        std::string & blueprint_ref,
        std::string & reason) {
    const auto value = json::parse(observed, nullptr, false);
    if (value.is_discarded() || !value.is_object()) {
        reason = "procedure observation is not a canonical object";
        return false;
    }
    const auto read = [&](const char * key, std::string & destination) {
        if (!value.contains(key) || !value.at(key).is_string()) return false;
        destination = value.at(key).get<std::string>();
        return !destination.empty() && destination.size() <= 512U;
    };
    if (!read("procedure_ref", procedure_ref) || !read("blueprint_ref", blueprint_ref)) {
        reason = "procedure observation requires bounded procedure_ref and blueprint_ref";
        return false;
    }
    return true;
}

void initialize_result(
        const common_flydelta_procedure_contract & contract,
        common_flydelta_oracle_result & result) {
    result = {};
    result.oracle_ref = contract.procedure_ref.empty()
        ? "flydelta://oracle/procedure" : contract.procedure_ref;
    result.oracle_revision = contract.procedure_revision.empty()
        ? "v1" : contract.procedure_revision;
    result.evaluator_ref = "flydelta://evaluator/procedure";
    result.evaluator_revision = "v1";
    result.confidence = 1.0f;
}

} // namespace

common_flydelta_oracle_evaluator common_flydelta_make_procedure_oracle(
        common_flydelta_procedure_contract contract) {
    return [contract = std::move(contract)](
            const common_flydelta_oracle_request & request,
            const std::string & observed,
            common_flydelta_oracle_result & result,
            std::string & error) {
        error.clear();
        if (request.semantic_kind != "procedure" &&
                request.semantic_kind != "blueprint" &&
                request.semantic_kind != "procedure_blueprint") {
            return false;
        }
        if (!request.expected_contract_ref.empty() &&
                request.expected_contract_ref != contract.procedure_ref) {
            return false;
        }
        if (!request.expected_contract_revision.empty() &&
                request.expected_contract_revision != contract.procedure_revision) {
            return false;
        }
        initialize_result(contract, result);
        result.policy_revision = request.policy_revision;

        if (contract.schema_version != 1 || contract.procedure_ref.empty() ||
                contract.procedure_revision.empty() || contract.blueprint_ref.empty() ||
                contract.blueprint_revision.empty()) {
            result.known = false;
            result.verdict = common_flydelta_oracle_verdict::unknown;
            result.reason = "procedure contract is incomplete";
            return true;
        }

        std::string procedure_ref;
        std::string blueprint_ref;
        std::string identity_reason;
        if (!canonical_identity(observed, procedure_ref, blueprint_ref, identity_reason)) {
            result.known = false;
            result.verdict = common_flydelta_oracle_verdict::unknown;
            result.reason = identity_reason;
            return true;
        }
        if (procedure_ref != contract.procedure_ref) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.violation_kind = common_flydelta_oracle_violation_kind::contract_violation;
            result.violation_code = "procedure.wrong_procedure";
            result.reason = "canonical observation belongs to a different procedure";
            result.checks.push_back({result.violation_code, result.verdict, true, {"procedure"}, contract.procedure_ref});
            return true;
        }
        if (blueprint_ref != contract.blueprint_ref) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.violation_kind = common_flydelta_oracle_violation_kind::contract_violation;
            result.violation_code = "procedure.wrong_blueprint";
            result.reason = "canonical observation belongs to a different blueprint";
            result.checks.push_back({result.violation_code, result.verdict, true, {"blueprint"}, contract.blueprint_ref});
            return true;
        }

        auto workflow = common_flydelta_make_workflow_oracle(contract.workflow);
        common_flydelta_oracle_request workflow_request = request;
        workflow_request.semantic_kind = "workflow";
        workflow_request.expected_contract_ref = contract.workflow.workflow_ref;
        workflow_request.expected_contract_revision = contract.workflow.workflow_revision;
        common_flydelta_oracle_result workflow_result;
        if (!workflow(workflow_request, observed, workflow_result, error)) {
            result.known = false;
            result.verdict = common_flydelta_oracle_verdict::unknown;
            result.reason = error.empty() ? "procedure workflow evaluator declined" : error;
            return true;
        }
        result = std::move(workflow_result);
        result.oracle_ref = contract.procedure_ref;
        result.oracle_revision = contract.procedure_revision;
        result.evaluator_ref = "flydelta://evaluator/procedure";
        result.evaluator_revision = "v1";
        if (result.known && result.verdict == common_flydelta_oracle_verdict::violated &&
                result.violation_code.rfind("workflow.", 0) == 0) {
            result.violation_code = "procedure." + result.violation_code;
            if (!result.checks.empty()) result.checks.back().code = result.violation_code;
        }
        return true;
    };
}

