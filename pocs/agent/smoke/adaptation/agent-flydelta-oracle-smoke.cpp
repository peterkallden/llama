#include "agent-flydelta-dataset-repair-host.h"

#include "agent/adaptation/flydelta/flydelta-semantic-decision.h"
#include "agent/adaptation/flydelta/oracles/flydelta-oracle-suite.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

using json = nlohmann::ordered_json;

namespace {

common_flydelta_oracle_request host_dataset_request(bool applicable = true) {
    common_flydelta_oracle_request request;
    request.oracle_ref = "flydelta://oracle/dataset-operation-host";
    request.oracle_revision = "v1";
    request.policy_revision = "policy-v1";
    request.phase = common_flydelta_oracle_phase::synthesis;
    request.concept_key = "dataset.grouped_sum";
    request.behavior_key = "dataset.grouped_sum";
    request.semantic_kind = "host_dataset";
    request.applicable = applicable;
    request.expected_decision_available = applicable;
    request.expected_decision.operation = "aggregate";
    request.expected_decision.dataset = "dataset://local/sales";
    request.expected_decision.group_by = {"region"};
    request.expected_decision.aggregate_function = "sum";
    request.expected_decision.aggregate_field = "amount";
    return request;
}

int fail(const std::string & reason) {
    std::cerr << "flydelta_oracle_smoke=failed reason=" << reason << '\n';
    return 1;
}

} // namespace

int main() {
    agent_flydelta_dataset_repair_host host;
    std::string error;
    if (!host.open("oracle", error)) return fail("host fixture: " + error);

    common_flydelta_oracle_suite_request suite;
    common_flydelta_oracle_probe target;
    target.probe_id = "grouped-sum-target";
    target.kind = common_flydelta_oracle_probe_kind::target;
    target.request = host_dataset_request();
    target.expected_verdict = common_flydelta_oracle_verdict::satisfied;
    common_flydelta_oracle_probe control;
    control.probe_id = "grouped-sum-control";
    control.kind = common_flydelta_oracle_probe_kind::control;
    control.request = host_dataset_request(false);
    control.expected_verdict = common_flydelta_oracle_verdict::not_applicable;
    suite.probes = {target, control};
    suite.evaluators.host_supported = [&host](
            const common_flydelta_oracle_request & request,
            const std::string & observed,
            common_flydelta_oracle_result & result,
            std::string & evaluator_error) {
        evaluator_error.clear();
        if (request.semantic_kind != "host_dataset") return false;
        result = {};
        result.oracle_ref = request.oracle_ref;
        result.oracle_revision = request.oracle_revision;
        result.policy_revision = request.policy_revision;
        if (!request.applicable) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::not_applicable;
            result.confidence = 1.0f;
            result.reason = "host applicability rule excluded the control";
            return true;
        }
        common_flydelta_semantic_decision actual;
        common_flydelta_semantic_decision_status status;
        std::string parse_error;
        if (!common_flydelta_parse_semantic_decision(
                observed, actual, status, parse_error)) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.confidence = 1.0f;
            result.reason = "host could not parse the semantic tool call: " + parse_error;
            return true;
        }
        if (!common_flydelta_semantic_decision_equal(request.expected_decision, actual)) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.confidence = 1.0f;
            result.reason = "host parsed a different semantic decision";
            return true;
        }
        const json parsed = json::parse(observed, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object() ||
                !parsed.contains("name") || !parsed["name"].is_string() ||
                !parsed.contains("arguments") || !parsed["arguments"].is_object()) {
            evaluator_error = "host oracle requires a canonical tool call";
            return false;
        }
        common_tool_execution_result execution;
        if (!host.execute_call(
                parsed["name"].get<std::string>(), parsed["arguments"], execution, evaluator_error)) {
            result.known = true;
            result.verdict = common_flydelta_oracle_verdict::violated;
            result.confidence = 0.95f;
            result.reason = "host rejected or could not execute the canonical call";
            evaluator_error.clear();
            return true;
        }
        result.known = true;
        result.verdict = common_flydelta_oracle_verdict::satisfied;
        result.confidence = 1.0f;
        result.evidence_ref = "execution://smoke/host-dataset-oracle";
        result.reason = "native host registry normalized and executed the call";
        return true;
    };
    suite.runner = [](const common_flydelta_oracle_probe & probe,
            bool candidate, std::string & observed, std::string & runner_error) {
        runner_error.clear();
        if (probe.kind == common_flydelta_oracle_probe_kind::control) {
            observed = "control is outside the concept scope";
        } else if (candidate) {
            observed = R"({"name":"data.aggregate","arguments":{"dataset":"dataset://local/sales","group_by":["region"],"measures":[{"function":"sum","column":"amount"}]}})";
        } else {
            observed = R"({"name":"data.filter","arguments":{"dataset":"dataset://local/sales","predicate":"region == north"}})";
        }
        return true;
    };

    common_flydelta_oracle_suite_result result;
    if (!common_flydelta_run_oracle_suite(suite, result, error)) {
        host.close();
        return fail(error);
    }
    const bool passed = result.semantically_helped && result.safe_to_continue &&
        result.candidate_success_rate == 1.0f && result.control_retention == 1.0f;
    std::cout << "flydelta_oracle_smoke=" << (passed ? "completed" : "failed")
              << " strength=host_supported"
              << " probes=" << result.probes.size()
              << " baseline_success_rate=" << result.baseline_success_rate
              << " candidate_success_rate=" << result.candidate_success_rate
              << " intervention_gain=" << result.intervention_gain
              << " false_intervention_rate=" << result.false_intervention_rate
              << " control_retention=" << result.control_retention
              << " semantically_helped=" << (result.semantically_helped ? "yes" : "no")
              << " safe_to_continue=" << (result.safe_to_continue ? "yes" : "no")
              << " learning_credit=none promotion=false\n";
    host.close();
    return passed ? 0 : 1;
}
