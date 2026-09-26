#include "agent/adaptation/flydelta/oracles/flydelta-dataset-operation-oracle.h"
#include "agent/adaptation/flydelta/oracles/flydelta-oracle-astar-proposer.h"
#include "agent/adaptation/flydelta/oracles/flydelta-oracle-suite.h"

#include <string>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

common_flydelta_oracle_request aggregate_request(bool applicable = true) {
    common_flydelta_oracle_request request;
    request.oracle_ref = "oracle://test/dataset-operation";
    request.oracle_revision = "v1";
    request.policy_revision = "policy-v1";
    request.concept_key = "dataset.grouped_sum";
    request.behavior_key = "dataset.grouped_sum";
    request.semantic_kind = "dataset_operation";
    request.applicable = applicable;
    request.expected_decision_available = applicable;
    request.expected_decision.operation = "aggregate";
    request.expected_decision.dataset = "sales.csv";
    request.expected_decision.group_by = {"region"};
    request.expected_decision.aggregate_function = "sum";
    request.expected_decision.aggregate_field = "amount";
    return request;
}

} // namespace

int main() {
    std::string error;
    const auto request = aggregate_request();
    common_flydelta_oracle_result result;
    CHECK(common_flydelta_dataset_operation_oracle(
        request,
        R"({"name":"data.aggregate","arguments":{"dataset":"sales.csv","group_by":["region"],"measure":"amount"}})",
        result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::satisfied);
    CHECK(result.oracle_revision == "v1");
    CHECK(common_flydelta_dataset_operation_oracle(
        request,
        R"({"name":"data.describe","arguments":{"dataset":"sales.csv","column":"amount"}})",
        result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::violated);
    // A valid but different dataset operation is still a deterministic
    // violation when it belongs to the semantic-decision IR.
    CHECK(common_flydelta_dataset_operation_oracle(
        request,
        R"({"operation":"filter","dataset":"sales.csv","predicate":"region == north"})",
        result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::violated);
    CHECK(common_flydelta_dataset_operation_oracle(
        aggregate_request(false), "not applicable", result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::not_applicable);

    common_flydelta_oracle_evaluator_chain evaluators;
    evaluators.deterministic = common_flydelta_dataset_operation_oracle;
    evaluators.host_supported = [](
            const common_flydelta_oracle_request & host_request,
            const std::string & observed,
            common_flydelta_oracle_result & host_result,
            std::string & host_error) {
        host_error.clear();
        if (host_request.semantic_kind != "host_dataset") return false;
        host_result = {};
        host_result.known = observed == "host-verified";
        host_result.verdict = host_result.known
            ? common_flydelta_oracle_verdict::satisfied
            : common_flydelta_oracle_verdict::unknown;
        host_result.confidence = host_result.known ? 0.9f : 0.0f;
        host_result.oracle_ref = "oracle://test/host-dataset";
        host_result.reason = host_result.known ? "host callback verified" : "host callback unresolved";
        return true;
    };
    common_flydelta_oracle_request host_request;
    host_request.semantic_kind = "host_dataset";
    CHECK(common_flydelta_oracle_evaluate(
        evaluators, host_request, "host-verified", result, error));
    CHECK(result.known && result.strength == common_flydelta_oracle_strength::host_supported);

    common_flydelta_oracle_probe target;
    target.probe_id = "target";
    target.kind = common_flydelta_oracle_probe_kind::target;
    target.request = request;
    target.expected_verdict = common_flydelta_oracle_verdict::satisfied;
    common_flydelta_oracle_probe control;
    control.probe_id = "control";
    control.kind = common_flydelta_oracle_probe_kind::control;
    control.request = aggregate_request(false);
    control.expected_verdict = common_flydelta_oracle_verdict::not_applicable;
    common_flydelta_oracle_suite_request suite_request;
    suite_request.probes = {target, control};
    suite_request.evaluators.deterministic = common_flydelta_dataset_operation_oracle;
    suite_request.runner = [](
            const common_flydelta_oracle_probe & probe,
            bool candidate,
            std::string & observed,
            std::string & runner_error) {
        runner_error.clear();
        if (probe.kind == common_flydelta_oracle_probe_kind::control) {
            observed = "control output";
        } else if (candidate) {
            observed = R"({"operation":"aggregate","dataset":"sales.csv","group_by":["region"],"measure":"amount"})";
        } else {
            observed = R"({"operation":"filter","dataset":"sales.csv","predicate":"region == north"})";
        }
        return true;
    };
    common_flydelta_oracle_suite_result suite;
    CHECK(common_flydelta_run_oracle_suite(suite_request, suite, error));
    CHECK(suite.semantically_helped && suite.safe_to_continue);
    CHECK(suite.intervention_gain > 0.49f && suite.control_retention == 1.0f);

    common_flydelta_astar_request astar;
    astar.start_state = "start";
    astar.goal_state = "goal";
    astar.max_expansions = 8;
    astar.max_path_length = 4;
    astar.expand = [](const std::string & state,
            std::vector<common_flydelta_astar_successor> & successors,
            std::string & astar_error) {
        astar_error.clear();
        if (state == "start") successors.push_back({"middle", "advance", 1.0f, 1.0f});
        if (state == "middle") successors.push_back({"goal", "finish", 1.0f, 0.0f});
        return true;
    };
    common_flydelta_astar_result astar_result;
    CHECK(common_flydelta_astar_propose(astar, astar_result, error));
    CHECK(astar_result.found && astar_result.states.size() == 3 && astar_result.actions.size() == 2);
    CHECK(astar_result.total_cost == 2.0f);
    return 0;
}
