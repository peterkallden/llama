#include "agent/adaptation/flydelta/oracles/dataset-operation.h"
#include "agent/adaptation/flydelta/oracles/astar-proposer.h"
#include "agent/adaptation/flydelta/oracles/suite.h"
#include "agent/adaptation/flydelta/oracles/tool-contract.h"
#include "agent/adaptation/flydelta/oracles/openapi-operation.h"
#include "agent-openapi-flydelta-oracle.h"

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
    request.expected_contract_kind = "semantic_decision";
    request.expected_contract_ref = "flydelta://contract/dataset-operation";
    request.expected_contract_revision = "v1";
    request.expected_contract_fingerprint = "dataset-contract-v1";
    request.applicable = applicable;
    request.expected_decision_available = applicable;
    request.expected_decision.operation = "aggregate";
    request.expected_decision.dataset = "dataset://local/sales";
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
        R"({"name":"data.aggregate","arguments":{"dataset":"dataset://local/sales","group_by":["region"],"measure":"amount"}})",
        result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::satisfied);
    CHECK(result.violation_kind == common_flydelta_oracle_violation_kind::none);
    CHECK(result.oracle_revision == "v1");
    CHECK(result.evaluator_ref == "flydelta://evaluator/dataset-operation");
    CHECK(result.evaluator_revision == "v1");
    CHECK(result.checks.size() == 1 &&
        result.checks.front().code == "dataset.semantic_decision" &&
        result.checks.front().verdict == common_flydelta_oracle_verdict::satisfied);
    CHECK(common_flydelta_dataset_operation_oracle(
        request,
        R"({"name":"statistics.describe","arguments":{"dataset":"dataset://local/sales","column":"amount"}})",
        result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::violated);
    CHECK(result.violation_kind == common_flydelta_oracle_violation_kind::wrong_tool);
    CHECK(result.violation_code == "dataset.wrong_operation");
    CHECK(result.checks.size() == 1 &&
        result.checks.front().code == "dataset.wrong_operation");
    // A valid but different dataset operation is still a deterministic
    // violation when it belongs to the semantic-decision IR.
    CHECK(common_flydelta_dataset_operation_oracle(
        request,
        R"({"operation":"filter","dataset":"dataset://local/sales","predicate":"region == north"})",
        result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::violated);
    CHECK(result.violation_kind == common_flydelta_oracle_violation_kind::wrong_tool);
    CHECK(common_flydelta_dataset_operation_oracle(
        request,
        R"({"name":"data.aggregate","arguments":{"dataset":"dataset://local/sales","measure":"amount"}})",
        result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::violated);
    CHECK(result.violation_kind == common_flydelta_oracle_violation_kind::missing_required_grouping);
    CHECK(result.violation_dimensions.size() == 1 &&
        result.violation_dimensions.front() == "grouping");
    CHECK(result.violation_code == "dataset.missing_required_grouping");
    CHECK(common_flydelta_dataset_operation_oracle(
        request, "not valid json", result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::violated);
    CHECK(result.violation_kind == common_flydelta_oracle_violation_kind::unattributable);
    CHECK(common_flydelta_dataset_operation_oracle(
        aggregate_request(false), "not applicable", result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::not_applicable);
    CHECK(result.violation_kind == common_flydelta_oracle_violation_kind::none);

    const auto default_registry = common_flydelta_make_default_oracle_registry();
    CHECK(default_registry.evaluators.size() == 1);
    CHECK(common_flydelta_oracle_evaluate(
        default_registry, request,
        R"({"name":"data.aggregate","arguments":{"dataset":"dataset://local/sales","group_by":["region"],"measure":"amount"}})",
        result, error));
    CHECK(result.known && result.verdict == common_flydelta_oracle_verdict::satisfied);
    CHECK(result.evaluator_ref == "flydelta://evaluator/dataset-operation");

    common_flydelta_oracle_registry scoped_registry;
    CHECK(common_flydelta_register_oracle_evaluator(scoped_registry, {
        common_flydelta_oracle_strength::host_supported,
        "flydelta://evaluator/test-host",
        "v1",
        "host_dataset",
        "host_contract",
        "host://dataset-contract",
        "v1",
        [](const common_flydelta_oracle_request & host_request,
                const std::string & host_observed,
                common_flydelta_oracle_result & host_result,
                std::string & host_error) {
            host_error.clear();
            if (host_request.semantic_kind != "host_dataset") return false;
            host_result = {};
            host_result.known = host_observed == "host-verified";
            host_result.verdict = host_result.known
                ? common_flydelta_oracle_verdict::satisfied
                : common_flydelta_oracle_verdict::unknown;
            host_result.oracle_ref = "oracle://test/host-dataset";
            return true;
        },
    }, error));
    common_flydelta_oracle_request scoped_request;
    scoped_request.semantic_kind = "host_dataset";
    scoped_request.expected_contract_kind = "host_contract";
    scoped_request.expected_contract_ref = "host://dataset-contract";
    scoped_request.expected_contract_revision = "v1";
    scoped_request.oracle_revision = "v1";
    CHECK(common_flydelta_oracle_evaluate(
        scoped_registry, scoped_request, "host-verified", result, error));
    CHECK(result.known && result.strength == common_flydelta_oracle_strength::host_supported);
    CHECK(result.evaluator_ref == "flydelta://evaluator/test-host");

    common_flydelta_oracle_request wrong_contract = scoped_request;
    wrong_contract.expected_contract_ref = "host://other-contract";
    CHECK(!common_flydelta_oracle_evaluate(
        scoped_registry, wrong_contract, "host-verified", result, error));
    CHECK(error.find("no registered Oracle evaluator") != std::string::npos);
    common_flydelta_oracle_request wrong_contract_revision = scoped_request;
    wrong_contract_revision.expected_contract_revision = "v2";
    CHECK(!common_flydelta_oracle_evaluate(
        scoped_registry, wrong_contract_revision, "host-verified", result, error));

    common_flydelta_oracle_registry ambiguous_registry;
    const auto host_registration = common_flydelta_oracle_evaluator_registration{
        common_flydelta_oracle_strength::host_supported,
        "flydelta://evaluator/ambiguous-a",
        "v1",
        "host_dataset",
        "host_contract",
        "host://dataset-contract",
        "v1",
        [](const common_flydelta_oracle_request &, const std::string &,
                common_flydelta_oracle_result & host_result, std::string & host_error) {
            host_error.clear();
            host_result = {};
            host_result.known = true;
            host_result.verdict = common_flydelta_oracle_verdict::satisfied;
            return true;
        },
    };
    CHECK(common_flydelta_register_oracle_evaluator(
        ambiguous_registry, host_registration, error));
    auto second_registration = host_registration;
    second_registration.evaluator_ref = "flydelta://evaluator/ambiguous-b";
    second_registration.evaluator_revision = "v2";
    CHECK(common_flydelta_register_oracle_evaluator(
        ambiguous_registry, second_registration, error));
    common_flydelta_oracle_request ambiguous_request = scoped_request;
    ambiguous_request.oracle_revision.clear();
    CHECK(!common_flydelta_oracle_evaluate(
        ambiguous_registry, ambiguous_request, "host-verified", result, error));
    CHECK(error.find("ambiguous") != std::string::npos);

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
    suite_request.registry = common_flydelta_make_default_oracle_registry();
    suite_request.runner = [](
            const common_flydelta_oracle_probe & probe,
            bool candidate,
            std::string & observed,
            std::string & runner_error) {
        runner_error.clear();
        if (probe.kind == common_flydelta_oracle_probe_kind::control) {
            observed = "control output";
        } else if (candidate) {
            observed = R"({"operation":"aggregate","dataset":"dataset://local/sales","group_by":["region"],"measure":"amount"})";
        } else {
            observed = R"({"operation":"filter","dataset":"dataset://local/sales","predicate":"region == north"})";
        }
        return true;
    };
    common_flydelta_oracle_suite_result suite;
    CHECK(common_flydelta_run_oracle_suite(suite_request, suite, error));
    CHECK(suite.semantically_helped && suite.safe_to_continue);
    CHECK(suite.intervention_gain > 0.49f && suite.control_retention == 1.0f);
    CHECK(suite.probes.size() == 2);
    CHECK(suite.probes.front().baseline.evaluator_ref ==
        "flydelta://evaluator/dataset-operation");
    CHECK(suite.probes.front().candidate.evaluator_revision == "v1");

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

    common_flydelta_oracle_suite_report durable_report;
    durable_report.id = "flydelta://oracle-suite/test-report";
    durable_report.candidate_id = "flydelta://candidate/test";
    durable_report.evaluation_revision = "evaluation:test-v1";
    durable_report.model_profile_id = "model:test";
    durable_report.oracle_ref = "flydelta://oracle/dataset-operation";
    durable_report.oracle_revision = "v1";
    durable_report.policy_revision = "policy-v1";
    durable_report.observations.push_back({
        "fixture-report", "intended", "fixture-target", "verifier-v1",
        "semantic_decision", "flydelta://contract/dataset-operation", "v1",
        "dataset-contract-v1", "helped", false, false, true, true,
        "flydelta://evaluator/dataset-operation",
        "flydelta://evaluator/dataset-operation"});
    durable_report.baseline_success_rate = 0.0f;
    durable_report.candidate_success_rate = 1.0f;
    durable_report.intervention_gain = 1.0f;
    durable_report.control_retention = 1.0f;
    durable_report.semantically_helped = true;
    durable_report.safe_to_continue = true;
    CHECK(common_flydelta_oracle_suite_report_validate(durable_report, error));
    common_flydelta_oracle_suite_report reloaded_report;
    CHECK(common_flydelta_oracle_suite_report_from_json(
        common_flydelta_oracle_suite_report_to_json(durable_report),
        reloaded_report, error));
    CHECK(reloaded_report.id == durable_report.id);
    CHECK(reloaded_report.observations.size() == 1);
    CHECK(reloaded_report.intervention_gain == 1.0f);
    CHECK(reloaded_report.observations.front().expected_contract_ref ==
        "flydelta://contract/dataset-operation");
    CHECK(reloaded_report.observations.front().candidate_evaluator_ref ==
        "flydelta://evaluator/dataset-operation");

    common_flydelta_model_tool_contract tool_contract;
    tool_contract.provider_ref = "openapi://openalex";
    tool_contract.exposed_tool_name = "openalex.listWorks";
    tool_contract.contract_ref = "tool://openalex/listWorks";
    tool_contract.contract_revision = "v3";
    tool_contract.contract_fingerprint = "sha256:model-facing-list-works";
    tool_contract.model_input_schema_json = R"({
        "type":"object",
        "additionalProperties":false,
        "required":["search","select"],
        "properties":{
            "search":{"type":"string","minLength":1},
            "select":{"type":"string","minLength":1},
            "per_page":{"type":"integer","minimum":1,"maximum":100}
        }
    })";
    tool_contract.host_input_schema_json = tool_contract.model_input_schema_json;

    common_flydelta_oracle_result tool_result;
    CHECK(common_flydelta_validate_model_tool_text(
        tool_contract,
        common_agent_tool_output_format::jsonl,
        R"({"tool":"openalex.listWorks","arguments":{"search":"machine learning","select":"id, display_name","per_page":1}})",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::satisfied);
    CHECK(tool_result.normalized_arguments_json.find("machine learning") != std::string::npos);

    CHECK(common_flydelta_validate_model_tool_text(
        tool_contract,
        common_agent_tool_output_format::compact_dsl,
        R"(open! openalex.listWorks search="machine learning" select="id, display_name" per_page=1)",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::satisfied);

    CHECK(common_flydelta_validate_model_tool_text(
        tool_contract,
        common_agent_tool_output_format::jsonl,
        R"({"tool":"openalex.listWorks","arguments":{"search":"machine learning"}})",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::violated &&
        tool_result.violation_code == "tool.missing_required_parameter");

    CHECK(common_flydelta_validate_model_tool_text(
        tool_contract,
        common_agent_tool_output_format::jsonl,
        R"({"tool":"openalex.listWorks","arguments":{"search":"machine learning","select":"id","unexpected":true}})",
        tool_result, error));
    CHECK(tool_result.violation_code == "tool.unknown_argument");

    CHECK(common_flydelta_validate_model_tool_text(
        tool_contract,
        common_agent_tool_output_format::jsonl,
        R"({"tool":"openalex.getAuthor","arguments":{"search":"machine learning","select":"id"}})",
        tool_result, error));
    CHECK(tool_result.violation_code == "tool.unknown_tool");

    CHECK(common_flydelta_validate_model_tool_text(
        tool_contract,
        common_agent_tool_output_format::compact_dsl,
        "open! openalex.listWorks search=",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.violation_code == "tool.malformed_call");

    common_flydelta_oracle_registry tool_registry;
    CHECK(common_flydelta_register_oracle_evaluator(tool_registry, {
        common_flydelta_oracle_strength::host_supported,
        "flydelta://evaluator/tool-contract",
        "v1",
        "tool_contract",
        "model_facing_tool_contract",
        tool_contract.contract_ref,
        tool_contract.contract_revision,
        common_flydelta_make_tool_contract_oracle(tool_contract),
    }, error));
    common_flydelta_oracle_request tool_request;
    tool_request.oracle_ref = "oracle://test/tool-contract";
    tool_request.oracle_revision = "v1";
    tool_request.semantic_kind = "tool_contract";
    tool_request.expected_contract_kind = "model_facing_tool_contract";
    tool_request.expected_contract_ref = tool_contract.contract_ref;
    tool_request.expected_contract_revision = tool_contract.contract_revision;
    tool_request.observed_format = "compact_dsl";
    CHECK(common_flydelta_oracle_evaluate(
        tool_registry,
        tool_request,
        R"(open! openalex.listWorks search="machine learning" select="id, display_name" per_page=1)",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::satisfied &&
        tool_result.evaluator_ref == "flydelta://evaluator/tool-contract");

    common_flydelta_openapi_operation_contract openapi_contract;
    openapi_contract.tool = tool_contract;
    openapi_contract.tool.contract_ref = "openapi://openalex/v1/operation/listWorks";
    openapi_contract.tool.contract_revision = "openapi-v1";
    openapi_contract.provider_id = "openalex";
    openapi_contract.operation_id = "listWorks";
    openapi_contract.method = "get";
    openapi_contract.path = "/works";
    openapi_contract.query_parameters = {"search", "select", "per_page"};
    openapi_contract.host_required_parameters = {"search", "select"};

    common_flydelta_oracle_registry openapi_registry;
    CHECK(common_flydelta_register_oracle_evaluator(openapi_registry, {
        common_flydelta_oracle_strength::host_supported,
        "flydelta://evaluator/openapi-operation",
        "v1",
        "openapi_operation",
        "openapi_operation_contract",
        openapi_contract.tool.contract_ref,
        openapi_contract.tool.contract_revision,
        common_flydelta_make_openapi_operation_oracle(openapi_contract),
    }, error));
    common_flydelta_oracle_request openapi_request;
    openapi_request.oracle_ref = "oracle://test/openapi-operation";
    openapi_request.oracle_revision = "v1";
    openapi_request.semantic_kind = "openapi_operation";
    openapi_request.expected_contract_kind = "openapi_operation_contract";
    openapi_request.expected_contract_ref = openapi_contract.tool.contract_ref;
    openapi_request.expected_contract_revision = openapi_contract.tool.contract_revision;
    openapi_request.observed_format = "jsonl";
    CHECK(common_flydelta_oracle_evaluate(
        openapi_registry,
        openapi_request,
        R"({"tool":"openalex.listWorks","arguments":{"search":"machine learning","select":"id, display_name","per_page":1}})",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::satisfied &&
        tool_result.checks.size() >= 5);

    CHECK(common_flydelta_oracle_evaluate(
        openapi_registry,
        openapi_request,
        R"({"name":"openalex.listWorks","arguments":{"search":"machine learning","select":"id, display_name","per_page":1}})",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::satisfied);

    CHECK(common_flydelta_oracle_evaluate(
        openapi_registry,
        openapi_request,
        R"({"tool":"openalex.getAuthor","arguments":{"search":"machine learning","select":"id"}})",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::violated &&
        tool_result.violation_code == "openapi.wrong_operation");

    CHECK(common_flydelta_oracle_evaluate(
        openapi_registry,
        openapi_request,
        R"({"tool":"other.listWorks","arguments":{"search":"machine learning","select":"id"}})",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::violated &&
        tool_result.violation_code == "openapi.wrong_provider");

    CHECK(common_flydelta_oracle_evaluate(
        openapi_registry,
        openapi_request,
        R"({"tool":"openalex.listWorks","arguments":{"search":"machine learning"}})",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::violated &&
        tool_result.violation_code == "openapi.missing_required_parameter");

    agent_openapi_catalog catalog;
    catalog.provider_id = "openalex";
    catalog.prefix = "openalex";
    agent_openapi_operation catalog_operation;
    catalog_operation.operation_id = "listWorks";
    catalog_operation.method = "GET";
    catalog_operation.path = "/works";
    catalog_operation.input_schema_json =
        R"({"type":"object","properties":{"search":{"type":"string"},"select":{"type":"string"}},"required":["search"]})";
    catalog_operation.host_required_parameters = {"search"};
    catalog.operations.push_back(catalog_operation);

    common_flydelta_oracle_registry catalog_registry;
    CHECK(register_agent_openapi_flydelta_oracles(
        catalog, catalog_registry, "openapi-v1", error));
    CHECK(catalog_registry.evaluators.size() == 1);
    common_flydelta_oracle_request catalog_request;
    catalog_request.oracle_ref =
        "flydelta://evaluator/openapi/openalex/listWorks";
    catalog_request.oracle_revision = "openapi-v1";
    catalog_request.semantic_kind = "openapi_operation";
    catalog_request.expected_contract_kind = "openapi_operation_contract";
    catalog_request.expected_contract_ref =
        "openapi://openalex/operation/listWorks";
    catalog_request.expected_contract_revision = "openapi-v1";
    CHECK(common_flydelta_oracle_evaluate(
        catalog_registry, catalog_request,
        R"({"name":"openalex.listWorks","arguments":{"search":"climate"}})",
        tool_result, error));
    CHECK(tool_result.known &&
        tool_result.verdict == common_flydelta_oracle_verdict::satisfied &&
        tool_result.checks.size() >= 5);
    return 0;
}
