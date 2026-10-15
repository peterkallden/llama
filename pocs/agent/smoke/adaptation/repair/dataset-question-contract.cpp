#include "../agent-flydelta-dataset-repair-host.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {
using json = nlohmann::ordered_json;

bool read_json(const char * path, json & value, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "could not open suite"; return false; }
    std::ostringstream contents;
    contents << input.rdbuf();
    value = json::parse(contents.str(), nullptr, false);
    if (value.is_discarded() || !value.is_object() || !value.contains("scenarios") ||
            !value["scenarios"].is_array()) {
        error = "dataset repair suite is invalid";
        return false;
    }
    return true;
}
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " SUITE_JSON\n";
        return 2;
    }
    json suite;
    std::string error;
    common_agent_generation_request task_prefix;
    task_prefix.messages = {{"system", "Return one DSL action."}, {"user", "Group sales by region."}};
    task_prefix.options.n_predict = 512;
    const auto forced = agent_flydelta_dataset_capture_context(task_prefix, "open! data.aggregate");
    if (forced.options.n_predict != 0 || forced.messages.size() != 3 ||
            forced.messages[0].content != task_prefix.messages[0].content ||
            forced.messages[1].content != task_prefix.messages[1].content ||
            forced.messages.back().role != "assistant" || forced.messages.back().content != "open! data.aggregate") {
        std::cerr << "forced capture changed the original task prefix or continuation\n"; return 1;
    }
    common_agent_teacher_forced_choice_request contrast;
    const std::string positive = "open! data.query dataset=\"dataset://local/sales\" limit=4";
    const std::string negative = "open! data.query dataset=\"dataset://local/sales\" limit=100";
    if (!agent_flydelta_dataset_dsl_contrast(contrast, positive, negative, error) ||
            contrast.choice_prefix + contrast.positive_continuation != positive ||
            contrast.choice_prefix + contrast.negative_continuation != negative ||
            contrast.choice_prefix != "open!") {
        std::cerr << "DSL teacher-scoring contrast did not preserve the model-facing calls\n";
        return 1;
    }
    if (agent_flydelta_dataset_dsl_contrast(contrast, positive, positive, error) ||
            agent_flydelta_dataset_dsl_contrast(contrast, positive, "{\"name\":\"data.query\"}", error) ||
            agent_flydelta_dataset_dsl_contrast(contrast, positive, "open! ", error)) {
        std::cerr << "DSL teacher-scoring accepted an invalid contrast\n";
        return 1;
    }
    const std::string observed_invalid = "open! data.query\nwhere[0].field: region\nwhere[0].field: amount";
    if (!agent_flydelta_dataset_dsl_contrast(contrast, positive, observed_invalid, error) ||
            contrast.choice_prefix + contrast.negative_continuation != observed_invalid) {
        std::cerr << "scoring replaced the observed invalid DSL continuation\n"; return 1;
    }
    if (!read_json(argv[1], suite, error)) { std::cerr << error << '\n'; return 1; }
    agent_flydelta_dataset_repair_host host;
    if (!host.open("contract", error)) { std::cerr << error << '\n'; return 1; }
    if (!host.select_model_tools({"data.query", "data.filter", "data.aggregate"}, error)) {
        host.close(); std::cerr << error << '\n'; return 1;
    }
    const std::string flat_call = "open! data.query\ndataset: dataset://local/sales\nselect: order_id, region\nlimit: 4";
    common_agent_tool_call parsed_flat_call;
    if (!host.parse_model_call(flat_call, parsed_flat_call, error) ||
            json::parse(parsed_flat_call.arguments_json)["select"] != json::array({"order_id", "region"})) {
        host.close(); std::cerr << "schema-aware flat call parsing failed: " << error << '\n'; return 1;
    }
    if (host.parse_model_call("open! web.fetch\nurl: https://example.org", parsed_flat_call, error)) {
        host.close(); std::cerr << "parser accepted a tool outside the model-facing catalog\n"; return 1;
    }
    if (!host.parse_model_call("open! data.query\nwhere[0].field: amount\nwhere[0].operator: >=\nwhere[0].value: 4",
            parsed_flat_call, error) || json::parse(parsed_flat_call.arguments_json)["where"][0]["operator"] != ">=") {
        host.close(); std::cerr << "parser rejected an exact unquoted schema enum: " << error << '\n'; return 1;
    }
    if (host.parse_model_call("open! data.query\nwhere[0].field: amount\nwhere[0].operator: =>\nwhere[0].value: 4",
            parsed_flat_call, error)) {
        host.close(); std::cerr << "parser accepted an undeclared operator enum\n"; return 1;
    }
    const json aggregate_arguments = {{"dataset", host.available_datasets().front().ref.uri},
        {"group_by", json::array({"region"})},
        {"measures", json::array({{{"function", "sum"}, {"column", "amount"}, {"as", "total_amount"}}})}};
    const auto aggregate_call = host.render_model_call("data.aggregate", aggregate_arguments, error);
    if (!host.parse_model_call(aggregate_call, parsed_flat_call, error) ||
            json::parse(parsed_flat_call.arguments_json) != aggregate_arguments) {
        host.close(); std::cerr << "flat aggregate directive roundtrip failed: " << error << '\n'; return 1;
    }
    const auto rejected_call = host.render_model_call("invalid_fixture_tool", json{
        {"where", {{"and", json::array({json{{"gt", {{"amount", 0}}}}, json{{"gte", {{"units", 1}}}}})}}}}, error);
    if (rejected_call.find("where.and[0].gt.amount: 0") == std::string::npos ||
            rejected_call.find("where.and[1].gte.units: 1") == std::string::npos ||
            host.parse_model_call(rejected_call, parsed_flat_call, error)) {
        host.close(); std::cerr << "rejected fixture rendering lost its arguments or gained tool authority\n"; return 1;
    }
    for (const auto * operation : {"data.query", "data.filter"}) {
        const auto condition_key = std::string(operation) == "data.query" ? "where" : "conditions";
        const json arguments = {{"dataset", host.available_datasets().front().ref.uri},
            {condition_key, json::array({{{"field", "region"}, {"operator", "in"},
                {"value", json::array({"north", "east"})}}})}};
        const auto rendered = host.render_model_call(operation, arguments, error);
        common_agent_tool_call roundtrip;
        if (!host.parse_model_call(rendered, roundtrip, error) ||
                json::parse(roundtrip.arguments_json) != arguments) {
            host.close(); std::cerr << "flat membership call roundtrip failed: " << error << '\n'; return 1;
        }
        common_tool_execution_result execution;
        if (!host.execute_call(operation, arguments, execution, error)) {
            host.close();
            std::cerr << "membership predicate execution failed: " << error << '\n';
            return 1;
        }
        const auto result = json::parse(execution.output, nullptr, false);
        if (result.is_discarded() || !result.contains("rows") || result["rows"].size() != 3) {
            host.close();
            std::cerr << operation << " membership predicate must retain the three north/east fixture rows\n";
            return 1;
        }
    }
    size_t verified = 0;
    size_t unavailable = 0;
    for (const auto & scenario : suite["scenarios"]) {
        const auto & steps = scenario.value("plan", json::object()).value("steps", json::array());
        if (steps.size() != 1) {
            std::cerr << "scenario " << scenario.value("id", "") << " must have exactly one plan step\n";
            host.close();
            return 1;
        }
        const auto & step = steps.front();
        const auto tool = step.value("tool", "");
        if (!host.has_tool(tool)) {
            ++unavailable;
            std::cout << "scenario=" << scenario.value("id", "")
                      << " canonical_execution=unavailable tool=" << tool << '\n';
            continue;
        }
        common_tool_execution_result result;
        if (!host.execute_step(step, result, error)) {
            std::cerr << "scenario " << scenario.value("id", "")
                      << " canonical host execution failed: " << error << '\n';
            host.close();
            return 1;
        }
        ++verified;
    }
    host.close();
    std::cout << "flydelta_dataset_question_repair_contract=passed"
              << " canonical_verified=" << verified
              << " canonical_unavailable=" << unavailable << '\n';
    return verified == 0 ? 1 : 0;
}
