#include "plan/plan-json.h"
#include "agent/tooling/catalog/tool-catalog.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

namespace {

using json = nlohmann::json;

bool read_json(const std::string & path, json & value, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "could not open suite: " + path; return false; }
    std::ostringstream text;
    text << input.rdbuf();
    value = json::parse(text.str(), nullptr, false);
    if (value.is_discarded()) { error = "suite is not valid JSON"; return false; }
    return true;
}

int fail(const std::string & error) {
    std::cerr << "dataset question suite contract failed: " << error << '\n';
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " SUITE_JSON\n";
        return 2;
    }

    json suite;
    std::string error;
    if (!read_json(argv[1], suite, error) || !suite.is_object()) return fail(error);
    if (suite.value("schema_version", 0) != 1 || suite.value("language", "") != "en" ||
            suite.value("dataset", "").rfind("dataset://", 0) != 0 ||
            !suite.contains("model_contract") || !suite["model_contract"].is_object() ||
            !suite.contains("scenarios") || !suite["scenarios"].is_array() ||
            suite["scenarios"].empty()) {
        return fail("suite identity, English model contract, or scenarios are invalid");
    }

    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("analysis", bootstrap, error)) return fail(error);

    std::set<std::string> expected_tools;
    std::set<std::string> scenario_ids;
    for (const auto & scenario : suite["scenarios"]) {
        if (!scenario.is_object() || !scenario.value("id", "").size() ||
                !scenario.value("question", "").size() ||
                !scenario.value("expected_tool", "").size() ||
                !scenario.contains("plan") || !scenario["plan"].is_object()) {
            return fail("scenario is missing id, English question, expected tool, or plan");
        }
        const auto id = scenario["id"].get<std::string>();
        const auto expected = scenario["expected_tool"].get<std::string>();
        if (!scenario_ids.insert(id).second || expected.empty()) {
            return fail("scenario ids must be unique and expected tools must be non-empty");
        }
        expected_tools.insert(expected);
        if (catalog.find_definition(expected) == nullptr) return fail("expected tool is not in catalog: " + expected);

        common_plan_state plan;
        std::vector<common_plan_operation> operations;
        if (!common_plan_parse_proposal_json(scenario["plan"].dump(), plan, operations, error)) {
            return fail(id + ": invalid plan: " + error);
        }
        bool found_expected = false;
        for (const auto & operation : operations) {
            if (!operation.step || !operation.step->tool_call) continue;
            const auto tool = operation.step->tool_call->name;
            if (catalog.find_definition(tool) == nullptr) return fail(id + ": plan uses unknown tool: " + tool);
            if (tool == expected) found_expected = true;
            const auto args = json::parse(operation.step->tool_call->arguments_json, nullptr, false);
            if (args.is_object() && args.contains("dataset") && args["dataset"].is_string() &&
                    args["dataset"].get<std::string>() != suite["dataset"].get<std::string>()) {
                return fail(id + ": plan dataset does not match suite dataset");
            }
        }
        if (!found_expected) return fail(id + ": expected tool is not present in plan");
    }
    if (suite["scenarios"].size() < 6 || expected_tools.size() < 6) {
        return fail("suite must cover six distinct dataset operations");
    }
    std::cout << "flydelta_dataset_question_suite=passed scenarios=" << suite["scenarios"].size()
              << " distinct_tools=" << expected_tools.size() << "\n";
    return 0;
}
