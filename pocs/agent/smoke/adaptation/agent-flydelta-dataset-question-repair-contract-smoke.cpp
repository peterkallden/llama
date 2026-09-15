#include "agent-flydelta-dataset-repair-host.h"

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
    if (!read_json(argv[1], suite, error)) { std::cerr << error << '\n'; return 1; }
    agent_flydelta_dataset_repair_host host;
    if (!host.open("contract", error)) { std::cerr << error << '\n'; return 1; }
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
