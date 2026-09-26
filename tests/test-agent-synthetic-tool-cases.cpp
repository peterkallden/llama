#include "agent/tooling/synthetic/synthetic-tool-cases.h"

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

static void check(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
        std::abort();
    }
}

#define CHECK(condition) check((condition), #condition, __LINE__)

int main() {
    common_tool_definition definition;
    definition.name = "dataset.inspect";
    definition.description = "Inspect a selected dataset.";
    definition.input_schema_json = R"({
        "type":"object",
        "additionalProperties":false,
        "required":["dataset","format"],
        "properties":{
            "dataset":{"type":"string","minLength":1},
            "format":{"type":"string","enum":["summary","schema"]}
        }
    })";
    definition.result_schema_json = R"({"type":"object","properties":{"columns":{"type":"array"}}})";

    std::vector<common_synthetic_tool_case> cases;
    std::string error;
    if (!common_synthetic_tool_cases_generate(
            definition, "data", "native", 42, 8, cases, error)) {
        std::fprintf(stderr, "generation error: %s\n", error.c_str());
        return 1;
    }
    CHECK(cases.size() == 5);
    CHECK(cases.front().model_input_schema_json == definition.input_schema_json);
    CHECK(cases.front().model_facing_contract.find("dataset.inspect") != std::string::npos);
    CHECK(cases.front().model_facing_contract.find("dataset:string") != std::string::npos);
    CHECK(cases.front().model_facing_prompt.find("Repair the invalid tool call") != std::string::npos);

    const auto first_json = common_synthetic_tool_case_to_json(cases.front());
    common_synthetic_tool_case round_trip;
    CHECK(common_synthetic_tool_case_from_json(first_json, 64 * 1024, round_trip, error));
    CHECK(common_synthetic_tool_case_to_json(round_trip) == first_json);

    for (auto & value : cases) {
        CHECK(value.host_verified);
        CHECK(!value.mutated_valid);
        CHECK(value.repair_valid);
        CHECK(common_synthetic_tool_case_validate(value, 64 * 1024, error));
        const auto encoded = nlohmann::json::parse(common_synthetic_tool_case_to_json(value));
        CHECK(encoded["model_facing_contract"].is_string());
        CHECK(encoded["model_input_schema"].is_string());
        CHECK(encoded["host_verified"] == true);
    }

    std::vector<common_synthetic_tool_case> repeated;
    CHECK(common_synthetic_tool_cases_generate(
        definition, "data", "native", 42, 8, repeated, error));
    CHECK(repeated.size() == cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        CHECK(common_synthetic_tool_case_to_json(cases[i]) ==
            common_synthetic_tool_case_to_json(repeated[i]));
    }

    const auto path = std::filesystem::temp_directory_path() / "llama-agent-synthetic-tools.jsonl";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    CHECK(common_synthetic_tool_cases_export_jsonl(cases, path, 512 * 1024, error));
    std::ifstream input(path, std::ios::binary);
    std::string line;
    size_t rows = 0;
    while (std::getline(input, line)) {
        CHECK(!line.empty());
        CHECK(nlohmann::json::parse(line).is_object());
        ++rows;
    }
    CHECK(rows == cases.size());
    std::filesystem::remove(path, ignored);

    common_tool_definition without_required = definition;
    without_required.input_schema_json = R"({"type":"object","additionalProperties":false,"properties":{"value":{"type":"string"}}})";
    cases.clear();
    CHECK(common_synthetic_tool_cases_generate(
        without_required, "data", "native", 42, 8, cases, error));
    CHECK(cases.size() == 3);
    for (const auto & value : cases) CHECK(value.mutation != common_synthetic_tool_mutation::missing_required);
    return 0;
}
