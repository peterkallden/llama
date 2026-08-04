#include "agent/tool-catalog.h"

#include <cassert>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <string>

int main() {
    common_tool_catalog catalog;
    common_tool_bootstrap_result first;
    std::string error;
    if (!catalog.bootstrap("memory", first, error)) {
        std::fprintf(stderr, "memory catalog bootstrap failed: %s\n", error.c_str());
        return 1;
    }
    assert(!first.definitions_created.empty());
    assert(catalog.find_definition("memory_search"));
    const auto * memory_remember = catalog.find_definition("memory_remember");
    if (!memory_remember) {
        std::fprintf(stderr, "missing memory_remember tool definition\n");
        return 1;
    }
    assert(memory_remember->requires_confirmation);
    assert(memory_remember->risk_class == common_tool_risk_class::memory_proposal);
    for (const auto * name : {"memory_get", "memory_propose_update", "memory_propose_forget"}) {
        const auto * definition = catalog.find_definition(name);
        if (!definition) {
            std::fprintf(stderr, "missing memory tool definition: %s\n", name);
            return 1;
        }
        const auto schema = nlohmann::json::parse(definition->input_schema_json);
        const auto & id = schema["properties"]["id"];
        assert(id.value("type", "") == "string");
        assert(id.value("minLength", 0) == 1);
        assert(id.value("maxLength", 0) == 256);
    }
    const auto * link_definition = catalog.find_definition("memory_link");
    if (!link_definition) {
        std::fprintf(stderr, "missing memory_link tool definition\n");
        return 1;
    }
    const auto link_schema = nlohmann::json::parse(link_definition->input_schema_json);
    assert(link_schema["properties"]["from"].value("minLength", 0) == 1);
    assert(link_schema["properties"]["to"].value("maxLength", 0) == 256);
    assert(link_schema["properties"]["relation"].value("minLength", 0) == 1);
    const auto * compact_definition = catalog.find_definition("memory_compact_propose");
    if (!compact_definition) {
        std::fprintf(stderr, "missing memory_compact_propose tool definition\n");
        return 1;
    }
    const auto compact_schema = nlohmann::json::parse(compact_definition->input_schema_json);
    assert(compact_schema["properties"]["source_ids"]["items"].value("maxLength", 0) == 256);
    const auto * web_search = catalog.find_definition("web_search");
    const auto * web_fetch = catalog.find_definition("web_fetch");
    if (!web_search || !web_fetch) {
        std::fprintf(stderr, "missing web tool definitions\n");
        return 1;
    }
    assert(web_search->executor_id == "builtin.web_search");
    assert(web_fetch->executor_id == "builtin.web_fetch");

    const auto * build = catalog.find_definition("development.build");
    const auto * test = catalog.find_definition("development.test");
    if (!build || !test) {
        std::fprintf(stderr, "missing developer tool definitions\n");
        return 1;
    }
    assert(build && test);
    assert(build->executor_id == "sandbox.development.build");
    assert(test->executor_id == "sandbox.development.test");
    assert(build->risk_class == common_tool_risk_class::sandbox_execution);
    assert(test->risk_class == common_tool_risk_class::sandbox_execution);
    assert(build->requires_confirmation && test->requires_confirmation);
    assert(build->input_schema_json.find("required\":[\"target\"]") != std::string::npos);
    assert(test->input_schema_json.find("required\":[\"target\"]") != std::string::npos);
    assert(build->policy_json.find("execution_class\":\"developer-build\"") != std::string::npos);
    assert(test->policy_json.find("filesystem\":\"workspace-write\"") != std::string::npos);

    const auto read = catalog.load_profile("memory-read", error);
    assert(error.empty());
    assert(read.size() == 8);
    for (const auto & definition : read) assert(definition.risk_class == common_tool_risk_class::local_read);

    common_tool_bootstrap_result second;
    assert(catalog.bootstrap("memory", second, error));
    assert(second.definitions_created.empty());
    assert(second.definitions_unchanged.size() >= first.definitions_created.size());
    assert(!catalog.bootstrap("not-a-profile", second, error));
    assert(error == "tool profile is unavailable: not-a-profile");
    return 0;
}
