#include "agent/agent-package-json.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <functional>
#include <set>

using json = nlohmann::ordered_json;

namespace {
bool strings(const json & value, std::vector<std::string> & out) {
    if (!value.is_array()) return false;
    out.clear();
    for (const auto & item : value) { if (!item.is_string()) return false; out.push_back(item.get<std::string>()); }
    return true;
}

bool valid_id(const std::string & id) {
    return !id.empty() && id.size() <= 128 && id.find("bootstrap:") == std::string::npos && id.find('\n') == std::string::npos;
}

bool validate(const common_agent_bootstrap_package & package, std::string & error) {
    if (package.name.empty() || package.name.size() > 128 || package.version.empty() || package.version.size() > 64) { error = "package name and version are required"; return false; }
    std::set<std::string> procedures, blueprints;
    for (const auto & procedure : package.procedures) {
        if (!valid_id(procedure.id) || procedure.content.empty() || procedure.content.size() > 8192 || procedure.summary.size() > 1024 || procedure.importance < 0 || procedure.importance > 1 || procedure.confidence < 0 || procedure.confidence > 1 || !procedures.insert(procedure.id).second) { error = "invalid or duplicate procedure definition"; return false; }
    }
    for (const auto & blueprint : package.blueprints) {
        if (!valid_id(blueprint.id) || blueprint.goal.empty() || blueprint.goal.size() > 4096 || blueprint.success_criteria.empty() || blueprint.success_criteria.size() > 4096 || blueprint.steps.empty() || blueprint.steps.size() > 64 || !blueprints.insert(blueprint.id).second) { error = "invalid or duplicate blueprint definition"; return false; }
        std::set<std::string> steps;
        for (const auto & step : blueprint.steps) {
            if (!valid_id(step.id) || step.title.empty() || step.objective.empty() || step.tool_call || step.selected_tool || !steps.insert(step.id).second) { error = "invalid blueprint step or tool binding"; return false; }
        }
        for (const auto & step : blueprint.steps) for (const auto & dependency : step.depends_on) if (!steps.count(dependency) || dependency == step.id) { error = "blueprint dependency references an unknown step"; return false; }
        // Every dependency points to an earlier-free DAG; bounded DFS detects cycles.
        std::set<std::string> visiting, visited;
        std::function<bool(const std::string &)> visit = [&](const std::string & id) {
            if (visiting.count(id)) return false;
            if (visited.count(id)) return true;
            visiting.insert(id);
            const auto it = std::find_if(blueprint.steps.begin(), blueprint.steps.end(), [&](const auto & step) { return step.id == id; });
            for (const auto & dep : it->depends_on) if (!visit(dep)) return false;
            visiting.erase(id); visited.insert(id); return true;
        };
        for (const auto & step : blueprint.steps) if (!visit(step.id)) { error = "blueprint dependencies contain a cycle"; return false; }
    }
    return true;
}
}

bool common_agent_package_parse_json(const std::string & text, common_agent_bootstrap_package & package, std::string & error) {
    try {
        const auto root = json::parse(text, nullptr, false);
        if (!root.is_object() || root.value("schema_version", 0) != 1) { error = "package requires supported schema_version"; return false; }
        package = {};
        package.name = root.value("name", std::string{});
        package.version = root.value("version", std::string{});
        if (root.contains("procedures")) {
            if (!root["procedures"].is_array()) { error = "package procedures must be an array"; return false; }
            for (const auto & item : root["procedures"]) {
                if (!item.is_object()) { error = "package procedure must be an object"; return false; }
                common_agent_bootstrap_procedure procedure;
                procedure.id = item.value("id", std::string{}); procedure.content = item.value("content", std::string{}); procedure.summary = item.value("summary", procedure.id);
                procedure.importance = item.value("importance", 0.8f); procedure.confidence = item.value("confidence", 1.0f);
                package.procedures.push_back(std::move(procedure));
            }
        }
        if (root.contains("blueprints")) {
            if (!root["blueprints"].is_array()) { error = "package blueprints must be an array"; return false; }
            for (const auto & item : root["blueprints"]) {
                if (!item.is_object() || !item.contains("steps") || !item["steps"].is_array()) { error = "package blueprint requires steps"; return false; }
                common_agent_bootstrap_blueprint blueprint;
                blueprint.id = item.value("id", std::string{}); blueprint.selection_description = item.value("selection_description", std::string{});
                blueprint.goal = item.value("goal", std::string{}); blueprint.success_criteria = item.value("success_criteria", std::string{});
                if (item.contains("next_action")) { if (!item["next_action"].is_string()) { error = "next_action must be a string"; return false; } blueprint.next_action = item["next_action"].get<std::string>(); }
                for (const auto & source : item["steps"]) {
                    if (!source.is_object()) { error = "package step must be an object"; return false; }
                    common_plan_step step; step.id = source.value("id", std::string{}); step.title = source.value("title", std::string{}); step.objective = source.value("objective", std::string{}); step.optional = source.value("optional", false);
                    if (source.contains("depends_on") && !strings(source["depends_on"], step.depends_on)) { error = "step dependencies must be strings"; return false; }
                    if (source.contains("required_evidence") && !strings(source["required_evidence"], step.required_evidence)) { error = "step required_evidence must be strings"; return false; }
                    blueprint.steps.push_back(std::move(step));
                }
                package.blueprints.push_back(std::move(blueprint));
            }
        }
        return validate(package, error);
    } catch (const std::exception & e) { error = std::string("invalid package JSON: ") + e.what(); return false; }
}

bool common_agent_package_to_json(const common_agent_bootstrap_package & package, std::string & text, std::string & error, bool pretty) {
    if (!validate(package, error)) return false;
    json root = {{"schema_version", 1}, {"name", package.name}, {"version", package.version}, {"procedures", json::array()}, {"blueprints", json::array()}};
    for (const auto & procedure : package.procedures) root["procedures"].push_back({{"id", procedure.id}, {"summary", procedure.summary}, {"content", procedure.content}, {"importance", procedure.importance}, {"confidence", procedure.confidence}});
    for (const auto & blueprint : package.blueprints) {
        json value = {{"id", blueprint.id}, {"goal", blueprint.goal}, {"success_criteria", blueprint.success_criteria}, {"steps", json::array()}};
        if (!blueprint.selection_description.empty()) value["selection_description"] = blueprint.selection_description;
        if (blueprint.next_action) value["next_action"] = *blueprint.next_action;
        for (const auto & step : blueprint.steps) { json item = {{"id", step.id}, {"title", step.title}, {"objective", step.objective}}; if (step.optional) item["optional"] = true; if (!step.depends_on.empty()) item["depends_on"] = step.depends_on; if (!step.required_evidence.empty()) item["required_evidence"] = step.required_evidence; value["steps"].push_back(std::move(item)); }
        root["blueprints"].push_back(std::move(value));
    }
    text = pretty ? root.dump(2) + "\n" : root.dump(); error.clear(); return true;
}
