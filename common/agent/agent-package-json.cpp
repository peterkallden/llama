#include "agent/agent-package-json.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
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
    if (id.empty() || id.size() > 128 || id.find("bootstrap:") != std::string::npos) return false;
    const auto alnum = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; };
    if (!alnum(id.front())) return false;
    return std::all_of(id.begin() + 1, id.end(), [&](char c) { return alnum(c) || c == '.' || c == '_' || c == '-'; });
}

bool constraints(const json & value, std::vector<common_plan_constraint> & out) {
    if (!value.is_array()) return false;
    out.clear();
    for (const auto & item : value) {
        if (!item.is_object() || !item.contains("id") || !item["id"].is_string() || !item.contains("description") || !item["description"].is_string()) return false;
        out.push_back({item["id"].get<std::string>(), item["description"].get<std::string>(), item.value("hard", true)});
    }
    return true;
}

bool assumptions(const json & value, std::vector<common_plan_assumption> & out) {
    if (!value.is_array()) return false;
    out.clear();
    for (const auto & item : value) {
        if (!item.is_object() || !item.contains("id") || !item["id"].is_string() || !item.contains("statement") || !item["statement"].is_string()) return false;
        common_plan_assumption assumption;
        assumption.id = item["id"].get<std::string>();
        assumption.statement = item["statement"].get<std::string>();
        assumption.confidence = item.value("confidence", 0.5f);
        assumption.valid = true;
        out.push_back(std::move(assumption));
    }
    return true;
}

bool validate(const common_agent_bootstrap_package & package, std::string & error) {
    if (package.name.empty() || package.name.size() > 128 || package.version.empty() || package.version.size() > 64) { error = "package name and version are required"; return false; }
    std::set<std::string> procedures, blueprints;
    for (const auto & procedure : package.procedures) {
        if (!valid_id(procedure.id) || procedure.content.empty() || procedure.content.size() > 8192 || procedure.summary.size() > 1024 || procedure.importance < 0 || procedure.importance > 1 || procedure.confidence < 0 || procedure.confidence > 1 || !procedures.insert(procedure.id).second) { error = "invalid or duplicate procedure definition"; return false; }
    }
    for (const auto & blueprint : package.blueprints) {
        if (!valid_id(blueprint.id) || blueprint.source_revision.size() > 128 || blueprint.goal.empty() || blueprint.goal.size() > 4096 || blueprint.success_criteria.empty() || blueprint.success_criteria.size() > 4096 || blueprint.steps.empty() || blueprint.steps.size() > 64 || !blueprints.insert(blueprint.id).second) { error = "invalid or duplicate blueprint definition"; return false; }
        std::set<std::string> steps;
        for (const auto & step : blueprint.steps) {
            if (!valid_id(step.id) || step.title.empty() || step.objective.empty() || step.tool_call || step.selected_tool || !steps.insert(step.id).second) { error = "invalid blueprint step or tool binding"; return false; }
        }
        for (const auto & step : blueprint.steps) for (const auto & dependency : step.depends_on) if (!steps.count(dependency) || dependency == step.id) { error = "blueprint dependency references an unknown step"; return false; }
        std::set<std::string> constraints, assumptions;
        for (const auto & capability : blueprint.required_capabilities) if (capability.empty() || capability.size() > 128) { error = "invalid blueprint required capability"; return false; }
        for (const auto & constraint : blueprint.constraints) if (!valid_id(constraint.id) || constraint.description.empty() || constraint.description.size() > 4096 || !constraints.insert(constraint.id).second) { error = "invalid or duplicate blueprint constraint"; return false; }
        for (const auto & assumption : blueprint.assumptions) if (!valid_id(assumption.id) || assumption.statement.empty() || assumption.statement.size() > 4096 || assumption.confidence < 0 || assumption.confidence > 1 || !assumptions.insert(assumption.id).second) { error = "invalid or duplicate blueprint assumption"; return false; }
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
                blueprint.id = item.value("id", std::string{}); blueprint.source_revision = item.value("source_revision", std::string{}); blueprint.selection_description = item.value("selection_description", std::string{});
                blueprint.purpose = item.value("purpose", std::string{});
                blueprint.goal = item.value("goal", std::string{}); blueprint.success_criteria = item.value("success_criteria", std::string{});
                if (item.contains("required_capabilities") && !strings(item["required_capabilities"], blueprint.required_capabilities)) { error = "blueprint required_capabilities must be strings"; return false; }
                if (item.contains("next_action")) { if (!item["next_action"].is_string()) { error = "next_action must be a string"; return false; } blueprint.next_action = item["next_action"].get<std::string>(); }
                if (item.contains("constraints") && !constraints(item["constraints"], blueprint.constraints)) { error = "blueprint constraints are invalid"; return false; }
                if (item.contains("assumptions") && !assumptions(item["assumptions"], blueprint.assumptions)) { error = "blueprint assumptions are invalid"; return false; }
                for (const auto & source : item["steps"]) {
                    if (!source.is_object()) { error = "package step must be an object"; return false; }
                    common_plan_step step; step.id = source.value("id", std::string{}); step.title = source.value("title", std::string{}); step.objective = source.value("objective", std::string{}); step.intended_contribution = source.value("contribution", step.objective); step.optional = source.value("optional", false);
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
        if (!blueprint.source_revision.empty()) value["source_revision"] = blueprint.source_revision;
        if (!blueprint.selection_description.empty()) value["selection_description"] = blueprint.selection_description;
        if (!blueprint.purpose.empty()) value["purpose"] = blueprint.purpose;
        if (!blueprint.required_capabilities.empty()) value["required_capabilities"] = blueprint.required_capabilities;
        if (blueprint.next_action) value["next_action"] = *blueprint.next_action;
        if (!blueprint.constraints.empty()) { value["constraints"] = json::array(); for (const auto & constraint : blueprint.constraints) value["constraints"].push_back({{"id", constraint.id}, {"description", constraint.description}, {"hard", constraint.hard}}); }
        if (!blueprint.assumptions.empty()) { value["assumptions"] = json::array(); for (const auto & assumption : blueprint.assumptions) value["assumptions"].push_back({{"id", assumption.id}, {"statement", assumption.statement}, {"confidence", assumption.confidence}}); }
        for (const auto & step : blueprint.steps) { json item = {{"id", step.id}, {"title", step.title}, {"objective", step.objective}}; if (!step.intended_contribution.empty() && step.intended_contribution != step.objective) item["contribution"] = step.intended_contribution; if (step.optional) item["optional"] = true; if (!step.depends_on.empty()) item["depends_on"] = step.depends_on; if (!step.required_evidence.empty()) item["required_evidence"] = step.required_evidence; value["steps"].push_back(std::move(item)); }
        root["blueprints"].push_back(std::move(value));
    }
    text = pretty ? root.dump(2) + "\n" : root.dump(); error.clear(); return true;
}
