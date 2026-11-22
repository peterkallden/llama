#pragma once

#include "agent/tooling/adapters/families/graph-adapters.h"
#include <memory>

// Compose existing read adapters with the same host scope and repository binding.
inline common_tool_execution_result common_change_impact(const common_native_tool_bindings & bindings,
        const common_adapter_json & args, uint32_t timeout_ms, bool collect_impact = true) {
    using json = common_adapter_json;
    common_tool_catalog catalog;
    common_tool_bootstrap_result boot;
    std::string error;
    if (!catalog.bootstrap("code-analysis", boot, error))
        return common_adapter_execution_failure("tool.code.change_impact.catalog", error, "Analysis catalog unavailable.");
    common_tool_registry tools;
    for (const char * name : {"code.symbols", "code.references", "code.impact", "repository.read", "repository.search", "graph.find", "graph.search"}) {
        const auto * definition = catalog.find_definition(name);
        bool installed = false;
        if (!definition) continue;
        if (std::string(name).find("code.") == 0)
            common_try_register_code_tool_adapter(*definition, bindings, tools, installed, error);
        else if (std::string(name).find("repository.") == 0)
            common_try_register_repository_tool_adapter(*definition, bindings, tools, installed, error);
        else common_try_register_graph_tool_adapter(*definition, bindings, tools, installed, error);
    }
    const auto started = std::chrono::steady_clock::now();
    size_t requests = 0, retained = 0;
    json gaps = json::array(), evidence = json::array(), sections = json::array();
    auto query = [&](const std::string & name, const json & input) -> json {
        if (++requests > 12 || std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(timeout_ms)) {
            gaps.push_back({{"tool", name}, {"reason", "shared_query_budget_exhausted"}}); return nullptr;
        }
        const auto query_started = std::chrono::steady_clock::now();
        const auto result = tools.execute({name, input.dump()});
        if (!result.ok) {
            gaps.push_back({{"tool", name}, {"reason", "unavailable_or_failed"}}); return nullptr;
        }
        const auto output = json::parse(result.output, nullptr, false);
        if (!output.is_object() || result.output.size() > 48000 - retained) {
            gaps.push_back({{"tool", name}, {"reason", "invalid_result_or_output_budget"}}); return nullptr;
        }
        sections.push_back({{"tool", name}, {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - query_started).count()}, {"evidence_ref", "e" + std::to_string(evidence.size() + 1)},
            {"complete", output.value("complete", false)}, {"truncated", output.value("truncated", false)},
            {"limitations", output.value("limitations", json::array())},
            {"continuations", output.value("continuations", json::array())}});
        if (output.value("truncated", false) || (output.contains("complete") && !output.value("complete", false)))
            gaps.push_back({{"tool", name}, {"reason", "partial_child_coverage"}, {"section", sections.size() - 1}});
        retained += result.output.size();
        evidence.push_back({{"id", "e" + std::to_string(evidence.size() + 1)}, {"tool", name}, {"arguments", input}, {"observation", output}});
        return output;
    };
    const auto path = args.at("path").get<std::string>();
    const auto symbol = args.at("symbol").get<std::string>();
    const auto before = common_repository_observation(bindings.repository_root, path);
    auto definitions = query("code.symbols", {{"symbol", symbol}, {"path_hint", path}, {"max_results", 16}});
    json target = {{"path", path}, {"symbol", symbol}, {"resolution", "unresolved"}};
    if (args.contains("line")) {
        target["line"] = args["line"]; target["column"] = args.value("column", 1);
        target["resolution"] = "explicit_location";
    } else if (definitions.is_object() && definitions.value("semantic", false) && definitions.contains("items")) {
        json matches = json::array();
        for (const auto & item : definitions["items"])
            if (item.value("path", std::string{}) == path && item.value("name", std::string{}) == symbol) matches.push_back(item);
        if (matches.size() == 1 && matches[0].contains("line") && definitions.value("complete", false) && !definitions.value("truncated", true)) {
            target["line"] = matches[0]["line"]; target["column"] = matches[0].value("column", 1);
            target["resolution"] = "unique_semantic_definition";
        } else { target["candidates"] = matches; gaps.push_back({{"reason", "ambiguous_or_incomplete_symbol_resolution"}}); }
    }
    const int line = target.value("line", 1);
    const auto read_args = json{{"path", path}, {"start_line", std::max(1, line - 3)}, {"end_line", line + 12}};
    const auto source = query("repository.read", read_args);
    json affected = json::array(), referenced_tests = json::array();
    if (collect_impact && target.contains("line")) {
        const json location = {{"symbol", symbol}, {"path_hint", path}, {"line", target["line"]}, {"column", target["column"]}};
        auto refs = location; refs["max_results"] = 16; refs["include_context"] = true;
        const auto references = query("code.references", refs);
        if (references.is_object() && references.value("semantic", false))
            for (const auto & item : references.value("items", json::array())) {
                const auto file = item.value("path", std::string{});
                if (file.compare(0, 6, "tests/") == 0)
                    referenced_tests.push_back({{"location", item}, {"evidence_ref", "e" + std::to_string(evidence.size())},
                        {"basis", "semantic_reference_in_tests_directory"}});
            }
        // Reserve separate traversal budgets so outgoing library calls cannot crowd out callers.
        std::set<std::string> seen;
        for (const auto * direction : {"callers", "callees"}) {
            auto impact = location; impact["direction"] = direction;
            impact["max_depth"] = args.value("max_depth", 2); impact["max_nodes"] = 12; impact["max_edges"] = 16;
            const auto calls = query("code.impact", impact);
            if (calls.is_object() && calls.contains("visualization"))
                for (auto node : calls["visualization"].value("nodes", json::array())) {
                    const auto id = node.value("id", std::string{});
                    if (!seen.insert(id).second) continue;
                    node["discovery_direction"] = direction;
                    node["evidence_ref"] = "e" + std::to_string(evidence.size());
                    affected.push_back(std::move(node));
                }
        }
    } else if (collect_impact) gaps.push_back({{"reason", "provide_explicit_line_and_column_for_relationship_analysis"}});
    const auto basename = std::filesystem::path(path).filename().generic_string();
    auto knowledge = collect_impact ? query("graph.find", {{"query", basename}, {"limit", 8}}) : json(nullptr);
    if (knowledge.is_object() && knowledge.contains("nodes") && knowledge["nodes"].is_array()) {
        json matched = json::array();
        for (const auto & node : knowledge["nodes"])
            if (node.value("path", std::string{}) == path) matched.push_back(node);
        knowledge["nodes"] = std::move(matched);
        knowledge["matching"] = "exact_repository_path";
    }
    if (knowledge.is_object() && knowledge.contains("nodes") && knowledge["nodes"].is_array()) {
        size_t expanded = 0;
        for (const auto & node : knowledge["nodes"]) {
            if (++expanded > 2) break;
            if (node.contains("id") && node["id"].is_string())
                query("graph.search", {{"from", node["id"]}, {"limit", 4}});
        }
    }
    auto tests = collect_impact ? query("repository.search", {{"query", symbol}, {"path", "tests"}, {"max_results", 8}, {"context_lines", 2}}) : json(nullptr);

    if (tests.is_object()) tests["semantic_reference_candidates"] = referenced_tests;

    const auto after_source = query("repository.read", read_args);
    const auto after = common_repository_observation(bindings.repository_root, path);
    const bool unchanged = source.is_object() && after_source.is_object() && source.contains("sha256") &&
        after_source.contains("sha256") && source["sha256"] == after_source["sha256"] &&
        before.value("status", std::string{}) == "observed" && after.value("status", std::string{}) == "observed" &&
        before["revision"] == after["revision"];
    gaps.push_back({{"reason", "Project-wide semantic index coverage and freshness are unknown; source checks are not an atomic snapshot."}});
    if (collect_impact) gaps.push_back({{"reason", "Contracts require interpretation. Test candidates are text matches under tests, not verified test coverage; tests in other directories are not searched."}});
    if (!collect_impact) gaps.push_back({{"reason", "Contract collection reads only the bounded target excerpt and symbol candidates; callers, tests and saved graph notes are not collected."}});
    return common_adapter_success_json({{"schema", "code-change-impact.v1"}, {"target", target},
        {"intent", args.value("intent", std::string{})}, {"affected_symbols", affected},
        {"contracts", json::array()}, {"test_candidates", tests}, {"knowledge_candidates", knowledge},
        {"knowledge_gaps", gaps}, {"evidence", evidence}, {"source", before}, {"source_after", after},
        {"target_content_unchanged_during_query", unchanged}, {"queries_used", requests},
        {"sections", sections},
        {"coverage", {{"complete", false}, {"revision_bound", false}, {"limitations", gaps}}}});
}
