#pragma once

#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/tooling/adapters/families/diagnostics-adapters.h"
#include <ctime>
#include <set>

inline common_tool_execution_result common_failure_summary(const common_native_tool_bindings & bindings,
        const common_adapter_json & args) {
    using json = common_adapter_json;
    if (!bindings.resource_runtime.store)
        return common_adapter_execution_failure("tool.diagnostics.failure_summary.unavailable", "resource store unavailable", "Execution receipts are unavailable.");
    auto & store = *bindings.resource_runtime.store;
    const auto authority = make_agent_resource_read_authority(bindings.resource_runtime, std::time(nullptr));
    auto read = [&](const std::string & uri, size_t limit, std::string & text, std::string & error) {
        agent_resource_descriptor descriptor;
        if (!store.stat(uri, authority, descriptor, error)) return false;
        if (descriptor.size_bytes > limit) { error = "resource exceeds diagnostic budget"; return false; }
        std::string bytes;
        if (!store.read_bytes(uri, authority, limit + 1, bytes, error)) return false;
        if (bytes.size() > limit) { error = "resource exceeds diagnostic budget"; return false; }
        text.assign(bytes.begin(), bytes.end()); return true;
    };
    common_tool_catalog catalog; common_tool_bootstrap_result boot; std::string error;
    if (!catalog.bootstrap("analysis", boot, error))
        return common_adapter_execution_failure("tool.diagnostics.failure_summary.catalog", error, "Diagnostic catalog unavailable.");
    common_tool_registry parsers;
    for (const auto * name : {"diagnostics.compile", "diagnostics.test_failures"}) {
        bool installed = false;
        common_try_register_diagnostics_tool_adapter(*catalog.find_definition(name), bindings, parsers, installed, error);
    }
    auto summarize = [&](const std::string & uri, json & output) {
        agent_resource_descriptor receipt_descriptor;
        if (!store.stat(uri, authority, receipt_descriptor, error)) return false;
        if (receipt_descriptor.metadata.purpose != "development_execution_receipt" ||
                receipt_descriptor.lineage.derivation != "development-execution-receipt:v1") {
            error = "resource is not a retained development execution receipt"; return false;
        }
        std::string text;
        if (!read(uri, 65536, text, error)) return false;
        const auto receipt = json::parse(text, nullptr, false);
        if (!receipt.is_object() || receipt.value("schema", std::string{}) != "development-execution-receipt.v1") {
            error = "expected a development execution receipt"; return false;
        }
        output = {{"receipt_ref", uri}, {"observations", json::array()}, {"coverage_gaps", json::array()}};
        for (const auto * key : {"source_snapshot", "logical_target", "configuration", "test_filter", "status", "exit_code", "started_at", "completed_at"})
            if (receipt.contains(key)) output[key] = receipt[key];
        size_t count = 0;
        for (const auto & resource : receipt.value("output_resources", json::array())) {
            if (++count > 2) { output["coverage_gaps"].push_back("Additional output resources omitted."); break; }
            const auto ref = resource.value("uri", std::string{});
            if (!read(ref, 65536, text, error)) return false;
            const auto compile = parsers.execute({"diagnostics.compile", json{{"output", text.empty() ? "\n" : text}}.dump()});
            const auto tests = parsers.execute({"diagnostics.test_failures", json{{"result", text.empty() ? "\n" : text}}.dump()});
            if (!compile.ok || !tests.ok) { error = "diagnostic parser failed"; return false; }
            output["observations"].push_back({{"resource_ref", ref}, {"compiler", json::parse(compile.output)},
                {"failure_groups", json::parse(tests.output)}, {"ordering", "within_stream_only"}});
        }
        output["coverage_gaps"].push_back("Retained output may be an excerpt; stream order is not causal order. No root cause is inferred.");
        if (!count) output["coverage_gaps"].push_back("No retained output resources.");
        return true;
    };
    json current, previous = nullptr;
    if (!summarize(args.at("receipt_ref").get<std::string>(), current) ||
            (args.contains("previous_receipt_ref") && !summarize(args.at("previous_receipt_ref").get<std::string>(), previous)))
        return common_adapter_execution_failure("tool.diagnostics.failure_summary.invalid_receipt", error, "Receipt or referenced output could not be read in the current scope.");
    json comparison = nullptr;
    if (previous.is_object()) {
        auto keys = [](const json & report) {
            std::set<std::string> result;
            for (const auto & stream : report["observations"]) {
                for (const auto & entry : stream["compiler"]["diagnostics"]) result.insert(entry.dump());
                for (const auto & entry : stream["failure_groups"]["failure_groups"])
                    result.insert(json{{"classification", entry["classification"]}, {"message", entry["message"]}}.dump());
            }
            return result;
        };
        auto now = keys(current), old = keys(previous);
        comparison = {{"added", json::array()}, {"removed", json::array()}, {"persisting", json::array()}};
        for (const auto & key : now) comparison[old.count(key) ? "persisting" : "added"].push_back(json::parse(key));
        for (const auto & key : old) if (!now.count(key)) comparison["removed"].push_back(json::parse(key));
        comparison["same_context"] = current.contains("source_snapshot") && current["source_snapshot"].is_object() && previous.contains("source_snapshot") && previous["source_snapshot"].is_object() && current.value("source_snapshot", json(nullptr)) == previous.value("source_snapshot", json(nullptr)) &&
            current.value("logical_target", json(nullptr)) == previous.value("logical_target", json(nullptr)) &&
            current.value("configuration", json(nullptr)) == previous.value("configuration", json(nullptr)) &&
            current.value("test_filter", json(nullptr)) == previous.value("test_filter", json(nullptr));
        comparison["interpretation"] = "Removed diagnostics are absent from retained output, not proof of a repaired cause.";
    }
    const json response = {{"schema", "failure-summary.v1"}, {"current", current}, {"previous", previous},
        {"comparison", comparison}, {"hypotheses", json::array()}, {"complete", false}};
    if (response.dump().size() > 60000)
        return common_adapter_limit_failure("tool.diagnostics.failure_summary.output_limit", "diagnostic summary too large", "Narrow the retained diagnostic inputs.");
    return common_adapter_success_json(response);
}
