#include "memory/memory-in-memory.h"
#include "memory/memory-tool-service.h"

#include <cassert>

int main() {
    std::string error;

    common_memory_tool_search_arguments_contract search_contract;
    assert(common_memory_parse_tool_search_arguments_value(
        nlohmann::ordered_json::object({
            {"query", "resident runtime"},
            {"limit", 3},
        }),
        search_contract,
        error));
    assert(search_contract.query == "resident runtime");
    assert(search_contract.limit.has_value() && *search_contract.limit == 3);

    assert(!common_memory_parse_tool_search_arguments_value(
        nlohmann::ordered_json::object({
            {"query", "resident runtime"},
            {"unexpected", true},
        }),
        search_contract,
        error));
    assert(error == "unsupported argument: unexpected");

    common_memory_tool_remember_arguments_contract remember_contract;
    assert(common_memory_parse_tool_remember_arguments_value(
        nlohmann::ordered_json::object({
            {"kind", "procedure"},
            {"content", "Verify project scope before reading stored evidence."},
            {"importance", 0.8},
            {"confidence", 0.7},
            {"rationale", "Frequently reused repair step."},
        }),
        remember_contract,
        error));
    assert(remember_contract.kind == "procedure");
    assert(remember_contract.content == "Verify project scope before reading stored evidence.");
    assert(remember_contract.importance.has_value());
    assert(remember_contract.confidence.has_value());
    assert(remember_contract.rationale.has_value());

    common_memory_in_memory_store store;
    common_memory_record record;
    record.id = "memory-1";
    record.kind = common_memory_kind::fact;
    record.content = "Resident runtime keeps a loaded model session alive across turns.";
    record.summary = record.content;
    record.scope = common_memory_scope::session;
    record.namespace_id = "namespace-a";
    record.session_id = "session-a";
    record.project_id = "project-a";
    record.turn_id = "turn-a";
    assert(store.open("", error));
    assert(store.put(record, error));

    common_memory_tool_service service(store);
    common_memory_tool_context context;
    context.query_defaults.scope = common_memory_scope::session;
    context.query_defaults.namespace_id = "namespace-a";
    context.query_defaults.session_id = "session-a";
    context.query_defaults.project_id = "project-a";
    context.query_defaults.turn_id = "turn-a";
    context.query_defaults.limit = 4;
    context.query_defaults.token_budget = 64;
    context.max_search_limit = 8;
    context.allow_write_proposals = true;
    context.now = 123456;

    common_memory_tool_search_result search_result;
    assert(service.search(
        context,
        R"({"query":"loaded model session","limit":2})",
        search_result,
        error));
    assert(search_result.query == "loaded model session");
    assert(!search_result.hits.empty());
    assert(search_result.hits[0].memory.id == "memory-1");

    common_memory_tool_remember_result remember_result;
    assert(service.remember_proposal(
        context,
        R"({"kind":"procedure","content":"Summarize runtime evidence before proposing a repair.","importance":0.9,"confidence":0.8,"rationale":"Reusable reflection fallback."})",
        remember_result,
        error));
    assert(remember_result.proposal.namespace_id == "namespace-a");
    assert(remember_result.proposal.session_id == "session-a");
    assert(remember_result.proposal.project_id == "project-a");
    assert(remember_result.proposal.turn_id == "turn-a");
    assert(remember_result.proposal.global_opt_in == false);

    return 0;
}
