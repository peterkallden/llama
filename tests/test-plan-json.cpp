#include "plan/plan-json.h"
#include "plan/plan-bindings.h"
#include "plan/plan-context.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <nlohmann/json.hpp>

int main() {
    common_plan_state plan;
    plan.id = "p";
    std::vector<common_plan_operation> operations;
    std::string error;
    auto same_json_object = [](const std::string & lhs, const std::string & rhs) {
        return nlohmann::json::parse(lhs, nullptr, false) ==
               nlohmann::json::parse(rhs, nullptr, false);
    };

    const auto compact = R"({"purpose":"inspect the current implementation","goal":"inspect bindings","steps":[{"id":"search","contribution":"find the relevant implementation","tool":"repository.search","args":{"query":"plan bindings"}},{"id":"read","tool":{"name":"repository.read","arguments":{"path":{"$from_step":"search","$json_pointer":"/matches/0/path"}}},"after":"search"},{"id":"answer","mode":"final","after":"read"}]})";
    assert(common_plan_parse_proposal_json(compact, plan, operations, error));
    assert(operations.size() == 3);
    assert(same_json_object(operations[0].step->tool_call->arguments_json, R"({"query":"plan bindings"})"));
    assert(plan.purpose == "inspect the current implementation");
    assert(operations[0].step->intended_contribution == "find the relevant implementation");
    assert(operations[1].step->depends_on == std::vector<std::string>{"search"});
    assert(common_plan_step_effective_mode(*operations[2].step) == common_plan_step_mode::final_response);

    const auto shorthand_binding = R"({"goal":"aggregate a document table","steps":[{"id":"table","tool":"document.table","args":{"resource":"r1","table":"Budget summary"}},{"id":"sum","after":["table"],"tool":"data.aggregate","args":{"dataset":"$table.dataset","measures":[{"function":"sum","column":"amount"}]}}]})";
    assert(common_plan_parse_proposal_json(shorthand_binding, plan, operations, error));
    const auto shorthand_args = nlohmann::json::parse(operations[1].step->tool_call->arguments_json, nullptr, false);
    assert(shorthand_args["dataset"].is_object());
    assert(shorthand_args["dataset"].value("$from_step", "") == "table");
    assert(shorthand_args["dataset"].value("$json_pointer", "") == "/dataset");

    const auto normalized = R"({"goal":"calculate","steps":[{"id":"calculate","tool":"calculator","args":{"operation":"multiply","operands":[{"value":17},{"value":23}]}}]})";
    assert(common_plan_parse_proposal_json(normalized, plan, operations, error));
    assert(operations.size() == 2); // native final synthesis
    assert(same_json_object(operations[0].step->tool_call->arguments_json, R"({"expression":"17 * 23"})"));
    assert(operations[1].step->id == "answer");

    const auto repaired_integer = R"({"goal":"inspect","steps":[{"id":"search","tool":"repository.search","args":{"query":"plan","max_results":"16"}}]})";
    assert(common_plan_parse_proposal_json(repaired_integer, plan, operations, error));
    const auto repaired_actual = nlohmann::json::parse(operations[0].step->tool_call->arguments_json, nullptr, false);
    assert(repaired_actual.is_object());
    assert(repaired_actual.value("query", std::string()) == "plan");
    assert(repaired_actual.contains("max_results"));
    assert(repaired_actual["max_results"].is_number_integer());
    assert(repaired_actual["max_results"].get<int>() == 16);

    common_plan_state evidence_plan;
    std::vector<common_plan_operation> evidence_operations;
    assert(common_plan_parse_proposal_json(
        R"({"goal":"read","steps":[{"id":"read","title":"Read","objective":"Read evidence","required_evidence":["tool:read:repository.search"],"tool":"repository.search","args":{"query":"resource"}}]})",
        evidence_plan, evidence_operations, error));
    assert(evidence_operations.front().step->required_evidence == std::vector<std::string>{"tool:read:repository.search"});
    std::string merged_arguments;
    assert(common_plan_merge_tool_arguments_json(
        R"({"id":"r1","representation":"text","offset":0,"max_bytes":1048576})",
        R"({"max_bytes":8192})",
        merged_arguments,
        error));
    const auto merged = nlohmann::json::parse(merged_arguments, nullptr, false);
    assert(merged.value("id", std::string()) == "r1");
    assert(merged.value("representation", std::string()) == "text");
    assert(merged.value("offset", -1) == 0);
    assert(merged.value("max_bytes", 0) == 8192);

    const auto wrapped_tool_arguments = R"({"goal":"inspect memory","steps":[{"id":"search","tool":"memory_search","args":{"tool":{"name":"memory_search","arguments":{"query":"regression procedure","limit":"2"}}}}]})";
    assert(common_plan_parse_proposal_json(wrapped_tool_arguments, plan, operations, error));
    const auto wrapped_actual = nlohmann::json::parse(operations[0].step->tool_call->arguments_json, nullptr, false);
    assert(wrapped_actual.is_object());
    assert(wrapped_actual.value("query", std::string()) == "regression procedure");
    assert(wrapped_actual.contains("limit"));
    assert(wrapped_actual["limit"].is_number_integer());
    assert(wrapped_actual["limit"].get<int>() == 2);

    const auto sibling_tool_argument = R"({"goal":"inspect memory","steps":[{"id":"search","tool":"memory_search","args":{"tool":"memory_search","query":"regression procedure","limit":"2"}}]})";
    assert(common_plan_parse_proposal_json(sibling_tool_argument, plan, operations, error));
    const auto sibling_actual = nlohmann::json::parse(operations[0].step->tool_call->arguments_json, nullptr, false);
    assert(sibling_actual.is_object());
    assert(sibling_actual.value("query", std::string()) == "regression procedure");
    assert(sibling_actual.contains("limit"));
    assert(sibling_actual["limit"].is_number_integer());
    assert(sibling_actual["limit"].get<int>() == 2);

    const auto compact_without_ids = R"({"goal":"inspect","steps":[{"tool":"repository.search","args":{"query":"planner"}},{"tool":{"name":"repository.read","arguments":{"path":{"$from_step":"step_1","$json_pointer":"/matches/0/path"}}}},{"mode":"reasoning"}]})";
    assert(common_plan_parse_proposal_json(compact_without_ids, plan, operations, error));
    assert(operations.size() == 4); // native final synthesis
    assert(operations[0].step->id == "step_1");
    assert(operations[1].step->id == "step_2");
    assert(operations[2].step->id == "step_3");
    assert(operations[1].step->depends_on == std::vector<std::string>{"step_1"});
    assert(operations[2].step->depends_on == std::vector<std::string>{"step_2"});
    assert(operations[3].step->id == "answer");
    assert(operations[3].step->depends_on == std::vector<std::string>({"step_1", "step_2", "step_3"}));

    // Simple model plans do not own internal IDs or dependencies. A malformed
    // self-dependency from a small model is ignored at this boundary, while
    // the host creates the sequential chain and resolves $previous.
    const auto simple_dataflow = R"({"goal":"read and aggregate","steps":[{"tool":"document.table","args":{"resource":"r1","table":"Budget summary"},"as":"table"},{"tool":"data.aggregate","args":{"dataset":"$previous.dataset","measures":[{"function":"sum","column":"amount"}]},"after":["step_2"]}]})";
    assert(common_plan_parse_proposal_json(simple_dataflow, plan, operations, error));
    assert(operations[0].step->id == "step_1");
    assert(operations[1].step->id == "step_2");
    assert(operations[1].step->depends_on == std::vector<std::string>{"step_1"});
    const auto simple_binding = nlohmann::json::parse(operations[1].step->tool_call->arguments_json, nullptr, false);
    assert(simple_binding["dataset"].value("$from_step", "") == "step_1");
    assert(simple_binding["dataset"].value("$json_pointer", "") == "/dataset");

    // Semantic aliases are model-facing only; the canonical plan retains the
    // same host-owned step IDs and supports a later multi-input join.
    const auto aliased_dataflow = R"({"goal":"join tables","steps":[{"as":"budget","tool":"document.table","args":{"resource":"r1","table":"Budget"}},{"as":"forecast","tool":"document.table","args":{"resource":"r1","table":"Forecast"}},{"tool":"data.join","args":{"left":"$budget.dataset","right":"$forecast.dataset"}}]})";
    assert(common_plan_parse_proposal_json(aliased_dataflow, plan, operations, error));
    assert(operations[0].step->id == "step_1");
    assert(operations[1].step->id == "step_2");
    assert(operations[2].step->id == "step_3");
    const auto join_args = nlohmann::json::parse(operations[2].step->tool_call->arguments_json, nullptr, false);
    assert(join_args["left"].value("$from_step", "") == "step_1");
    assert(join_args["right"].value("$from_step", "") == "step_2");

    const auto unknown_alias = R"({"goal":"aggregate a table","steps":[{"tool":"document.table","args":{"resource":"r1","table":"Budget"}},{"tool":"data.aggregate","args":{"dataset":"$table.dataset","measures":[{"function":"sum","column":"amount"}]}}]})";
    assert(!common_plan_parse_proposal_json(unknown_alias, plan, operations, error));
    assert(error.find("plan.binding.unknown_alias") != std::string::npos);

    const auto invalid_reference = R"({"goal":"aggregate a table","steps":[{"tool":"document.table","args":{"resource":"r1","table":"Budget"}},{"tool":"data.aggregate","args":{"dataset":"$tables[0].dataset","measures":[{"function":"sum","column":"amount"}]}}]})";
    assert(!common_plan_parse_proposal_json(invalid_reference, plan, operations, error));
    assert(error.find("plan.binding.invalid_syntax") != std::string::npos);

    const auto invalid_reference_shape = R"({"goal":"aggregate a table","steps":[{"tool":"document.table","args":{"resource":"r1","table":"Budget"}},{"tool":"data.aggregate","args":{"dataset":"$table","measures":[{"function":"sum","column":"amount"}]}}]})";
    assert(!common_plan_parse_proposal_json(invalid_reference_shape, plan, operations, error));
    assert(error.find("plan.binding.invalid_syntax") != std::string::npos);
    const auto indexed_reference = R"({"goal":"select the first dataset","steps":[{"tool":"dataset.list","args":{},"as":"candidates"},{"tool":"dataset.inspect","args":{"dataset":"$candidates.datasets[0]"}}]})";
    assert(common_plan_parse_proposal_json(indexed_reference, plan, operations, error));
    const auto indexed_args = nlohmann::json::parse(operations[1].step->tool_call->arguments_json, nullptr, false);
    assert(indexed_args["dataset"].value("$from_step", "") == "step_1");
    assert(indexed_args["dataset"].value("$json_pointer", "") == "/datasets/0");
    const auto invalid_indexed_reference = R"({"goal":"select a dataset","steps":[{"tool":"dataset.list","args":{},"as":"candidates"},{"tool":"dataset.inspect","args":{"dataset":"$candidates.datasets[wrong]"}}]})";
    assert(!common_plan_parse_proposal_json(invalid_indexed_reference, plan, operations, error));
    assert(error.find("plan.binding.invalid_syntax") != std::string::npos);

    const auto literal_alias_name = R"({"goal":"use a literal","steps":[{"tool":"document.table","args":{"resource":"r1","table":"table"},"as":"table"}]})";
    assert(common_plan_parse_proposal_json(literal_alias_name, plan, operations, error));
    const auto literal_args = nlohmann::json::parse(operations[0].step->tool_call->arguments_json, nullptr, false);
    assert(literal_args.value("table", std::string()) == "table");

    const auto missing_reference_prefix = R"({"goal":"aggregate a table","steps":[{"tool":"document.table","args":{"resource":"r1","table":"Budget"},"as":"table"},{"tool":"data.aggregate","args":{"dataset":"table.dataset","measures":[{"function":"sum","column":"amount"}]}}]})";
    assert(!common_plan_parse_proposal_json(missing_reference_prefix, plan, operations, error));
    assert(error.find("plan.binding.alias_used_as_literal") != std::string::npos);

    // The prior full proposal format remains accepted for persisted or older callers.
    const auto legacy = R"({"goal":"answer","success_criteria":"clear","next_action":"draft","operations":[{"kind":"add_step","reason_summary":"tool use","evidence_ids":[],"step":{"id":"s1","title":"Calc","objective":"compute","depends_on":[],"required_evidence":[],"tool":{"name":"calculator","arguments_json":"{'operation':'multiply','operands':[{'value':17},{'value':23}]}"}}}]})";
    assert(common_plan_parse_proposal_json(legacy, plan, operations, error));
    assert(same_json_object(operations[0].step->tool_call->arguments_json, R"({"expression":"17 * 23"})"));
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[]})", plan, operations, error));
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[{}]})", plan, operations, error));
    assert(error.find("final synthesis is host-owned") != std::string::npos);
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[{"id":"invalid","mode":"tool"}]})", plan, operations, error));
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[{"id":"search","tool":"repository.search","args":{"query":"x"}},{"id":"search","mode":"final"}]})", plan, operations, error));
    assert(error == "duplicate step id");
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[{"id":"answer","tool":"calculator","args":{"expression":"6 * 7"}}]})", plan, operations, error));
    assert(error == "native final step id conflicts with proposed step");
    assert(common_plan_proposal_json_schema().find("arguments_json") == std::string::npos);
    assert(common_plan_proposal_json_schema().find(R"("required":["id"])") == std::string::npos);
    assert(common_plan_proposal_json_schema().find("oneOf") == std::string::npos);
    const auto model_schema = nlohmann::json::parse(
        common_plan_model_facing_json_schema({"document.table", "data.aggregate"}), nullptr, false);
    assert(model_schema.is_object());
    const auto model_step_schema = model_schema["properties"]["steps"]["items"];
    assert(model_step_schema["properties"].contains("as"));
    assert(!model_step_schema["properties"].contains("id"));
    assert(!model_step_schema["properties"].contains("depends_on"));
    assert(model_step_schema["properties"]["tool"]["enum"].size() == 2);

    common_plan_state materialize_plan;
    common_plan_step completed_search{"search", "Search", "Find evidence"};
    completed_search.status = common_plan_step_status::completed;
    materialize_plan.steps.push_back(completed_search);
    common_plan_observation search_observation;
    search_observation.id = "tool:search:1";
    search_observation.source = "tool:search:1";
    search_observation.summary = R"({"matches":[{"path":"pocs/agent/agent-runtime-host.cpp"}]})";
    search_observation.confidence = 1.0f;
    materialize_plan.observations.push_back(search_observation);
    common_plan_step read_step{"read", "Read", "Open the file"};
    read_step.tool_call = common_plan_tool_call{
        "repository.read",
        R"({"path":{"$from_step":"search","$json_pointer":"/matches/0/path"}})",
    };
    std::string materialized_arguments_json;
    assert(common_plan_materialize_tool_arguments(
        materialize_plan,
        read_step,
        read_step.tool_call->arguments_json,
        materialized_arguments_json,
        error));
    assert(same_json_object(materialized_arguments_json, R"({"path":"pocs/agent/agent-runtime-host.cpp"})"));

    common_plan_step table_result{"table", "Table", "Resolve table"};
    table_result.status = common_plan_step_status::completed;
    table_result.tool_call = common_plan_tool_call{"document.table", "{}"};
    materialize_plan.steps.push_back(table_result);
    common_plan_observation table_observation{
        "tool:table:1", "document.table", R"({"dataset":"d1","name":"Budget summary"})", 1.0f, {}, {}, 0};
    materialize_plan.observations.push_back(table_observation);
    common_plan_step aggregate_step{"aggregate", "Aggregate", "Sum amount"};
    aggregate_step.depends_on = {"table"};
    aggregate_step.tool_call = common_plan_tool_call{
        "data.aggregate",
        R"({"measures":[{"function":"sum","column":"amount"}]})",
    };
    const auto dataflow_resolver = [](const std::string & tool_name,
            common_plan_tool_dataflow_contract & contract,
            std::string & contract_error) {
        if (tool_name == "document.table") {
            return common_plan_dataflow_contract_from_schemas(
                tool_name,
                R"({"type":"object","properties":{"resource":{"type":"string","x-agent-type":"resource_ref"}}})",
                R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref"}}})",
                contract, contract_error);
        }
        if (tool_name == "data.aggregate") {
            return common_plan_dataflow_contract_from_schemas(
                tool_name,
                R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref"}},"required":["dataset"]})",
                R"({"type":"object","properties":{"rows":{"type":"array"},"dataset":{"type":"string","x-agent-type":"dataset_ref"}}})",
                contract, contract_error);
        }
        contract_error.clear();
        return false;
    };
    assert(common_plan_materialize_tool_arguments(
        materialize_plan,
        aggregate_step,
        aggregate_step.tool_call->arguments_json,
        materialized_arguments_json,
        error,
        dataflow_resolver));
    assert(nlohmann::json::parse(materialized_arguments_json, nullptr, false).value("dataset", "") == "d1");

    common_plan_step mismatch_source{"mismatch-source", "Source", "Produce a resource"};
    mismatch_source.status = common_plan_step_status::completed;
    mismatch_source.tool_call = common_plan_tool_call{"resource.producer", "{}"};
    materialize_plan.steps.push_back(mismatch_source);
    materialize_plan.observations.push_back({
        "tool:mismatch-source:1", "resource.producer", R"({"resource":"r1"})", 1.0f, {}, {}, 0});
    common_plan_step mismatch_target{"mismatch-target", "Target", "Consume a dataset"};
    mismatch_target.depends_on = {"mismatch-source"};
    mismatch_target.tool_call = common_plan_tool_call{
        "data.aggregate",
        R"({"dataset":{"$from_step":"mismatch-source","$json_pointer":"/resource"}})"};
    const auto mismatch_resolver = [dataflow_resolver](const std::string & tool_name,
            common_plan_tool_dataflow_contract & contract,
            std::string & contract_error) {
        if (tool_name == "resource.producer") {
            return common_plan_dataflow_contract_from_schemas(
                tool_name,
                R"({"type":"object"})",
                R"({"type":"object","properties":{"resource":{"type":"string","x-agent-type":"resource_ref"}}})",
                contract, contract_error);
        }
        return dataflow_resolver(tool_name, contract, contract_error);
    };
    assert(!common_plan_materialize_tool_arguments(
        materialize_plan,
        mismatch_target,
        mismatch_target.tool_call->arguments_json,
        materialized_arguments_json,
        error,
        mismatch_resolver));
    assert(error.find("plan.binding.incompatible_types") != std::string::npos);

    common_plan_step ambiguous_target{"ambiguous", "Join", "Join two datasets"};
    ambiguous_target.depends_on = {"table"};
    ambiguous_target.tool_call = common_plan_tool_call{
        "data.join", R"({"on":[{"left":"id","right":"id"}]})"};
    const auto join_resolver = [dataflow_resolver](const std::string & tool_name,
            common_plan_tool_dataflow_contract & contract,
            std::string & contract_error) {
        if (tool_name == "data.join") {
            return common_plan_dataflow_contract_from_schemas(
                tool_name,
                R"({"type":"object","properties":{"left":{"type":"string","x-agent-type":"dataset_ref"},"right":{"type":"string","x-agent-type":"dataset_ref"}},"required":["left","right"]})",
                R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref"}}})",
                contract, contract_error);
        }
        return dataflow_resolver(tool_name, contract, contract_error);
    };
    assert(!common_plan_materialize_tool_arguments(
        materialize_plan,
        ambiguous_target,
        ambiguous_target.tool_call->arguments_json,
        materialized_arguments_json,
        error,
        join_resolver));
    assert(error.find("plan.binding.ambiguous_autowire") != std::string::npos);

    common_plan_state context_plan;
    context_plan.goal = "Use only relevant evidence";
    common_plan_step search{"search", "Search", "Find evidence"};
    search.result_summary = "matching file found";
    context_plan.steps.push_back(search);
    common_plan_step reasoning{"reason", "Reason", "Use the selected search result."};
    reasoning.depends_on = {"search"};
    context_plan.steps.push_back(reasoning);
    common_plan_observation matching_observation;
    matching_observation.id = "tool:search";
    matching_observation.source = "tool:search";
    matching_observation.summary = "matching file found";
    matching_observation.confidence = 1.0f;
    context_plan.observations.push_back(matching_observation);
    common_plan_observation unrelated_observation;
    unrelated_observation.id = "tool:other";
    unrelated_observation.source = "tool:other";
    unrelated_observation.summary = "unrelated result";
    unrelated_observation.confidence = 1.0f;
    context_plan.observations.push_back(unrelated_observation);
    const auto step_context = common_plan_render_step_context(context_plan, context_plan.steps.back());
    assert(step_context.find("matching file found") != std::string::npos);
    assert(step_context.find("unrelated result") == std::string::npos);

    std::string normalized_document_arguments;
    assert(common_plan_normalize_tool_arguments_json(
        "document.table",
        R"({"table_name":"Budget summary","resource":"agent-resource://turn/t/document.json"})",
        normalized_document_arguments,
        error));
    const auto normalized_document = nlohmann::json::parse(normalized_document_arguments, nullptr, false);
    assert(normalized_document.is_object());
    assert(normalized_document.value("table", std::string()) == "Budget summary");
    assert(!normalized_document.contains("table_name"));
    std::string normalized_document_again;
    assert(common_plan_normalize_tool_arguments_json(
        "document.table", normalized_document_arguments, normalized_document_again, error));
    assert(same_json_object(normalized_document_arguments, normalized_document_again));

    const auto proposal_schema = nlohmann::json::parse(common_plan_proposal_json_schema(), nullptr, false);
    assert(proposal_schema.is_object());
    const auto step_schema = proposal_schema["properties"]["steps"]["items"];
    assert(step_schema["properties"]["tool"].value("type", std::string()) == "string");
    assert(step_schema["properties"]["args"].value("type", std::string()) == "object");
    assert(!step_schema["properties"]["tool"].contains("properties"));
    std::string compact_schema_error;
    const auto compact_schema = common_render_compact_plan_schema(
        common_plan_proposal_json_schema(), compact_schema_error);
    assert(compact_schema_error.empty());
    assert(compact_schema.find("required: goal:string; steps:object[]") != std::string::npos);
    assert(compact_schema.find("steps: step[]") != std::string::npos);
    assert(compact_schema.find("step fields:") != std::string::npos);
    assert(compact_schema.find("tool?:string") != std::string::npos);
    assert(compact_schema.find("host assigns step IDs") != std::string::npos);
    assert(compact_schema.find("$previous.field") != std::string::npos);
    return 0;
}
