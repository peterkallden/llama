#include "agent/tool-catalog.h"
#include "agent/catalog/model-projection.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <set>

using json = nlohmann::ordered_json;

namespace {

common_tool_definition tool(
        const char * name, const char * description, const char * input, const char * result,
        const char * executor, common_tool_risk_class risk, bool confirmation = false,
        uint32_t timeout_ms = 1000, size_t max_result_bytes = 16384, const char * policy = "{}") {
    common_tool_definition definition;
    definition.name = name;
    definition.description = description;
    definition.input_schema_json = input;
    if (std::string(name) == "resource_inspect") {
        definition.model_input_schema_json = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string","minLength":1,"maxLength":64}}})";
    } else if (std::string(name) == "resource_read") {
        definition.model_input_schema_json = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string","minLength":1,"maxLength":64},"representation":{"type":"string","enum":["text","bytes"],"default":"text"},"offset":{"type":"integer","minimum":0,"maximum":1073741824},"max_bytes":{"type":"integer","minimum":1,"maximum":32768}}})";
    }
    definition.result_schema_json = result;
    definition.executor_id = executor;
    definition.risk_class = risk;
    definition.requires_confirmation = confirmation;
    definition.timeout_ms = timeout_ms;
    definition.max_result_bytes = max_result_bytes;
    definition.policy_json = policy;
    return definition;
}

std::vector<common_tool_definition> builtin_definitions() {
    const char * empty = R"({"type":"object","additionalProperties":false})";
    const char * object = R"({"type":"object"})";
    std::vector<common_tool_definition> definitions = {
        tool("calculator", "Evaluate a bounded arithmetic expression.", R"({"type":"object","additionalProperties":false,"required":["expression"],"properties":{"expression":{"type":"string","minLength":1,"maxLength":256}}})", object, "builtin.calculator", common_tool_risk_class::local_read),
        tool("time_now", "Return the current UTC time.", R"({"type":"object","additionalProperties":false,"properties":{"timezone":{"type":"string","enum":["UTC"]}}})", object, "builtin.time_now", common_tool_risk_class::local_read),
        tool("memory_search", "Search memories available to the current runtime scope.", R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string","minLength":1,"maxLength":1024},"limit":{"type":"integer","minimum":1,"maximum":8}}})", object, "builtin.memory_search", common_tool_risk_class::local_read),
        tool("memory_get", "Get a previously retrieved memory by its opaque id.", R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string","minLength":1,"maxLength":256}}})", object, "builtin.memory_get", common_tool_risk_class::local_read),
        tool("memory_inspect", "Inspect bounded memory statistics for the current scope.", empty, object, "builtin.memory_inspect", common_tool_risk_class::local_read),
        tool("memory_conflict_check", "Find potentially conflicting memories in the current scope.", R"({"type":"object","additionalProperties":false,"required":["content"],"properties":{"content":{"type":"string","minLength":1,"maxLength":2048}}})", object, "builtin.memory_conflict_check", common_tool_risk_class::local_read),
        tool("memory_remember", "Propose a policy-gated persistent memory.", R"({"type":"object","additionalProperties":false,"required":["kind","content"],"properties":{"kind":{"type":"string","enum":["fact","preference","procedure","constraint","decision","goal","observation","reflection","episode"]},"content":{"type":"string","minLength":1,"maxLength":512},"importance":{"type":"number","minimum":0,"maximum":1},"confidence":{"type":"number","minimum":0,"maximum":1},"rationale":{"type":"string","minLength":1,"maxLength":240}}})", object, "builtin.memory_remember", common_tool_risk_class::memory_proposal, true),
        tool("memory_propose_update", "Propose a version-checked memory correction or merge.", R"({"type":"object","additionalProperties":false,"required":["id","expected_version","content","rationale"],"properties":{"id":{"type":"string","minLength":1,"maxLength":256},"expected_version":{"type":"integer","minimum":0},"content":{"type":"string","minLength":1,"maxLength":2048},"rationale":{"type":"string","minLength":1,"maxLength":512}}})", object, "builtin.memory_propose_update", common_tool_risk_class::memory_proposal, true),
        tool("memory_propose_forget", "Propose archival of a memory; never hard-delete it.", R"({"type":"object","additionalProperties":false,"required":["id","rationale"],"properties":{"id":{"type":"string","minLength":1,"maxLength":256},"rationale":{"type":"string","minLength":1,"maxLength":512}}})", object, "builtin.memory_propose_forget", common_tool_risk_class::memory_proposal, true),
        tool("memory_link", "Propose a typed relationship between two memories or a memory and a plan.", R"({"type":"object","additionalProperties":false,"required":["from","relation","to","rationale"],"properties":{"from":{"type":"string","minLength":1,"maxLength":256},"relation":{"type":"string","minLength":1,"maxLength":64},"to":{"type":"string","minLength":1,"maxLength":256},"rationale":{"type":"string","minLength":1,"maxLength":512}}})", object, "builtin.memory_link", common_tool_risk_class::memory_proposal, true),
        tool("memory_compact_propose", "Propose consolidation of supplied source memories into one summary.", R"({"type":"object","additionalProperties":false,"required":["source_ids","content","rationale"],"properties":{"source_ids":{"type":"array","minItems":2,"maxItems":8,"items":{"type":"string","minLength":1,"maxLength":256}},"content":{"type":"string","minLength":1,"maxLength":2048},"rationale":{"type":"string","minLength":1,"maxLength":512}}})", object, "builtin.memory_compact_propose", common_tool_risk_class::memory_proposal, true),
        tool("plan_get", "Return the plan bound to the current runtime turn.", R"({"type":"object","additionalProperties":false,"properties":{"include_completed":{"type":"boolean"},"include_history":{"type":"boolean"}}})", object, "builtin.plan_get", common_tool_risk_class::local_read),
        tool("plan_propose", "Propose version-checked plan operations for native policy evaluation.", R"({"type":"object","additionalProperties":false,"required":["expected_version","operations"],"properties":{"expected_version":{"type":"integer","minimum":0},"operations":{"type":"array","minItems":1,"maxItems":8,"items":{"type":"object"}}}})", object, "builtin.plan_propose", common_tool_risk_class::plan_proposal, true),
        tool("repository.list", "List a bounded directory tree inside the runtime repository root.", R"({"type":"object","additionalProperties":false,"properties":{"path":{"type":"string","maxLength":512},"depth":{"type":"integer","minimum":0,"maximum":3}}})", object, "builtin.repository.list", common_tool_risk_class::local_read),
        tool("repository.search", "Search bounded text files inside the runtime repository root.", R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string","minLength":1,"maxLength":256},"path":{"type":"string","maxLength":512},"max_results":{"type":"integer","minimum":1,"maximum":32}}})", object, "builtin.repository.search", common_tool_risk_class::local_read),
        tool("repository.read", "Read a bounded line range from a text file inside the runtime repository root.", R"({"type":"object","additionalProperties":false,"required":["path"],"properties":{"path":{"type":"string","minLength":1,"maxLength":512},"start_line":{"type":"integer","minimum":1,"maximum":1000000},"end_line":{"type":"integer","minimum":1,"maximum":1000000}}})", object, "builtin.repository.read", common_tool_risk_class::local_read),
        tool("repository.diff", "Return a bounded read-only Git working-tree diff summary.", R"({"type":"object","additionalProperties":false})", object, "builtin.repository.diff", common_tool_risk_class::local_read),
        tool("repository.log", "Return a bounded read-only Git commit log.", R"({"type":"object","additionalProperties":false,"properties":{"limit":{"type":"integer","minimum":1,"maximum":20}}})", object, "builtin.repository.log", common_tool_risk_class::local_read),
        tool("repository.status", "Return the bounded Git working-tree status.", R"({"type":"object","additionalProperties":false})", object, "builtin.repository.status", common_tool_risk_class::local_read),
        tool("repository.changed_files", "List bounded Git working-tree changes.", R"({"type":"object","additionalProperties":false})", object, "builtin.repository.changed_files", common_tool_risk_class::local_read),
        tool("workspace.list", "List a bounded directory tree inside the host-owned workspace.", R"({"type":"object","additionalProperties":false,"properties":{"path":{"type":"string","maxLength":512},"depth":{"type":"integer","minimum":0,"maximum":3}}})", object, "builtin.workspace.list", common_tool_risk_class::local_read),
        tool("workspace.read", "Read a bounded line range from a workspace text file.", R"({"type":"object","additionalProperties":false,"required":["path"],"properties":{"path":{"type":"string","minLength":1,"maxLength":512},"start_line":{"type":"integer","minimum":1,"maximum":1000000},"end_line":{"type":"integer","minimum":1,"maximum":1000000}}})", object, "builtin.workspace.read", common_tool_risk_class::local_read),
        tool("workspace.search", "Search bounded text files inside the host-owned workspace.", R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string","minLength":1,"maxLength":256},"path":{"type":"string","maxLength":512},"max_results":{"type":"integer","minimum":1,"maximum":32}}})", object, "builtin.workspace.search", common_tool_risk_class::local_read),
        tool("workspace.patch", "Apply bounded, hash-checked line operations to a host-owned workspace file.", R"({"type":"object","additionalProperties":false,"required":["path","operations"],"properties":{"path":{"type":"string","minLength":1,"maxLength":512},"expected_hash":{"type":"string","maxLength":128},"operations":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object"}}}})", object, "builtin.workspace.patch", common_tool_risk_class::sandbox_execution, true, 5000, 65536),
        tool("diagnostics.compile", "Parse bounded compiler output into normalized diagnostics.", R"({"type":"object","additionalProperties":false,"required":["output"],"properties":{"output":{"type":"string","minLength":1,"maxLength":65536}}})", object, "builtin.diagnostics.compile", common_tool_risk_class::local_read, false, 5000, 65536),
        tool("diagnostics.symbol", "Find bounded symbol definitions using the host diagnostics provider or a text fallback.", R"({"type":"object","additionalProperties":false,"required":["symbol"],"properties":{"symbol":{"type":"string","minLength":1,"maxLength":256},"path_hint":{"type":"string","maxLength":512},"max_results":{"type":"integer","minimum":1,"maximum":64}}})", object, "builtin.diagnostics.symbol", common_tool_risk_class::local_read, false, 10000, 65536),
        tool("diagnostics.references", "Find bounded symbol references using the host diagnostics provider or a text fallback.", R"({"type":"object","additionalProperties":false,"required":["symbol"],"properties":{"symbol":{"type":"string","minLength":1,"maxLength":256},"definition_path":{"type":"string","maxLength":512},"definition_line":{"type":"integer","minimum":1,"maximum":1000000},"definition_column":{"type":"integer","minimum":1,"maximum":1000000},"max_results":{"type":"integer","minimum":1,"maximum":128}}})", object, "builtin.diagnostics.references", common_tool_risk_class::local_read, false, 10000, 65536),
        tool("diagnostics.call_hierarchy", "Find bounded callers and callees through the host semantic diagnostics provider.", R"({"type":"object","additionalProperties":false,"required":["symbol"],"properties":{"symbol":{"type":"string","minLength":1,"maxLength":256},"path_hint":{"type":"string","maxLength":512},"line":{"type":"integer","minimum":1,"maximum":1000000},"column":{"type":"integer","minimum":1,"maximum":1000000},"definition_path":{"type":"string","maxLength":512},"definition_line":{"type":"integer","minimum":1,"maximum":1000000},"definition_column":{"type":"integer","minimum":1,"maximum":1000000},"direction":{"type":"string","enum":["callers","callees","both"]},"max_depth":{"type":"integer","minimum":1,"maximum":8},"max_results":{"type":"integer","minimum":1,"maximum":128}}})", object, "builtin.diagnostics.call_hierarchy", common_tool_risk_class::local_read, false, 10000, 65536),
        tool("diagnostics.test_failures", "Group and summarize bounded test failures from a test result.", R"({"type":"object","additionalProperties":false,"required":["result"],"properties":{"result":{"type":"string","minLength":1,"maxLength":65536}}})", object, "builtin.diagnostics.test_failures", common_tool_risk_class::local_read, false, 5000, 65536),
        tool("diagnostics.format", "Check bounded formatter output and return a patch proposal when present.", R"({"type":"object","additionalProperties":false,"required":["output"],"properties":{"output":{"type":"string","minLength":1,"maxLength":65536}}})", object, "builtin.diagnostics.format", common_tool_risk_class::local_read, false, 5000, 65536),
        tool("diagnostics.include_graph", "Build a bounded include graph from compiler-style dependency output.", R"({"type":"object","additionalProperties":false,"required":["output"],"properties":{"output":{"type":"string","minLength":1,"maxLength":65536}}})", object, "builtin.diagnostics.include_graph", common_tool_risk_class::local_read, false, 5000, 65536),
        tool("dataset.list", "List bounded CSV, JSON and Parquet files under the host-owned workspace.", R"({"type":"object","additionalProperties":false,"properties":{"path":{"type":"string","maxLength":512},"max_results":{"type":"integer","minimum":1,"maximum":256}}})", object, "builtin.dataset.list", common_tool_risk_class::local_read),
        tool("dataset.inspect", "Inspect bounded metadata for a dataset reference, an acquired resource, or legacy file.", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"},"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"path":{"type":"string","minLength":1,"maxLength":512}}})", R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref"},"resource":{"type":"string","x-agent-type":"resource_ref"},"path":{"type":"string"},"format":{"type":"string"},"size_bytes":{"type":"integer"},"name":{"type":"string"},"rows":{"type":"integer"},"columns":{"type":"array","items":{"type":"object"}}}})", "builtin.dataset.inspect", common_tool_risk_class::local_read),
        tool("dataset.schema", "Return a bounded typed schema for a dataset reference, an acquired resource, or legacy CSV file.", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"},"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"path":{"type":"string","minLength":1,"maxLength":512},"max_rows":{"type":"integer","minimum":1,"maximum":10000}}})", R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref"},"columns":{"type":"array","items":{"type":"object"}}}})", "builtin.dataset.schema", common_tool_risk_class::local_read),
        tool("dataset.sample", "Return a bounded sample from a dataset reference, an acquired resource, or legacy CSV file.", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"},"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"path":{"type":"string","minLength":1,"maxLength":512},"rows":{"type":"integer","minimum":1,"maximum":100}}})", R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref"},"columns":{"type":"array","items":{"type":"string"}},"rows":{"type":"array","items":{"type":"object"}},"scan_truncated":{"type":"boolean"},"result_truncated":{"type":"boolean"}}})", "builtin.dataset.sample", common_tool_risk_class::local_read),
        tool("document.tables", "List bounded tables exposed by a host-owned document representation. Names are model-friendly aliases; node IDs and indexes are stable host addresses.", R"({"type":"object","additionalProperties":false,"required":["resource"],"properties":{"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"max_results":{"type":"integer","minimum":1,"maximum":64}}})", R"({"type":"object","additionalProperties":false,"required":["resource","tables","truncated"],"properties":{"resource":{"type":"string","x-agent-type":"resource_ref"},"tables":{"type":"array","maxItems":64,"items":{"type":"object","additionalProperties":false,"required":["index","name","node_id"],"properties":{"index":{"type":"integer","minimum":0},"name":{"type":"string","maxLength":256},"caption":{"type":"string","maxLength":512},"node_id":{"type":"string","x-agent-type":"table_ref","maxLength":512}}}},"truncated":{"type":"boolean"}}})", "builtin.document.tables", common_tool_risk_class::local_read),
        tool("document.table", "Resolve one document table by a unique name, index or node ID and return its bounded dataset handle.", R"({"type":"object","additionalProperties":false,"required":["resource"],"properties":{"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"table":{"type":"string","minLength":1,"maxLength":256,"description":"Unique table name or caption."},"table_index":{"type":"integer","minimum":0,"maximum":1000000},"node_id":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"table_ref"}}})", R"({"type":"object","additionalProperties":false,"required":["table_index","name","node_id","dataset","source_resource"],"properties":{"table_index":{"type":"integer","minimum":0},"name":{"type":"string","maxLength":256},"node_id":{"type":"string","x-agent-type":"table_ref","maxLength":512},"dataset":{"type":"string","x-agent-type":"dataset_ref","minLength":1,"maxLength":512},"source_resource":{"type":"string","x-agent-type":"resource_ref","minLength":1,"maxLength":512}}})", "builtin.document.table", common_tool_risk_class::local_read),
        tool("dataset.validate", "Validate a host-approved dataset against bounded declarative rules.", R"({"type":"object","additionalProperties":false,"required":["dataset","rules"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref","x-agent-inferable":true},"rules":{"type":"array","minItems":1,"maxItems":32,"items":{"type":"object"}}}})", R"({"type":"object","required":["valid","violations"],"properties":{"valid":{"type":"boolean"},"violations":{"type":"array","items":{"type":"object"}}}})", "builtin.dataset.validate", common_tool_risk_class::local_read, false, 30000, 65536),
        tool("data.query", "Run a bounded backend-neutral structured data query. where accepts canonical predicates, a bounded model-friendly object form, or a simple AND expression.", R"({"type":"object","additionalProperties":false,"required":["dataset"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref","x-agent-inferable":true},"select":{"type":"array","maxItems":32,"items":{"type":"string","maxLength":128}},"where":{"maxLength":2048,"maxItems":16,"items":{"type":"object"}},"order_by":{"type":"array","maxItems":8,"items":{"type":"object"}},"distinct":{"type":"boolean"},"offset":{"type":"integer","minimum":0,"maximum":100000},"limit":{"type":"integer","minimum":1,"maximum":1000},"max_scan_rows":{"type":"integer","minimum":1,"maximum":100000},"max_result_rows":{"type":"integer","minimum":1,"maximum":10000},"materialize":{"type":"boolean"},"result_dataset":{"type":"string","minLength":1,"maxLength":512}}})", R"({"type":"object","properties":{"rows":{"type":"array","items":{"type":"object"}},"dataset":{"type":"string","x-agent-type":"dataset_ref"},"scan_truncated":{"type":"boolean"},"result_truncated":{"type":"boolean"}}})", "builtin.data.query", common_tool_risk_class::local_read, false, 30000, 65536),
        tool("data.filter", "Filter a bounded dataset with declarative predicates. conditions accepts canonical predicates, a bounded model-friendly object form, or a simple AND expression.", R"({"type":"object","additionalProperties":false,"required":["dataset","conditions"],"properties":{"dataset":{"type":"string","maxLength":2048,"maxItems":16,"items":{"type":"object"},"x-agent-type":"dataset_ref","x-agent-inferable":true},"conditions":{"maxLength":2048,"maxItems":16,"items":{"type":"object"}},"limit":{"type":"integer","minimum":1,"maximum":1000},"max_scan_rows":{"type":"integer","minimum":1,"maximum":100000},"max_result_rows":{"type":"integer","minimum":1,"maximum":10000},"materialize":{"type":"boolean"},"result_dataset":{"type":"string","minLength":1,"maxLength":512}}})", R"({"type":"object","properties":{"rows":{"type":"array","items":{"type":"object"}},"dataset":{"type":"string","x-agent-type":"dataset_ref"},"scan_truncated":{"type":"boolean"},"result_truncated":{"type":"boolean"}}})", "builtin.data.filter", common_tool_risk_class::local_read, false, 30000, 65536),
        tool("data.aggregate", "Aggregate a bounded dataset with host-approved functions.", R"({"type":"object","additionalProperties":false,"required":["dataset","measures"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref","x-agent-inferable":true},"group_by":{"type":"array","maxItems":16,"items":{"type":"string","maxLength":128}},"measures":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object","additionalProperties":false,"required":["function"],"properties":{"function":{"type":"string","enum":["count","sum","avg","min","max"]},"column":{"type":"string","minLength":1,"maxLength":128},"as":{"type":"string","minLength":1,"maxLength":128}}}},"max_scan_rows":{"type":"integer","minimum":1,"maximum":100000},"max_result_rows":{"type":"integer","minimum":1,"maximum":10000},"materialize":{"type":"boolean"},"result_dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"}}})", R"({"type":"object","additionalProperties":false,"properties":{"rows":{"type":"array","maxItems":10000,"items":{"type":"object"}},"dataset":{"type":"string","x-agent-type":"dataset_ref","maxLength":512},"materialized":{"type":"boolean"},"scan_truncated":{"type":"boolean"},"result_truncated":{"type":"boolean"}}})", "builtin.data.aggregate", common_tool_risk_class::local_read, false, 30000, 65536),
        tool("data.join", "Join two bounded datasets using declarative key mappings.", R"({"type":"object","additionalProperties":false,"required":["left","right","on"],"properties":{"left":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"right":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"type":{"type":"string","enum":["inner","left"]},"on":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object"}},"max_scan_rows":{"type":"integer","minimum":1,"maximum":100000},"max_result_rows":{"type":"integer","minimum":1,"maximum":10000},"materialize":{"type":"boolean"},"result_dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"}}})", R"({"type":"object","additionalProperties":false,"properties":{"rows":{"type":"array","maxItems":10000,"items":{"type":"object"}},"dataset":{"type":"string","x-agent-type":"dataset_ref","maxLength":512},"materialized":{"type":"boolean"},"scan_truncated":{"type":"boolean"},"result_truncated":{"type":"boolean"}}})", "builtin.data.join", common_tool_risk_class::local_read, false, 30000, 65536),
        tool("data.transform", "Transform a bounded dataset with host-approved column operations.", R"({"type":"object","additionalProperties":false,"required":["dataset","operations"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref","x-agent-inferable":true},"operations":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object"}},"materialize":{"type":"boolean"},"result_dataset":{"type":"string","minLength":1,"maxLength":512}}})", R"({"type":"object","properties":{"rows":{"type":"array","items":{"type":"object"}},"dataset":{"type":"string","x-agent-type":"dataset_ref"},"scan_truncated":{"type":"boolean"},"result_truncated":{"type":"boolean"}}})", "builtin.data.transform", common_tool_risk_class::local_read, false, 30000, 65536),
        tool("statistics.describe", "Describe selected or schema-declared numeric columns in a bounded dataset. Returns count, null_count, min, max, mean and population stddev; optional group_by returns one bounded description per group.", R"({"type":"object","additionalProperties":false,"required":["dataset"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref","x-agent-inferable":true},"columns":{"type":"array","maxItems":32,"items":{"type":"string","maxLength":128}},"group_by":{"type":"array","maxItems":16,"items":{"type":"string","maxLength":128}},"max_scan_rows":{"type":"integer","minimum":1,"maximum":100000}}})", R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref"},"rows":{"type":"array","items":{"type":"object"}},"scan_truncated":{"type":"boolean"}}})", "builtin.statistics.describe", common_tool_risk_class::local_read, false, 30000, 65536),
        tool("statistics.outliers", "Identify bounded numeric outliers in a dataset using the host-owned IQR method. Optional group_by computes thresholds independently per group.", R"({"type":"object","additionalProperties":false,"required":["dataset"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref","x-agent-inferable":true},"columns":{"type":"array","maxItems":32,"items":{"type":"string","maxLength":128}},"column":{"type":"string","minLength":1,"maxLength":128},"group_by":{"type":"array","maxItems":16,"items":{"type":"string","maxLength":128}},"method":{"type":"string","enum":["iqr"]},"multiplier":{"type":"number","minimum":0.1,"maximum":10.0},"max_scan_rows":{"type":"integer","minimum":1,"maximum":100000}}})", R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref"},"rows":{"type":"array","items":{"type":"object"}},"scan_truncated":{"type":"boolean"},"result_truncated":{"type":"boolean"}}})", "builtin.statistics.outliers", common_tool_risk_class::local_read, false, 30000, 65536),
        tool("statistics.value_counts", "Return bounded value frequencies for one dataset column, including nulls.", R"({"type":"object","additionalProperties":false,"required":["dataset","column"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref","x-agent-inferable":true},"column":{"type":"string","minLength":1,"maxLength":128},"limit":{"type":"integer","minimum":1,"maximum":1000},"max_scan_rows":{"type":"integer","minimum":1,"maximum":100000}}})", R"({"type":"object","additionalProperties":false,"required":["column","values","distinct_count","null_count"],"properties":{"column":{"type":"string"},"values":{"type":"array","maxItems":1000,"items":{"type":"object"}},"distinct_count":{"type":"integer"},"null_count":{"type":"integer"},"scanned_rows":{"type":"integer"},"scan_truncated":{"type":"boolean"},"result_truncated":{"type":"boolean"}}})", "builtin.statistics.value_counts", common_tool_risk_class::local_read, false, 30000, 65536),
        tool("artifact.export", "Publish bounded text content or export a bounded dataset as CSV into a host-owned resource artifact.", R"({"type":"object","additionalProperties":false,"properties":{"name":{"type":"string","minLength":1,"maxLength":256},"content":{"type":"string","maxLength":65536},"mime_type":{"type":"string","maxLength":128},"source_dataset":{"type":"string","minLength":1,"maxLength":512},"format":{"type":"string","enum":["csv"]},"max_rows":{"type":"integer","minimum":1,"maximum":10000}},"anyOf":[{"required":["name","content"]},{"required":["source_dataset"]}]})", object, "builtin.artifact.export", common_tool_risk_class::local_read, false, 30000, 8192),
        tool("development.build", "Build a named host-approved target through the sandbox execution runtime.", R"({"type":"object","additionalProperties":false,"required":["target"],"properties":{"target":{"type":"string","minLength":1,"maxLength":256},"configuration":{"type":"string","enum":["Debug","Release","RelWithDebInfo","MinSizeRel"]},"resource_refs":{"type":"array","maxItems":32,"items":{"type":"string","minLength":1,"maxLength":1024}}}})", R"({"type":"object","additionalProperties":false,"required":["status"],"properties":{"status":{"type":"string","enum":["queued","running","completed","failed","cancelled","timed_out"]},"exit_code":{"type":"integer"},"summary":{"type":"string"},"artifacts":{"type":"array"}}})", "sandbox.development.build", common_tool_risk_class::sandbox_execution, true, 120000, 65536, R"({"execution":"sandbox","execution_class":"developer-build","network":"none","filesystem":"workspace-write","host_owned":true})"),
        tool("development.test", "Run a named host-approved test selection through the sandbox execution runtime.", R"({"type":"object","additionalProperties":false,"required":["target"],"properties":{"target":{"type":"string","minLength":1,"maxLength":256},"configuration":{"type":"string","enum":["Debug","Release","RelWithDebInfo","MinSizeRel"]},"filter":{"type":"string","maxLength":256},"timeout_ms":{"type":"integer","minimum":1000,"maximum":1800000},"resource_refs":{"type":"array","maxItems":32,"items":{"type":"string","minLength":1,"maxLength":1024}}}})", R"({"type":"object","additionalProperties":false,"required":["status"],"properties":{"status":{"type":"string","enum":["queued","running","completed","failed","cancelled","timed_out"]},"exit_code":{"type":"integer"},"summary":{"type":"string"},"artifacts":{"type":"array"}}})", "sandbox.development.test", common_tool_risk_class::sandbox_execution, true, 120000, 65536, R"({"execution":"sandbox","execution_class":"developer-build","network":"none","filesystem":"workspace-write","host_owned":true})"),
        tool("resource_inspect", "Inspect a bounded host-owned resource descriptor and available representations by opaque URI.", R"({"type":"object","additionalProperties":false,"required":["uri"],"properties":{"uri":{"type":"string","minLength":1,"maxLength":512}}})", object, "builtin.resource_inspect", common_tool_risk_class::local_read),
        tool("resource_read", "Read a bounded host-owned text or base64 byte representation by opaque URI. If representation is omitted, text is preferred.", R"({"type":"object","additionalProperties":false,"required":["uri"],"properties":{"uri":{"type":"string","minLength":1,"maxLength":512},"representation":{"type":"string","minLength":1,"maxLength":64,"default":"text","description":"Requested representation. Defaults to text when omitted."},"offset":{"type":"integer","minimum":0,"maximum":1073741824},"max_bytes":{"type":"integer","minimum":1,"maximum":32768}}})", object, "builtin.resource_read", common_tool_risk_class::local_read),
        tool("web_search", "Search the public web through a bounded HTTPS provider and return result candidates.", R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string","minLength":1,"maxLength":512},"limit":{"type":"integer","minimum":1,"maximum":8},"site":{"type":"string","minLength":1,"maxLength":256}}})", object, "builtin.web_search", common_tool_risk_class::network_read, false, 10000, 65536, R"({"https_only":true,"provider":"duckduckgo-lite","block_private_networks":true})"),
        tool("web_fetch", "Fetch a public HTTPS URL through the native safe HTTP client.", R"({"type":"object","additionalProperties":false,"required":["url"],"properties":{"url":{"type":"string","minLength":9,"maxLength":2048},"max_bytes":{"type":"integer","minimum":1,"maximum":500000},"extract":{"type":"string","enum":["text"]}}})", object, "builtin.web_fetch", common_tool_risk_class::network_read, false, 10000, 65536, R"({"https_only":true,"max_redirects":3,"block_private_networks":true})"),
    };

    const auto set_model_schema = [&](const char * name, const char * schema) {
        for (auto & definition : definitions) {
            if (definition.name == name) {
                definition.model_input_schema_json = schema;
                return;
            }
        }
    };
    const auto set_model_result_schema = [&](const char * name, const char * schema) {
        for (auto & definition : definitions) {
            if (definition.name == name) {
                definition.model_result_schema_json = schema;
                return;
            }
        }
    };
    set_model_schema("document.tables", R"({"type":"object","additionalProperties":false,"required":["resource"],"properties":{"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"max_results":{"type":"integer","minimum":1,"maximum":64}}})");
    // The model-facing view selects a table by its semantic name. Stable
    // node IDs and numeric indexes remain available only to advanced host
    // callers through the full execution schema.
    set_model_schema("document.table", R"({"type":"object","additionalProperties":false,"required":["resource","table"],"properties":{"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"table":{"type":"string","minLength":1,"maxLength":256}}})");
    set_model_schema("dataset.inspect", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"},"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"path":{"type":"string","minLength":1,"maxLength":512}}})");
    set_model_schema("dataset.schema", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"},"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"path":{"type":"string","minLength":1,"maxLength":512},"max_rows":{"type":"integer","minimum":1,"maximum":10000}}})");
    set_model_schema("dataset.sample", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"},"resource":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"resource_ref"},"path":{"type":"string","minLength":1,"maxLength":512},"rows":{"type":"integer","minimum":1,"maximum":100}}})");
    set_model_schema("dataset.validate", R"({"type":"object","additionalProperties":false,"required":["rules"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"},"rules":{"type":"array","minItems":1,"maxItems":32,"items":{"type":"object"}}}})");
    set_model_schema("data.query", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"select":{"type":"array","maxItems":32,"items":{"type":"string","maxLength":128}},"where":{"type":"object"},"order_by":{"type":"array","maxItems":8,"items":{"type":"object"}},"distinct":{"type":"boolean"},"limit":{"type":"integer","minimum":1,"maximum":1000}}})");
    set_model_schema("data.filter", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","maxLength":2048,"x-agent-type":"dataset_ref"},"conditions":{"type":"object"},"limit":{"type":"integer","minimum":1,"maximum":1000}}})");
    set_model_schema("data.aggregate", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"measures":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object","additionalProperties":false,"required":["function"],"properties":{"function":{"type":"string","enum":["count","sum","avg","min","max"]},"column":{"type":"string","minLength":1,"maxLength":128},"as":{"type":"string","minLength":1,"maxLength":128}}}},"group_by":{"type":"array","maxItems":16,"items":{"type":"string","maxLength":128}}}})");
    set_model_schema("data.join", R"({"type":"object","additionalProperties":false,"required":["left","right","on"],"properties":{"left":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"right":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"type":{"type":"string","enum":["inner","left"]},"on":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object"}}}})");
    set_model_schema("data.transform", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"operations":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object"}}}})");
    set_model_schema("statistics.describe", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"columns":{"type":"array","maxItems":32,"items":{"type":"string","maxLength":128}},"group_by":{"type":"array","maxItems":16,"items":{"type":"string","maxLength":128}}}})");
    set_model_schema("statistics.outliers", R"({"type":"object","additionalProperties":false,"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"columns":{"type":"array","maxItems":32,"items":{"type":"string","maxLength":128}},"column":{"type":"string","minLength":1,"maxLength":128},"group_by":{"type":"array","maxItems":16,"items":{"type":"string","maxLength":128}},"method":{"type":"string","enum":["iqr"]},"multiplier":{"type":"number","minimum":0.1,"maximum":10.0}}})");
    set_model_schema("statistics.value_counts", R"({"type":"object","additionalProperties":false,"required":["column"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"column":{"type":"string","minLength":1,"maxLength":128},"limit":{"type":"integer","minimum":1,"maximum":1000}}})");
    set_model_schema("artifact.export", R"({"type":"object","additionalProperties":false,"properties":{"name":{"type":"string","minLength":1,"maxLength":256},"content":{"type":"string","maxLength":65536},"mime_type":{"type":"string","maxLength":128},"source_dataset":{"type":"string","minLength":1,"maxLength":512,"x-agent-type":"dataset_ref"},"format":{"type":"string","enum":["csv"]}},"anyOf":[{"required":["name","content"]},{"required":["source_dataset"]}]})");
    // Most tools need no hand-written model input schema. Generate a
    // conservative projection from the host contract and keep explicit
    // schemas above only for tools with genuinely different model semantics.
    for (auto & definition : definitions) {
        if (definition.model_input_schema_json.empty()) {
            definition.model_input_schema_json = common_tool_default_model_input_projection(definition);
        }
    }
    // Derive model-facing inference hints from field capabilities in the full
    // host contract. The host still decides at runtime whether exactly one
    // compatible source exists. Alternative schemas remain explicit until
    // their binding semantics are modeled more precisely.
    for (auto & definition : definitions) {
        if (definition.model_input_schema_json.empty()) continue;
        auto schema = json::parse(definition.model_input_schema_json, nullptr, false);
        const auto host_schema = json::parse(definition.input_schema_json, nullptr, false);
        if (!schema.is_object() || !host_schema.is_object() ||
                schema.contains("anyOf") || schema.contains("oneOf")) continue;
        auto properties = schema.value("properties", json::object());
        const auto host_properties = host_schema.value("properties", json::object());
        if (!properties.is_object() || !host_properties.is_object()) continue;
        json inferable = json::array();
        for (auto it = properties.begin(); it != properties.end(); ++it) {
            const auto host_it = host_properties.find(it.key());
            if (!it.value().is_object() || host_it == host_properties.end() ||
                    !host_it.value().is_object()) continue;
            if (host_it.value().value("x-agent-inferable", false)) {
                inferable.push_back(it.key());
                it.value()["x-agent-inferable"] = true;
            }
        }
        if (!inferable.empty()) schema["x-agent-autowire-fields"] = std::move(inferable);
        definition.model_input_schema_json = schema.dump();
    }
    // Generate a conservative result projection when a tool has not supplied
    // an explicit one. Explicit model_result_schema_json remains the escape
    // hatch for tools whose evidence semantics need a custom view.
    for (auto & definition : definitions) {
        if (definition.model_result_schema_json.empty()) {
            definition.model_result_schema_json = common_tool_default_model_result_projection(definition);
        }
    }
    // Keep execution/result schemas authoritative while presenting only
    // useful chaining and evidence fields to the model.
    set_model_result_schema("document.tables", R"({"type":"object","properties":{"resource":{"type":"string","x-agent-type":"resource_ref"},"tables":{"type":"array","items":{"type":"object"}}}})");
    set_model_result_schema("document.table", R"({"type":"object","properties":{"name":{"type":"string"},"dataset":{"type":"string","x-agent-type":"dataset_ref"},"source_resource":{"type":"string","x-agent-type":"resource_ref"}}})");
    set_model_result_schema("dataset.inspect", R"({"type":"object","properties":{"dataset":{"type":"string","x-agent-type":"dataset_ref"},"resource":{"type":"string","x-agent-type":"resource_ref"},"name":{"type":"string"},"rows":{"type":"integer"},"columns":{"type":"array","items":{"type":"object"}}}})");
    set_model_result_schema("dataset.schema", R"({"type":"object","properties":{"columns":{"type":"array","items":{"type":"object"}}}})");
    set_model_result_schema("dataset.sample", R"({"type":"object","properties":{"columns":{"type":"array","items":{"type":"string"}},"rows":{"type":"array","items":{"type":"object"}}}})");
    set_model_result_schema("dataset.validate", R"({"type":"object","properties":{"valid":{"type":"boolean"},"violations":{"type":"array","items":{"type":"object"}}}})");
    return definitions;
}

common_tool_profile profile(const char * id, const char * description, std::initializer_list<const char *> tools) {
    common_tool_profile value;
    value.id = id;
    value.description = description;
    for (const auto * name : tools) value.members.push_back({name, 1, true, "{}"});
    return value;
}

common_tool_profile policy_profile(
        common_tool_profile value,
        bool allow_network,
        bool allow_policy_gated_writes) {
    value.allow_network = allow_network;
    value.allow_policy_gated_writes = allow_policy_gated_writes;
    return value;
}

std::vector<common_tool_profile> builtin_profiles() {
    return {
        policy_profile(profile("minimal", "Local deterministic utility tools.", {"calculator", "time_now"}), false, false),
        policy_profile(profile("memory-read", "Read-only scoped memory, plan and resource inspection.", {"calculator", "time_now", "memory_search", "memory_get", "memory_inspect", "memory_conflict_check", "plan_get", "resource_inspect", "resource_read"}), false, false),
        policy_profile(profile("memory", "Memory inspection plus policy-gated memory, plan and resource proposals.", {"calculator", "time_now", "memory_search", "memory_get", "memory_inspect", "memory_conflict_check", "memory_remember", "memory_propose_update", "memory_propose_forget", "memory_link", "memory_compact_propose", "plan_get", "plan_propose", "resource_inspect", "resource_read"}), false, true),
        policy_profile(profile("analysis", "Bounded deliberate inspection of resources, documents, datasets and statistics.", {"calculator", "time_now", "memory_search", "memory_get", "memory_inspect", "memory_conflict_check", "plan_get", "repository.search", "repository.read", "repository.list", "resource_inspect", "resource_read", "document.tables", "document.table", "dataset.inspect", "dataset.schema", "dataset.sample", "dataset.validate", "data.query", "data.filter", "data.aggregate", "data.join", "data.transform", "statistics.describe", "statistics.outliers", "statistics.value_counts", "web_search", "web_fetch"}), true, false),
        policy_profile(profile("research", "Memory profile with host-owned acquisition plus bounded resource, document, dataset and statistics inspection.", {"calculator", "time_now", "memory_search", "memory_get", "memory_inspect", "memory_conflict_check", "memory_remember", "plan_get", "plan_propose", "repository.search", "repository.read", "repository.list", "repository.diff", "resource_inspect", "resource_read", "document.tables", "document.table", "dataset.inspect", "dataset.schema", "dataset.sample", "dataset.validate", "data.query", "data.filter", "data.aggregate", "data.join", "data.transform", "statistics.describe", "statistics.outliers", "statistics.value_counts", "web_search", "web_fetch"}), true, true),
        policy_profile(profile("developer-read", "Read-only workspace, repository and resource inspection tools.", {"calculator", "time_now", "repository.list", "repository.search", "repository.read", "repository.diff", "repository.log", "repository.status", "repository.changed_files", "workspace.list", "workspace.read", "workspace.search", "diagnostics.symbol", "diagnostics.references", "diagnostics.call_hierarchy", "resource_inspect", "resource_read", "document.tables", "document.table"}), false, false),
        policy_profile(profile("all-configured", "All catalogued tools, still subject to host policy and confirmation.", {"calculator", "time_now", "memory_search", "memory_get", "memory_inspect", "memory_conflict_check", "memory_remember", "memory_propose_update", "memory_propose_forget", "memory_link", "memory_compact_propose", "plan_get", "plan_propose", "repository.list", "repository.search", "repository.read", "repository.diff", "repository.log", "repository.status", "repository.changed_files", "workspace.list", "workspace.read", "workspace.search", "workspace.patch", "development.build", "development.test", "diagnostics.compile", "diagnostics.symbol", "diagnostics.references", "diagnostics.call_hierarchy", "diagnostics.test_failures", "diagnostics.format", "diagnostics.include_graph", "dataset.list", "dataset.inspect", "dataset.schema", "dataset.sample", "dataset.validate", "data.query", "data.filter", "data.aggregate", "data.join", "data.transform", "statistics.describe", "statistics.outliers", "statistics.value_counts", "artifact.export", "resource_inspect", "resource_read", "document.tables", "document.table", "web_search", "web_fetch"}), true, true),
    };
}

bool validate(const common_tool_definition & definition, std::string & error) {
    if (definition.name.empty() || definition.executor_id.empty()) { error = "tool definition requires a name and native executor id"; return false; }
    const auto input = json::parse(definition.input_schema_json, nullptr, false);
    const auto model_input = definition.model_input_schema_json.empty()
        ? input
        : json::parse(definition.model_input_schema_json, nullptr, false);
    const auto result = json::parse(definition.result_schema_json, nullptr, false);
    const auto model_result = definition.model_result_schema_json.empty()
        ? result
        : json::parse(definition.model_result_schema_json, nullptr, false);
    const auto policy = json::parse(definition.policy_json, nullptr, false);
    if (!input.is_object() || input.value("type", std::string()) != "object" ||
            !model_input.is_object() || model_input.value("type", std::string()) != "object" ||
            !result.is_object() || !model_result.is_object() || !policy.is_object()) {
        error = "tool definition has invalid JSON schema or policy";
        return false;
    }
    return true;
}

} // namespace

std::string common_tool_model_input_schema(const common_tool_definition & definition) {
    return definition.model_input_schema_json.empty()
        ? definition.input_schema_json
        : definition.model_input_schema_json;
}

std::string common_tool_model_result_schema(const common_tool_definition & definition) {
    return definition.model_result_schema_json.empty()
        ? definition.result_schema_json
        : definition.model_result_schema_json;
}

const char * common_tool_risk_class_name(common_tool_risk_class value) {
    switch (value) {
        case common_tool_risk_class::local_read: return "local_read";
        case common_tool_risk_class::memory_proposal: return "memory_proposal";
        case common_tool_risk_class::plan_proposal: return "plan_proposal";
        case common_tool_risk_class::network_read: return "network_read";
        case common_tool_risk_class::sandbox_execution: return "sandbox_execution";
    }
    return "unknown";
}

bool common_tool_catalog::bootstrap(
        const std::string & profile_id,
        common_tool_bootstrap_result & result,
        std::string & error,
        const std::map<std::string, std::vector<std::string>> & configured_capabilities,
        const std::map<std::string, common_tool_profile> & configured_profiles) {
    result = {};
    const auto all_definitions = builtin_definitions();
    const auto all_profiles = builtin_profiles();
    const auto selected = profile_id.empty() ? "minimal" : profile_id;
    for (const auto & definition : all_definitions) {
        if (!validate(definition, error)) return false;
        const auto key = definition.name + "@" + std::to_string(definition.version);
        if (definitions.count(key)) result.definitions_unchanged.push_back(key);
        else { definitions.emplace(key, definition); result.definitions_created.push_back(key); }
    }
    for (const auto & profile : all_profiles) {
        if (profiles.count(profile.id)) result.profiles_unchanged.push_back(profile.id);
        else { profiles.emplace(profile.id, profile); result.profiles_created.push_back(profile.id); }
    }
    for (const auto & capability : configured_capabilities) {
        if (capability.first.empty()) {
            error = "configured tool capability id must not be empty";
            return false;
        }
        for (const auto & tool_name : capability.second) {
            if (tool_name.empty() || find_definition(tool_name) == nullptr) {
                error = "configured tool capability references an unavailable tool: " + tool_name;
                return false;
            }
        }
        capabilities[capability.first] = capability.second;
    }
    for (const auto & configured : configured_profiles) {
        common_tool_profile profile = configured.second;
        profile.id = configured.first;
        if (profile.id.empty()) {
            error = "configured tool profile id must not be empty";
            return false;
        }
        for (const auto & capability : profile.include_capabilities) {
            if (!capabilities.count(capability)) {
                error = "configured tool profile references an unavailable capability: " + capability;
                return false;
            }
        }
        for (const auto & capability : profile.exclude_capabilities) {
            if (!capabilities.count(capability)) {
                error = "configured tool profile references an unavailable capability: " + capability;
                return false;
            }
        }
        profiles[profile.id] = std::move(profile);
    }
    if (!find_profile(selected)) {
        error = "tool profile is unavailable: " + selected;
        return false;
    }
    error.clear();
    return true;
}

const common_tool_definition * common_tool_catalog::find_definition(const std::string & name, uint32_t version) const {
    const auto it = definitions.find(name + "@" + std::to_string(version));
    return it == definitions.end() ? nullptr : &it->second;
}

const common_tool_profile * common_tool_catalog::find_profile(const std::string & id) const {
    const auto it = profiles.find(id);
    return it == profiles.end() ? nullptr : &it->second;
}

std::vector<common_tool_definition> common_tool_catalog::load_profile(const std::string & id, std::string & error) const {
    common_tool_profile_snapshot snapshot;
    if (!resolve_profile(id, snapshot, error)) return {};
    return snapshot.tools;
}

bool common_tool_catalog::resolve_profile(
        const std::string & id,
        common_tool_profile_snapshot & snapshot,
        std::string & error) const {
    snapshot = {};
    const auto * profile = find_profile(id);
    if (!profile || !profile->enabled) { error = "tool profile is unavailable"; return false; }
    snapshot.id = profile->id;
    snapshot.allow_network = profile->allow_network;
    snapshot.allow_policy_gated_writes = profile->allow_policy_gated_writes;
    std::vector<common_tool_definition> loaded;
    for (const auto & member : profile->members) {
        if (!member.enabled) continue;
        const auto * definition = find_definition(member.tool_name, member.tool_version);
        if (!definition || !definition->enabled) { error = "tool profile references an unavailable tool"; return false; }
        loaded.push_back(*definition);
    }
    std::set<std::string> loaded_names;
    for (const auto & definition : loaded) loaded_names.insert(definition.name);
    for (const auto & capability_id : profile->include_capabilities) {
        const auto capability = capabilities.find(capability_id);
        if (capability == capabilities.end()) {
            error = "tool profile references an unavailable capability";
            return false;
        }
        for (const auto & tool_name : capability->second) {
            const auto * definition = find_definition(tool_name);
            if (!definition || !definition->enabled) {
                error = "tool capability references an unavailable tool";
                return false;
            }
            if (loaded_names.insert(definition->name).second) loaded.push_back(*definition);
        }
    }
    if (!profile->exclude_capabilities.empty()) {
        std::set<std::string> excluded_names;
        for (const auto & capability_id : profile->exclude_capabilities) {
            const auto capability = capabilities.find(capability_id);
            if (capability == capabilities.end()) {
                error = "tool profile references an unavailable excluded capability";
                return false;
            }
            excluded_names.insert(capability->second.begin(), capability->second.end());
        }
        loaded.erase(std::remove_if(loaded.begin(), loaded.end(), [&excluded_names](const common_tool_definition & definition) {
            return excluded_names.count(definition.name) != 0;
        }), loaded.end());
    }
    error.clear();
    snapshot.tools = std::move(loaded);
    return true;
}

bool resolve_common_tool_profile_snapshot(
        const std::string & profile_id,
        const std::map<std::string, std::vector<std::string>> & configured_capabilities,
        const std::map<std::string, common_tool_profile> & configured_profiles,
        common_tool_profile_snapshot & snapshot,
        std::string & error) {
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap(
            profile_id,
            bootstrap,
            error,
            configured_capabilities,
            configured_profiles)) {
        return false;
    }
    return catalog.resolve_profile(
        profile_id.empty() ? std::string("minimal") : profile_id,
        snapshot,
        error);
}
