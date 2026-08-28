#include "agent/tooling/adapters/tool-adapters.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>

class test_resource_store final : public agent_resource_store {
public:
    bool put_text(const agent_resource_put_request & request, agent_resource_descriptor & out, std::string &) override {
        last_request = request;
        out.uri = "resource://test/" + request.name;
        out.name = request.name;
        out.mime_type = request.mime_type;
        out.scope = request.scope;
        return true;
    }
    bool read_text(const std::string &, const agent_resource_read_authority &, size_t, std::string &, std::string &) const override { return false; }
    bool stat(const std::string &, const agent_resource_read_authority &, agent_resource_descriptor &, std::string &) const override { return false; }
    bool list(const agent_resource_read_authority &, std::vector<agent_resource_descriptor> &, std::string &) const override { return false; }
    agent_resource_put_request last_request;
};

class test_data_store final : public common_agent_data_store {
public:
    bool get_dataset_descriptor(const std::string & dataset_uri,
            common_agent_dataset_descriptor & descriptor, std::string & error) override {
        if (dataset_uri != stored_descriptor.ref.uri) {
            error = "unknown dataset";
            return false;
        }
        descriptor = stored_descriptor;
        return true;
    }
    bool list_dataset_descriptors(std::vector<common_agent_dataset_descriptor> & descriptors, std::string &) override {
        descriptors = {stored_descriptor};
        return true;
    }
    bool find_dataset_by_name(const std::string & name,
            common_agent_dataset_descriptor & descriptor, std::string & error) override {
        if (name != stored_descriptor.ref.name && name != "sales") {
            error = "dataset name was not found";
            return false;
        }
        descriptor = stored_descriptor;
        return true;
    }
    bool execute(const std::string & operation, const std::string & request_json, std::string & result, std::string & error) override {
        last_operation = operation;
        last_request = request_json;
        if (operation == "data.query") { result = R"({"columns":["name","value"],"rows":[{"name":"alpha","value":1},{"name":"beta","value":2}],"row_count":2,"scan_truncated":false,"result_truncated":false})"; return true; }
        if (operation == "data.filter") { result = R"({"rows":[{"name":"alpha"}],"row_count":1})"; return true; }
        if (operation == "data.aggregate") { result = R"({"rows":[{"group":"all","count":2}]})"; return true; }
        if (operation == "data.join") { result = R"({"rows":[{"id":"1","name":"alpha"}]})"; return true; }
        if (operation == "data.transform") { result = R"({"dataset":"dataset://derived/1"})"; return true; }
        if (operation == "statistics.describe") { result = R"({"columns":[{"name":"value","count":2,"mean":1.5}]})"; return true; }
        if (operation == "statistics.outliers") { result = R"({"columns":[]})"; return true; }
        if (operation == "statistics.value_counts") { result = R"({"column":"region","values":[{"value":"north","count":2}]})"; return true; }
        error = "unexpected data operation";
        return false;
    }

    std::string last_operation;
    std::string last_request;
    common_agent_dataset_descriptor stored_descriptor = [] {
        common_agent_dataset_descriptor value;
        value.ref.uri = "dataset://analysis/sales";
        value.ref.name = "Sales";
        value.ref.row_count = 2;
        value.ref.column_count = 2;
        value.ref.source_resource_uri = "resource://uploads/sales.xlsx";
        value.ref.source_representation = "tabular-dataset";
        value.source_sheet_name = "Sales";
        value.source_range = "A1:B3";
        value.import_processor_id = "xlsx-workbook-json-v1";
        value.columns = {{"name", common_agent_dataset_column_type::string, true},
                         {"value", common_agent_dataset_column_type::decimal, true}};
        return value;
    }();
};

int main() {
    std::string error;
    common_memory_in_memory_store memories;
    assert(memories.open("", error));
    common_memory_record memory;
    memory.id = "memory-1";
    memory.kind = common_memory_kind::fact;
    memory.content = "The plan store uses optimistic version checks.";
    memory.scope = common_memory_scope::session;
    memory.session_id = "session-1";
    assert(memories.put(memory, error));

    common_plan_in_memory_store plans;
    assert(plans.open("", error));
    common_plan_state plan;
    plan.id = "plan-1";
    plan.goal = "Verify native adapters";
    assert(plans.create(plan, error));

    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    assert(catalog.bootstrap("memory-read", bootstrap, error));
    common_tool_registry registry;
    common_native_tool_bindings bindings;
    bindings.memory_store = &memories;
    bindings.plan_store = &plans;
    bindings.memory_query.scope = common_memory_scope::session;
    bindings.memory_query.session_id = "session-1";
    std::string active_plan_id = "plan-1";
    bindings.plan_id = &active_plan_id;
    common_tool_adapter_result adapters;
    assert(common_register_native_tool_adapters(catalog, "memory-read", bindings, registry, adapters, error));
    assert(adapters.registered.size() == 7);
    auto result = registry.execute({"calculator", R"({"expression":"(18 + 2) * 3"})"});
    assert(result.ok);
    auto output = result.output;
    assert(output == R"({"value":60.0})" || output == R"({"value":60})");
    result = registry.execute({"memory_search", R"({"query":"optimistic checks"})"});
    assert(result.ok && result.output.find("memory-1") != std::string::npos);
    result = registry.execute({"memory_get", R"({"id":"memory-1"})"});
    assert(result.ok && result.output.find("optimistic version checks") != std::string::npos);
    result = registry.execute({"memory_get", R"({"memory_id":"memory-1"})"});
    assert(result.ok && result.output.find("optimistic version checks") != std::string::npos);
    result = registry.execute({"memory_inspect", "{}"});
    assert(result.ok && result.output.find("\"count\":1") != std::string::npos && result.output.find("fact") != std::string::npos);
    result = registry.execute({"memory_conflict_check", R"({"content":"The plan store uses optimistic version checks."})"});
    assert(result.ok && result.output.find("\"conflict\":true") != std::string::npos && result.output.find("memory-1") != std::string::npos);
    result = registry.execute({"memory_get", R"("memory-1")"});
    assert(result.ok && result.output.find("optimistic version checks") != std::string::npos);
    result = registry.execute({"plan_get", "{}"});
    assert(result.ok && result.output.find("plan-1") != std::string::npos);
    result = registry.execute({"memory_remember", "{}"});
    assert(!result.ok);

    common_tool_registry proposal_registry;
    common_native_tool_bindings proposal_bindings;
    proposal_bindings.memory_remember_proposal = [](const std::string &) { return common_tool_execution_result::success(R"({"decision":"accept"})"); };
    assert(common_register_native_tool_adapters(catalog, "memory", proposal_bindings, proposal_registry, adapters, error));
    assert(proposal_registry.is_policy_gated("memory_remember"));
    result = proposal_registry.execute({"memory_remember", R"({"kind":"fact","content":"verified"})"});
    assert(result.ok && result.output.find("accept") != std::string::npos);

    const auto repository = std::filesystem::temp_directory_path() / "llama-agent-repository-tool-test";
    std::filesystem::create_directories(repository / "src");
    { std::ofstream file(repository / "src" / "sample.txt"); file << "alpha\nneedle in a haystack\n"; }
    const auto git_init = "git -C \"" + repository.string() + "\" init -q";
    assert(std::system(git_init.c_str()) == 0);
    common_tool_catalog research_catalog;
    assert(research_catalog.bootstrap("research", bootstrap, error));
    common_tool_registry repository_registry;
    common_native_tool_bindings repository_bindings;
    repository_bindings.repository_root = repository.string();
    repository_bindings.web_search = [](const std::string & input) {
        if (input.find("\"query\":\"llama\"") == std::string::npos) {
            return common_tool_execution_result::failure("tool.web_search.unexpected_query", common_tool_failure_class::validation, false, "Unexpected search query.", "unexpected search query");
        }
        return common_tool_execution_result::success(R"({"results":[{"title":"llama.cpp","url":"https://example.com/llama","snippet":"native tools","source":"test"}],"provider":"test"})");
    };
    repository_bindings.web_fetch = [](const std::string & input) {
        if (input.find("\"url\":\"https://example.com/llama\"") == std::string::npos) {
            return common_tool_execution_result::failure("tool.web_fetch.unexpected_url", common_tool_failure_class::validation, false, "Unexpected fetch URL.", "unexpected fetch url");
        }
        return common_tool_execution_result::success(R"({"url":"https://example.com/llama","final_url":"https://example.com/llama","status":200,"content_type":"text/html","title":"llama.cpp","text":"native tools","truncated":false})");
    };
    assert(common_register_native_tool_adapters(research_catalog, "research", repository_bindings, repository_registry, adapters, error));
    result = repository_registry.execute({"repository.list", R"({"path":"src","depth":1})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos);
    result = repository_registry.execute({"repository.search", R"({"query":"needle","path":"src"})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos && result.output.find("needle") != std::string::npos);
    result = repository_registry.execute({"repository.read", R"({"path":"src/sample.txt","start_line":2,"end_line":2})"});
    assert(result.ok && result.output.find("needle in a haystack") != std::string::npos);
    result = repository_registry.execute({"web_search", R"({"query":"llama","limit":1})"});
    assert(result.ok && result.output.find("https://example.com/llama") != std::string::npos);
    result = repository_registry.execute({"web_fetch", R"({"url":"https://example.com/llama","max_bytes":4096})"});
    assert(result.ok && result.output.find("\"status\":200") != std::string::npos && result.output.find("native tools") != std::string::npos);
    result = repository_registry.execute({"repository.read", R"({"path":"../outside.txt"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);

    common_tool_catalog developer_catalog;
    assert(developer_catalog.bootstrap("developer-read", bootstrap, error));
    common_tool_registry developer_registry;
    common_tool_adapter_result developer_adapters;
    assert(common_register_native_tool_adapters(developer_catalog, "developer-read", repository_bindings, developer_registry, developer_adapters, error));
    result = developer_registry.execute({"workspace.list", R"({"path":"src","depth":1})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos);
    result = developer_registry.execute({"workspace.search", R"({"query":"needle","path":"src"})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos);
    result = developer_registry.execute({"workspace.read", R"({"path":"src/sample.txt","start_line":1,"end_line":1})"});
    assert(result.ok && result.output.find("alpha") != std::string::npos);
    result = developer_registry.execute({"repository.status", "{}"});
    assert(result.ok && result.output.find("src") != std::string::npos);
    result = developer_registry.execute({"repository.changed_files", "{}"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos);

    std::filesystem::create_directories(repository / "datasets");
    { std::ofstream file(repository / "datasets" / "sample.csv"); file << "name,value\nalpha,1\nbeta,2\n"; }
    common_tool_profile foundation_profile;
    foundation_profile.id = "developer-foundation";
    foundation_profile.members = {
        {"workspace.patch", 1, true, "{}"},
        {"diagnostics.compile", 1, true, "{}"},
        {"dataset.list", 1, true, "{}"},
        {"dataset.select", 1, true, "{}"},
        {"dataset.inspect", 1, true, "{}"},
        {"dataset.schema", 1, true, "{}"},
        {"dataset.sample", 1, true, "{}"},
        {"document.tables", 1, true, "{}"},
        {"document.table", 1, true, "{}"},
        {"dataset.validate", 1, true, "{}"},
        {"data.query", 1, true, "{}"},
        {"data.filter", 1, true, "{}"},
        {"data.aggregate", 1, true, "{}"},
        {"data.join", 1, true, "{}"},
        {"data.transform", 1, true, "{}"},
        {"statistics.describe", 1, true, "{}"},
        {"statistics.outliers", 1, true, "{}"},
        {"statistics.value_counts", 1, true, "{}"},
        {"diagnostics.test_failures", 1, true, "{}"},
        {"diagnostics.symbol", 1, true, "{}"},
        {"diagnostics.references", 1, true, "{}"},
        {"diagnostics.call_hierarchy", 1, true, "{}"},
        {"diagnostics.format", 1, true, "{}"},
        {"diagnostics.include_graph", 1, true, "{}"},
        {"diagnostics.native_crash", 1, true, "{}"},
        {"artifact.export", 1, true, "{}"},
    };
    foundation_profile.allow_policy_gated_writes = true;
    std::map<std::string, common_tool_profile> foundation_profiles;
    foundation_profiles.emplace(foundation_profile.id, foundation_profile);
    common_tool_catalog foundation_catalog;
    assert(foundation_catalog.bootstrap(foundation_profile.id, bootstrap, error, {}, foundation_profiles));
    common_tool_registry foundation_registry;
    common_native_tool_bindings foundation_bindings = repository_bindings;
    test_resource_store foundation_resources;
    foundation_bindings.resource_runtime.store = &foundation_resources;
    test_data_store foundation_data;
    foundation_bindings.data_store = &foundation_data;
    foundation_bindings.document_tables = [](const std::string & input) {
        return common_tool_execution_result::success(
            std::string(R"({"resource":"agent-resource://document/report.json","tables":[{"index":0,"name":"Population","node_id":"document-node://table/0","dataset":"dataset://report/table/0"}]})") +
            "\n" + input);
    };
    foundation_bindings.document_table = [](const std::string & input) {
        if (input.find("Budget summary") == std::string::npos &&
                input.find("table_index") == std::string::npos) {
            return common_tool_execution_result::failure(
                "tool.document.table.not_found", common_tool_failure_class::not_found,
                false, "Document table was not found.", "document table was not found");
        }
        return common_tool_execution_result::success(
            R"({"table_index":1,"name":"Budget summary","dataset":"dataset://report/table/1"})");
    };
    foundation_bindings.resource_runtime.namespace_id = "test";
    foundation_bindings.resource_runtime.session_id = "session-1";
    assert(common_register_native_tool_adapters(foundation_catalog, foundation_profile.id, foundation_bindings, foundation_registry, adapters, error));
    result = foundation_registry.execute({"workspace.patch", R"({"path":"src/sample.txt","operations":[{"type":"replace_range","start_line":1,"end_line":1,"content":"patched"}]})"});
    assert(result.ok && result.output.find("content_hash") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.compile", R"({"output":"src/sample.cpp:7:3: error: missing symbol\n"})"});
    assert(result.ok && result.output.find("missing symbol") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.test_failures", R"({"result":"PASS one\nFAILED two: assertion\n"})"});
    assert(result.ok && result.output.find("assertion") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.symbol", R"({"symbol":"needle","path_hint":"src/sample.txt"})"});
    assert(result.ok && result.output.find("text-fallback") != std::string::npos && result.output.find("sample.txt") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.references", R"({"symbol":"needle","definition_path":"src"})"});
    assert(result.ok && result.output.find("\"references\"") != std::string::npos && result.output.find("sample.txt") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.call_hierarchy", R"({"symbol":"needle"})"});
    assert(!result.ok && result.failure_code == "tool.diagnostics.call_hierarchy.unavailable");
    result = foundation_registry.execute({"diagnostics.native_crash", R"({"executable":"bin/agent","dump":"artifacts/core"})"});
    assert(!result.ok && result.failure_code == "tool.diagnostics.native_crash.backend_unavailable");
    result = foundation_registry.execute({"diagnostics.test_failures", R"({"result":"FAILED src/a.cpp:42 assertion expected 1\nFAILED src/b.cpp:42 assertion expected 2\nctest: timeout after 120000ms\n"})"});
    assert(result.ok && result.output.find("assertion_failure") != std::string::npos && result.output.find("timeout") != std::string::npos && result.output.find("\"count\":2") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.format", R"({"output":"src/sample.cpp would reformat\n"})"});
    assert(result.ok && result.output.find("formatted") != std::string::npos && result.output.find("sample.cpp") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.include_graph", R"({"output":"a.h -> b.h\nb.h -> c.h\n"})"});
    assert(result.ok && result.output.find("a.h") != std::string::npos && result.output.find("c.h") != std::string::npos);
    result = foundation_registry.execute({"dataset.list", R"({"path":"datasets"})"});
    assert(result.ok && result.output.find("Sales") != std::string::npos);
    result = foundation_registry.execute({"dataset.inspect", R"({"path":"datasets/sample.csv"})"});
    assert(result.ok && result.output.find("csv") != std::string::npos);
    result = foundation_registry.execute({"dataset.schema", R"({"path":"datasets/sample.csv"})"});
    assert(result.ok && result.output.find("name") != std::string::npos && result.output.find("value") != std::string::npos);
    result = foundation_registry.execute({"dataset.sample", R"({"path":"datasets/sample.csv","rows":1})"});
    assert(result.ok && result.output.find("alpha") != std::string::npos);
    result = foundation_registry.execute({"dataset.inspect", R"({"dataset":"dataset://analysis/sales"})"});
    assert(result.ok && result.output.find("resource://uploads/sales.xlsx") != std::string::npos &&
           result.output.find("Sales") != std::string::npos);
    result = foundation_registry.execute({"dataset.schema", R"({"dataset":"dataset://analysis/sales"})"});
    assert(result.ok && result.output.find("decimal") != std::string::npos && result.output.find("value") != std::string::npos);
    result = foundation_registry.execute({"dataset.sample", R"({"dataset":"dataset://analysis/sales","rows":1})"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           foundation_data.last_request.find("dataset://analysis/sales") != std::string::npos);
    result = foundation_registry.execute({"document.tables", R"({"resource":"agent-resource://document/report.json"})"});
    assert(result.ok && result.output.find("Population") != std::string::npos);
    result = foundation_registry.execute({"document.table", R"({"resource":"agent-resource://document/report.json","table":"Budget summary"})"});
    assert(result.ok && result.output.find("dataset://report/table/1") != std::string::npos);
    result = foundation_registry.execute({"document.table", R"({"resource":"agent-resource://document/report.json","table":"Budget summary","table_index":1})"});
    assert(!result.ok && result.failure_code == "tool.document.table.invalid_locator");

    // First-class dataset tools do not require a repository root. The host
    // data-store binding is sufficient; legacy path tools remain unavailable.
    common_native_tool_bindings dataset_only_bindings;
    test_data_store dataset_only_data;
    dataset_only_bindings.data_store = &dataset_only_data;
    common_tool_registry dataset_only_registry;
    common_tool_adapter_result dataset_only_adapters;
    assert(common_register_native_tool_adapters(foundation_catalog, foundation_profile.id,
        dataset_only_bindings, dataset_only_registry, dataset_only_adapters, error));
    result = dataset_only_registry.execute({"dataset.inspect", R"({"dataset":"dataset://analysis/sales"})"});
    assert(result.ok && result.output.find("resource://uploads/sales.xlsx") != std::string::npos);
    result = dataset_only_registry.execute({"dataset.schema", R"({"dataset":"dataset://analysis/sales"})"});
    assert(result.ok && result.output.find("decimal") != std::string::npos);
    result = dataset_only_registry.execute({"dataset.sample", R"({"dataset":"dataset://analysis/sales","rows":1})"});
    assert(result.ok && dataset_only_data.last_operation == "data.query");
    result = dataset_only_registry.execute({"dataset.list", "{}"});
    assert(result.ok && result.output.find("dataset://analysis/sales") != std::string::npos);
    result = dataset_only_registry.execute({"dataset.select", R"({"name":"sales"})"});
    assert(result.ok && result.output.find("dataset://analysis/sales") != std::string::npos);
    result = dataset_only_registry.execute({"dataset.select", R"({"name":"missing"})"});
    assert(!result.ok && result.failure_code == "tool.dataset.select.not_found" &&
           result.detail.find("choose one of: Sales (dataset://analysis/sales)") != std::string::npos);
    result = foundation_registry.execute({"dataset.inspect", R"({"dataset":"dataset://missing"})"});
    assert(!result.ok && result.failure_code == "tool.dataset.unavailable");
    result = foundation_registry.execute({"dataset.validate", R"({"dataset":"datasets/sample.csv","rules":[{"type":"not_null","column":"name"},{"type":"unique","column":"name"}]})"});
    assert(result.ok && result.output.find("\"valid\":true") != std::string::npos);
    result = foundation_registry.execute({"data.query", R"({"dataset":"tool-events","limit":10})"});
    assert(result.ok && foundation_data.last_operation == "data.query");
    result = foundation_registry.execute({"data.query", R"({"dataset":"tool-events","where":{"status":"failed"}})"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           foundation_data.last_request.find(R"("where":[{"field":"status","operator":"=","value":"failed"}])") != std::string::npos);
    result = foundation_registry.execute({"data.query", R"({"dataset":"tool-events","where":"status -eq \"failed\" and attempt -gt 1"})"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           foundation_data.last_request.find(R"("field":"status")") != std::string::npos &&
           foundation_data.last_request.find(R"("operator":"=","value":"failed")") != std::string::npos &&
           foundation_data.last_request.find(R"("field":"attempt","operator":">","value":1)") != std::string::npos);
    result = foundation_registry.execute({"data.filter", R"({"dataset":"tool-events","conditions":[{"field":"status","operator":"=","value":"failed"}]})"});
    assert(result.ok && foundation_data.last_operation == "data.filter");
    result = foundation_registry.execute({"data.filter", R"({"dataset":"tool-events","conditions":{"status":{"in":["failed","blocked"]}}})"});
    assert(result.ok && foundation_data.last_operation == "data.filter" &&
           foundation_data.last_request.find(R"("operator":"in","value":["failed","blocked"])") != std::string::npos);
    result = foundation_registry.execute({"data.aggregate", R"({"dataset":"tool-events","measures":[{"function":"count","column":"*"}]})"});
    assert(result.ok && foundation_data.last_operation == "data.aggregate");
    result = foundation_registry.execute({"data.join", R"({"left":"a","right":"b","on":[{"left":"id","right":"id"}]})"});
    assert(result.ok && foundation_data.last_operation == "data.join");
    result = foundation_registry.execute({"data.transform", R"({"dataset":"a","operations":[{"type":"rename","from":"x","to":"y"}]})"});
    assert(result.ok && foundation_data.last_operation == "data.transform");
    result = foundation_registry.execute({"statistics.describe", R"({"dataset":"a","columns":["value"]})"});
    assert(result.ok && foundation_data.last_operation == "statistics.describe");
    result = foundation_registry.execute({"statistics.describe", R"({"dataset":"a","column":"value","group_by":"region"})"});
    assert(result.ok && foundation_data.last_operation == "statistics.describe" &&
           foundation_data.last_request.find(R"("columns":["value"])" ) != std::string::npos &&
           foundation_data.last_request.find(R"("group_by":["region"])" ) != std::string::npos);
    result = foundation_registry.execute({"statistics.outliers", R"({"dataset":"a","column":"value","group_by":"region"})"});
    assert(result.ok && foundation_data.last_operation == "statistics.outliers" &&
           foundation_data.last_request.find(R"("columns":["value"])" ) != std::string::npos &&
           foundation_data.last_request.find(R"("group_by":["region"])" ) != std::string::npos);
    result = foundation_registry.execute({"statistics.value_counts", R"({"dataset":"a","column":"region","limit":5})"});
    assert(result.ok && foundation_data.last_operation == "statistics.value_counts");
    result = foundation_registry.execute({"data.aggregate", R"({"dataset":"a","group_by":"region","sum":"amount"})"});
    assert(result.ok && foundation_data.last_operation == "data.aggregate" &&
           foundation_data.last_request.find(R"("group_by":["region"])" ) != std::string::npos &&
           foundation_data.last_request.find(R"("function":"sum","column":"amount")" ) != std::string::npos);
    result = foundation_registry.execute({"data.aggregate", R"json({"dataset":"a","select":"sum(amount)"})json"});
    assert(result.ok && foundation_data.last_operation == "data.aggregate" &&
           foundation_data.last_request.find(R"("measures":[{"function":"sum","column":"amount"}])") != std::string::npos &&
           foundation_data.last_request.find(R"("select")") == std::string::npos);
    result = foundation_registry.execute({"data.join", R"({"left":"a","right":"b","on":"id"})"});
    assert(result.ok && foundation_data.last_operation == "data.join" &&
           foundation_data.last_request.find(R"("left":"id","right":"id")" ) != std::string::npos);
    result = foundation_registry.execute({"dataset.sample", R"({"dataset":"dataset://analysis/sales","limit":1})"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           foundation_data.last_request.find(R"("limit":1)" ) != std::string::npos);
    result = foundation_registry.execute({"artifact.export", R"({"name":"summary.txt","content":"artifact result"})"});
    assert(result.ok && result.output.find("resource://") != std::string::npos);
    result = foundation_registry.execute({"artifact.export", R"({"source_dataset":"dataset://analysis/sales","format":"csv","name":"sales.csv"})"});
    assert(result.ok && result.output.find("dataset://analysis/sales") != std::string::npos &&
           foundation_resources.last_request.mime_type == "text/csv" &&
           foundation_resources.last_request.text == "name,value\nalpha,1\nbeta,2\n" &&
           foundation_resources.last_request.lineage.parent_uri == "dataset://analysis/sales" &&
           foundation_resources.last_request.lineage.derivation == "artifact.export:dataset-csv");
    result = foundation_registry.execute({"artifact.export", R"({"source_dataset":"dataset://analysis/sales","format":"json","name":"sales.json"})"});
    assert(!result.ok);

    common_tool_profile execution_profile;
    execution_profile.id = "developer-execution";
    execution_profile.members = {{"development.build", 1, true, "{}"}, {"development.test", 1, true, "{}"}};
    execution_profile.allow_policy_gated_writes = true;
    std::map<std::string, common_tool_profile> configured_profiles;
    configured_profiles.emplace(execution_profile.id, execution_profile);
    common_tool_catalog execution_catalog;
    assert(execution_catalog.bootstrap("developer-execution", bootstrap, error, {}, configured_profiles));
    common_tool_registry execution_registry;
    common_native_tool_bindings execution_bindings;
    common_agent_sandbox_request captured_request;
    execution_bindings.sandbox_execute = [&captured_request](common_agent_sandbox_request request) {
        captured_request = std::move(request);
        return common_tool_execution_result::success(R"({"status":"completed","exit_code":0})");
    };
    assert(common_register_native_tool_adapters(execution_catalog, "developer-execution", execution_bindings, execution_registry, adapters, error));
    assert(execution_registry.is_policy_gated("development.build"));
    result = execution_registry.execute({"development.build", R"({"target":"llama-agent","configuration":"Debug"})"});
    assert(result.ok && captured_request.command.program == "agent.development.build" && captured_request.command.arguments[0] == "llama-agent");
    result = execution_registry.execute({"development.test", R"({"target":"agent-smoke","filter":"workspace"})"});
    assert(result.ok && captured_request.command.program == "agent.development.test" && captured_request.command.arguments[1] == "workspace");
    result = execution_registry.execute({"development.test", R"({"target":"agent-smoke","resource_refs":["resource://input.txt"]})"});
    assert(result.ok && captured_request.workspace.input_resources.size() == 1 &&
            captured_request.workspace.input_resources[0].uri == "resource://input.txt");
    common_tool_registry native_network_registry;
    common_native_tool_bindings native_network_bindings;
    assert(common_register_native_tool_adapters(research_catalog, "research", native_network_bindings, native_network_registry, adapters, error));
    result = native_network_registry.execute({"web_fetch", R"({"url":"http://127.0.0.1/test"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::network);
    std::filesystem::remove_all(repository);
    return 0;
}
