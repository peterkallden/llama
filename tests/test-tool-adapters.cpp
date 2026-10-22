#include "agent/tooling/adapters/tool-adapters.h"
#include "tools/agent/resource/dispatch/agent-resource-processor-dispatch.h"
#include "agent/tooling/bridge/tool-chat-bridge.h"
#include "agent/tooling/contracts/tool-result-contracts.h"
#include "agent/artifact/artifact-contract.h"
#include "hash/hash.h"
#include "base64.hpp"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

class test_resource_store final : public agent_resource_store {
public:
    bool put_text(const agent_resource_put_request & request, agent_resource_descriptor & out, std::string & error) override {
        if(!fail_mime.empty()&&request.mime_type==fail_mime){error="injected publication failure";return false;}
        last_request = request;
        requests.push_back(request);
        out.uri = "resource://test/" + request.name;
        out.name = request.name;
        out.mime_type = request.mime_type;
        out.scope = request.scope;
        out.metadata = request.metadata;
        out.lineage = request.lineage;
        out.source_provider = request.source_provider;
        out.source_tool = request.source_tool;
        out.size_bytes = request.text.size();
        out.sha256 = std::to_string(request.text.size());
        resources[out.uri] = out;
        available_uris.insert(out.uri);
        contents[out.uri] = request.bytes_are_authoritative ? request.bytes : request.text;
        return true;
    }
    bool put_bytes(const agent_resource_put_request & request, agent_resource_descriptor & out, std::string & error) override {
        auto text_request=request;
        text_request.text=request.bytes;
        text_request.bytes_are_authoritative=false;
        if (!put_text(text_request,out,error)) return false;
        out.sha256=hash_sha256_hex(request.bytes.data(),request.bytes.size());
        resources[out.uri]=out;
        return true;
    }
    bool read_text(const std::string & uri, const agent_resource_read_authority &, size_t max_bytes,
            std::string & out, std::string & error) const override {
        const auto it=contents.find(uri);
        if(it!=contents.end()){out=it->second.substr(0,max_bytes);return true;}
        if (uri != "resource://session/long-text") {
            error = "resource was not found";
            return false;
        }
        out.assign(std::min<size_t>(max_bytes, 16000), 'x');
        return true;
    }
    bool read_bytes(const std::string & uri,const agent_resource_read_authority & authority,size_t max_bytes,std::string & out,std::string & error) const override {
        return read_text(uri,authority,max_bytes,out,error);
    }
    bool stat(const std::string & uri, const agent_resource_read_authority &, agent_resource_descriptor & out, std::string & error) const override {
        const auto stored = resources.find(uri);
        if (stored != resources.end()) {
            out = stored->second;
            return true;
        }
        if (available_uris.count(uri) == 0) {
            error = "resource was not found";
            return false;
        }
        out.uri = uri;
        out.name = uri == "resource://session/long-text" ? "long.txt" : "sales.xlsx";
        out.mime_type = uri == "resource://session/long-text"
            ? "text/plain"
            : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
        out.size_bytes = uri == "resource://session/long-text" ? 16000 : 0;
        return true;
    }
    bool list(const agent_resource_read_authority &, std::vector<agent_resource_descriptor> & out, std::string &) const override {
        out.clear();for(const auto & item:resources)out.push_back(item.second);return true;
    }
    agent_resource_put_request last_request;
    std::string fail_mime;
    std::vector<agent_resource_put_request> requests;
    std::set<std::string> available_uris;
    std::map<std::string,agent_resource_descriptor> resources;
    std::map<std::string,std::string> contents;
};

class test_investigation_store final : public common_agent_data_store {
public:
    common_agent_dataset_descriptor source;
    std::map<std::string, common_investigation_case_revision> cases;
    std::vector<common_investigation_report_index_entry> reports;
    std::string virtual_page = R"({"columns":["event.id","event.type","@timestamp","user.id","host.name","@source.index","@source.id"],"rows":[["doc-2","authentication","1970-01-01T00:00:00.300Z","alice","db2","logs-2026.10","doc-2"]],"complete":true,"scan_truncated":false,"result_truncated":false})";

    bool authorize_investigation_source(const std::string & source_id, const common_agent_scope &, std::string & error) const override {
        if (source_id != source.ref.source_provider) { error = "source is not authorized"; return false; }
        error.clear(); return true;
    }

    bool execute(const std::string &, const std::string &, std::string &, std::string & error) override {
        error = "not used by investigation adapter contract";
        return false;
    }
    bool execute_scoped(const std::string & operation, const std::string &, const common_agent_scope &,
            bool, std::string & result, std::string & error) override {
        if (operation != "data.query") { error = "unexpected virtual operation"; return false; }
        result = virtual_page; error.clear(); return true;
    }
    bool get_dataset_descriptor(const std::string & uri, common_agent_dataset_descriptor & out,
            std::string & error) override {
        if (uri != source.ref.uri) { error = "virtual dataset not found"; return false; }
        out = source; error.clear(); return true;
    }
    bool list_dataset_descriptors(std::vector<common_agent_dataset_descriptor> & out,
            std::string & error) override {
        out = {source}; error.clear(); return true;
    }
    bool get_investigation_case(const std::string & id, const common_agent_scope & scope,
            common_investigation_case_revision & out, std::string & error) override {
        const auto found = cases.find(id);
        if (found == cases.end() || found->second.scope.namespace_id != scope.namespace_id ||
                found->second.scope.project_id != scope.project_id ||
                (scope.project_id.empty() && found->second.scope.session_id != scope.session_id)) {
            error = "case is unavailable in this scope"; return false;
        }
        out = found->second; error.clear(); return true;
    }
    bool list_investigation_cases(const common_agent_scope & scope, size_t limit,
            const std::string & after, std::vector<common_investigation_case_summary> & out,
            std::string & cursor, std::string & error) override {
        out.clear(); cursor.clear();
        for (const auto & item : cases) {
            const auto & value = item.second;
            if (value.scope.namespace_id != scope.namespace_id || value.scope.project_id != scope.project_id ||
                    (scope.project_id.empty() && value.scope.session_id != scope.session_id) || item.first <= after) continue;
            out.push_back({value.case_id, value.revision, value.title, value.question,
                value.source_id, value.window_start_ms, value.window_end_ms});
        }
        if (out.size() > limit) { out.resize(limit); cursor = out.back().case_id; }
        error.clear(); return true;
    }
    bool publish_investigation_case(const common_investigation_case_revision & input,
            uint64_t expected, const std::string &, common_investigation_case_revision & out,
            std::string & error) override {
        const auto found = cases.find(input.case_id);
        const uint64_t current = found == cases.end() ? 0 : found->second.revision;
        if (current != expected || input.revision != expected + 1) {
            error = "investigation case revision conflict"; return false;
        }
        out = input; cases[input.case_id] = out; error.clear(); return true;
    }
    bool publish_investigation_report(const common_agent_scope & scope,
            const common_investigation_report_index_entry & input,
            common_investigation_report_index_entry & out, std::string & error) override {
        const auto found = cases.find(input.case_id);
        if (found == cases.end() || found->second.scope.namespace_id != scope.namespace_id ||
                found->second.scope.project_id != scope.project_id ||
                (scope.project_id.empty() && found->second.scope.session_id != scope.session_id)) {
            error = "report is unavailable in this scope"; return false;
        }
        const auto existing = std::find_if(reports.begin(), reports.end(), [&](const auto & item) {
            return item.case_id == input.case_id && item.case_revision == input.case_revision &&
                item.content_fingerprint == input.content_fingerprint;
        });
        if (existing != reports.end()) { out = *existing; error.clear(); return true; }
        reports.push_back(input); out = input; error.clear(); return true;
    }
    bool list_investigation_reports(const common_agent_scope & scope, const std::string & case_id,
            uint64_t revision, std::vector<common_investigation_report_index_entry> & out,
            std::string & error) override {
        out.clear();
        const auto found = cases.find(case_id);
        if (found == cases.end() || found->second.scope.namespace_id != scope.namespace_id ||
                found->second.scope.project_id != scope.project_id ||
                (scope.project_id.empty() && found->second.scope.session_id != scope.session_id)) {
            error = "report is unavailable in this scope"; return false;
        }
        for (const auto & item : reports) if (item.case_id == case_id && item.case_revision == revision)
            out.push_back(item);
        error.clear(); return true;
    }
    void seed(common_investigation_case_revision value) { cases[value.case_id] = std::move(value); }
};

class test_docx_processing_provider final : public agent_resource_processing_provider {
public:
    test_docx_processing_provider(
            test_resource_store & store,
            bool & called,
            agent_resource_processing_binding_request & observed_request)
        : store_(store), called_(called), observed_request_(observed_request) {}

    agent_resource_processing_result process(
            const agent_resource_processing_binding_request & request) const override {
        called_ = true;
        observed_request_ = request;
        agent_resource_processing_result result;
        result.success = true;
        const bool markdown = request.target_representation == "markdown";
        agent_resource_processing_output output;
        output.name = markdown ? "pandoc-output.md" : "pandoc-output.docx";
        output.mime_type = request.target_media_type;
        output.bytes = markdown ? (request.source_uri.find("empty.docx")!=std::string::npos ? "" : "# Imported DOCX\n\n| Name | Value |\n|---|---|\n| alpha | 1 |\n") : "test-docx-bytes";
        output.metadata.limitations = markdown ? "Images or unsupported Word formatting may not be preserved." : "";
        agent_resource_put_request put;put.name=output.name;put.mime_type=output.mime_type;
        put.bytes=output.bytes;put.bytes_are_authoritative=true;put.metadata=output.metadata;
        agent_resource_descriptor resource;std::string error;
        assert(store_.put_bytes(put,resource,error));
        result.resources.push_back(std::move(resource));
        return result;
    }

private:
    test_resource_store & store_;
    bool & called_;
    agent_resource_processing_binding_request & observed_request_;
};

class test_data_store final : public common_agent_data_store {
public:
    bool get_dataset_descriptor(const std::string & dataset_uri,
            common_agent_dataset_descriptor & descriptor, std::string & error) override {
        if (dataset_uri == "dataset://analysis/csv-sales") {
            descriptor = stored_descriptor;
            descriptor.ref.uri = dataset_uri;
            descriptor.ref.source_resource_uri = "resource://uploads/sales.csv";
            descriptor.ref.source_representation = "csv:dataset";
            descriptor.origin.kind = "csv_resource";
            descriptor.import_processor_id = "csv-resource-import-v1";
            return true;
        }
        if (dataset_uri != stored_descriptor.ref.uri && dataset_uri != "dataset://tool-events") {
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
    // Report export feeds Markdown to the processor and requests DOCX. The
    // configured DOCX policy must therefore be selected for text/markdown,
    // not only for DOCX inputs that are being read.
    std::map<std::string, agent_resource_processor_execution_policy> pandoc_policies;
    agent_resource_processor_execution_policy pandoc_policy;
    pandoc_policy.execution = "local_required";
    pandoc_policy.backend = "local";
    pandoc_policy.executable = "pandoc";
    pandoc_policies.emplace("docx.text", pandoc_policy);
    const auto markdown_to_docx_dispatch = resolve_agent_resource_processor_dispatch({
        pandoc_policies, "text/markdown", "", true, false});
    assert(markdown_to_docx_dispatch.has_policy(agent_resource_processor_kind::pandoc));
    assert(markdown_to_docx_dispatch.wants_local_for(agent_resource_processor_kind::pandoc));

    const auto fetched_payload = common_tool_web_fetch_inline_result_to_json({
        "https://example.test/openapi.json", "https://example.test/openapi.json", 200,
        "application/json", "spec", "{}", 2, false, "resource://session/spec-1"});
    assert(fetched_payload.value("resource_ref", std::string{}) == "resource://session/spec-1");
    common_tool_web_fetch_result large_fetch{
        "https://example.test/catalogue", "https://example.test/catalogue", 200,
        "text/plain", {}, std::string(1023, 'x') + "€" + std::string(8192, 'x'), false};
    const auto bounded_fetch = common_tool_web_fetch_make_inline_result(large_fetch);
    const auto bounded_fetch_json = common_tool_web_fetch_inline_result_to_json(bounded_fetch);
    assert(bounded_fetch.text_excerpt.size() == 1023); // preserve UTF-8 at the 1 KiB boundary
    assert(bounded_fetch.text_length == 9218 && bounded_fetch.preview_truncated);
    assert(!bounded_fetch.truncated && bounded_fetch.resource_ref.empty());
    assert(!bounded_fetch_json.contains("text"));
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
    auto result = registry.execute({"math.calculate", R"({"expression":"(18 + 2) * 3"})"});
    assert(result.ok);
    auto output = result.output;
    assert(output == R"({"value":60.0})" || output == R"({"value":60})");
    result = registry.execute({"memory.search", R"({"query":"optimistic checks"})"});
    assert(result.ok && result.output.find("memory-1") != std::string::npos);
    result = registry.execute({"memory.get", R"({"id":"memory-1"})"});
    assert(result.ok && result.output.find("optimistic version checks") != std::string::npos);
    result = registry.execute({"memory.get", R"({"memory_id":"memory-1"})"});
    assert(result.ok && result.output.find("optimistic version checks") != std::string::npos);
    result = registry.execute({"memory.inspect", "{}"});
    assert(result.ok && result.output.find("\"count\":1") != std::string::npos && result.output.find("fact") != std::string::npos);
    result = registry.execute({"memory.conflict_check", R"({"content":"The plan store uses optimistic version checks."})"});
    assert(result.ok && result.output.find("\"conflict\":true") != std::string::npos && result.output.find("memory-1") != std::string::npos);
    result = registry.execute({"memory.get", R"("memory-1")"});
    assert(result.ok && result.output.find("optimistic version checks") != std::string::npos);
    result = registry.execute({"plan.get", "{}"});
    assert(result.ok && result.output.find("plan-1") != std::string::npos);
    result = registry.execute({"memory.remember", "{}"});
    assert(!result.ok);

    common_tool_registry proposal_registry;
    common_native_tool_bindings proposal_bindings;
    proposal_bindings.memory_remember_proposal = [](const std::string &) { return common_tool_execution_result::success(R"({"decision":"accept"})"); };
    assert(common_register_native_tool_adapters(catalog, "memory", proposal_bindings, proposal_registry, adapters, error));
    assert(proposal_registry.is_policy_gated("memory.remember"));
    result = proposal_registry.execute({"memory.remember", R"({"kind":"fact","content":"verified"})"});
    assert(result.ok && result.output.find("accept") != std::string::npos);

    const auto repository = std::filesystem::temp_directory_path() /
        ("llama-agent-repository-tool-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
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
        if (input.find("\"url\":\"https://example.com/llama") == std::string::npos) {
            return common_tool_execution_result::failure("tool.web_fetch.unexpected_url", common_tool_failure_class::validation, false, "Unexpected fetch URL.", "unexpected fetch url");
        }
        auto fetched = common_tool_execution_result::success(R"({"url":"https://example.com/llama","final_url":"https://example.com/llama","status":200,"content_type":"text/html","title":"llama.cpp","text":"native tools","truncated":false})");
        fetched.http.provider = "web.fetch";
        fetched.http.operation = "fetch";
        fetched.http.method = "GET";
        fetched.http.requested_url = "https://example.com/llama?token=redacted";
        fetched.http.final_url = "https://example.com/llama";
        fetched.http.status = 200;
        fetched.http.content_type = "text/html";
        fetched.http.response_bytes = 128;
        fetched.http.duration_ms = 23;
        return fetched;
    };
    assert(common_register_native_tool_adapters(research_catalog, "research", repository_bindings, repository_registry, adapters, error));
    assert(repository_registry.contains("research.workspace.publish"));
    assert(!repository_registry.is_read_only("research.workspace.publish"));
    assert(repository_registry.is_policy_gated("research.workspace.publish"));
    result = repository_registry.execute({"repository.list", R"({"path":"src","depth":1})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos);
    result = repository_registry.execute({"repository.search", R"({"query":"needle","path":"src"})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos && result.output.find("needle") != std::string::npos);
    result = repository_registry.execute({"repository.search", R"({"query":"needle","path":"src/sample.txt"})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos && result.output.find("needle") != std::string::npos);
    result = repository_registry.execute({"repository.search", R"({"query":"needle","path":"src/missing.txt"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::not_found);
    result = repository_registry.execute({"repository.read", R"({"path":"src/sample.txt","start_line":2,"end_line":2})"});
    assert(result.ok && result.output.find("needle in a haystack") != std::string::npos);
    result = repository_registry.execute({"web.search", R"({"query":"llama","limit":1})"});
    assert(result.ok && result.output.find("https://example.com/llama") != std::string::npos);
    result = repository_registry.execute({"web.fetch", R"({"url":"https://example.com/llama","max_bytes":4096})"});
    assert(result.ok && result.output.find("\"status\":200") != std::string::npos && result.output.find("native tools") != std::string::npos &&
           result.http.provider == "web.fetch" && result.http.status == 200 &&
           result.http.response_bytes == 128 && result.http.duration_ms == 23);
    const auto fetch_schema = nlohmann::ordered_json::parse(research_catalog.find_definition("web.fetch")->input_schema_json);
    assert(fetch_schema["properties"]["max_bytes"]["default"] == 10 * 1024 * 1024);
    assert(repository_registry.validate({"web.fetch", R"({"url":"https://example.com/llama","max_bytes":10485760})"}, error));
    assert(!repository_registry.validate({"web.fetch", R"({"url":"https://example.com/llama","max_bytes":10485761})"}, error));
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
    std::filesystem::create_directories(repository / "long-status");
    for (size_t i = 0; i < 400; ++i) {
        std::ofstream file(repository / "long-status" / (std::to_string(i) + std::string(180, 'x')));
        file << "large bounded Git status fixture\n";
    }
    result = developer_registry.execute({"repository.changed_files", "{}"});
    assert(result.ok);
    const auto large_status = nlohmann::ordered_json::parse(result.output);
    assert(large_status["truncated"] == true && !large_status["files"].empty());
    assert(result.output.size() <= developer_catalog.find_definition("repository.changed_files")->max_result_bytes);

    std::filesystem::create_directories(repository / "datasets");
    { std::ofstream file(repository / "datasets" / "sample.csv"); file << "name,value\nalpha,1\nbeta,2\n"; }
    { std::ofstream file(repository / "datasets" / "duplicate-combo.csv"); file << "region,code,value\nnorth,A,1\nnorth,A,2\nnorth,B,3\n"; }
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
        {"document.create", 1, true, "{}"},
        {"document.open", 1, true, "{}"},
        {"document.import", 1, true, "{}"},
        {"document.save", 1, true, "{}"},
        {"document.propose_patch", 1, true, "{}"},
        {"document.apply_patch", 1, true, "{}"},
        {"openapi.connect", 1, true, "{}"},
        {"resource.read", 1, true, "{}"},
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
    foundation_resources.available_uris.insert("resource://uploads/sales.xlsx");
    foundation_resources.available_uris.insert("resource://session/long-text");
    test_data_store foundation_data;
    foundation_bindings.data_store = &foundation_data;
    foundation_bindings.dataset_from_resource = [](const std::string & resource, const std::string & operation) {
        if (resource != "resource://uploads/sales.csv" || operation != "inspect") {
            return common_tool_execution_result::failure(
                "tool.dataset.resource_unavailable", common_tool_failure_class::not_found,
                false, "CSV resource is unavailable.", "unexpected resource materialization request");
        }
        return common_tool_execution_result::success(R"({"dataset":"dataset://analysis/csv-sales"})");
    };
    foundation_bindings.openapi_connect = [](const std::string & input) {
        const auto arguments = nlohmann::json::parse(input, nullptr, false);
        if (!arguments.is_object() || arguments.value("spec_resource", std::string{}).empty()) {
            return common_tool_execution_result::failure(
                "tool.openapi.connect.invalid_arguments", common_tool_failure_class::validation,
                false, "OpenAPI connection requires a fetched spec resource.", "missing spec_resource");
        }
        return common_tool_execution_result::success(
            R"({"provider_id":"dynamic-test","operation_count":1,"available_next_turn":true})");
    };
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
    foundation_bindings.resource_runtime.turn_id = "turn-1";
    common_agent_artifact_profile report_profile;
    report_profile.id = "research";
    report_profile.revision = "v1";
    report_profile.title = "Research report";
    report_profile.purpose = "Grounded findings and explicit gaps.";
    report_profile.sections = {{"findings", "Findings", "Supported findings."},
        {"limitations", "Limitations", "Unresolved gaps."}};
    std::string report_profile_json;
    {
        nlohmann::ordered_json value = {{"id", report_profile.id}, {"revision", report_profile.revision},
            {"title", report_profile.title}, {"purpose", report_profile.purpose}, {"sections", nlohmann::ordered_json::array()}};
        for (const auto & section : report_profile.sections) value["sections"].push_back({
            {"id", section.id}, {"title", section.title}, {"purpose", section.purpose}});
        report_profile_json = value.dump();
    }
    common_plan_state report_plan;
    report_plan.id = "report-plan";
    report_plan.namespace_id = "test";
    report_plan.session_id = "session-1";
    report_plan.scope = common_plan_scope::session;
    report_plan.goal = "Compose a source-grounded report";
    report_plan.success_criteria = "Preserve gaps and provenance.";
    report_plan.version = 1;
    common_plan_workflow_definition report_workflow;
    report_workflow.workflow_ref = "workflow://artifact/research-report";
    report_workflow.workflow_revision = "v1";
    common_plan_workflow_transition report_transition;
    report_transition.id = "compose-findings";
    report_transition.kind = common_plan_workflow_transition_kind::artifact_compose;
    report_transition.artifact_profile_ref = report_profile.id;
    report_transition.artifact_profile_revision = report_profile.revision;
    report_transition.artifact_profile_json = report_profile_json;
    report_transition.artifact_section_id = "findings";
    report_workflow.transitions.push_back(report_transition);
    report_plan.workflow_definition = report_workflow;
    common_agent_artifact_snapshot empty_report_snapshot;
    assert(common_agent_artifact_snapshot_from_plan(report_plan, report_profile, empty_report_snapshot, error));
    std::string empty_report_snapshot_json;
    assert(common_agent_artifact_snapshot_to_json(empty_report_snapshot, empty_report_snapshot_json, error));
    report_plan.observations.push_back({"artifact-snapshot:report-plan", "artifact_snapshot",
        empty_report_snapshot_json, 1.0f, {}, {}, 1, {}});
    report_plan.observations.push_back({"artifact-section:findings", "artifact_section",
        nlohmann::json{{"id","findings"},{"title","Findings"},{"snapshot_fingerprint",empty_report_snapshot.fingerprint},{"blocks",nlohmann::json::array({{{"id","findings:1"},{"kind","observation"},{"text","No source evidence was supplied."},{"validation","gap"},{"validation_note","empty snapshot"}}})}}.dump(),
        1.0f, {}, {}, 2, {}});
    report_plan.observations.push_back({"artifact-section:limitations", "artifact_section",
        nlohmann::json{{"id","limitations"},{"title","Limitations"},{"snapshot_fingerprint",empty_report_snapshot.fingerprint},{"blocks",nlohmann::json::array({{{"id","limitations:1"},{"kind","observation"},{"text","Coverage is incomplete."},{"validation","gap"},{"validation_note","empty snapshot"}}})}}.dump(),
        1.0f, {}, {}, 3, {}});
    assert(plans.create(report_plan, error));
    foundation_bindings.plan_store = &plans;
    std::string active_report_plan_id = report_plan.id;
    foundation_bindings.plan_id = &active_report_plan_id;
    bool docx_processing_called = false;
    agent_resource_processing_binding_request docx_processing_request;
    foundation_bindings.resource_processing_provider_factory = [&](const auto &) {
        return std::make_shared<test_docx_processing_provider>(foundation_resources,
            docx_processing_called, docx_processing_request);
    };
    foundation_bindings.diagnostics_symbol = [](const std::string &) {
        return common_tool_execution_result::success(
            R"({"backend":"clangd","semantic":true,"definitions":[],"count":0,"truncated":false})");
    };
    assert(common_register_native_tool_adapters(foundation_catalog, foundation_profile.id, foundation_bindings, foundation_registry, adapters, error));
    {
        result=foundation_registry.execute({"document.create",nlohmann::json{{"title","Working notes"},{"operation_id","create-document-1"}}.dump()});
        assert(result.ok);
        const auto created=nlohmann::json::parse(result.output);
        assert(created.value("revision",uint64_t(0))==1&&created.value("title",std::string{})=="Working notes"&&!created.value("reused",true));
        const auto created_uri=created.value("document_uri",std::string{});
        result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",created_uri}}.dump()});
        assert(result.ok);
        auto opened=nlohmann::json::parse(result.output);
        assert(opened.value("kind",std::string{})=="document"&&opened.value("title",std::string{})=="Working notes");
        const auto empty_markdown=opened.value("markdown",std::string{});
        auto renamed=nlohmann::json{{"document_uri",created_uri},{"expected_revision",1},{"title","Research draft"},
            {"markdown",empty_markdown},{"operation_id","create-document-rename-1"}};
        result=foundation_registry.execute({"document.save",renamed.dump()});
        assert(result.ok&&nlohmann::json::parse(result.output).value("revision",uint64_t(0))==2);
        const auto renamed_uri=nlohmann::json::parse(result.output).value("document_uri",std::string{});
        result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",renamed_uri}}.dump()});
        assert(result.ok);
        opened=nlohmann::json::parse(result.output);
        assert(opened.value("title",std::string{})=="Research draft"&&opened.value("markdown",std::string{})==empty_markdown);
        result=foundation_registry.execute({"document.create",nlohmann::json{{"title","Working notes"},{"operation_id","create-document-1"}}.dump()});
        assert(result.ok);
        const auto retried=nlohmann::json::parse(result.output);
        assert(retried.value("reused",false)&&retried.value("document_uri",std::string{})==created_uri&&retried.value("revision",uint64_t(0))==1);
        result=foundation_registry.execute({"document.create",nlohmann::json{{"title","Different title"},{"operation_id","create-document-1"}}.dump()});
        assert(!result.ok&&result.failure_class==common_tool_failure_class::conflict);
        const auto collision_create=foundation_registry.execute({"document.create",nlohmann::json{{"title","Collision test"},{"operation_id","create-collision-document"}}.dump()});
        assert(collision_create.ok);
        const auto collision_uri=nlohmann::json::parse(collision_create.output).value("document_uri",std::string{});
        const auto collision_save=nlohmann::json{{"document_uri",collision_uri},{"expected_revision",1},{"title","First\nSecond"},
            {"markdown","Body"},{"operation_id","title-boundary-test"}};
        result=foundation_registry.execute({"document.save",collision_save.dump()});
        assert(result.ok);
        auto different_input=collision_save;different_input["title"]="First";different_input["markdown"]="Second\nBody";
        result=foundation_registry.execute({"document.save",different_input.dump()});
        assert(!result.ok&&result.failure_class==common_tool_failure_class::conflict);
        auto unframed_input=collision_save;unframed_input.erase("title");
        unframed_input["markdown"]=nlohmann::json{{"title","First\nSecond"},{"markdown","Body"}}.dump();
        result=foundation_registry.execute({"document.save",unframed_input.dump()});
        assert(!result.ok&&result.failure_class==common_tool_failure_class::conflict);
        bool legacy_operation_found=false;
        for(auto & resource:foundation_resources.contents){
            auto operation_record=nlohmann::json::parse(resource.second,nullptr,false);
            if(!operation_record.is_object()||operation_record.value("operation_id",std::string{})!="title-boundary-test"||!operation_record.contains("input_fingerprint"))continue;
            const std::string legacy_input="First\nSecond\nBody";
            operation_record.erase("input_format");
            operation_record["input_fingerprint"]="sha256:"+hash_sha256_hex(legacy_input.data(),legacy_input.size());
            resource.second=operation_record.dump();legacy_operation_found=true;
        }
        assert(legacy_operation_found);
        result=foundation_registry.execute({"document.save",collision_save.dump()});
        assert(result.ok);
        result=foundation_registry.execute({"document.save",different_input.dump()});
        assert(!result.ok&&result.failure_class==common_tool_failure_class::conflict);
        unframed_input["markdown"]="First\nSecond\nBody";
        result=foundation_registry.execute({"document.save",unframed_input.dump()});
        assert(!result.ok&&result.failure_class==common_tool_failure_class::conflict);
    }
    {
        agent_resource_descriptor uploaded;
        uploaded.uri = "resource://session/notes.txt";
        uploaded.name = "notes.txt";
        uploaded.mime_type = "text/plain";
        uploaded.scope = common_runtime_resource_scope::session;
        uploaded.size_bytes = 35;
        foundation_resources.resources[uploaded.uri] = uploaded;
        foundation_resources.available_uris.insert(uploaded.uri);
        foundation_resources.contents[uploaded.uri] = "# literal heading\n\n*literal text*\n";
        result = foundation_registry.execute({"document.import", nlohmann::json{{"source_uri",uploaded.uri},{"operation_id","import-notes-1"}}.dump()});
        assert(result.ok);
        const auto imported = nlohmann::json::parse(result.output);
        assert(imported.value("revision",uint64_t(0)) == 1 && imported.value("original_uri",std::string{}) == uploaded.uri);
        const auto imported_uri = imported.value("document_uri",std::string{});
        result = foundation_registry.execute({"document.import", nlohmann::json{{"source_uri",uploaded.uri},{"operation_id","import-notes-1"}}.dump()});
        assert(result.ok && nlohmann::json::parse(result.output).value("reused",false));
        agent_resource_descriptor other_upload=uploaded;
        other_upload.uri="resource://session/other.txt";other_upload.name="other.txt";
        foundation_resources.resources[other_upload.uri]=other_upload;foundation_resources.available_uris.insert(other_upload.uri);
        foundation_resources.contents[other_upload.uri]="Different source.";
        result=foundation_registry.execute({"document.import",nlohmann::json{{"source_uri",other_upload.uri},{"operation_id","import-notes-1"}}.dump()});
        assert(!result.ok&&result.failure_class==common_tool_failure_class::conflict);
        result = foundation_registry.execute({"document.open", nlohmann::json{{"document_uri",imported_uri}}.dump()});
        assert(result.ok);
        const auto opened = nlohmann::json::parse(result.output);
        assert(opened.value("kind",std::string{}) == "document");
        assert(opened.value("markdown",std::string{}).find("```text") != std::string::npos);
        assert(opened.value("markdown",std::string{}).find("# literal heading") != std::string::npos);
        auto retry = foundation_registry.execute({"document.import", nlohmann::json{{"source_uri",uploaded.uri},{"operation_id","import-notes-2"}}.dump()});
        assert(retry.ok && nlohmann::json::parse(retry.output).value("reused",false));
        auto edited = opened.value("markdown",std::string{}) + "\nA user paragraph.\n";
        result = foundation_registry.execute({"document.save",nlohmann::json{{"document_uri",imported_uri},{"expected_revision",1},
            {"markdown",edited},{"operation_id","import-notes-save-1"}}.dump()});
        assert(result.ok && nlohmann::json::parse(result.output).value("revision",uint64_t(0)) == 2);
        const auto saved_uri = nlohmann::json::parse(result.output).value("document_uri",std::string{});
        result = foundation_registry.execute({"artifact.export",nlohmann::json{{"source_document",saved_uri},{"format","markdown"}}.dump()});
        assert(result.ok && nlohmann::json::parse(result.output).value("document_kind",std::string{}) == "document");
        const auto generic_export = nlohmann::json::parse(result.output);
        const auto generic_markdown = foundation_resources.contents.at(generic_export.value("markdown_resource",std::string{}));
        assert(generic_markdown.find("Coverage incomplete") == std::string::npos);
        assert(generic_markdown.find("A user paragraph.") != std::string::npos);
        result=foundation_registry.execute({"artifact.export",nlohmann::json{{"source_document",saved_uri},{"format","docx"}}.dump()});
        assert(result.ok);
        const auto docx_export=nlohmann::json::parse(result.output);
        assert(foundation_resources.contents.at(docx_export["resource"].get<std::string>())=="test-docx-bytes");
        const auto before_docx_retry=foundation_resources.requests.size();
        result=foundation_registry.execute({"artifact.export",nlohmann::json{{"source_document",saved_uri},{"format","docx"}}.dump()});
        assert(result.ok && foundation_resources.requests.size()==before_docx_retry);

        auto install_upload = [&](const std::string & name, const std::string & content) {
            auto descriptor=uploaded;descriptor.uri="resource://session/"+name;descriptor.name=name;
            descriptor.size_bytes=content.size();
            foundation_resources.resources[descriptor.uri]=descriptor;
            foundation_resources.contents[descriptor.uri]=content;
            return descriptor.uri;
        };
        const auto markdown_source=install_upload("actual.md","# A heading\n\nA paragraph.");
        result=foundation_registry.execute({"document.import",nlohmann::json{{"source_uri",markdown_source},{"operation_id","import-markdown"}}.dump()});
        assert(result.ok);
        auto markdown_document_uri=nlohmann::json::parse(result.output)["document_uri"];
        result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",markdown_document_uri}}.dump()});
        assert(result.ok && nlohmann::json::parse(result.output)["markdown"].get<std::string>().find("# A heading")==0);
        const auto empty_source=install_upload("empty.md","");
        result=foundation_registry.execute({"document.import",nlohmann::json{{"source_uri",empty_source},{"operation_id","import-empty"}}.dump()});
        assert(result.ok);
        const auto empty_document_uri=nlohmann::json::parse(result.output)["document_uri"];
        result=foundation_registry.execute({"document.propose_patch",nlohmann::json{{"document_uri",empty_document_uri},{"expected_revision",1},
            {"operation","append"},{"text","First paragraph.\n\nSecond paragraph."},{"summary","Add two paragraphs."}}.dump()});
        assert(result.ok);
        const auto empty_proposal=nlohmann::json::parse(result.output)["proposal_uri"];
        result=foundation_registry.execute({"document.apply_patch",nlohmann::json{{"document_uri",empty_document_uri},{"expected_revision",1},
            {"proposal_uri",empty_proposal},{"operation_id","append-to-empty"}}.dump()});
        assert(result.ok && nlohmann::json::parse(result.output)["revision"]==2);
        const auto empty_text=install_upload("empty.txt","");
        result=foundation_registry.execute({"document.import",nlohmann::json{{"source_uri",empty_text},{"operation_id","import-empty-txt"}}.dump()});
        assert(result.ok);
        result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",nlohmann::json::parse(result.output)["document_uri"]}}.dump()});
        assert(result.ok && nlohmann::json::parse(result.output)["markdown"]=="");
        for(const auto * failed_mime : {"application/vnd.llama-agent.document-operation+json","application/vnd.llama-agent.document-head+json"}) {
            const std::string suffix=std::string(failed_mime).find("head")!=std::string::npos ? "head" : "operation";
            const auto source_uri=install_upload("interrupt-"+suffix+".md","Restorable document.");
            const auto args=nlohmann::json{{"source_uri",source_uri},{"operation_id","import-interrupt-"+suffix}};
            foundation_resources.fail_mime=failed_mime;
            result=foundation_registry.execute({"document.import",args.dump()});assert(!result.ok);
            foundation_resources.fail_mime.clear();
            result=foundation_registry.execute({"document.import",args.dump()});assert(result.ok);
            const auto recovered=nlohmann::json::parse(result.output);
            bool published=false;std::string head_uri;
            for(const auto & entry : foundation_resources.resources)
                if(entry.second.metadata.processing_cache_key=="document-head:"+recovered["document_id"].get<std::string>()){published=true;head_uri=entry.first;}
            assert(published);
            const auto head_contents=foundation_resources.contents.at(head_uri);
            foundation_resources.contents[head_uri]="{}";
            result=foundation_registry.execute({"document.import",args.dump()});
            assert(!result.ok && foundation_resources.contents.at(head_uri)=="{}");
            foundation_resources.contents[head_uri]=head_contents;
        }

        agent_resource_descriptor uploaded_docx;
        uploaded_docx.uri = "resource://session/word.docx";
        uploaded_docx.name = "word.docx";
        uploaded_docx.mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
        uploaded_docx.scope = common_runtime_resource_scope::session;
        uploaded_docx.size_bytes = 20;
        foundation_resources.resources[uploaded_docx.uri] = uploaded_docx;
        foundation_resources.available_uris.insert(uploaded_docx.uri);
        foundation_resources.contents[uploaded_docx.uri] = "docx bytes";
        result = foundation_registry.execute({"document.import",nlohmann::json{{"source_uri",uploaded_docx.uri},{"operation_id","import-docx-1"}}.dump()});
        assert(result.ok && docx_processing_request.target_representation == "markdown");
        const auto imported_docx = nlohmann::json::parse(result.output);
        result=foundation_registry.execute({"document.import",nlohmann::json{{"source_uri",uploaded_docx.uri},{"operation_id","import-docx-1"}}.dump()});
        assert(result.ok && nlohmann::json::parse(result.output)["limitations"]==imported_docx["limitations"] && !imported_docx["limitations"].empty());
        result = foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",imported_docx.value("document_uri",std::string{})}}.dump()});
        assert(result.ok && nlohmann::json::parse(result.output).value("markdown",std::string{}).find("| Name | Value |") != std::string::npos);
        uploaded_docx.uri="resource://session/empty.docx";uploaded_docx.name="empty.docx";
        foundation_resources.resources[uploaded_docx.uri]=uploaded_docx;
        foundation_resources.contents[uploaded_docx.uri]="empty docx bytes";
        result=foundation_registry.execute({"document.import",nlohmann::json{{"source_uri",uploaded_docx.uri},{"operation_id","import-empty-docx"}}.dump()});
        assert(result.ok);
        const auto empty_docx_uri=nlohmann::json::parse(result.output)["document_uri"];
        result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",empty_docx_uri}}.dump()});
        assert(result.ok && nlohmann::json::parse(result.output)["markdown"]=="");
    }
    result = foundation_registry.execute({"openapi.connect", R"({"spec_resource":"resource://session/openapi"})"});
    assert(result.ok && result.output.find("dynamic-test") != std::string::npos &&
           result.output.find("available_next_turn") != std::string::npos);
    result = foundation_registry.execute({"resource.read", R"({"uri":"resource://session/long-text","representation":"text"})"});
    assert(result.ok && result.resource_refs.size() == 1);
    assert(result.resource_refs.front().uri == "resource://session/long-text");
    assert(result.resource_refs.front().size_bytes == 16000);
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
    result = foundation_registry.execute({"diagnostics.format", R"({"output":"src/sample.cpp:1:10: warning: formatting differs\n"})"});
    assert(result.ok && result.output.find("\"formatted\":false") != std::string::npos && result.output.find("formatting differs") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.include_graph", R"({"output":"a.h -> b.h\nb.h -> c.h\n"})"});
    assert(result.ok && result.output.find("a.h") != std::string::npos && result.output.find("c.h") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.include_graph", R"({"output":". a.h\n.. b.h\n... c.h\n. d.h\n"})"});
    assert(result.ok && result.output.find("\"from\":\"a.h\",\"to\":\"b.h\"") != std::string::npos && result.output.find("\"from\":\"b.h\",\"to\":\"c.h\"") != std::string::npos);
    result = foundation_registry.execute({"diagnostics.include_graph", R"({"output":"source.o: source.cpp header.h\n"})"});
    assert(!result.ok && result.failure_code == "tool.diagnostics.include_graph.unrecognized_output");
    result = foundation_registry.execute({"dataset.list", "{}"});
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
    result = foundation_registry.execute({"dataset.sample", R"({"resource":"resource://uploads/sales.csv","rows":1})"});
    assert(result.ok && foundation_data.last_operation == "data.query");
    const auto resource_sample_query = nlohmann::json::parse(foundation_data.last_request);
    assert(resource_sample_query["dataset"] == "dataset://analysis/csv-sales" &&
        resource_sample_query["limit"] == 1 && resource_sample_query["max_result_rows"] == 1);
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
           result.raw_diagnostic.find("choose one of: Sales (dataset://analysis/sales)") != std::string::npos);
    result = foundation_registry.execute({"dataset.inspect", R"({"dataset":"dataset://missing"})"});
    assert(!result.ok && result.failure_code == "tool.dataset.unavailable");
    const std::string valid_rules = R"([{"type":"not_null","column":"name"},{"type":"unique","column":"name"}])";
    result = foundation_registry.execute({"dataset.validate", std::string(R"({"path":"datasets/sample.csv","rules":)") + valid_rules + "}"});
    assert(result.ok && result.output.find("\"valid\":true") != std::string::npos &&
           result.output.find("\"rows_checked\":2") != std::string::npos);
    result = foundation_registry.execute({"dataset.validate", R"({"path":"datasets/sample.csv","rules":[{"type":"numeric_range","column":"value","min":1,"max":2},{"type":"allowed_values","column":"name","values":["alpha","beta"]},{"type":"unique_combination","columns":["name","value"]}]})"});
    assert(result.ok && result.output.find("\"valid\":true") != std::string::npos);
    result = foundation_registry.execute({"dataset.validate", R"({"path":"datasets/sample.csv","rules":[{"type":"numeric_range","column":"value","min":2}]})"});
    assert(result.ok && result.output.find("\"valid\":false") != std::string::npos &&
           result.output.find("numeric_range:value") != std::string::npos);
    result = foundation_registry.execute({"dataset.validate", R"({"path":"datasets/sample.csv","rules":[{"type":"allowed_values","column":"name","values":["alpha"]}]})"});
    assert(result.ok && result.output.find("\"valid\":false") != std::string::npos &&
           result.output.find("allowed_values:name") != std::string::npos);
    result = foundation_registry.execute({"dataset.validate", R"({"path":"datasets/duplicate-combo.csv","rules":[{"type":"unique_combination","columns":["region","code"]}]})"});
    assert(result.ok && result.output.find("\"valid\":false") != std::string::npos &&
           result.output.find("unique_combination:region,code") != std::string::npos);
    result = foundation_registry.execute({"dataset.validate", std::string(R"({"dataset":"dataset://analysis/csv-sales","rules":)") + valid_rules + "}"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           result.output.find("\"valid\":true") != std::string::npos);
    result = foundation_registry.execute({"dataset.validate", std::string(R"({"dataset":"dataset://analysis/sales","rules":)") + valid_rules + "}"});
    assert(!result.ok && result.failure_code == "tool.dataset.validate.unsupported_format");
    result = foundation_registry.execute({"dataset.validate", std::string(R"({"resource":"resource://uploads/sales.csv","rules":)") + valid_rules + "}"});
    assert(result.ok && result.output.find("\"valid\":true") != std::string::npos);
    result = foundation_registry.execute({"dataset.validate", R"({"path":"datasets/sample.csv","dataset":"dataset://analysis/sales","rules":[{"type":"unique","column":"name"}]})"});
    assert(!result.ok);
    result = foundation_registry.execute({"dataset.validate", R"({"rules":[{"type":"unique","column":"name"}]})"});
    assert(!result.ok);
    result = foundation_registry.execute({"dataset.validate", R"({"path":"../outside.csv","rules":[{"type":"unique","column":"name"}]})"});
    assert(!result.ok);
    result = foundation_registry.execute({"dataset.validate", R"({"path":"datasets/sample.csv","rules":[{"type":"regex","column":"name"}]})"});
    assert(!result.ok);
    result = foundation_registry.execute({"dataset.validate", R"({"path":"datasets/sample.csv","rules":[{"type":"unique"}]})"});
    assert(!result.ok);
    result = foundation_registry.execute({"dataset.validate", R"({"path":"datasets/sample.csv","rules":[{"type":"unique","column":"absent"}]})"});
    assert(!result.ok && result.failure_code == "tool.dataset.validate.invalid_rule");
    for (const auto * invalid : {
            R"({"type":"numeric_range","column":"value","min":3,"max":1})",
            R"({"type":"allowed_values","column":"name","values":[]})",
            R"({"type":"unique_combination","columns":["name","absent"]})",
            R"({"type":"numeric_range","column":"value","min":1,"unexpected":true})"}) {
        result = foundation_registry.execute({"dataset.validate",
            std::string(R"({"path":"datasets/sample.csv","rules":[)") + invalid + "]}"});
        assert(!result.ok && (result.failure_code == "tool.dataset.validate.invalid_rule" ||
            result.failure_code == "tool.invalid_arguments"));
    }
    common_registered_tool throwing_adapter;
    throwing_adapter.name = "throwing_adapter";
    throwing_adapter.version = 1;
    throwing_adapter.executor_id = "test.throwing_adapter";
    throwing_adapter.arguments_schema = R"({"type":"object"})";
    throwing_adapter.handler = [](const std::string &) -> common_tool_execution_result {
        throw std::runtime_error("adapter failure fixture");
    };
    assert(foundation_registry.register_tool(std::move(throwing_adapter), error));
    result = foundation_registry.execute({"throwing_adapter", "{}"});
    assert(!result.ok && result.failure_code == "tool.executor_exception" &&
           result.raw_diagnostic == "adapter failure fixture");
    const auto saved_descriptor = foundation_data.stored_descriptor;
    common_tool_catalog data_catalog;
    common_tool_bootstrap_result data_bootstrap;
    if (!data_catalog.bootstrap("analysis", data_bootstrap, error)) return 1;
    common_tool_registry data_registry;
    common_tool_adapter_result data_adapters;
    if (!common_register_native_tool_adapters(
            data_catalog, "analysis", foundation_bindings, data_registry,
            data_adapters, error) || !data_registry.contains("data.join")) return 1;
    assert(data_registry.contains("artifact.export"));
    assert(!data_registry.is_read_only("artifact.export"));
    assert(data_registry.is_policy_gated("artifact.export"));
    std::vector<common_chat_tool> data_chat_tools;
    assert(common_tool_profile_to_chat_tools(
        data_catalog, "analysis", data_registry, data_chat_tools, error));
    assert(std::any_of(data_chat_tools.begin(), data_chat_tools.end(), [](const auto & tool) {
        return tool.name == "artifact.export";
    }));
    common_chat_msg unapproved_export;
    unapproved_export.role = "assistant";
    unapproved_export.tool_calls.push_back({
        "artifact.export", R"({"name":"unapproved.txt","content":"not authorized"})", ""});
    common_tool_chat_dispatch_result unapproved_dispatch;
    assert(common_tool_dispatch_chat_calls(
        unapproved_export, data_registry, 1, unapproved_dispatch, error));
    assert(unapproved_dispatch.executed == 0);
    assert(data_registry.contains("artifact.export"));
    assert(!data_registry.is_read_only("artifact.export"));
    assert(data_registry.is_policy_gated("artifact.export"));
    foundation_data.stored_descriptor.origin.kind = "derived";
    foundation_data.stored_descriptor.ref.uri = "dataset://agent/turn/turn-1/step-1";
    result = foundation_registry.execute({"data.query", R"({"dataset":"dataset://agent/turn/turn-1/step-1"})"});
    assert(result.ok && foundation_data.last_operation == "data.query");
    foundation_data.stored_descriptor.ref.uri = "dataset://agent/turn/turn-2/step-1";
    result = foundation_registry.execute({"data.query", R"({"dataset":"dataset://agent/turn/turn-2/step-1"})"});
    assert(!result.ok && result.failure_code == "tool.dataset.out_of_scope");
    result = data_registry.execute({"data.join", R"({"left":"dataset://agent/turn/turn-2/step-1","right":"dataset://agent/turn/turn-2/step-1","on":[{"left":"id","right":"id"}]})"});
    if (result.ok || result.failure_code != "tool.dataset.out_of_scope") return 1;
    foundation_data.stored_descriptor = saved_descriptor;
    foundation_data.stored_descriptor.ref.source_representation =
        (repository / "datasets" / "sample.csv").string();
    result = data_registry.execute({"data.join", R"({"left":"dataset://analysis/sales","right":"dataset://analysis/sales","on":[{"left":"name","right":"name"}]})"});
    if (!result.ok || foundation_data.last_operation != "data.join") return 1;
    foundation_data.stored_descriptor = saved_descriptor;
    result = foundation_registry.execute({"data.query", R"({"dataset":"dataset://tool-events","limit":10})"});
    assert(result.ok && foundation_data.last_operation == "data.query");
    result = foundation_registry.execute({"data.query", R"({"dataset":"dataset://tool-events","where":{"status":"failed"}})"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           foundation_data.last_request.find(R"("where":[{"field":"status","operator":"=","value":"failed"}])") != std::string::npos);
    result = foundation_registry.execute({"data.query", R"({"dataset":"dataset://tool-events","where":{"title":{"contains":"GDP"}}})"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           foundation_data.last_request.find(R"("where":[{"field":"title","operator":"contains","value":"GDP"}])") != std::string::npos);
    result = foundation_registry.execute({"data.query", R"({"dataset":"dataset://tool-events","where":[{"field":"title","operator":"contains","value":"GDP"}]})"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           foundation_data.last_request.find(R"("operator":"contains","value":"GDP")") != std::string::npos);
    result = foundation_registry.execute({"data.query", R"({"dataset":"dataset://tool-events","where":"status -eq \"failed\" and attempt -gt 1"})"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           foundation_data.last_request.find(R"("field":"status")") != std::string::npos &&
           foundation_data.last_request.find(R"("operator":"=","value":"failed")") != std::string::npos &&
           foundation_data.last_request.find(R"("field":"attempt","operator":">","value":1)") != std::string::npos);
    result = foundation_registry.execute({"data.filter", R"({"dataset":"dataset://tool-events","conditions":[{"field":"status","operator":"=","value":"failed"}]})"});
    assert(result.ok && foundation_data.last_operation == "data.filter");
    result = foundation_registry.execute({"data.filter", R"({"dataset":"dataset://tool-events","conditions":{"status":{"in":["failed","blocked"]}}})"});
    assert(result.ok && foundation_data.last_operation == "data.filter" &&
           foundation_data.last_request.find(R"("operator":"in","value":["failed","blocked"])") != std::string::npos);
    result = foundation_registry.execute({"data.aggregate", R"({"dataset":"dataset://tool-events","measures":[{"function":"count","column":"*"}]})"});
    assert(result.ok && foundation_data.last_operation == "data.aggregate");
    result = foundation_registry.execute({"data.join", R"({"left":"dataset://analysis/sales","right":"dataset://analysis/sales","on":[{"left":"id","right":"id"}]})"});
    assert(result.ok && foundation_data.last_operation == "data.join");
    result = foundation_registry.execute({"data.transform", R"({"dataset":"dataset://analysis/sales","operations":[{"type":"rename","from":"x","to":"y"}]})"});
    assert(result.ok && foundation_data.last_operation == "data.transform");
    result = foundation_registry.execute({"statistics.describe", R"({"dataset":"dataset://analysis/sales","columns":["value"]})"});
    assert(result.ok && foundation_data.last_operation == "statistics.describe");
    result = foundation_registry.execute({"statistics.describe", R"({"dataset":"dataset://analysis/sales","column":"value","group_by":"region"})"});
    assert(result.ok && foundation_data.last_operation == "statistics.describe" &&
           foundation_data.last_request.find(R"("columns":["value"])" ) != std::string::npos &&
           foundation_data.last_request.find(R"("group_by":["region"])" ) != std::string::npos);
    result = foundation_registry.execute({"statistics.describe", R"({"dataset":"dataset://analysis/sales","columns":"value"})"});
    assert(result.ok && foundation_data.last_operation == "statistics.describe" &&
           foundation_data.last_request.find(R"("columns":["value"])") != std::string::npos);
    result = foundation_registry.execute({"statistics.outliers", R"({"dataset":"dataset://analysis/sales","column":"value","group_by":"region"})"});
    assert(result.ok && foundation_data.last_operation == "statistics.outliers" &&
           foundation_data.last_request.find(R"("columns":["value"])" ) != std::string::npos &&
           foundation_data.last_request.find(R"("group_by":["region"])" ) != std::string::npos);
    result = foundation_registry.execute({"statistics.value_counts", R"({"dataset":"dataset://analysis/sales","column":"region","limit":5})"});
    assert(result.ok && foundation_data.last_operation == "statistics.value_counts");
    result = foundation_registry.execute({"data.aggregate", R"({"dataset":"dataset://analysis/sales","group_by":"region","sum":"amount"})"});
    assert(result.ok && foundation_data.last_operation == "data.aggregate" &&
           foundation_data.last_request.find(R"("group_by":["region"])" ) != std::string::npos &&
           foundation_data.last_request.find(R"("function":"sum","column":"amount")" ) != std::string::npos);
    result = foundation_registry.execute({"data.aggregate", R"json({"dataset":"dataset://analysis/sales","select":"sum(amount)"})json"});
    assert(result.ok && foundation_data.last_operation == "data.aggregate" &&
           foundation_data.last_request.find(R"("measures":[{"function":"sum","column":"amount"}])") != std::string::npos &&
           foundation_data.last_request.find(R"("select")") == std::string::npos);
    result = foundation_registry.execute({"data.join", R"({"left":"dataset://analysis/sales","right":"dataset://analysis/sales","on":"id"})"});
    assert(result.ok && foundation_data.last_operation == "data.join" &&
           foundation_data.last_request.find(R"("left":"id","right":"id")" ) != std::string::npos);
    result = foundation_registry.execute({"dataset.sample", R"({"dataset":"dataset://analysis/sales","limit":1})"});
    assert(result.ok && foundation_data.last_operation == "data.query" &&
           foundation_data.last_request.find(R"("limit":1)" ) != std::string::npos);
    const auto writes_before_text_export = foundation_resources.requests.size();
    result = foundation_registry.execute({"artifact.export", R"({"name":"summary.txt","content":"artifact result"})"});
    assert(result.ok && result.output.find("resource://") != std::string::npos);
    result = foundation_registry.execute({"artifact.export", R"({"format":"markdown","name":"grounded-report"})"});
    assert(result.ok);
    const auto report_export = nlohmann::json::parse(result.output);
    assert(!report_export.value("resource", std::string{}).empty() &&
        !report_export.value("report_resource", std::string{}).empty() &&
        !report_export.value("snapshot_resource", std::string{}).empty() &&
        !report_export.value("markdown_resource", std::string{}).empty() &&
        !report_export.value("manifest_resource", std::string{}).empty() &&
        report_export.value("report_id", std::string{}) == "report:report-plan" &&
        report_export.value("revision", uint64_t(0)) == 1 &&
        report_export.contains("coverage_complete") && !report_export["coverage_complete"].get<bool>());
    assert(foundation_resources.requests.size() == writes_before_text_export + 5 &&
        foundation_resources.requests[writes_before_text_export + 1].mime_type == "application/json" &&
        foundation_resources.requests[writes_before_text_export + 2].mime_type == "application/vnd.llama-agent.report+json" &&
        foundation_resources.requests[writes_before_text_export + 3].mime_type == "text/markdown" &&
        foundation_resources.requests[writes_before_text_export + 3].text.find("Coverage incomplete") != std::string::npos &&
        foundation_resources.requests[writes_before_text_export + 4].text.find(report_export.value("export_id", std::string{})) != std::string::npos);
    const auto writes_before_retry=foundation_resources.requests.size();
    // Simulate a publication predating renderer versioning. Retry and format
    // changes must preserve that stored document rather than rebuilding it.
    const auto published_report_uri=report_export.value("report_resource",std::string{});
    auto legacy_report=nlohmann::json::parse(foundation_resources.contents.at(published_report_uri));
    legacy_report.erase("markdown_renderer_version");
    foundation_resources.contents[published_report_uri]=legacy_report.dump();
    result = foundation_registry.execute({"artifact.export", R"({"format":"markdown","name":"grounded-report"})"});
    assert(result.ok);
    const auto retried_report=nlohmann::json::parse(result.output);
    assert(retried_report.value("manifest_resource",std::string{})==report_export.value("manifest_resource",std::string{}));
    assert(foundation_resources.requests.size()==writes_before_retry);
    result = foundation_registry.execute({"artifact.export", R"({"format":"docx","name":"grounded-report"})"});
    assert(result.ok);
    assert(docx_processing_called);
    assert(docx_processing_request.source_uri == retried_report.value("markdown_resource", std::string{}));
    assert(docx_processing_request.target_representation == "docx");
    assert(docx_processing_request.target_media_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
    assert(docx_processing_request.purpose == agent_resource_processing_purpose::artifact_generation);
    const auto docx_export = nlohmann::json::parse(result.output);
    assert(docx_export.value("format", std::string{}) == "docx" &&
        docx_export.value("mime_type", std::string{}) == "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
    assert(docx_export.value("report_resource",std::string{})==published_report_uri);
    assert(foundation_resources.contents.at(published_report_uri)==legacy_report.dump());
    assert(std::any_of(foundation_resources.requests.begin(), foundation_resources.requests.end(), [](const auto & request) {
        return request.source_tool == "artifact.export" &&
            request.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" &&
            request.text == "test-docx-bytes";
    }));
    const auto document_uri=report_export.value("report_resource",std::string{});
    result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",document_uri}}.dump()});
    assert(result.ok);
    auto opened_document=nlohmann::json::parse(result.output);
    assert(opened_document.value("revision",uint64_t(0))==1&&opened_document.contains("markdown"));
    assert(opened_document.contains("rendered_markdown"));
    assert(!opened_document["rendered_markdown"].get<std::string>().empty());
    assert(opened_document["document"]["id"] == report_export["report_id"]);
    auto edited_markdown=opened_document.value("markdown",std::string{});
    edited_markdown+="\n## User edits\n\nUser-edited report text.\n";
    const auto save_args=nlohmann::json{{"document_uri",document_uri},{"expected_revision",1},
        {"markdown",edited_markdown},{"operation_id","adapter-test-save-1"}};
    result=foundation_registry.execute({"document.save",save_args.dump()});
    assert(result.ok);
    const auto saved_document=nlohmann::json::parse(result.output);
    assert(saved_document.value("revision",uint64_t(0))==2&&!saved_document.value("reused",false));
    assert(saved_document["diff"].value("base_revision",uint64_t(0))==1&&
        saved_document["diff"]["blocks"].size()>=1&&
        saved_document["diff"]["blocks"][0].value("new_text",std::string{}).find("User-edited")!=std::string::npos);
    result=foundation_registry.execute({"document.save",save_args.dump()});
    assert(result.ok&&nlohmann::json::parse(result.output).value("reused",false));
    auto conflict_args=save_args;conflict_args["operation_id"]="adapter-test-save-conflict";
    result=foundation_registry.execute({"document.save",conflict_args.dump()});
    assert(!result.ok&&result.failure_class==common_tool_failure_class::conflict);
    // An unchanged legacy revision is a no-op, but still reserves its operation
    // ID so a retry cannot silently substitute a different payload.
    auto install_editor_fixture=[&](const std::string & id){
        auto fixture=legacy_report;fixture["id"]=id;fixture["revision"]=1;fixture["schema_version"]=1;fixture.erase("content_fingerprint");
        agent_resource_put_request put;put.name=id+".json";put.mime_type="application/vnd.llama-agent.report+json";put.text=fixture.dump();
        agent_resource_descriptor resource;std::string error;
        assert(foundation_resources.put_text(put,resource,error));
        return resource.uri;
    };
    const auto no_op_uri=install_editor_fixture("report-no-op");
    result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",no_op_uri}}.dump()});assert(result.ok);
    auto no_op_args=nlohmann::json{{"document_uri",no_op_uri},{"expected_revision",1},
        {"markdown",nlohmann::json::parse(result.output)["markdown"]},{"operation_id","unchanged-save"}};
    result=foundation_registry.execute({"document.save",no_op_args.dump()});assert(result.ok&&nlohmann::json::parse(result.output)["revision"]==1);
    const auto no_op_fingerprint=nlohmann::json::parse(result.output)["content_fingerprint"];
    result=foundation_registry.execute({"document.save",no_op_args.dump()});assert(result.ok);
    assert(nlohmann::json::parse(result.output)["content_fingerprint"]==no_op_fingerprint);
    auto changed_no_op=no_op_args;changed_no_op["markdown"]=no_op_args["markdown"].get<std::string>()+"\n## New section\n\nChanged.\n";
    result=foundation_registry.execute({"document.save",changed_no_op.dump()});assert(!result.ok&&result.failure_class==common_tool_failure_class::conflict);

    // A report resource written before its operation/head is not a published
    // revision. Retrying from the original base remains possible.
    for(const auto * failing_mime:{"application/vnd.llama-agent.document-operation+json","application/vnd.llama-agent.document-head+json"}){
        const auto interrupted_uri=install_editor_fixture(std::string(failing_mime).find("head")!=std::string::npos?"report-head-failure":"report-operation-failure");
        auto interrupted_args=changed_no_op;interrupted_args["document_uri"]=interrupted_uri;interrupted_args["operation_id"]="interrupted-save";
        foundation_resources.fail_mime=failing_mime;
        result=foundation_registry.execute({"document.save",interrupted_args.dump()});assert(!result.ok);
        foundation_resources.fail_mime.clear();
        result=foundation_registry.execute({"document.save",interrupted_args.dump()});
        assert(result.ok&&nlohmann::json::parse(result.output)["revision"]==2);
    }

    const auto large_uri=install_editor_fixture("report-large-proposal");
    result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",large_uri}}.dump()});assert(result.ok);
    const auto large_markdown=nlohmann::json::parse(result.output)["markdown"].get<std::string>()+"\n## Long clarification\n\n"+std::string(70000,'a')+"\n";
    result=foundation_registry.execute({"document.propose_patch",nlohmann::json{{"document_uri",large_uri},{"expected_revision",1},{"markdown",large_markdown},{"summary","Large but bounded clarification."}}.dump()});assert(result.ok);
    const auto large_proposal=nlohmann::json::parse(result.output);assert(large_proposal.contains("diff"));
    result=foundation_registry.execute({"document.apply_patch",nlohmann::json{{"document_uri",large_uri},{"expected_revision",1},
        {"proposal_uri",large_proposal["proposal_uri"]},{"operation_id","large-proposal-apply"}}.dump()});assert(result.ok);
    const auto large_saved_uri=nlohmann::json::parse(result.output)["document_uri"].get<std::string>();
    result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",large_saved_uri}}.dump()});assert(result.ok);
    const auto large_opened=nlohmann::json::parse(result.output);
    assert(large_opened["rendered_markdown"].get<std::string>().size() > 65536);
    assert(large_opened["document"]["sections"].back()["blocks"][0]["author"]=="model");
    assert(large_opened["document"]["sections"].back()["blocks"][0]["review_state"]=="unreviewed");
    const auto & target_section=large_opened["document"]["sections"].back();
    auto block_proposal=nlohmann::json{{"document_uri",large_saved_uri},{"expected_revision",2},
        {"operation","replace"},{"section_id",target_section["id"]},{"block_id",target_section["blocks"][0]["id"]},
        {"text","A bounded replacement."},{"summary","Replace only the selected block."}};
    result=foundation_registry.execute({"document.propose_patch",block_proposal.dump()});assert(result.ok);
    auto unknown_block=block_proposal;unknown_block["block_id"]="not-a-report-block";
    result=foundation_registry.execute({"document.propose_patch",unknown_block.dump()});assert(!result.ok);
    auto escaping_block=block_proposal;escaping_block["text"]="Changed.\n\n## Another section\n\nUnrelated content.";
    result=foundation_registry.execute({"document.propose_patch",escaping_block.dump()});assert(!result.ok);
    // A newer report composition supersedes an earlier editor head. Further
    // edits must continue its revision rather than becoming permanently stale.
    auto recomposed_json=nlohmann::json::parse(foundation_resources.contents.at(large_saved_uri));
    recomposed_json["revision"]=3;recomposed_json["editor_parent_uri"]="";recomposed_json["editor_operation_id"]="";recomposed_json["editor_markdown_uri"]="";
    common_agent_artifact_document recomposed;std::string recomposition_error,recomposed_markdown,recomposed_editable;
    assert(common_agent_artifact_document_from_json(recomposed_json.dump(),recomposed,recomposition_error));
    assert(common_agent_artifact_render_markdown(recomposed,recomposed_markdown,recomposition_error));
    assert(common_agent_artifact_render_editable_markdown(recomposed,recomposed_editable,recomposition_error));
    auto publish_fixture=[&](const std::string & name,const std::string & mime,const std::string & text,const std::string & key){
        agent_resource_put_request put;put.name=name;put.mime_type=mime;put.text=text;put.metadata.processing_cache_key=key;
        agent_resource_descriptor resource;assert(foundation_resources.put_text(put,resource,recomposition_error));return resource.uri;
    };
    const auto recomposed_uri=publish_fixture("recomposed-r3.json","application/vnd.llama-agent.report+json",recomposed_json.dump(),"");
    const auto recomposed_md_uri=publish_fixture("recomposed-r3.md","text/markdown",recomposed_markdown,"");
    publish_fixture("recomposed-r3-manifest.json","application/json",nlohmann::json{{"report_id",recomposed.id},{"revision",3},
        {"report_resource",recomposed_uri},{"markdown_resource",recomposed_md_uri},{"resource",recomposed_md_uri},
        {"snapshot_fingerprint",recomposed.snapshot_fingerprint}}.dump(),"artifact-export:"+recomposed.id+":r3:fixture:manifest");
    result=foundation_registry.execute({"document.save",nlohmann::json{{"document_uri",recomposed_uri},{"expected_revision",3},
        {"markdown",recomposed_editable+"\n## After recomposition\n\nAnother edit.\n"},{"operation_id","after-recomposition"}}.dump()});
    assert(result.ok&&nlohmann::json::parse(result.output)["revision"]==4);
    result=foundation_registry.execute({"document.open",nlohmann::json{{"document_uri",saved_document.value("document_uri",std::string{})}}.dump()});
    assert(result.ok&&nlohmann::json::parse(result.output).value("markdown",std::string{}).find("User-edited report text.")!=std::string::npos);
    result=foundation_registry.execute({"artifact.export",nlohmann::json{{"source_document",saved_document.value("document_uri",std::string{})},{"format","markdown"}}.dump()});
    assert(result.ok&&nlohmann::json::parse(result.output).value("report_resource",std::string{})==saved_document.value("document_uri",std::string{}));
    const auto saved_document_export=nlohmann::json::parse(result.output);
    const auto saved_export_markdown=foundation_resources.contents.at(saved_document_export.value("markdown_resource",std::string{}));
    assert(saved_export_markdown.find("User-edited report text.")!=std::string::npos&&
        saved_export_markdown.find("Coverage incomplete")!=std::string::npos);
    auto proposal_markdown=edited_markdown+"\n## Reviewer note\n\nProposed clarification.\n";
    result=foundation_registry.execute({"document.propose_patch",nlohmann::json{{"document_uri",saved_document.value("document_uri",std::string{})},
        {"expected_revision",2},{"markdown",proposal_markdown},{"summary","Clarify the edited result."}}.dump()});
    assert(result.ok);
    const auto proposal=nlohmann::json::parse(result.output);
    result=foundation_registry.execute({"document.apply_patch",nlohmann::json{{"document_uri",saved_document.value("document_uri",std::string{})},
        {"proposal_uri",proposal.value("proposal_uri",std::string{})},{"expected_revision",2},{"operation_id","adapter-test-apply-1"}}.dump()});
    assert(result.ok&&nlohmann::json::parse(result.output).value("revision",uint64_t(0))==3);
    result=foundation_registry.execute({"document.save",save_args.dump()});
    assert(result.ok&&nlohmann::json::parse(result.output).value("revision",uint64_t(0))==2&&nlohmann::json::parse(result.output).value("reused",false));
    auto stale_apply=nlohmann::json{{"document_uri",saved_document.value("document_uri",std::string{})},
        {"proposal_uri",proposal.value("proposal_uri",std::string{})},{"expected_revision",2},{"operation_id","adapter-test-stale-apply"}};
    result=foundation_registry.execute({"document.apply_patch",stale_apply.dump()});
    assert(!result.ok&&result.failure_class==common_tool_failure_class::conflict);
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
    result = native_network_registry.execute({"web.fetch", R"({"url":"http://127.0.0.1/test"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::network);
    result = native_network_registry.execute({"web.fetch", R"({"url":"https://example.org/data","accept":"application/json\r\nX-Injected: true"})"});
    assert(!result.ok && result.failure_code == "tool.web_fetch.out_of_bounds" &&
           result.failure_class == common_tool_failure_class::validation);

    test_investigation_store investigation_store;
    investigation_store.source.ref.uri = "dataset://investigation/logs/security";
    investigation_store.source.ref.name = "security";
    investigation_store.source.ref.source_provider = "logs";
    investigation_store.source.ref.source_request_json = nlohmann::json{{"relation_profiles", nlohmann::json::array({
        {{"id", "auth-observed-v1"}, {"event_type", "authentication"},
            {"from_kind", "user"}, {"to_kind", "host"}, {"relation_kind", "authenticated_to"},
            {"from_role", "user"}, {"to_role", "host"}}})}}.dump();
    investigation_store.source.ref.storage_mode = common_agent_dataset_storage_mode::virtual_;
    investigation_store.source.columns = {{"event.id", common_agent_dataset_column_type::string, false},
        {"event.type", common_agent_dataset_column_type::string, false},
        {"@timestamp", common_agent_dataset_column_type::string, true},
        {"user.id", common_agent_dataset_column_type::string, true},
        {"host.name", common_agent_dataset_column_type::string, true},
        {"@source.index", common_agent_dataset_column_type::string, false},
        {"@source.id", common_agent_dataset_column_type::string, false}};
    common_investigation_case_revision investigation_case;
    investigation_case.case_id = "graph-case";
    investigation_case.title = "Authentication chain";
    investigation_case.question = "Which host received the login?";
    investigation_case.source_id = "logs";
    investigation_case.window_start_ms = 100;
    investigation_case.window_end_ms = 1000;
    investigation_case.scope.namespace_id = "tenant-a";
    investigation_case.scope.project_id = "project-a";
    investigation_case.scope.session_id = "session-a";
    investigation_case.entities = {
        {"user:logs:alice", "logs", common_investigation_entity_kind::user, "alice", "alice", "", 0, 0},
        {"host:logs:app1", "logs", common_investigation_entity_kind::host, "app1", "app1", "", 0, 0},
    };
    investigation_case.events = {{"event:1", "logs", "doc-1", "authentication", {200, 210, 0, true},
        {"user:logs:alice", "host:logs:app1"}, "resource://raw/doc-1", "resource://normalized/doc-1", {}, {}}};
    investigation_case.relations = {{"rel:1", "logs", "user:logs:alice", "host:logs:app1",
        common_investigation_relation_kind::authenticated_to, 200, 200, true, 1,
        "auth-v1", "1", "observed", {"event:1"}}};
    investigation_store.seed(investigation_case);
    common_tool_catalog investigation_catalog;
    assert(investigation_catalog.bootstrap("investigation", bootstrap, error));
    common_native_tool_bindings investigation_bindings;
    investigation_bindings.data_store = &investigation_store;
    investigation_bindings.tool_scope = investigation_case.scope;
    test_resource_store investigation_resources;
    investigation_bindings.resource_runtime.store = &investigation_resources;
    common_tool_registry investigation_registry;
    assert(common_register_native_tool_adapters(investigation_catalog, "investigation",
        investigation_bindings, investigation_registry, adapters, error));
    assert(!investigation_registry.is_read_only("investigation.case.create") &&
        investigation_registry.is_policy_gated("investigation.case.create"));
    assert(!investigation_registry.is_read_only("investigation.collect") &&
        investigation_registry.is_policy_gated("investigation.collect"));
    assert(!investigation_registry.is_read_only("investigation.claim.hypothesize") &&
        investigation_registry.is_policy_gated("investigation.claim.hypothesize"));
    assert(!investigation_registry.is_read_only("investigation.report.compose") &&
        investigation_registry.is_policy_gated("investigation.report.compose"));
    assert(investigation_registry.is_read_only("investigation.case.open") &&
        !investigation_registry.is_policy_gated("investigation.case.open"));
    result = investigation_registry.execute({"investigation.case.open", R"({"case_id":"graph-case"})"});
    assert(result.ok && result.output.find("first_entity_id") != std::string::npos &&
        result.output.find("\"counts\"") != std::string::npos &&
        result.output.find("\"events\":[") == std::string::npos);
    result = investigation_registry.execute({"investigation.case.list", R"({"limit":10})"});
    assert(result.ok && result.output.find("graph-case") != std::string::npos);
    result = investigation_registry.execute({"investigation.case.create",
        R"({"title":"New case","question":"What happened?","source_id":"logs","window_start_ms":100,"window_end_ms":1000})"});
    assert(result.ok && result.output.find("case-") != std::string::npos);
    result = investigation_registry.execute({"investigation.graph.expand",
        R"({"case_id":"graph-case","case_revision":1,"center_entity_id":"user:logs:alice"})"});
    assert(result.ok && result.output.find("host:logs:app1") != std::string::npos &&
        result.output.find("event:1") != std::string::npos);
    const auto timestamp_column = std::find_if(investigation_store.source.columns.begin(),
        investigation_store.source.columns.end(), [](const auto & column) { return column.name == "@timestamp"; });
    const auto saved_timestamp_column = *timestamp_column;
    investigation_store.source.columns.erase(timestamp_column);
    result = investigation_registry.execute({"investigation.collect",
        R"({"case_id":"graph-case","expected_revision":1,"dataset":"dataset://investigation/logs/security","identifier_field":"user.id","identifier_value":"alice","limit":20})"});
    assert(!result.ok && result.failure_code == "tool.investigation.timestamp_unmapped" &&
        investigation_store.cases.at("graph-case").revision == 1);
    investigation_store.source.columns.push_back(saved_timestamp_column);
    result = investigation_registry.execute({"investigation.collect",
        R"({"case_id":"graph-case","expected_revision":1,"dataset":"dataset://investigation/logs/security","identifier_field":"user.id","identifier_value":"alice","limit":20})"});
    assert(result.ok && result.output.find("collected_events") != std::string::npos &&
        investigation_store.cases.at("graph-case").events.size() == 2 &&
        investigation_store.cases.at("graph-case").events[1].source_document_id == "doc-2" &&
        investigation_store.cases.at("graph-case").events[1].source_index == "logs-2026.10" &&
        investigation_store.cases.at("graph-case").events[1].time.has_event_time &&
        investigation_store.cases.at("graph-case").events[1].time.event_time_ms == 300 &&
        investigation_store.cases.at("graph-case").events[1].entity_roles.at("user") == "entity:logs:user:alice" &&
        investigation_store.cases.at("graph-case").events[1].entity_roles.at("host") == "entity:logs:host:db2" &&
        investigation_store.cases.at("graph-case").entities.size() == 4 &&
        investigation_resources.requests.size() == 2);
    assert(std::any_of(investigation_store.cases.at("graph-case").relations.begin(),
        investigation_store.cases.at("graph-case").relations.end(), [](const auto & relation) {
            return std::find(relation.evidence_event_ids.begin(), relation.evidence_event_ids.end(), "event:1") !=
                relation.evidence_event_ids.end();
        }));
    const auto & newly_collected_event = investigation_store.cases.at("graph-case").events[1];
    if (newly_collected_event.entity_roles.at("user") != "entity:logs:user:alice" ||
            newly_collected_event.entity_roles.at("host") != "entity:logs:host:db2") {
        std::cerr << "new authentication events must preserve user and host endpoint roles\n";
        return 20;
    }
    const auto collected_auth_relation = std::find_if(investigation_store.cases.at("graph-case").relations.begin(),
        investigation_store.cases.at("graph-case").relations.end(), [](const auto & relation) {
            return !relation.evidence_event_ids.empty() && relation.evidence_event_ids[0] ==
                "event:logs:logs-2026.10:doc-2";
        });
    if (collected_auth_relation == investigation_store.cases.at("graph-case").relations.end() ||
            collected_auth_relation->from_entity_id != "entity:logs:user:alice" ||
            collected_auth_relation->to_entity_id != "entity:logs:host:db2") {
        std::cerr << "configured authentication profiles must produce the declared directed endpoints\n";
        return 21;
    }
    result = investigation_registry.execute({"investigation.analyze",
        R"({"case_id":"graph-case","analysis":"graph_metrics"})"});
    assert(result.ok && result.output.find("betweenness_normalized") != std::string::npos &&
        result.output.find("\"complete\":true") != std::string::npos);
    result = investigation_registry.execute({"investigation.analyze",
        R"({"case_id":"graph-case","analysis":"temporal_path","start_entity_id":"user:logs:alice","goal_entity_id":"host:logs:app1"})"});
    assert(result.ok && result.output.find("event:1") != std::string::npos &&
        result.output.find("shortest_hop_count") != std::string::npos);
    const auto evidence_relation = std::find_if(investigation_store.cases.at("graph-case").relations.begin(),
        investigation_store.cases.at("graph-case").relations.end(), [](const auto & relation) {
            return std::find(relation.evidence_event_ids.begin(), relation.evidence_event_ids.end(), "event:1") !=
                relation.evidence_event_ids.end();
        });
    assert(evidence_relation != investigation_store.cases.at("graph-case").relations.end());
    investigation_store.cases.at("graph-case").events[1].entity_ids.push_back("user:logs:alice");
    evidence_relation->evidence_event_ids.push_back(investigation_store.cases.at("graph-case").events[1].id);
    result = investigation_registry.execute({"investigation.events",
        nlohmann::json{{"case_id", "graph-case"}, {"case_revision", 2}, {"relation_id", evidence_relation->id}, {"limit", 1}}.dump()});
    assert(result.ok && result.output.find("doc-1") != std::string::npos && result.output.find("next_cursor") != std::string::npos &&
        result.output.find("\"has_more\":true") != std::string::npos);
    const auto event_page = nlohmann::json::parse(result.output);
    result = investigation_registry.execute({"investigation.events", nlohmann::json{
        {"case_id", "graph-case"}, {"case_revision", 2}, {"relation_id", evidence_relation->id},
        {"limit", 1}, {"cursor", event_page.at("next_cursor")}}.dump()});
    assert(result.ok && result.output.find("doc-2") != std::string::npos && result.output.find("\"has_more\":false") != std::string::npos);
    result = investigation_registry.execute({"investigation.events", nlohmann::json{
        {"case_id", "graph-case"}, {"case_revision", 1}, {"relation_id", evidence_relation->id}}.dump()});
    assert(!result.ok && result.failure_code == "tool.investigation.stale_cursor");
    result = investigation_registry.execute({"investigation.case.items", nlohmann::json{
        {"case_id", "graph-case"}, {"case_revision", 2}, {"type", "entities"}, {"limit", 1}}.dump()});
    assert(result.ok && result.output.find("\"has_more\":true") != std::string::npos);
    const auto item_page = nlohmann::json::parse(result.output);
    result = investigation_registry.execute({"investigation.case.items", nlohmann::json{
        {"case_id", "graph-case"}, {"case_revision", 2}, {"type", "entities"},
        {"limit", 1}, {"cursor", item_page.at("next_cursor")}}.dump()});
    assert(result.ok && result.output.find("\"case_revision\":2") != std::string::npos &&
        result.output.find("\"items\":[") != std::string::npos);
    result = investigation_registry.execute({"investigation.case.items",
        R"({"case_id":"graph-case","case_revision":"2","type":"entities"})"});
    assert(!result.ok);
    result = investigation_registry.execute({"investigation.claim.hypothesize",
        R"({"case_id":"graph-case","expected_revision":2,"claim_id":"h-1","text":"The login may be unusual.","supporting_event_ids":["event:1"]})"});
    assert(result.ok && result.output.find("hypothesis") != std::string::npos &&
        investigation_store.cases.at("graph-case").revision == 3 &&
        investigation_store.cases.at("graph-case").claims[0].state == common_investigation_claim_state::hypothesis);
    agent_resource_put_request report_snapshot_request;
    report_snapshot_request.name="selected-visualization.png";
    report_snapshot_request.mime_type="image/png";
    report_snapshot_request.bytes=base64::decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg==");
    report_snapshot_request.bytes_are_authoritative=true;
    report_snapshot_request.scope=common_runtime_resource_scope::session;
    report_snapshot_request.source_provider="native";
    report_snapshot_request.source_tool="visualization.snapshot";
    report_snapshot_request.lineage.parent_uri="resource://test/visualization.json";
    report_snapshot_request.lineage.chunk_count=1;
    report_snapshot_request.lineage.byte_length=report_snapshot_request.bytes.size();
    report_snapshot_request.lineage.derivation="visualization.snapshot:png-v1";
    agent_resource_descriptor report_snapshot;
    assert(investigation_resources.put_bytes(report_snapshot_request,report_snapshot,error));
    result = investigation_registry.execute({"investigation.report.compose", nlohmann::json{
        {"case_id", "graph-case"}, {"expected_revision", 3},
        {"visualization_uris", {report_snapshot.uri}}}.dump()});
    assert(result.ok && result.output.find("report_uri") != std::string::npos &&
        investigation_resources.requests.size() == 6 &&
        investigation_resources.requests.back().text.find("## Timeline") != std::string::npos &&
        investigation_resources.requests.back().text.find("## Visualization snapshots") != std::string::npos &&
        investigation_resources.requests.back().text.find(report_snapshot.uri) != std::string::npos &&
        investigation_resources.requests.back().text.find("authenticated_to") != std::string::npos &&
        investigation_resources.requests.back().text.find("Hypothesis:") != std::string::npos &&
        investigation_resources.requests.back().text.find("event:1") != std::string::npos);
    const auto composed_report = nlohmann::json::parse(result.output);
    assert(std::any_of(result.resource_refs.begin(), result.resource_refs.end(), [&](const auto & resource) {
        return resource.uri == composed_report.at("report_uri").get<std::string>() &&
            resource.mime_type == "application/vnd.llama-agent.document+json";
    }));
    common_agent_artifact_document reopened_investigation_report;
    assert(common_agent_artifact_document_from_json(
        investigation_resources.contents.at(composed_report.at("report_uri").get<std::string>()),
        reopened_investigation_report, error));
    assert(reopened_investigation_report.kind == common_agent_artifact_document_kind::document &&
        reopened_investigation_report.snapshot_uri == composed_report.at("source_manifest_uri") &&
        reopened_investigation_report.sections.size() == 1);
    std::string reopened_report_markdown;
    assert(common_agent_artifact_render_markdown(reopened_investigation_report, reopened_report_markdown, error) &&
        reopened_report_markdown.find(report_snapshot.uri) != std::string::npos &&
        reopened_report_markdown.find("## Timeline") != std::string::npos);
    result = investigation_registry.execute({"investigation.case.items", nlohmann::json{
        {"case_id", "graph-case"}, {"case_revision", 3}, {"type", "reports"}, {"limit", 10}}.dump()});
    assert(result.ok);
    const auto indexed_reports = nlohmann::json::parse(result.output);
    assert(indexed_reports.at("items").size() == 1 &&
        indexed_reports.at("items")[0].at("report_uri") == composed_report.at("report_uri") &&
        indexed_reports.at("items")[0].at("case_revision") == 3);
    auto second_indexed_report = investigation_store.reports.front();
    second_indexed_report.content_fingerprint = "sha256:second-report";
    second_indexed_report.report_uri = "resource://reports/second.json";
    investigation_store.reports.push_back(second_indexed_report);
    result = investigation_registry.execute({"investigation.case.items", nlohmann::json{
        {"case_id", "graph-case"}, {"case_revision", 3}, {"type", "reports"}, {"limit", 1}}.dump()});
    assert(result.ok);
    const auto first_report_page = nlohmann::json::parse(result.output);
    assert(first_report_page.at("items").size() == 1 && first_report_page.at("has_more") == true);
    result = investigation_registry.execute({"investigation.case.items", nlohmann::json{
        {"case_id", "graph-case"}, {"case_revision", 3}, {"type", "reports"}, {"limit", 1},
        {"cursor", first_report_page.at("next_cursor")}}.dump()});
    assert(result.ok && nlohmann::json::parse(result.output).at("items").size() == 1 &&
        nlohmann::json::parse(result.output).at("has_more") == false);
    auto foreign_scope = investigation_bindings;
    foreign_scope.tool_scope.namespace_id = "tenant-b";
    common_tool_registry foreign_investigation_registry;
    assert(common_register_native_tool_adapters(investigation_catalog, "investigation",
        foreign_scope, foreign_investigation_registry, adapters, error));
    result = foreign_investigation_registry.execute({"investigation.case.open",
        R"({"case_id":"graph-case"})"});
    assert(!result.ok && result.failure_code == "tool.investigation.case_unavailable");

    auto collision_case = investigation_case;
    collision_case.case_id = "index-id-collision-case";
    collision_case.entities.clear(); collision_case.events.clear(); collision_case.relations.clear();
    collision_case.claims.clear(); collision_case.gaps.clear(); collision_case.revision = 1;
    collision_case.fingerprint.clear();
    investigation_store.seed(collision_case);
    investigation_store.virtual_page = R"({"columns":["event.id","event.type","@timestamp","user.id","host.name","@source.index","@source.id"],"rows":[["same-id","authentication","1970-01-01T00:00:00.300Z","alice","app-a","logs-a","same-id"],["same-id","not_authentication","1970-01-01T00:00:00.400Z","alice","app-b","logs-b","same-id"]],"complete":true,"scan_truncated":false,"result_truncated":false})";
    result = investigation_registry.execute({"investigation.collect", R"({"case_id":"index-id-collision-case","expected_revision":1,"dataset":"dataset://investigation/logs/security","identifier_field":"user.id","identifier_value":"alice","limit":20})"});
    const auto & collision_events = investigation_store.cases.at("index-id-collision-case").events;
    assert(result.ok && collision_events.size() == 2 && collision_events[0].source_document_id == "same-id" &&
        collision_events[1].source_document_id == "same-id" && collision_events[0].source_index != collision_events[1].source_index &&
        collision_events[0].id != collision_events[1].id);
    assert(std::none_of(investigation_store.cases.at("index-id-collision-case").relations.begin(),
        investigation_store.cases.at("index-id-collision-case").relations.end(), [&](const auto & relation) {
            return std::find(relation.evidence_event_ids.begin(), relation.evidence_event_ids.end(), collision_events[1].id) !=
                relation.evidence_event_ids.end();
        }));
    const auto & collision_gaps = investigation_store.cases.at("index-id-collision-case").gaps;
    if (std::find(collision_gaps.begin(), collision_gaps.end(),
            "one_or_more_events_have_no_declared_relation_profile_and_remain_unlinked_observations") == collision_gaps.end() ||
            result.output.find("unlinked_observations") == std::string::npos) {
        std::cerr << "unmatched profile limitations must survive relation building and publication\n";
        return 22;
    }
    auto logical_case = collision_case;
    logical_case.case_id = "logical-field-case";
    investigation_store.seed(logical_case);
    investigation_store.source.columns = {{"event_id", common_agent_dataset_column_type::string, false},
        {"event_type", common_agent_dataset_column_type::string, false},
        {"event_time", common_agent_dataset_column_type::string, false},
        {"user_id", common_agent_dataset_column_type::string, false},
        {"host_name", common_agent_dataset_column_type::string, false},
        {"process_id", common_agent_dataset_column_type::string, false},
        {"process_start_time", common_agent_dataset_column_type::string, false}};
    investigation_store.virtual_page = R"({"columns":["event_id","event_type","event_time","user_id","host_name","process_id","process_start_time"],"rows":[["logical-doc","authentication",300,"alice","app-logical","42","start-1"]],"complete":false,"scan_truncated":true,"result_truncated":true})";
    result = investigation_registry.execute({"investigation.collect", R"({"case_id":"logical-field-case","expected_revision":1,"dataset":"dataset://investigation/logs/security","identifier_field":"user_id","identifier_value":"alice","limit":20})"});
    const auto & logical_events = investigation_store.cases.at("logical-field-case").events;
    if (!result.ok || logical_events.size() != 1 ||
            logical_events[0].entity_roles.at("host") != "entity:logs:host:app-logical" ||
            logical_events[0].entity_roles.at("process") != "entity:logs:process:app-logical:42@start-1" ||
            result.output.find("source page is partial") == std::string::npos) {
        std::cerr << "logical source mappings and partial collection limitations must reach the published case\n";
        return 23;
    }
    const auto normalized_evidence = nlohmann::json::parse(investigation_resources.requests.back().text);
    if (normalized_evidence.at("events")[0].at("entity_roles").at("host") !=
            logical_events[0].entity_roles.at("host") ||
            normalized_evidence.at("events")[0].at("collected_time_ms") != logical_events[0].time.collected_time_ms) {
        std::cerr << "immutable normalization evidence must preserve endpoint roles and collection time\n";
        return 24;
    }
    std::filesystem::remove_all(repository);
    return 0;
}
