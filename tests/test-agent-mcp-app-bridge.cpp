#include "tools/agent/tooling/agent-tool-provider.h"
#include "tools/agent/runtime/agent-runtime-tooling.h"
#include "tools/agent/mcp/agent-mcp-protocol.h"
#include "tools/agent/mcp/agent-mcp-server-tool-registry.h"
#include "agent/tooling/schema/tool-output-codec.h"

#include <cassert>
#include <algorithm>
#include <cstdio>

namespace {

class fake_mcp_client final : public agent_mcp_tool_client {
public:
    bool list_tools(const agent_tool_context &, std::vector<mcp_agent_tool_definition> & tools, std::string & error) override {
        tools = definitions;
        error.clear();
        return true;
    }
    bool call_tool(const agent_tool_context &, const mcp_agent_tool_definition & tool,
                   const std::string &, mcp_agent_tool_call_result & result, std::string & error) override {
        called = tool.name;
        result.ok = true;
        result.structured_content_json = R"({"columns":["region","amount"]})";
        error.clear();
        return true;
    }
    bool read_resource(const std::string & uri, mcp_agent_resource_read_result & result, std::string & error) override {
        read_uri = uri;
        result.resource.uri = uri;
        result.resource.name = "Explorer UI";
        result.resource.mime_type = "text/html;profile=mcp-app";
        result.text_content = "<html>app</html>";
        error.clear();
        return true;
    }

    std::vector<mcp_agent_tool_definition> definitions;
    std::string called;
    std::string read_uri;
};

class graph_fixture_store final : public common_agent_data_store {
public:
    common_investigation_case_revision investigation;
    bool execute(const std::string &, const std::string &, std::string &, std::string &) override { return false; }
    bool get_investigation_case(const std::string & id, const common_agent_scope & scope,
            common_investigation_case_revision & out, std::string & error) override {
        if (id != investigation.case_id || scope.namespace_id != investigation.scope.namespace_id ||
                scope.project_id != investigation.scope.project_id) { error = "case is unavailable"; return false; }
        out = investigation; error.clear(); return true;
    }
    bool authorize_investigation_source(const std::string & source, const common_agent_scope &,
            std::string & error) const override {
        if (source != "logs") { error = "source denied"; return false; }
        error.clear(); return true;
    }
};

mcp_agent_tool_definition tool(std::string name, bool app_visible) {
    mcp_agent_tool_definition result;
    result.provider_id = "catalog";
    result.name = std::move(name);
    result.description = "test tool";
    result.input_schema_json = R"({"type":"object","properties":{},"additionalProperties":false})";
    result.read_only = true;
    result.read_only_declared = true;
    if (app_visible) result.ui_visibility = {"app"};
    return result;
}

} // namespace

int main() {
    const auto init = make_mcp_initialize_params();
    assert(init["capabilities"]["extensions"]["io.modelcontextprotocol/ui"]["mimeTypes"][0] ==
        "text/html;profile=mcp-app");
    mcp_agent_tool_definition parsed_definition;
    std::string error;
    agent_mcp_json definition_json = agent_mcp_json::object();
    definition_json["name"] = "dataset.schema";
    definition_json["annotations"]["readOnlyHint"] = true;
    definition_json["_meta"]["ui"]["resourceUri"] = "ui://catalog/data";
    definition_json["_meta"]["ui"]["visibility"] = agent_mcp_json::array({"app"});
    assert(parse_mcp_tool_definition("catalog", definition_json, parsed_definition, error));
    assert(parsed_definition.read_only && parsed_definition.read_only_hint && parsed_definition.read_only_declared);
    assert(parsed_definition.ui_resource_uri == "ui://catalog/data");
    assert(parsed_definition.ui_visibility == std::vector<std::string>{"app"});
    assert(parse_mcp_tool_definition("catalog", {{"name", "unknown.read"}}, parsed_definition, error));
    assert(parsed_definition.read_only && !parsed_definition.read_only_declared);
    assert(parsed_definition.ui_visibility == std::vector<std::string>{"model"});
    definition_json = agent_mcp_json::object();
    definition_json["name"] = "write";
    definition_json["annotations"]["readOnlyHint"] = false;
    assert(parse_mcp_tool_definition("catalog", definition_json, parsed_definition, error));
    assert(!parsed_definition.read_only && !parsed_definition.read_only_hint && parsed_definition.read_only_declared);
    definition_json = {{"name", "defaults"}, {"_meta", {{"ui", {{"resourceUri", "ui://catalog/defaults"}}}}}};
    assert(parse_mcp_tool_definition("catalog", definition_json, parsed_definition, error));
    assert(parsed_definition.ui_visibility == (std::vector<std::string>{"model", "app"}));

    agent_mcp_server_tool_registry registry;
    agent_mcp_server_tool listed_tool;
    listed_tool.name = "dataset.schema";
    listed_tool.description = "dataset schema";
    listed_tool.ui_metadata = agent_mcp_first_party_app_metadata(listed_tool.name);
    listed_tool.handler = [](const agent_mcp_json &, agent_mcp_server_tool_result &, std::string &) { return true; };
    assert(registry.register_tool(std::move(listed_tool), error));
    const auto listed = agent_mcp_render_tools_list_result(registry);
    assert(listed["tools"][0]["_meta"]["ui"]["resourceUri"] == "ui://llama-agent/apps/data-explorer");
    assert(listed["tools"][0]["_meta"]["ui"]["visibility"] == agent_mcp_json::array({"model", "app"}));

    const auto resources = agent_mcp_first_party_app_resources();
    assert(resources["resources"].size() >= 2);
    bool has_document_editor=false;
    bool has_investigation_workspace=false;
    for(const auto & resource:resources["resources"]) {
        const auto uri=resource.value("uri",std::string{});
        has_document_editor=has_document_editor||uri=="ui://llama-agent/apps/document-editor";
        has_investigation_workspace=has_investigation_workspace||uri=="ui://llama-agent/apps/investigation-workspace";
    }
    assert(has_document_editor==(resources["resources"].size()>=3 && resources["resources"].back().value("uri",std::string{})=="ui://llama-agent/apps/document-editor"));
    assert(has_investigation_workspace==(agent_mcp_first_party_app_metadata("investigate.case.open").value("resourceUri","")=="ui://llama-agent/apps/investigation-workspace"));
    assert(resources["resources"][0]["mimeType"] == "text/html;profile=mcp-app");
    assert(agent_mcp_first_party_app_metadata("dataset.schema").value("resourceUri", "") ==
        "ui://llama-agent/apps/data-explorer");
    assert(agent_mcp_first_party_app_metadata("plan.get").value("resourceUri", "") ==
        "ui://llama-agent/apps/workflow-inspector");
    agent_mcp_json app_resource;
    assert(agent_mcp_read_first_party_app_resource("ui://llama-agent/apps/data-explorer", app_resource, error));
    assert(app_resource["contents"][0]["text"].get<std::string>().find("dataset.schema") != std::string::npos);
    if (has_investigation_workspace) {
        assert(agent_mcp_read_first_party_app_resource("ui://llama-agent/apps/investigation-workspace", app_resource, error));
        assert(app_resource["contents"][0]["text"].get<std::string>().find("Investigation workspace") != std::string::npos);
    }
    assert(!agent_mcp_read_first_party_app_resource("ui://llama-agent/apps/unknown", app_resource, error));

    fake_mcp_client client;
    auto app = tool("explorer", true);
    app.ui_resource_uri = "ui://catalog/explorer";
    app.ui_metadata_json = R"({"resourceUri":"ui://catalog/explorer","visibility":["app"]})";
    auto app_operation = tool("schema", true);
    app_operation.read_only_hint = false; // remote hints do not grant or revoke host authority
    app_operation.read_only_declared = true;
    auto visible_both = tool("list", true);
    visible_both.ui_visibility = {"model", "app"};
    visible_both.ui_metadata_json = R"({"visibility":["model","app"]})";
    auto write_app = tool("dataset.write", true);
    write_app.writes_plan = true;
    auto hidden = tool("hidden", false);
    client.definitions = {app, app_operation, visible_both, write_app, hidden};

    mcp_agent_tool_provider provider("catalog", client, {}, {}, "mcp", {"schema", "list", "dataset.write"});
    agent_tool_context context;
    auto view = provider.resolve_tools(context, error);
    assert(view && error.empty());
    assert(view->chat_tools().size() == 1 && view->chat_tools()[0].name == "catalog_list");
    assert(view->app_tools().size() == 4);
    assert(!view->exposes_tool("catalog_hidden"));
    assert(!view->exposes_tool("catalog_schema"));
    assert(!view->validate({"test", "catalog_schema", "{}"}, error));
    assert(view->is_mcp_app_read_only("catalog_schema"));
    assert(!view->is_mcp_app_read_only("catalog_dataset.write"));
    assert(!view->is_web_read_only("catalog_schema"));
    assert(view->is_web_read_only("catalog_list"));

    agent_tool_result resource;
    assert(view->read_mcp_app_resource("catalog_explorer", "ui://catalog/explorer", resource, error));
    assert(resource.ok && client.read_uri == "ui://catalog/explorer");
    assert(resource.content_json.find("<html>app</html>") != std::string::npos);
    assert(!view->read_mcp_app_resource("explorer", "ui://other/arbitrary", resource, error));

    agent_tool_result call_result;
    assert(view->call_mcp_app_tool("catalog_explorer", {"app", "schema", "{}"}, call_result, error));
    assert(call_result.ok && client.called == "schema");
    assert(!view->call_mcp_app_tool("catalog_explorer", {"app", "dataset.write", "{}"}, call_result, error));
    assert(!view->call_mcp_app_tool("dataset.schema", {"app", "dataset.schema", "{}"}, call_result, error));

    // The composite runtime must preserve the same boundary: app-only tools
    // are app-callable through their provider owner, but are not model-callable.
    composite_agent_tool_provider composite;
    composite.add_provider(provider);
    auto composite_view = composite.resolve_tools(context, error);
    assert(composite_view && error.empty());
    assert(!composite_view->exposes_tool("catalog_schema"));
    assert(!composite_view->validate({"test", "catalog_schema", "{}"}, error));
    assert(!composite_view->call({"test", "catalog_schema", "{}"}, error).ok);
    assert(!composite_view->read_mcp_app_resource("missing-app", "ui://missing", resource, error));
    assert(!error.empty());
    assert(composite_view->call_mcp_app_tool(
        "catalog_explorer", {"app", "schema", "{}"}, call_result, error));
    assert(call_result.ok && client.called == "schema");

    agent_tool_context restricted_context = context;
    restricted_context.allowed_exposed_tool_names = {"catalog_explorer", "catalog_list"};
    auto restricted_view = provider.resolve_tools(restricted_context, error);
    assert(restricted_view && error.empty());
    assert(restricted_view->app_tools().size() == 2);
    assert(!restricted_view->is_mcp_app_read_only("catalog_schema"));
    assert(!restricted_view->call_mcp_app_tool(
        "catalog_explorer", {"app", "schema", "{}"}, call_result, error));

    auto unannotated = tool("unannotated.read", true);
    unannotated.read_only_declared = false;
    client.definitions.push_back(std::move(unannotated));
    view = provider.resolve_tools(context, error);
    assert(view && error.empty());
    assert(!view->call_mcp_app_tool("catalog_explorer", {"app", "unannotated.read", "{}"}, call_result, error));
    // First-party app-only metadata is a presentation rule as well as an
    // execution boundary. Factory-supplied reviewer flags are cleared for
    // ordinary native calls by the provider itself.
    common_tool_catalog native_catalog;
    common_tool_bootstrap_result native_bootstrap;
    if (!native_catalog.bootstrap("investigation",native_bootstrap,error)) return 20;
    native_agent_tool_provider native(native_catalog,[](const agent_tool_context &,common_native_tool_bindings & bindings,std::string &) {
        bindings.investigation_reviewer_id="must-not-leak-to-model"; return true;
    });
    agent_tool_context user_context;
    user_context.profile_id="investigation";
    user_context.scope.namespace_id="tenant-a"; user_context.scope.session_id="session-a";
    user_context.allow_policy_gated_writes=true; user_context.max_calls=10;
    auto native_view=native.resolve_tools(user_context,error);
    if (!native_view) return 21;
    agent_tool_call review_call{"review","investigate.association.review",
        R"({"case_id":"case-a","expected_revision":1,"idempotency_key":"user-review","actor_ids":["a","b"],"kind":"identity","decision":"same_actor"})"};
    auto denied=native_view->call(review_call,error);
    if (denied.ok || denied.failure_code != "tool.investigation.user_review_required") return 22;
    if (has_investigation_workspace) {
        const auto & model_tools=native_view->chat_tools();
        if (std::any_of(model_tools.begin(),model_tools.end(),[](const auto & t) { return t.name=="investigate.association.review"; })) return 23;
        const auto & app_tools=native_view->app_tools();
        if (std::none_of(app_tools.begin(),app_tools.end(),[](const auto & t) { return t.name=="investigate.association.review"; })) return 24;
        common_agent_runtime_tooling presented;
        presented.tools = model_tools;
        presented.app_tools = app_tools;
        for (const char * name : {"visualize.snapshot", "investigate.association.review"}) {
            const auto * exposed = presented.find_presented_tool(name);
            if (!exposed || exposed->mcp_ui_metadata.empty() || !exposed->mcp_app_write_authorized ||
                    !native_view->is_mcp_app_write_authorized(name)) return 26;
            if (std::any_of(presented.tools.begin(), presented.tools.end(),
                    [&](const auto & tool) { return tool.name == name; })) return 27;
        }
        if (presented.find_presented_tool("not.registered") != nullptr) return 28;
        composite_agent_tool_provider native_composite;
        native_composite.add_provider(native);
        auto composed_native_view = native_composite.resolve_tools(user_context, error);
        if (!composed_native_view) return 29;
        for (const char * name : {"visualize.snapshot", "investigate.association.review"}) {
            if (composed_native_view->exposes_tool(name) ||
                    !composed_native_view->is_mcp_app_write_authorized(name)) return 30;
            std::string provider_id, server_name;
            if (!composed_native_view->app_tool_provenance(name, provider_id, server_name) ||
                    !provider_id.empty() || server_name != name) return 31;
            if (composed_native_view->call({"model-denied", name, "{}"}, error).ok) return 32;
        }
        agent_tool_result reviewed;
        if (native_view->call_mcp_app_tool("investigate.case.open",review_call,reviewed,error) ||
            reviewed.failure_code != "tool.investigation.store_unavailable") return 25;
        // The app route reached its host executor; this fixture intentionally
        // has no data store, so it cannot publish a decision.
    }
    // A graph source is a delegated read, never an additional authority.
    graph_fixture_store graph_store;
    int code_calls = 0;
    common_tool_catalog graph_catalog;
    common_tool_bootstrap_result graph_bootstrap;
    if (!graph_catalog.bootstrap("code-analysis", graph_bootstrap, error)) return 40;
    native_agent_tool_provider graph_provider(graph_catalog, [&](const agent_tool_context & tool_context,
            common_native_tool_bindings & bindings, std::string &) {
        bindings.data_store = &graph_store; bindings.tool_scope = tool_context.scope;
        bindings.resource_runtime.namespace_id = tool_context.scope.namespace_id;
        bindings.resource_runtime.project_id = tool_context.scope.project_id;
        bindings.resource_runtime.session_id = tool_context.scope.session_id;
        bindings.code_calls = [&](const std::string &) {
            ++code_calls;
            return common_tool_execution_result::success(
                R"({"backend":"clangd","callers":[],"callees":[],"count":0,"truncated":false})");
        };
        return true;
    });
    agent_tool_context graph_context;
    graph_context.profile_id = "code-analysis"; graph_context.scope.namespace_id = "tenant-a";
    graph_context.scope.project_id = "project-a"; graph_context.max_calls = 10;
    graph_context.allowed_exposed_tool_names = {"graph.source"};
    auto graph_view = graph_provider.resolve_tools(graph_context, error);
    if (!graph_view) return 41;
    const agent_tool_call graph_call{"source", "graph.source", R"({"source":"code","arguments":{"symbol":"f","path_hint":"a.cpp","line":1}})"};
    auto unavailable = graph_view->call(graph_call, error);
    if (unavailable.ok || unavailable.failure_class != common_tool_failure_class::policy || code_calls != 0) return 42;
    graph_context.allowed_exposed_tool_names.push_back("code.calls");
    graph_view = graph_provider.resolve_tools(graph_context, error);
    if (!graph_view) return 43;
    const auto source_definition = std::find_if(graph_view->chat_tools().begin(), graph_view->chat_tools().end(),
        [](const auto & tool) { return tool.name == "graph.source"; });
    if (source_definition == graph_view->chat_tools().end() ||
            !common_compact_dsl_schema_supported(source_definition->parameters, error)) return 53;
    common_agent_tool_call source_dsl;
    if (!common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl,
            "open! graph.source\nsource: code\narguments.symbol: f\narguments.path_hint: a.cpp\narguments.line: 1",
            graph_view->chat_tools(), source_dsl, error)) return 54;
    if (!graph_view->validate({"dsl", source_dsl.name, source_dsl.arguments_json}, error)) return 55;
    const auto observation = graph_view->call(graph_call, error);
    if (!observation.ok || code_calls != 1) return 44;
    const auto payload = nlohmann::json::parse(observation.content_json, nullptr, false);
    if (payload.is_discarded() || payload["result"]["source_tool"] != "code.calls" ||
            payload["result"]["persisted"] != false || !payload["result"].contains("visualization")) return 45;
    const auto invalid = graph_view->call({"invalid", "graph.source", R"({"source":"code","arguments":{"symbol":"f"}})"}, error);
    if (invalid.ok || code_calls != 1) return 46;
    const auto no_investigation = graph_view->call({"case", "graph.source", R"({"source":"investigation","arguments":{}})"}, error);
    if (no_investigation.ok || no_investigation.failure_class != common_tool_failure_class::policy) return 47;
    auto & case_value = graph_store.investigation;
    case_value.case_id = "case-a"; case_value.title = "Case"; case_value.question = "Who accessed the host?";
    case_value.source_id = "logs"; case_value.window_start_ms = 100; case_value.window_end_ms = 1000;
    case_value.scope.namespace_id = "tenant-a"; case_value.scope.project_id = "project-a"; case_value.scope.session_id = "session-a";
    case_value.entities = {{"alice", "logs", common_investigation_entity_kind::user, "alice", "Alice", "", 0, 0},
        {"host", "logs", common_investigation_entity_kind::host, "host", "Host", "", 0, 0}};
    case_value.events = {{"event:1", "logs", "doc-1", "authentication", {200, 210, 0, true},
        {"alice", "host"}, "resource://raw/doc-1", "resource://normalized/doc-1", {}, {}}};
    case_value.relations = {{"rel:1", "logs", "alice", "host", common_investigation_relation_kind::authenticated_to,
        200, 200, true, 1, "auth-v1", "1", "observed", {"event:1"}}};
    if (!graph_catalog.bootstrap("investigation", graph_bootstrap, error)) return 48;
    graph_context.profile_id = "investigation"; graph_context.scope.session_id = "session-a";
    graph_context.allowed_exposed_tool_names = {"graph.source", "investigate.graph.expand"};
    graph_view = graph_provider.resolve_tools(graph_context, error);
    if (!graph_view) return 49;
    const agent_tool_call case_call{"case", "graph.source", R"({"source":"investigation","arguments":{"case_id":"case-a","case_revision":1,"center_entity_id":"alice"}})"};
    const auto case_graph = graph_view->call(case_call, error);
    if (!case_graph.ok) { std::fprintf(stderr, "case graph failed: %s\n", error.c_str()); return 50; }
    const auto case_payload = nlohmann::json::parse(case_graph.content_json, nullptr, false);
    if (case_payload.is_discarded() || case_payload["result"]["visualization"]["edges"].size() != 1 ||
            case_payload["result"]["visualization"]["edges"][0]["provenance_refs"][0] != "event:1") return 51;
    graph_context.scope.project_id = "other-project";
    graph_view = graph_provider.resolve_tools(graph_context, error);
    if (!graph_view || graph_view->call(case_call, error).ok) return 52;
    return 0;
}
