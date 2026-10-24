#include "tools/agent/tooling/agent-tool-provider.h"
#include "tools/agent/runtime/agent-runtime-tooling.h"
#include "tools/agent/mcp/agent-mcp-protocol.h"
#include "tools/agent/mcp/agent-mcp-server-tool-registry.h"

#include <cassert>
#include <algorithm>

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
    assert(has_investigation_workspace==(agent_mcp_first_party_app_metadata("investigation.case.open").value("resourceUri","")=="ui://llama-agent/apps/investigation-workspace"));
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
    agent_tool_call review_call{"review","investigation.association.review",
        R"({"case_id":"case-a","expected_revision":1,"idempotency_key":"user-review","actor_ids":["a","b"],"kind":"identity","decision":"same_actor"})"};
    auto denied=native_view->call(review_call,error);
    if (denied.ok || denied.failure_code != "tool.investigation.user_review_required") return 22;
    if (has_investigation_workspace) {
        const auto & model_tools=native_view->chat_tools();
        if (std::any_of(model_tools.begin(),model_tools.end(),[](const auto & t) { return t.name=="investigation.association.review"; })) return 23;
        const auto & app_tools=native_view->app_tools();
        if (std::none_of(app_tools.begin(),app_tools.end(),[](const auto & t) { return t.name=="investigation.association.review"; })) return 24;
        common_agent_runtime_tooling presented;
        presented.tools = model_tools;
        presented.app_tools = app_tools;
        for (const char * name : {"visualization.snapshot", "investigation.association.review"}) {
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
        for (const char * name : {"visualization.snapshot", "investigation.association.review"}) {
            if (composed_native_view->exposes_tool(name) ||
                    !composed_native_view->is_mcp_app_write_authorized(name)) return 30;
            std::string provider_id, server_name;
            if (!composed_native_view->app_tool_provenance(name, provider_id, server_name) ||
                    !provider_id.empty() || server_name != name) return 31;
            if (composed_native_view->call({"model-denied", name, "{}"}, error).ok) return 32;
        }
        agent_tool_result reviewed;
        if (native_view->call_mcp_app_tool("investigation.case.open",review_call,reviewed,error) ||
            reviewed.failure_code != "tool.investigation.store_unavailable") return 25;
        // The app route reached its host executor; this fixture intentionally
        // has no data store, so it cannot publish a decision.
    }
    return 0;
}
