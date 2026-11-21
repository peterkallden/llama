#include "agent/investigation-context.h"
#include "plan/plan-context.h"
#include <stdexcept>
#include <algorithm>

using json = nlohmann::ordered_json;
static void require(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }

int main() {
    {
        std::vector<common_chat_tool> tools{{"investigate.case.create", "Create",
            R"({"type":"object","additionalProperties":false,"anyOf":[{"required":["source_id"]},{"required":["resource_uri"]}],"properties":{"source_id":{"type":"string","maxLength":128,"x-agent-source-kind":"investigation_search"},"resource_uri":{"type":"string"}}})"},
            {"other.tool", "Unrelated source identifier", R"({"type":"object","properties":{"source_id":{"type":"string"}}})"}};
        std::string projection_error;
        require(common_agent_project_investigation_source_choices(tools, {"logs", "logs"}, projection_error), projection_error);
        auto projected = json::parse(tools.front().parameters);
        require(projected["properties"]["source_id"]["enum"] == json::array({"logs"}) &&
                !json::parse(tools.back().parameters)["properties"]["source_id"].contains("enum"),
            "source projection confused dataset handles or unrelated identifier fields with configured sources");
        require(common_agent_project_investigation_source_choices(tools, {"logs", "other"}, projection_error), projection_error);
        require(json::parse(tools.front().parameters)["properties"]["source_id"]["enum"] == json::array({"logs"}),
            "source projection widened an existing host restriction");
        require(common_agent_project_investigation_source_choices(tools, {}, projection_error), projection_error);
        projected = json::parse(tools.front().parameters);
        require(!projected["properties"].contains("source_id") &&
                projected["properties"].contains("resource_uri") &&
                projected["anyOf"] == json::array({json{{"required", json::array({"resource_uri"})}}}),
            "absence of authorized search sources disabled resource-backed investigation");
    }
    common_investigation_case_revision current;
    current.case_id="case"; current.title="Test"; current.source_id="search";
    current.scope.namespace_id="local"; current.scope.session_id="session";
    current.window_start_ms=100; current.window_end_ms=1000;
    common_investigation_source source;
    source.id="provider-search"; source.provider_id="search";
    source.kind=common_investigation_source_kind::search_connector;
    source.processing_status="not_collected"; source.complete=false;
    current.sources={source};
    common_agent_request request;
    request.prompt="Investigate the selected identity";
    common_investigation_context_binding binding;
    binding.case_id=current.case_id; binding.revision=1; binding.source_ids={source.id};
    request.investigation_context=binding;
    request.tool_argument_bindings.push_back({"investigate.events",R"({"case_id":"case","case_revision":1})","investigation-context",false});
    request.investigation_context_reader=[&](const auto & selected,json & projection,std::string & error) {
        return common_investigation_project_context(current,selected,projection,error);
    };
    std::string error;
    auto unopened=request;unopened.investigation_context->revision=0;
    require(common_agent_refresh_investigation_context(unopened,nullptr,error) && unopened.investigation_context_json.empty(),
        "case.open was blocked before it could establish a verified baseline");
    require(common_agent_refresh_investigation_context(request,nullptr,error),error);
    require(request.investigation_context_json.find("not_collected")!=std::string::npos &&
        request.prompt=="Investigate the selected identity","case context leaked into user text");
    common_plan_state plan; plan.id="workflow";
    common_plan_step open{"open","Open","Open selected case"};
    open.status=common_plan_step_status::completed;
    open.tool_call=common_plan_tool_call{"investigate.case.open",R"({"case_id":"case"})"};
    plan.steps={open};
    plan.observations.push_back({"tool:open:investigate.case.open","investigate.case.open",
        R"({"ok":true,"result":{"case":{"case_id":"case","revision":1}}})",1.0f});
    auto collect=open;collect.id="collect";
    collect.tool_call=common_plan_tool_call{"investigate.collect",R"({"case_id":"case","expected_revision":1})"};
    plan.steps.push_back(collect);
    plan.observations.push_back({"tool:collect:investigate.collect","investigate.collect",
        R"({"ok":true,"result":{"case_id":"case","revision":2}})",1.0f});
    current.revision=2;current.sources.front().processing_status="query_complete";current.sources.front().complete=true;
    require(common_agent_refresh_investigation_context(request,&plan,error),error);
    require(request.investigation_context->revision==2 && request.investigation_context_json.find("query_complete")!=std::string::npos &&
        request.investigation_context_json.find("not_collected")==std::string::npos,"live projection retained old readiness");
    require(json::parse(request.tool_argument_bindings.front().arguments_json)["case_revision"]==2,
        "verified publication left stale host argument bindings");

    require(request.investigation_source_kinds && request.investigation_source_kinds->size()==1 &&
        request.investigation_source_kinds->front()==common_investigation_source_kind::search_connector,"source eligibility lost selected source kinds");
    std::reverse(plan.steps.begin(),plan.steps.end());
    require(common_agent_refresh_investigation_context(request,&plan,error),
        "publication replay followed planner display order instead of recorded execution order");
    std::reverse(plan.steps.begin(),plan.steps.end());
    auto repair_request=request;
    require(common_agent_bind_investigation_tool(repair_request,"investigate.case.items",
        R"({"type":"object","properties":{"case_id":{"type":"string"},"case_revision":{"type":"integer"},"cursor":{"type":"string"}}})",error),error);
    const auto repair_fields=json::parse(repair_request.tool_argument_bindings.back().arguments_json);
    require(repair_fields["case_id"]=="case" && repair_fields["case_revision"]==2 &&
        !repair_fields.contains("cursor"),"repair did not bind the verified baseline or fabricated a cursor");
    const auto saved=request.investigation_context_json;
    current.revision=3;
    require(!common_agent_refresh_investigation_context(request,&plan,error) && request.investigation_context->revision==2 &&
        request.investigation_context_json==saved,"live read silently rebased an external publication");
    auto external=open;external.id="external";plan.steps.push_back(external);
    plan.observations.push_back({"tool:external:investigate.case.open","investigate.case.open",
        R"({"ok":true,"result":{"case":{"case_id":"case","revision":3}}})",1.0f});
    require(!common_agent_refresh_investigation_context(request,&plan,error),"read observation authorized an external revision");
    plan.steps.pop_back();plan.observations.pop_back();current.revision=2;
    plan.route_binding=common_plan_route_binding{};
    plan.route_binding->workflow_step_start_index=1; // A later linked route inherits the same case chain.
    auto restored=request;restored.investigation_context=binding;
    require(common_agent_refresh_investigation_context(restored,&plan,error) && restored.investigation_context->revision==2,
        "resume did not replay its original publication chain");
    restored.investigation_context_reader={};restored.investigation_context=binding;
    restored.investigation_context_json="old snapshot";
    require(common_agent_refresh_investigation_context(restored,&plan,error) && restored.investigation_context_json.empty(),
        "missing scoped reader retained a stale snapshot");
    const auto text=common_agent_render_investigation_context(request);
    require(text.find("supersedes older case snapshots")!=std::string::npos && text.find("query_complete")!=std::string::npos,
        "current projection lost its authority boundary");
}
