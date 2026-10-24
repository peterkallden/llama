#include "tools/agent/daemon/agent-template-library.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "plan/plan-blueprint.h"
#include "tools/agent/runtime/agent-route-compiler.h"
#include "agent/agent-package-json.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <iostream>
static void check(bool ok,const std::string & why){if(!ok){std::cerr<<why<<'\n';std::exit(1);}}
int main(){
    std::string error;common_memory_in_memory_store memories;common_plan_in_memory_store plans;
    check(memories.open("",error)&&plans.open("",error),error);
    auto resources=make_agent_resource_store({"in-memory","","in-memory",""},error);check(bool(resources),error);
    const auto dir=std::filesystem::temp_directory_path()/("template-library-contract-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));std::filesystem::create_directories(dir);
    const auto path=dir/"package.json";
    common_agent_bootstrap_package package;package.name="methods";package.version="v1";
    common_agent_bootstrap_procedure procedure;procedure.id="method";procedure.content="Read authorized evidence.";package.procedures.push_back(procedure);
    common_agent_bootstrap_blueprint blueprint;blueprint.id="method";blueprint.goal="Inspect evidence";blueprint.success_criteria="Evidence inspected";
    common_plan_step step;step.id="inspect";step.title="Inspect";step.objective="Inspect evidence";step.status=common_plan_step_status::pending;
    common_agent_bootstrap_workflow workflow;workflow.id="clock";
    workflow.definition.workflow_ref="workflow://test/clock";
    workflow.definition.workflow_revision="v1";
    workflow.definition.family="test";
    workflow.definition.graph_revision="g1";
    workflow.definition.required_capabilities={"clock"};
    package.workflows.push_back(workflow);
    blueprint.workflow_policy=common_plan_workflow_policy::required;
    blueprint.workflow_bindings={{"workflow://test/clock","v1"}};
    blueprint.steps.push_back(step);blueprint.procedure_refs={"method"};package.blueprints.push_back(blueprint);
    auto write=[&]{std::string text;check(common_agent_package_to_json(package,text,error,false),error);std::ofstream(path)<<text;};write();
    std::vector<common_agent_template_library_config> configs={{"methods",path.string(),"host",{}, {}}};
    auto embed=[](const std::string &,std::vector<float> & v,std::string &){v={1.0f};return true;};
    common_agent_template_library_snapshot first;std::vector<common_blueprint_candidate> candidates;
    check(load_agent_template_libraries(configs,memories,plans,*resources,embed,first,candidates,error),error);
    check(first.revisions.size()==1&&candidates.size()==1,"library not published");
    const auto id=candidates.front().persisted_id;
    check(first.permits(id,"a","")&&first.permits(id,"b","p"),"host definition not shared");
    auto resolved=common_agent_template_definition(plans,id,&first,error);check(bool(resolved),error);
    check(resolved->namespace_id=="llama-agent-host","template ownership changed");
    check(!common_plan_template_scope_matches(*resolved,common_plan_scope::turn,"a","s","","t"),"generic scope matcher broadened");
    check(common_agent_template_definition_allowed(*resolved,common_plan_scope::turn,"a","s","","t",&first),"library resolver rejected approved definition");
    common_agent_runtime_tooling tooling;
    tooling.template_libraries=std::make_shared<common_agent_template_library_snapshot>(first);
    tooling.capabilities={"clock"};tooling.capability_tools={{"clock",{"time.now"}}};
    tooling.tools={{"time.now","Read time",R"({"type":"object"})"}};
    common_agent_request request;request.namespace_id="a";request.session_id="session-a";request.turn_id="turn-a";
    common_agent_scope scope;scope.namespace_id="a";scope.session_id="session-a";scope.turn_id="turn-a";
    common_agent_route_catalog catalog;
    check(compile_agent_route_catalog(request,plans,scope,candidates,tooling,{},catalog,error),error);
    check(catalog.candidates.size()==2 && catalog.candidates.back().workflow.has_value(), "shared workflow not compiled for caller");
    common_explicit_blueprint_selector selector(candidates.front().logical_id);
    common_blueprint_selection_config selection;selection.task_plan_id="selected-task";selection.session_id="session-a";
    selection.template_libraries=tooling.template_libraries;
    selection.selected_workflow=common_plan_workflow_binding{"workflow://test/clock","v1"};
    common_plan_route_binding route_binding;
    route_binding.blueprint_ref=candidates.front().logical_id;
    route_binding.blueprint_revision=resolved->source_revision;
    route_binding.workflow_ref="workflow://test/clock";route_binding.workflow_revision="v1";route_binding.graph_revision="g1";
    selection.route_binding=route_binding;
    common_blueprint_selection_result outcome;
    check(common_agent_select_and_instantiate_blueprint(plans,request,selector,candidates,selection,outcome,error),error);
    check(outcome.outcome==common_blueprint_selection_outcome::instantiated, "shared blueprint was not instantiated");
    auto selected=plans.get("selected-task",error);
    check(selected && selected->namespace_id=="a" && selected->session_id=="session-a" &&
        selected->derived_from_plan_id==id && selected->workflow_definition.has_value(), "task leaked template ownership or lost workflow");
    check(common_agent_template_task_allowed(*selected,"a","",&first,error), error);
    auto foreign_route=*selected;foreign_route.route_binding->blueprint_ref="other:clock";
    check(!common_agent_template_task_allowed(foreign_route,"a","",&first,error), "foreign route was accepted");
    auto missing=tooling;missing.tools.clear();
    check(compile_agent_route_catalog(request,plans,scope,candidates,missing,{},catalog,error),error);
    check(catalog.candidates.size()==1, "library granted missing tool capability");
    common_agent_template_library_snapshot reopened;
    check(load_agent_template_libraries(configs,memories,plans,*resources,embed,reopened,candidates,error),error);
    check(reopened.revisions.size()==1&&candidates.front().persisted_id==id,"retry duplicated or changed revision");
    agent_resource_read_authority user;user.namespace_id="a";user.session_id="s";
    agent_resource_descriptor descriptor;
    check(!resources->stat(first.revisions[0].resource_uri,user,descriptor,error),"host import receipt leaked to user session");
    package.version="v2";package.procedures[0].content="Read and compare authorized evidence.";write();
    check(load_agent_template_libraries(configs,memories,plans,*resources,embed,reopened,candidates,error),error);
    check(reopened.revisions.size()==2&&candidates.front().persisted_id!=id&&reopened.permits(id,"a",""),"update lost pinned historical revision");
    common_plan_state task;
    check(common_plan_instantiate_blueprint(*resolved, "task-a", "session-a", task, error), error);
    task.namespace_id="a";
    check(common_agent_template_task_allowed(task,"a","",&reopened,error), error);
    check(!common_agent_template_task_allowed(task,"a","",nullptr,error), "missing library allowed resume");
    auto corrupt=task;corrupt.source_revision="other-revision";
    check(!common_agent_template_task_allowed(corrupt,"a","",&reopened,error), "wrong revision allowed resume");
    check(load_agent_template_libraries({},memories,plans,*resources,embed,reopened,candidates,error),error);
    check(!reopened.permits(id,"a","")&&candidates.empty(),"removed library remained executable");
    check(!common_agent_template_task_allowed(task,"a","",&reopened,error), "disabled library allowed resume");
    auto duplicate=package.workflows.front();duplicate.id="ambiguous-alias";
    package.workflows.push_back(duplicate);write();
    check(!load_agent_template_libraries(configs,memories,plans,*resources,embed,reopened,candidates,error), "ambiguous workflow origin was accepted");
    check(candidates.empty() && reopened.revisions.empty(), "failed import exposed candidates");
    package.workflows.pop_back();write();
    check(plans.erase(id,error),error);
    check(!load_agent_template_libraries(configs,memories,plans,*resources,embed,reopened,candidates,error),"missing historical definition silently accepted");
    check(reopened.revisions.empty(),"failed publication exposed a partial snapshot");
    std::filesystem::remove_all(dir);
}
