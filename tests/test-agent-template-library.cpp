#include "agent/template-library.h"
#include "tools/agent/host/agent-host-config.h"
#include "tools/agent/runtime/agent-plan-orchestration.h"
#include <cstdlib>
#include <iostream>
using json=nlohmann::ordered_json;
static void check(bool value,const std::string & why){if(!value){std::cerr<<why<<'\n';std::exit(1);}}
int main(){
    common_agent_scope scope;
    scope.namespace_id="a";scope.session_id="same-session";scope.turn_id="same-turn";
    const auto identity=make_agent_automatic_blueprint_plan_id(scope);
    check(identity==make_agent_automatic_blueprint_plan_id(scope), "task identity is unstable");
    auto other=scope;other.namespace_id="b";
    check(identity!=make_agent_automatic_blueprint_plan_id(other), "task identity collides across namespaces");
    other=scope;other.project_id="p";
    check(identity!=make_agent_automatic_blueprint_plan_id(other), "task identity collides across projects");
    other=scope;other.session_id="other-session";
    check(identity!=make_agent_automatic_blueprint_plan_id(other), "task identity collides across sessions");
    other=scope;other.turn_id="other-turn";
    check(identity!=make_agent_automatic_blueprint_plan_id(other), "task identity collides across turns");
    std::string error;std::vector<common_agent_template_library_config> configs;
    const auto input=json::array({{{"id","shared"},{"package","methods.json"}},
        {{"id","team"},{"package","team.json"},{"level","namespace"},{"namespace_id","team-a"}},
        {{"id","project"},{"package","project.json"},{"level","project"},{"namespace_id","team-a"},{"project_id","p1"}}});
    check(common_agent_template_library_config_parse(input,configs,error),error);
    agent_host_config host;
    check(parse_agent_host_config_json(json{{"bootstrap",{{"libraries", input}}}}, host, error), error);
    agent_host_config restored;
    check(parse_agent_host_config_json(agent_host_config_to_json(host), restored, error), error);
    check(restored.bootstrap_libraries.size() == 3, "host config lost library definitions");
    check(configs.size()==3 && configs[0].level=="host","wrong default import level");
    std::vector<common_agent_template_library_config> roundtrip;
    check(common_agent_template_library_config_parse(common_agent_template_library_config_json(configs),roundtrip,error),error);
    check(roundtrip[2].project_id=="p1","lost project owner");
    for(const auto & invalid:{json::array({{{"id","x"},{"package",12}}}),
            json::array({{{"id","x"},{"package","p"},{"level","session"}}}),
            json::array({{{"id","x"},{"package","p"},{"level","host"},{"namespace_id","n"}}}),
            json::array({{{"id","x"},{"package","p"},{"level","project"},{"namespace_id","n"}}}),
            json::array({{{"id","bootstrap.default"},{"package","p"}}}),
            json::array({{{"id","x"},{"package","p"}},{{"id","x"},{"package","q"}}})}){
        check(!common_agent_template_library_config_parse(invalid,roundtrip,error),"invalid import configuration accepted");
    }
    common_agent_template_library_manifest m;
    m.library_id="shared";m.level="host";m.package_name="methods";m.package_version="v1";
    m.fingerprint=std::string(64,'a');m.definition_prefix="library:shared:"+m.fingerprint+":";
    m.blueprints["method"]=m.definition_prefix+"blueprint:method";
    m.workflows.insert(m.definition_prefix+"workflow:method");m.procedures.insert(m.definition_prefix+"procedure:method");
    common_agent_template_library_manifest decoded;
    check(common_agent_template_library_manifest_parse(common_agent_template_library_manifest_json(m),decoded,error),error);
    common_agent_template_library_snapshot snapshot;snapshot.revisions.push_back(decoded);
    const auto blueprint=m.blueprints.at("method"),workflow=*m.workflows.begin();
    check(snapshot.permits(blueprint,"a","p") && snapshot.permits(blueprint,"b",""),"host library not shared");
    check(!snapshot.permits(blueprint,"","") && !snapshot.permits("bootstrap:foreign:global:blueprint:method","a","p"),"unregistered record gained authority");
    check(snapshot.same_revision(blueprint,workflow),"same package members disconnected");
    auto newer=m;newer.fingerprint=std::string(64,'b');newer.definition_prefix="library:shared:"+newer.fingerprint+":";
    newer.blueprints["method"]=newer.definition_prefix+"blueprint:method";newer.workflows.clear();newer.procedures.clear();snapshot.revisions.push_back(newer);
    check(!snapshot.same_revision(blueprint,newer.blueprints.at("method")),"revision collision accepted");
    m.level="namespace";m.namespace_id="a";
    check(common_agent_template_library_available(m,"a","") && !common_agent_template_library_available(m,"b",""),"namespace library leaked");
    m.level="project";m.project_id="p";
    check(common_agent_template_library_available(m,"a","p") && !common_agent_template_library_available(m,"a","q"),"project library leaked");
    auto bad=json::parse(common_agent_template_library_manifest_json(m));bad["blueprints"]["method"]="foreign-record";
    check(!common_agent_template_library_manifest_parse(bad.dump(),decoded,error),"foreign member accepted");
    bad=json::parse(common_agent_template_library_manifest_json(m));bad["library_id"]=false;
    check(!common_agent_template_library_manifest_parse(bad.dump(),decoded,error),"invalid manifest field type accepted");
}
