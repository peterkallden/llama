#include "plan/cozo/plan-cozo.h"
#include "plan/cozo/plan-cozo-schema.h"
#include <nlohmann/json.hpp>
extern "C" {
#include <cozo_c.h>
}
using json = nlohmann::ordered_json;
static std::string scope_name(common_plan_scope v) { return v == common_plan_scope::session ? "session" : v == common_plan_scope::project ? "project" : v == common_plan_scope::global ? "global" : "turn"; }
static common_plan_scope parse_scope(const std::string & v) { return v == "session" ? common_plan_scope::session : v == "project" ? common_plan_scope::project : v == "global" ? common_plan_scope::global : common_plan_scope::turn; }
static std::string kind_name(common_plan_kind v) { return v == common_plan_kind::blueprint ? "blueprint" : "task"; }
static common_plan_kind parse_kind(const std::string & v) { return v == "blueprint" ? common_plan_kind::blueprint : common_plan_kind::task; }
static std::string status_name(common_plan_status v) { switch(v) { case common_plan_status::proposed:return "proposed"; case common_plan_status::active:return "active"; case common_plan_status::completed:return "completed"; case common_plan_status::blocked:return "blocked"; case common_plan_status::failed:return "failed"; default:return "cancelled"; } }
static common_plan_status parse_status(const std::string & v) { if(v=="active")return common_plan_status::active; if(v=="completed")return common_plan_status::completed; if(v=="blocked")return common_plan_status::blocked; if(v=="failed")return common_plan_status::failed; if(v=="cancelled")return common_plan_status::cancelled; return common_plan_status::proposed; }
static std::string step_status_name(common_plan_step_status v) { switch(v) { case common_plan_step_status::pending:return "pending"; case common_plan_step_status::active:return "active"; case common_plan_step_status::completed:return "completed"; case common_plan_step_status::blocked:return "blocked"; case common_plan_step_status::skipped:return "skipped"; default:return "failed"; } }
static common_plan_step_status parse_step_status(const std::string & v) { if(v=="active")return common_plan_step_status::active; if(v=="completed")return common_plan_step_status::completed; if(v=="blocked")return common_plan_step_status::blocked; if(v=="skipped")return common_plan_step_status::skipped; if(v=="failed")return common_plan_step_status::failed; return common_plan_step_status::pending; }
static std::string step_mode_name(common_plan_step_mode v) { return v == common_plan_step_mode::tool ? "tool" : v == common_plan_step_mode::reasoning ? "reasoning" : "final_response"; }
static common_plan_step_mode parse_step_mode(const std::string & v) { return v == "tool" ? common_plan_step_mode::tool : v == "reasoning" ? common_plan_step_mode::reasoning : common_plan_step_mode::final_response; }
static json serialize_resource_ref(const common_runtime_resource_ref & resource) {
    json value = {
        {"uri", resource.uri},
        {"name", resource.name},
        {"description", resource.description},
        {"mime_type", resource.mime_type},
        {"size_bytes", resource.size_bytes},
        {"scope", common_runtime_resource_scope_name(resource.scope)},
        {"metadata", {
            {"purpose", resource.metadata.purpose},
            {"content_summary", resource.metadata.content_summary},
            {"usage_hint", resource.metadata.usage_hint},
            {"limitations", resource.metadata.limitations},
            {"keywords", resource.metadata.keywords},
            {"entities", resource.metadata.entities},
        }},
    };
    return value;
}

static common_runtime_resource_ref parse_resource_ref(const json & value) {
    common_runtime_resource_ref resource;
    resource.uri = value.value("uri", std::string{});
    resource.name = value.value("name", std::string{});
    resource.description = value.value("description", std::string{});
    resource.mime_type = value.value("mime_type", std::string{});
    resource.size_bytes = value.value("size_bytes", size_t(0));
    const auto scope_name = value.value("scope", std::string("turn"));
    resource.scope = scope_name == "session" ? common_runtime_resource_scope::session :
        scope_name == "project" ? common_runtime_resource_scope::project :
        common_runtime_resource_scope::turn;
    if (value.contains("metadata") && value["metadata"].is_object()) {
        const auto & metadata = value["metadata"];
        resource.metadata.purpose = metadata.value("purpose", std::string{});
        resource.metadata.content_summary = metadata.value("content_summary", std::string{});
        resource.metadata.usage_hint = metadata.value("usage_hint", std::string{});
        resource.metadata.limitations = metadata.value("limitations", std::string{});
        resource.metadata.keywords = metadata.value("keywords", std::vector<std::string>{});
        resource.metadata.entities = metadata.value("entities", std::vector<std::string>{});
    }
    return resource;
}

static json serialize(const common_plan_state & p) { json steps=json::array(), observations=json::array(), constraints=json::array(), assumptions=json::array(); for(const auto & s:p.steps) { json step={{"id",s.id},{"title",s.title},{"objective",s.objective},{"intended_contribution",s.intended_contribution},{"mode",step_mode_name(common_plan_step_effective_mode(s))},{"status",step_status_name(s.status)},{"depends_on",s.depends_on},{"blocked_by",s.blocked_by},{"required_evidence",s.required_evidence},{"source_memory_ids",s.source_memory_ids},{"optional",s.optional},{"generated_from_memory",s.generated_from_memory},{"created_at",s.created_at},{"updated_at",s.updated_at}}; if(s.selected_tool)step["selected_tool"]=*s.selected_tool; if(s.result_summary)step["result_summary"]=*s.result_summary; if(s.tool_call)step["tool_call"]={{"name",s.tool_call->name},{"arguments_json",s.tool_call->arguments_json}}; steps.push_back(std::move(step)); } for(const auto & observation:p.observations) { json resources = json::array(); for (const auto & resource : observation.resource_refs) resources.push_back(serialize_resource_ref(resource)); json datasets = json::array(); for (const auto & dataset : observation.dataset_refs) datasets.push_back({{"uri",dataset.uri},{"name",dataset.name},{"row_count",dataset.row_count},{"column_count",dataset.column_count},{"source_resource_uri",dataset.source_resource_uri},{"source_representation",dataset.source_representation}}); observations.push_back({{"id",observation.id},{"source",observation.source},{"summary",observation.summary},{"confidence",observation.confidence},{"evidence_ids",observation.evidence_ids},{"resource_refs",resources},{"dataset_refs",datasets},{"created_at",observation.created_at}}); } for(const auto & constraint:p.constraints) constraints.push_back({{"id",constraint.id},{"description",constraint.description},{"hard",constraint.hard}}); for(const auto & assumption:p.assumptions) assumptions.push_back({{"id",assumption.id},{"statement",assumption.statement},{"confidence",assumption.confidence},{"valid",assumption.valid},{"evidence_ids",assumption.evidence_ids}}); return {{"id",p.id},{"namespace_id",p.namespace_id},{"session_id",p.session_id},{"project_id",p.project_id},{"turn_id",p.turn_id},{"source_revision",p.source_revision},{"kind",kind_name(p.kind)},{"derived_from_plan_id",p.derived_from_plan_id},{"scope",scope_name(p.scope)},{"status",status_name(p.status)},{"purpose",p.purpose},{"goal",p.goal},{"success_criteria",p.success_criteria},{"required_capabilities",p.required_capabilities},{"steps",steps},{"observations",observations},{"constraints",constraints},{"assumptions",assumptions},{"active_step_id",p.active_step_id},{"next_action",p.next_action},{"version",p.version},{"created_at",p.created_at},{"updated_at",p.updated_at}}; }
static bool deserialize(const std::string & text, common_plan_state & p) {
    auto j = json::parse(text, nullptr, false);
    if (!j.is_object()) return false;

    p = {};
    p.id = j.value("id", std::string{});
    p.namespace_id = j.value("namespace_id", std::string("local"));
    p.session_id = j.value("session_id", std::string{});
    p.project_id = j.value("project_id", std::string{});
    p.turn_id = j.value("turn_id", std::string{});
    p.source_revision = j.value("source_revision", std::string{});
    p.kind = parse_kind(j.value("kind", std::string("task")));
    if (j.contains("derived_from_plan_id") && !j["derived_from_plan_id"].is_null()) p.derived_from_plan_id = j["derived_from_plan_id"].get<std::string>();
    p.scope = parse_scope(j.value("scope", std::string("turn")));
    p.status = parse_status(j.value("status", std::string("proposed")));
    p.goal = j.value("goal", std::string{});
    p.purpose = j.value("purpose", p.goal);
    p.success_criteria = j.value("success_criteria", std::string{});
    p.required_capabilities = j.value("required_capabilities", std::vector<std::string>{});
    if (j.contains("active_step_id") && !j["active_step_id"].is_null()) p.active_step_id = j["active_step_id"].get<std::string>();
    if (j.contains("next_action") && !j["next_action"].is_null()) p.next_action = j["next_action"].get<std::string>();
    p.version = j.value("version", uint64_t(0));
    p.created_at = j.value("created_at", int64_t(0));
    p.updated_at = j.value("updated_at", int64_t(0));

    if (j.contains("steps")) {
        for (const auto & s : j["steps"]) {
            common_plan_step x;
            x.id = s.value("id", std::string{});
            x.title = s.value("title", std::string{});
            x.objective = s.value("objective", std::string{});
            x.intended_contribution = s.value("intended_contribution", x.objective);
            x.mode = parse_step_mode(s.value("mode", std::string("final_response")));
            x.status = parse_step_status(s.value("status", std::string("pending")));
            x.depends_on = s.value("depends_on", std::vector<std::string>{});
            x.blocked_by = s.value("blocked_by", std::vector<std::string>{});
            x.required_evidence = s.value("required_evidence", std::vector<std::string>{});
            x.source_memory_ids = s.value("source_memory_ids", std::vector<std::string>{});
            x.optional = s.value("optional", false);
            x.generated_from_memory = s.value("generated_from_memory", false);
            x.created_at = s.value("created_at", int64_t(0));
            x.updated_at = s.value("updated_at", int64_t(0));
            if (s.contains("selected_tool") && s["selected_tool"].is_string()) x.selected_tool = s["selected_tool"].get<std::string>();
            if (s.contains("result_summary") && s["result_summary"].is_string()) x.result_summary = s["result_summary"].get<std::string>();
            if (s.contains("tool_call") && s["tool_call"].is_object() && s["tool_call"].value("name", std::string{}).size()) {
                x.tool_call = common_plan_tool_call{
                    s["tool_call"].value("name", std::string{}),
                    s["tool_call"].value("arguments_json", std::string("{}"))
                };
            }
            p.steps.push_back(std::move(x));
        }
    }

    if (j.contains("observations")) {
        for (const auto & observation : j["observations"]) {
            common_plan_observation value;
            value.id = observation.value("id", std::string{});
            value.source = observation.value("source", std::string{});
            value.summary = observation.value("summary", std::string{});
            value.confidence = observation.value("confidence", 0.5f);
            value.evidence_ids = observation.value("evidence_ids", std::vector<std::string>{});
            if (observation.contains("resource_refs") && observation["resource_refs"].is_array()) {
                for (const auto & resource : observation["resource_refs"]) {
                    if (resource.is_object()) value.resource_refs.push_back(parse_resource_ref(resource));
                }
            } if (observation.contains("dataset_refs") && observation["dataset_refs"].is_array()) { for (const auto & dataset : observation["dataset_refs"]) { if (!dataset.is_object()) continue; common_agent_dataset_ref ref; ref.uri=dataset.value("uri", std::string()); ref.name=dataset.value("name", std::string()); ref.row_count=dataset.value("row_count", size_t(0)); ref.column_count=dataset.value("column_count", size_t(0)); ref.source_resource_uri=dataset.value("source_resource_uri", std::string()); ref.source_representation=dataset.value("source_representation", std::string()); value.dataset_refs.push_back(std::move(ref)); } }
            value.created_at = observation.value("created_at", int64_t(0));
            p.observations.push_back(std::move(value));
        }
    }
    if (j.contains("constraints")) for (const auto & constraint : j["constraints"]) p.constraints.push_back({constraint.value("id", std::string{}), constraint.value("description", std::string{}), constraint.value("hard", true)});
    if (j.contains("assumptions")) for (const auto & assumption : j["assumptions"]) { common_plan_assumption value; value.id = assumption.value("id", std::string{}); value.statement = assumption.value("statement", std::string{}); value.confidence = assumption.value("confidence", 0.5f); value.valid = assumption.value("valid", true); value.evidence_ids = assumption.value("evidence_ids", std::vector<std::string>{}); p.assumptions.push_back(std::move(value)); }

    return !p.id.empty();
}
common_plan_cozo_store::common_plan_cozo_store() = default; common_plan_cozo_store::~common_plan_cozo_store(){close();}
bool common_plan_cozo_store::run(const std::string & script,const std::string & params,std::string & out,std::string & error)const{if(db_id<0){error="Cozo plan store is not open";return false;}char*r=cozo_run_query(db_id,script.c_str(),params.empty()?"{}":params.c_str(),false);if(!r){error="Cozo query failed";return false;}out=r;cozo_free_str(r);auto j=json::parse(out,nullptr,false);if(j.is_object()&&!j.value("ok",true)){error="Cozo query failed: "+j.value("message",std::string("unknown"));return false;}error.clear();return true;}
bool common_plan_cozo_store::reload_cache(std::string & error){ cache.close(); if(!cache.open("",error))return false; std::string out; if(!run("?[id, state_json] := *agent_plan[id, state_json]","{}",out,error))return false; auto plans=json::parse(out,nullptr,false); for(const auto &row:plans["rows"]){common_plan_state p;if(!deserialize(row[1].get<std::string>(),p)||!cache.create(p,error))return false;} if(!run("?[plan_id, sequence, event_json] := *agent_plan_event[plan_id, sequence, event_json]","{}",out,error))return false; std::map<std::string,std::vector<common_plan_event>> events; auto rows=json::parse(out,nullptr,false); for(const auto &row:rows["rows"]){auto value=json::parse(row[2].get<std::string>(),nullptr,false); common_plan_event event; event.sequence=row[1].get<uint64_t>(); if(value.is_object()){event.accepted=value.value("accepted",false);event.prior_version=value.value("prior_version",uint64_t(0));event.new_version=value.value("new_version",uint64_t(0));event.reason_summary=value.value("reason_summary",std::string{});} events[row[0].get<std::string>()].push_back(std::move(event));} for(auto &entry:events)if(!cache.restore_history(entry.first,std::move(entry.second),error))return false; error.clear();return true; }
bool common_plan_cozo_store::open(const std::string & path,std::string & error){close();int32_t id=-1;char*e=cozo_open_db("sqlite",(path.empty()?"plan.cozo":path).c_str(),"{}",&id);if(e){error=e;cozo_free_str(e);return false;}db_id=id;std::string out;if(!run("::relations","{}",out,error)){close();return false;}auto relations=json::parse(out,nullptr,false);bool has_plan=false,has_event=false;for(const auto &row:relations["rows"]){if(row.is_array()&&row[0].is_string()){has_plan|=row[0]=="agent_plan";has_event|=row[0]=="agent_plan_event";}}if(has_plan!=has_event){error="Cozo plan database has an incomplete schema";close();return false;}if(!has_plan&&!run(common_plan_cozo_schema_script(),"{}",out,error)){close();return false;}if(!reload_cache(error)){close();return false;}return true;}
void common_plan_cozo_store::close(){cache.close();if(db_id>=0){cozo_close_db(db_id);db_id=-1;}}
bool common_plan_cozo_store::persist_plan(const common_plan_state&p,std::string&error){std::string out;json rows=json::array({json::array({p.id,serialize(p).dump()})});return run("?[id, state_json] <- $rows :put agent_plan { id => state_json }",json({{"rows",rows}}).dump(),out,error);}
bool common_plan_cozo_store::create(const common_plan_state&p,std::string&error){if(!persist_plan(p,error))return false;return cache.create(p,error);}
std::optional<common_plan_state> common_plan_cozo_store::get(const std::string&id,std::string&error){return cache.get(id,error);}
std::vector<common_plan_state> common_plan_cozo_store::list(std::string&error){return cache.list(error);}
bool common_plan_cozo_store::apply(const common_plan_operation&o,common_plan_state&updated,std::string&error){auto prior=cache.get(o.plan_id,error);if(!prior)return false;if(!cache.apply(o,updated,error)){const auto original=error;std::string ignored;reload_cache(ignored);error=original;return false;}if(!persist_plan(updated,error)){const auto original=error;std::string ignored;reload_cache(ignored);error=original;return false;}auto events=cache.history(o.plan_id,error);const auto &e=events.back();std::string out;json rows=json::array({json::array({o.plan_id,e.sequence,json({{"accepted",e.accepted},{"prior_version",e.prior_version},{"new_version",e.new_version},{"operation",common_plan_operation_kind_name(o.kind)},{"reason_summary",e.reason_summary},{"evidence_ids",o.evidence_ids}}).dump()})});if(run("?[plan_id, sequence, event_json] <- $rows :put agent_plan_event { plan_id, sequence => event_json }",json({{"rows",rows}}).dump(),out,error))return true;const auto original=error;std::string rollback_error;persist_plan(*prior,rollback_error);std::string ignored;reload_cache(ignored);error=original;return false;}
std::vector<common_plan_event> common_plan_cozo_store::history(const std::string&id,std::string&error){return cache.history(id,error);}
bool common_plan_cozo_store::erase(const std::string&id,std::string&error){std::string out;if(!run("?[id] <- [[$id]] :delete agent_plan { id }",json({{"id",id}}).dump(),out,error))return false;if(!run("?[plan_id, sequence] := *agent_plan_event[plan_id, sequence, event_json], plan_id == $id :delete agent_plan_event { plan_id, sequence }",json({{"id",id}}).dump(),out,error))return false;return cache.erase(id,error);}
