#include "semi-oracle-store-cozo.h"

#include <cozo_c.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <limits>

using json = nlohmann::ordered_json;

namespace {
const char * schema = R"COZO(
{
 ?[ruleset_id, revision, payload] <- [['__probe__', 1, '{}']]
 :create agent_semi_oracle_rulesets { ruleset_id: String, revision: Int => payload: String }
}
{
 ?[ruleset_id, revision] <- [['__probe__', 1]]
 :delete agent_semi_oracle_rulesets { ruleset_id, revision }
}
{
 ?[session_id, ruleset_id, revision] <- [['__probe__', '__probe__', 1]]
 :create agent_semi_oracle_session_bindings { session_id: String => ruleset_id: String, revision: Int }
}
{
 ?[session_id] <- [['__probe__']]
 :delete agent_semi_oracle_session_bindings { session_id }
}
{
 ?[ruleset_id, revision] <- [['__probe__', 1]]
 :create agent_semi_oracle_host_catalog { ruleset_id: String => revision: Int }
}
{
 ?[ruleset_id] <- [['__probe__']]
 :delete agent_semi_oracle_host_catalog { ruleset_id }
}
{
 ?[evaluation_id, ruleset_id, revision, payload] <- [['__probe__', '__probe__', 1, '{}']]
 :create agent_semi_oracle_evaluations { evaluation_id: String => ruleset_id: String, revision: Int, payload: String }
}
{
 ?[evaluation_id] <- [['__probe__']]
 :delete agent_semi_oracle_evaluations { evaluation_id }
}
{
 ?[interaction_id, session_id, ruleset_id, revision, payload] <- [['__probe__', '__probe__', '__probe__', 1, '{}']]
 :create agent_semi_oracle_interactions { interaction_id: String => session_id: String, ruleset_id: String, revision: Int, payload: String }
}
{
 ?[interaction_id] <- [['__probe__']]
 :delete agent_semi_oracle_interactions { interaction_id }
}
)COZO";

const char * interaction_schema = R"COZO(
{
 ?[interaction_id, session_id, ruleset_id, revision, payload] <- [['__probe__', '__probe__', '__probe__', 1, '{}']]
 :create agent_semi_oracle_interactions { interaction_id: String => session_id: String, ruleset_id: String, revision: Int, payload: String }
}
{
 ?[interaction_id] <- [['__probe__']]
 :delete agent_semi_oracle_interactions { interaction_id }
}
)COZO";

bool relations_ready(const json & value) {
    if(!value.is_object()||!value.contains("rows")||!value["rows"].is_array()) return false;
    std::vector<std::string> found;
    for(const auto & row:value["rows"]) if(row.is_array()&&!row.empty()&&row[0].is_string()) found.push_back(row[0].get<std::string>());
    return std::find(found.begin(),found.end(),"agent_semi_oracle_rulesets")!=found.end() &&
        std::find(found.begin(),found.end(),"agent_semi_oracle_session_bindings")!=found.end() &&
        std::find(found.begin(),found.end(),"agent_semi_oracle_host_catalog")!=found.end() &&
        std::find(found.begin(),found.end(),"agent_semi_oracle_evaluations")!=found.end() &&
        std::find(found.begin(),found.end(),"agent_semi_oracle_interactions")!=found.end();
}

bool relation_exists(const json & value, const char * name) {
    if (!value.is_object() || !value.contains("rows") || !value["rows"].is_array()) return false;
    for (const auto & row : value["rows"]) {
        if (row.is_array() && !row.empty() && row[0].is_string() &&
                row[0].get<std::string>() == name) return true;
    }
    return false;
}
}

agent_semi_oracle_cozo_store::~agent_semi_oracle_cozo_store(){close();}
bool agent_semi_oracle_cozo_store::run(const std::string & q,const std::string & p,std::string & out,std::string & error) const {
    if(db_id_<0){error="semi-oracle Cozo store is not open";return false;}
    char * raw=cozo_run_query(db_id_,q.c_str(),p.empty()?"{}":p.c_str(),false);
    if(!raw){error="semi-oracle Cozo query failed without diagnostic output";return false;}
    out=raw;cozo_free_str(raw); const auto parsed=json::parse(out,nullptr,false);
    if(parsed.is_object()&&parsed.value("ok",true)==false){error=parsed.value("message",std::string("semi-oracle Cozo query failed"));return false;}
    return true;
}
bool agent_semi_oracle_cozo_store::open(const std::string & path,std::string & error){
    close();if(path.empty()){error="semi-oracle Cozo store requires a path";return false;}
    char * e=cozo_open_db("sqlite",path.c_str(),"{}",&db_id_);if(e){error=e;cozo_free_str(e);db_id_=-1;return false;}
    std::string raw;if(!run("::relations","{}",raw,error)){close();return false;}
    auto rel=json::parse(raw,nullptr,false);
    // Existing databases may predate the immutable interaction relation.  Do
    // not replay the full :create script against them: Cozo rejects duplicate
    // relation creation.  The original four relations are all-or-nothing;
    // later additions are independently migrated.
    if (!relation_exists(rel, "agent_semi_oracle_rulesets")) {
        if(!run(schema,"{}",raw,error)){close();return false;}
    } else if (!relation_exists(rel, "agent_semi_oracle_interactions")) {
        if(!run(interaction_schema,"{}",raw,error)){close();return false;}
    }
    if (!run("::relations", "{}", raw, error)) { close(); return false; }
    rel = json::parse(raw, nullptr, false);
    if (!relation_exists(rel, "agent_semi_oracle_game_states")) {
        const char * game_state_schema = R"COZO(
{
 ?[session_id, ruleset_id, revision, payload] <- [['__probe__', '__probe__', 1, '{}']]
 :create agent_semi_oracle_game_states { session_id: String, ruleset_id: String, revision: Int => payload: String }
}
{
 ?[session_id, ruleset_id, revision] <- [['__probe__', '__probe__', 1]]
 :delete agent_semi_oracle_game_states { session_id, ruleset_id, revision }
}
)COZO";
        if (!run(game_state_schema, "{}", raw, error)) { close(); return false; }
    }
    return true;
}
void agent_semi_oracle_cozo_store::close(){if(db_id_>=0){cozo_close_db(db_id_);db_id_=-1;}}

bool agent_semi_oracle_cozo_store::put_revision(const common_semi_oracle_ruleset & r,std::string & error){
    error.clear();if(!common_semi_oracle_validate_ruleset(r,error))return false;
    if(r.revision>static_cast<uint64_t>(std::numeric_limits<int64_t>::max())){error="RuleSet revision exceeds Cozo integer range";return false;}
    common_semi_oracle_ruleset old;bool found=false;
    if(!get_revision(r.id,r.revision,old,found,error))return false;
    const auto payload=common_semi_oracle_ruleset_to_json(r);
    if(found){if(common_semi_oracle_ruleset_to_json(old)==payload)return true;error="RuleSet revisions are immutable";return false;}
    const auto params=json{{"id",r.id},{"rev",r.revision},{"payload",payload}}.dump();std::string raw;
    return run("?[ruleset_id, revision, payload] <- [[$id, $rev, $payload]] :put agent_semi_oracle_rulesets { ruleset_id, revision => payload }",params,raw,error);
}

bool agent_semi_oracle_cozo_store::get_revision(const std::string & id,uint64_t rev,common_semi_oracle_ruleset & r,bool & found,std::string & error) const {
    found=false;error.clear();if(rev>static_cast<uint64_t>(std::numeric_limits<int64_t>::max())){error="RuleSet revision exceeds Cozo integer range";return false;}
    std::string raw;const auto params=json{{"id",id},{"rev",rev}}.dump();
    if(!run("?[payload] := *agent_semi_oracle_rulesets[$id, $rev, payload]",params,raw,error))return false;
    const auto j=json::parse(raw,nullptr,false);if(!j.is_object()||!j.contains("rows")||!j["rows"].is_array()){error="semi-oracle RuleSet query returned invalid rows";return false;}
    if(j["rows"].empty())return true;if(!j["rows"][0].is_array()||j["rows"][0].empty()||!j["rows"][0][0].is_string()){error="semi-oracle RuleSet row is invalid";return false;}
    if(!common_semi_oracle_ruleset_from_json(j["rows"][0][0].get<std::string>(),r,error))return false;found=true;return true;
}

bool agent_semi_oracle_cozo_store::promote_to_host(const std::string & id,uint64_t rev,std::string & error){
    common_semi_oracle_ruleset r;bool found=false;if(!get_revision(id,rev,r,found,error))return false;if(!found){error="cannot promote missing RuleSet revision";return false;}
    const auto semantic=r.semantic_fingerprint;r.scope=common_semi_oracle_scope::host;r.scope_id="host";
    if(r.semantic_fingerprint!=semantic){error="RuleSet promotion unexpectedly changed semantic fingerprint";return false;}
    const auto params=json{{"id",id},{"rev",rev},{"payload",common_semi_oracle_ruleset_to_json(r)}}.dump();std::string raw;
    if(!run("?[ruleset_id, revision, payload] <- [[$id, $rev, $payload]] :put agent_semi_oracle_rulesets { ruleset_id, revision => payload }",params,raw,error))return false;
    const auto catalog=json{{"id",id},{"rev",rev}}.dump();
    return run("?[ruleset_id, revision] <- [[$id, $rev]] :put agent_semi_oracle_host_catalog { ruleset_id => revision }",catalog,raw,error);
}

bool agent_semi_oracle_cozo_store::bind_session(const std::string & session,const std::string & id,uint64_t rev,std::string & error){
    if(session.empty()){error="session binding requires a session id";return false;}common_semi_oracle_ruleset r;bool found=false;if(!get_revision(id,rev,r,found,error))return false;if(!found){error="cannot bind session to missing RuleSet revision";return false;}
    std::string raw;return run("?[session_id, ruleset_id, revision] <- [[$session, $id, $rev]] :put agent_semi_oracle_session_bindings { session_id => ruleset_id, revision }",json{{"session",session},{"id",id},{"rev",rev}}.dump(),raw,error);
}
bool agent_semi_oracle_cozo_store::get_session_binding(const std::string & session,std::string & id,uint64_t & rev,bool & found,std::string & error) const {
    found=false;std::string raw;if(!run("?[ruleset_id, revision] := *agent_semi_oracle_session_bindings[$session, ruleset_id, revision]",json{{"session",session}}.dump(),raw,error))return false;
    const auto j=json::parse(raw,nullptr,false);if(!j.is_object()||!j.contains("rows")||!j["rows"].is_array()){error="semi-oracle binding query returned invalid rows";return false;}if(j["rows"].empty())return true;
    const auto & row=j["rows"][0];if(!row.is_array()||row.size()<2||!row[0].is_string()||!row[1].is_number_unsigned()&&!row[1].is_number_integer()){error="semi-oracle binding row is invalid";return false;}
    id=row[0].get<std::string>();rev=row[1].get<uint64_t>();found=true;return true;
}
bool agent_semi_oracle_cozo_store::get_session_game_state(
        const std::string & session, const std::string & id, uint64_t rev,
        std::string & state, bool & found, std::string & error) const {
    found = false; state.clear(); error.clear(); std::string raw;
    if (session.empty() || id.empty() || rev > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        error = "semi-oracle game-state lookup identity is invalid"; return false;
    }
    const auto params = json{{"session", session}, {"id", id}, {"rev", rev}}.dump();
    if (!run("?[payload] := *agent_semi_oracle_game_states[$session, $id, $rev, payload]", params, raw, error)) return false;
    const auto value = json::parse(raw, nullptr, false);
    if (!value.is_object() || !value.contains("rows") || !value["rows"].is_array()) {
        error = "semi-oracle game-state query returned invalid rows"; return false;
    }
    if (value["rows"].empty()) return true;
    const auto & row = value["rows"][0];
    if (!row.is_array() || row.empty() || !row[0].is_string()) {
        error = "semi-oracle game-state row is invalid"; return false;
    }
    state = row[0].get<std::string>(); found = true; return true;
}
bool agent_semi_oracle_cozo_store::put_session_game_state(
        const std::string & session, const std::string & id, uint64_t rev,
        const std::string & state, std::string & error) {
    error.clear();
    if (session.empty() || id.empty() || state.empty() ||
            rev > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        error = "semi-oracle game-state write identity or payload is invalid"; return false;
    }
    std::string raw;
    return run("?[session_id, ruleset_id, revision, payload] <- [[$session, $id, $rev, $payload]] :put agent_semi_oracle_game_states { session_id, ruleset_id, revision => payload }",
        json{{"session", session}, {"id", id}, {"rev", rev}, {"payload", state}}.dump(), raw, error);
}
bool agent_semi_oracle_cozo_store::append_evaluation(const std::string & eid,const common_semi_oracle_result & result,std::string & error){
    if(eid.empty()||result.ruleset_id.empty()){error="semi-oracle evaluation requires ids";return false;}
    const json payload={{"ruleset_id",result.ruleset_id},{"ruleset_revision",result.ruleset_revision},{"verdict",common_semi_oracle_verdict_name(result.verdict)},{"level",common_semi_oracle_level_name(result.level)},{"evaluator_ref",result.evaluator_ref},{"evaluator_revision",result.evaluator_revision},{"expansions",result.expansions},{"depth_reached",result.depth_reached},{"proof_json",result.proof_json},{"counterexample_json",result.counterexample_json},{"reason",result.reason},{"guidance_ref",result.guidance_ref}};
    std::string raw;return run("?[evaluation_id, ruleset_id, revision, payload] <- [[$eid, $id, $rev, $payload]] :put agent_semi_oracle_evaluations { evaluation_id => ruleset_id, revision, payload }",json{{"eid",eid},{"id",result.ruleset_id},{"rev",result.ruleset_revision},{"payload",payload.dump()}}.dump(),raw,error);
}
bool agent_semi_oracle_cozo_store::get_evaluation_payload(const std::string & eid,std::string & payload,bool & found,std::string & error) const {
    found=false;payload.clear();std::string raw;
    if(!run("?[payload] := *agent_semi_oracle_evaluations[$eid, ruleset_id, revision, payload]",json{{"eid",eid}}.dump(),raw,error))return false;
    const auto value=json::parse(raw,nullptr,false);
    if(!value.is_object()||!value.contains("rows")||!value["rows"].is_array()){error="semi-oracle evaluation query returned invalid rows";return false;}
    if(value["rows"].empty())return true;
    if(!value["rows"][0].is_array()||value["rows"][0].empty()||!value["rows"][0][0].is_string()){error="semi-oracle evaluation payload row is invalid";return false;}
    payload=value["rows"][0][0].get<std::string>();found=true;return true;
}
bool agent_semi_oracle_cozo_store::put_interaction(
        const common_semi_oracle_interaction & value, std::string & error) {
    if (!common_semi_oracle_interaction_validate(value, error)) return false;
    common_semi_oracle_interaction existing; bool found = false;
    if (!get_interaction(value.id, existing, found, error)) return false;
    if (found) {
        if (common_semi_oracle_interaction_to_json(existing) ==
                common_semi_oracle_interaction_to_json(value)) { error.clear(); return true; }
        error = "SemiOracle interactions are immutable"; return false;
    }
    std::string raw;
    return run("?[interaction_id, session_id, ruleset_id, revision, payload] <- [[$id, $session, $ruleset, $rev, $payload]] :put agent_semi_oracle_interactions { interaction_id => session_id, ruleset_id, revision, payload }",
        json{{"id", value.id}, {"session", value.session_id}, {"ruleset", value.ruleset_id},
            {"rev", value.ruleset_revision}, {"payload", common_semi_oracle_interaction_to_json(value)}}.dump(), raw, error);
}
bool agent_semi_oracle_cozo_store::get_interaction(
        const std::string & id, common_semi_oracle_interaction & value,
        bool & found, std::string & error) const {
    found = false; value = {}; std::string raw;
    if (!run("?[payload] := *agent_semi_oracle_interactions[$id, session_id, ruleset_id, revision, payload]", json{{"id", id}}.dump(), raw, error)) return false;
    const auto parsed = json::parse(raw, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("rows") || !parsed["rows"].is_array()) { error = "semi-oracle interaction query returned invalid rows"; return false; }
    if (parsed["rows"].empty()) return true;
    if (!parsed["rows"][0].is_array() || parsed["rows"][0].empty() || !parsed["rows"][0][0].is_string()) { error = "semi-oracle interaction row is invalid"; return false; }
    if (!common_semi_oracle_interaction_from_json(parsed["rows"][0][0].get<std::string>(), value, error)) return false;
    found = true; return true;
}
std::vector<std::string> agent_semi_oracle_cozo_store::list_host_rulesets(std::string & error) const {
    error.clear();std::vector<std::string> ids;std::string raw;if(!run("?[ruleset_id] := *agent_semi_oracle_host_catalog[ruleset_id, revision]","{}",raw,error))return ids;
    const auto j=json::parse(raw,nullptr,false);if(!j.is_object()||!j.contains("rows")||!j["rows"].is_array()){error="semi-oracle host catalog query returned invalid rows";return {};}
    for(const auto & row:j["rows"])if(row.is_array()&&!row.empty()&&row[0].is_string())ids.push_back(row[0].get<std::string>());
    std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());return ids;
}
