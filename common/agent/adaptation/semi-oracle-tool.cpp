#include "agent/adaptation/semi-oracle-tool.h"
#include "agent/adaptation/semi-oracle-sft-candidate.h"

#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace {
using json = nlohmann::ordered_json;
std::string opaque_id(const std::string & host, const std::string & session, const char * kind) {
    static std::atomic<uint64_t> sequence{0};
    const auto now = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    uint64_t hash = 1469598103934665603ULL;
    const auto input = host + "\n" + session + "\n" + kind + "\n" +
        std::to_string(now) + "\n" + std::to_string(sequence.fetch_add(1));
    for (const unsigned char c : input) { hash ^= c; hash *= 1099511628211ULL; }
    std::ostringstream out;out<<"semi-oracle://"<<kind<<"/"<<std::hex<<hash;return out.str();
}
}

common_semi_oracle_tool_service::common_semi_oracle_tool_service(
        common_semi_oracle_store & store, std::string host_namespace,
        std::string session_id)
    : store_(store), host_namespace_(std::move(host_namespace)),
      session_id_(std::move(session_id)) {}

bool common_semi_oracle_tool_service::define_ruleset(const std::string & dsl,
        common_semi_oracle_rule_authority authority,
        common_semi_oracle_ruleset & active, std::string & error) {
    error.clear();
    if(host_namespace_.empty()||session_id_.empty()){error="semi-oracle tool service requires host and session identity";return false;}
    std::string existing_id;uint64_t existing_revision=0;bool bound=false;
    if(!store_.get_session_binding(session_id_,existing_id,existing_revision,bound,error))return false;
    if(!bound) {
        if(!common_semi_oracle_parse_ruleset_dsl(dsl,opaque_id(host_namespace_,session_id_,"ruleset"),
                session_id_,active,error,authority))return false;
    } else {
        common_semi_oracle_ruleset current;bool found=false;
        if(!store_.get_revision(existing_id,existing_revision,current,found,error))return false;
        if(!found){error="active session RuleSet binding points to a missing revision";return false;}
        if(!common_semi_oracle_revise_ruleset(current,dsl,authority,session_id_,active,error))return false;
    }
    if(!store_.put_revision(active,error)||!store_.bind_session(session_id_,active.id,active.revision,error))return false;
    return true;
}

bool common_semi_oracle_tool_service::active_ruleset(
        common_semi_oracle_ruleset & ruleset,bool & found,std::string & error) const {
    std::string id;uint64_t revision=0;
    if(!store_.get_session_binding(session_id_,id,revision,found,error))return false;
    if(!found)return true;
    if(!store_.get_revision(id,revision,ruleset,found,error))return false;
    if(!found)error="active session RuleSet binding points to a missing revision";
    return error.empty();
}

bool common_semi_oracle_tool_service::action_contract(std::string & contract,std::string & error) const {
    common_semi_oracle_ruleset ruleset;bool found=false;
    if(!active_ruleset(ruleset,found,error))return false;
    if(!found){error="no active semi-oracle RuleSet is bound to this session";return false;}
    contract=common_semi_oracle_action_contract(ruleset);return true;
}

bool common_semi_oracle_tool_service::verify_action(const std::string & state_json,
        const std::string & action_dsl,common_semi_oracle_level level,
        const std::string & goal,common_semi_oracle_budget budget,
        const std::vector<std::string> & guidance,
        common_semi_oracle_result & result,std::string & error) {
    common_semi_oracle_ruleset ruleset;bool found=false;
    if(!active_ruleset(ruleset,found,error))return false;
    if(!found){error="no active semi-oracle RuleSet is bound to this session";return false;}
    common_semi_oracle_query query;
    query.ruleset_id=ruleset.id;query.ruleset_revision=ruleset.revision;
    query.level=level;query.goal=goal;query.state_json=state_json;
    query.action_dsl=action_dsl;query.budget=budget;query.guidance=guidance;
    if(!common_semi_oracle_evaluate(ruleset,query,result,error))return false;
    const auto evaluation_id=opaque_id(host_namespace_,session_id_,"evaluation");
    return store_.append_evaluation(evaluation_id,result,error);
}

bool common_semi_oracle_tool_service::rank_placements(const std::string & state_json,
        common_semi_oracle_budget budget,
        std::vector<common_semi_oracle_placement_action_value> & values,
        std::string & error) {
    common_semi_oracle_ruleset ruleset;bool found=false;
    if(!active_ruleset(ruleset,found,error))return false;
    if(!found){error="no active semi-oracle RuleSet is bound to this session";return false;}
    if(!common_semi_oracle_rank_placements(ruleset,state_json,budget,values,error))return false;

    json actions=json::array();
    for(const auto & value:values) {
        actions.push_back({{"player",value.player},{"row",value.row},{"column",value.column},
            {"value",static_cast<int>(value.value)},{"expansions",value.expansions},
            {"depth_reached",value.depth_reached},{"proof_json",value.proof_json}});
    }
    common_semi_oracle_result result;
    result.ruleset_id=ruleset.id;result.ruleset_revision=ruleset.revision;
    result.evaluator_ref="semi-oracle://evaluator/placement_game";
    result.evaluator_revision="minimax-v1";
    result.level=common_semi_oracle_level::bounded;
    const bool ranking_complete=!std::any_of(values.begin(),values.end(),[](const auto & value){
        return value.value==common_semi_oracle_placement_value::unknown;
    });
    result.verdict=ranking_complete?common_semi_oracle_verdict::satisfied:common_semi_oracle_verdict::unknown;
    result.proof_json=json{{"algorithm","memoized_minimax_v1"},
        {"state_fingerprint",values.front().state_fingerprint},
        {"action_contract_fingerprint",ruleset.action_contract_fingerprint},
        {"actions",std::move(actions)}}.dump();
    const std::string evaluation_id=opaque_id(host_namespace_,session_id_,"evaluation");
    if(!store_.append_evaluation(evaluation_id,result,error))return false;
    for(auto & value:values) {
        value.persisted_evaluation_ref=evaluation_id;
        value.ranking_complete=ranking_complete;
    }
    return true;
}

bool common_semi_oracle_tool_service::make_sft_candidate(
        const common_semi_oracle_placement_action_value & preferred,
        const std::string & approved_prompt, const std::string & redaction_policy_id,
        const std::string & redaction_method, bool curator_approved,
        common_training_candidate & candidate, std::string & error) const {
    common_semi_oracle_ruleset ruleset;
    bool found = false;
    if (!active_ruleset(ruleset, found, error)) return false;
    if (!found) { error = "no active semi-oracle RuleSet is bound to this session"; return false; }
    if (preferred.ruleset_id != ruleset.id || preferred.ruleset_revision != ruleset.revision) {
        error = "SFT action does not belong to the active session RuleSet revision";
        return false;
    }
    std::string payload;
    if (!store_.get_evaluation_payload(preferred.persisted_evaluation_ref, payload, found, error)) return false;
    if (!found) { error = "SFT action evaluation is not present in the configured semi-oracle store"; return false; }
    common_agent_scope scope;
    scope.namespace_id = host_namespace_;
    scope.session_id = session_id_;
    return common_semi_oracle_make_sft_candidate(ruleset, preferred, payload, scope,
        approved_prompt, redaction_policy_id, redaction_method, curator_approved, candidate, error);
}

bool common_semi_oracle_tool_service::promote_active_ruleset_to_host(std::string & error) {
    common_semi_oracle_ruleset ruleset;bool found=false;
    if(!active_ruleset(ruleset,found,error))return false;
    if(!found){error="no active semi-oracle RuleSet is bound to this session";return false;}
    // Promotion changes visibility only. It never upgrades inferred rule
    // authority or makes the RuleSet eligible for FlyDelta evidence.
    if(!common_semi_oracle_validate_ruleset(ruleset,error))return false;
    return store_.promote_to_host(ruleset.id,ruleset.revision,error);
}
