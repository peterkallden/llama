#include "agent/adaptation/semi-oracle-tool.h"
#include "agent/adaptation/semi-oracle-target.h"
#include "agent/adaptation/semi-oracle-sft-candidate.h"

#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>

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
        std::string session_id, common_semi_oracle_outcome_guidance_config guidance,
        std::string project_id)
    : store_(store), host_namespace_(std::move(host_namespace)),
      session_id_(std::move(session_id)), guidance_(std::move(guidance)), project_id_(std::move(project_id)) {}

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
    if(!store_.put_revision(active,error))return false;
    if (active.domain == common_semi_oracle_domain::placement_game) {
        const auto spec = json::parse(active.spec_json, nullptr, false);
        const auto players = spec.is_object()
            ? spec.value("players", std::vector<std::string>{}) : std::vector<std::string>{};
        const int rows = spec.is_object() ? spec.value("rows", 0) : 0;
        const int columns = spec.is_object() ? spec.value("columns", 0) : 0;
        if (players.size() != 2 || rows < 1 || columns < 1 || rows > 8 || columns > 8) {
            error = "validated placement_game RuleSet has invalid initial-state dimensions";
            return false;
        }
        json cells = json::array();
        for (int i = 0; i < rows * columns; ++i) cells.push_back(nullptr);
        const auto initial_state = json{{"cells", std::move(cells)}, {"turn", players[0]}}.dump();
        if (!store_.put_session_state(session_id_, active.id, active.revision,
                initial_state, error)) return false;
    }
    if(!store_.bind_session(session_id_,active.id,active.revision,error))return false;
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

bool common_semi_oracle_tool_service::initialize_active_state(
        const std::string & state_json, bool replace, std::string & error) {
    common_semi_oracle_ruleset ruleset; bool found = false;
    if (!active_ruleset(ruleset, found, error)) return false;
    if (!found) { error = "no active semi-oracle RuleSet is bound to this session"; return false; }
    if (!common_semi_oracle_validate_state(ruleset, state_json, error)) return false;
    std::string existing;
    if (!store_.get_session_state(session_id_, ruleset.id, ruleset.revision,
            existing, found, error)) return false;
    if (found && !replace) return true;
    return store_.put_session_state(session_id_, ruleset.id, ruleset.revision,
        state_json, error);
}

bool common_semi_oracle_tool_service::verify_action(const std::string & state_json,
        const std::string & action_dsl,common_semi_oracle_level level,
        const std::string & goal,common_semi_oracle_budget budget,
        const std::vector<std::string> & guidance,
        common_semi_oracle_result & result,std::string & error,
        common_semi_oracle_frontier_observer frontier_observer) {
    common_semi_oracle_ruleset ruleset;bool found=false;
    if(!active_ruleset(ruleset,found,error))return false;
    if(!found){error="no active semi-oracle RuleSet is bound to this session";return false;}
    common_semi_oracle_query query;
    query.ruleset_id=ruleset.id;query.ruleset_revision=ruleset.revision;
    query.level=level;query.goal=goal;query.state_json=state_json;
    query.action_dsl=action_dsl;query.budget=budget;query.guidance=guidance;
    query.search_policy.algorithm = guidance_.search_algorithm == "focal" &&
            ruleset.domain == common_semi_oracle_domain::target_hunt
        ? common_semi_oracle_search_algorithm::focal : common_semi_oracle_search_algorithm::exact;
    query.search_policy.suboptimality_bound = query.search_policy.algorithm ==
            common_semi_oracle_search_algorithm::focal ? guidance_.suboptimality_bound : 1.0;
    query.search_policy.exact_history_order = guidance_.exact_history_order;
    query.frontier_observer = std::move(frontier_observer);
    query.use_historical_order = guidance_.mode == "experimental";
    if(!common_semi_oracle_evaluate_with_history(store_, guidance_, host_namespace_,
            session_id_, ruleset, query, result, error, project_id_))return false;
    result.applied_search_policy = query.search_policy;
    const auto evaluation_id=opaque_id(host_namespace_,session_id_,"evaluation");
    if (!result.search_guidance_json.empty()) {
        auto metadata = json::parse(result.search_guidance_json, nullptr, false);
        if (metadata.is_object() && metadata.contains("diagnostics") && metadata["diagnostics"].is_object()) {
            metadata["diagnostics"]["evaluation_ref"] = evaluation_id;
            result.search_guidance_json = metadata.dump();
        }
    }
    return store_.append_evaluation(evaluation_id,result,error);
}

bool common_semi_oracle_tool_service::verify_active_placement_action(
        const std::string & action_dsl, common_semi_oracle_result & result,
        std::string & state_before, std::string & state_after, std::string & error) {
    state_before.clear(); state_after.clear();
    common_semi_oracle_ruleset ruleset; bool found = false;
    if (!active_ruleset(ruleset, found, error)) return false;
    if (!found) { error = "no active semi-oracle RuleSet is bound to this session"; return false; }
    if (ruleset.domain != common_semi_oracle_domain::placement_game) {
        error = "the active RuleSet is not a placement_game; this runtime action path requires host-owned game state";
        return false;
    }
    return verify_active_action(action_dsl, result, state_before, state_after, error);
}

bool common_semi_oracle_tool_service::verify_active_action(
        const std::string & action_dsl, common_semi_oracle_result & result,
        std::string & state_before, std::string & state_after, std::string & error) {
    state_before.clear(); state_after.clear();
    common_semi_oracle_ruleset ruleset; bool found = false;
    if (!active_ruleset(ruleset, found, error)) return false;
    if (!found) { error = "no active semi-oracle RuleSet is bound to this session"; return false; }
    if (!store_.get_session_state(session_id_, ruleset.id, ruleset.revision,
            state_before, found, error)) return false;
    if (!found) {
        error = "the active " + std::string(common_semi_oracle_domain_name(ruleset.domain)) +
            " RuleSet has no host-owned session state";
        return false;
    }
    if (!verify_action(state_before, action_dsl, common_semi_oracle_level::local,
            {}, {1, 1}, {}, result, error)) return false;
    if (result.verdict != common_semi_oracle_verdict::satisfied) return true;
    const auto proof = json::parse(result.proof_json, nullptr, false);
    if (!proof.is_object() || !proof.contains("next_state") || !proof["next_state"].is_object()) {
        error = "verified active-domain action did not return a host-verifiable next state";
        return false;
    }
    state_after = proof["next_state"].dump();
    if (!common_semi_oracle_validate_state(ruleset, state_after, error)) return false;
    return store_.put_session_state(session_id_, ruleset.id, ruleset.revision,
        state_after, error);
}

bool common_semi_oracle_tool_service::search_active_algebra_goal(
        common_semi_oracle_result & result, std::string & error) {
    return search_active_goal(result,error,{2048,16});
}

bool common_semi_oracle_tool_service::define_problem(const std::string & dsl,
        std::string & state, std::string & error) {
    common_semi_oracle_ruleset ruleset; bool found=false;
    if(!active_ruleset(ruleset,found,error))return false;
    if(!found||ruleset.domain!=common_semi_oracle_domain::target_hunt){error="problem definition requires an active target_hunt RuleSet";return false;}
    if(!common_semi_oracle_target_problem(ruleset,dsl,state,error))return false;
    auto data=json::parse(state);data["problem_id"]=opaque_id(host_namespace_,session_id_,"problem");state=data.dump();
    return initialize_active_state(state,true,error);
}

bool common_semi_oracle_tool_service::search_active_goal(common_semi_oracle_result & result,
        std::string & error, common_semi_oracle_budget budget, const std::vector<std::string> & host_order,
        common_semi_oracle_frontier_observer frontier_observer) {
    common_semi_oracle_ruleset ruleset;
    bool found = false;
    if (!active_ruleset(ruleset, found, error)) return false;
    if (!found || (ruleset.domain != common_semi_oracle_domain::linear_algebra && ruleset.domain != common_semi_oracle_domain::target_hunt)) {
        error = "goal search requires an active linear_algebra or target_hunt RuleSet";
        return false;
    }
    std::string state;
    if (!store_.get_session_state(session_id_, ruleset.id, ruleset.revision,
            state, found, error)) return false;
    if (!found) {
        error = "goal search requires host-owned active problem state";
        return false;
    }
    const auto parsed = json::parse(state, nullptr, false);
    if (ruleset.domain == common_semi_oracle_domain::target_hunt)
        return verify_action(state,"",common_semi_oracle_level::bounded,"reach:target",budget,host_order,result,error,
            std::move(frontier_observer));
    if (!parsed.is_object() || !parsed.contains("lhs") || !parsed["lhs"].is_string()) {
        error = "active algebra state is not a supported equation";
        return false;
    }
    budget.max_expansions = 2048;
    budget.max_depth = 16;
    return verify_action(state,
        "open! algebra.rewrite_equation\nlhs: " + parsed["lhs"].get<std::string>() + "\n",
        common_semi_oracle_level::bounded, "solve:x", budget, {}, result, error);
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

bool common_semi_oracle_tool_service::make_dpo_candidate(
        const common_semi_oracle_placement_action_value & chosen,
        const common_semi_oracle_placement_action_value & rejected,
        const std::string & approved_prompt, const std::string & redaction_policy_id,
        const std::string & redaction_method, bool curator_approved,
        common_training_candidate & candidate, std::string & error) const {
    common_semi_oracle_ruleset ruleset;
    bool found = false;
    if (!active_ruleset(ruleset, found, error)) return false;
    if (!found) { error = "no active session RuleSet is bound to this session"; return false; }
    if (chosen.ruleset_id != ruleset.id || chosen.ruleset_revision != ruleset.revision ||
            rejected.ruleset_id != ruleset.id || rejected.ruleset_revision != ruleset.revision) {
        error = "DPO actions do not belong to the active session RuleSet revision";
        return false;
    }
    std::string payload;
    if (!store_.get_evaluation_payload(chosen.persisted_evaluation_ref, payload, found, error)) return false;
    if (!found || chosen.persisted_evaluation_ref != rejected.persisted_evaluation_ref) {
        error = "DPO actions do not share a persisted evaluation in the configured semi-oracle store";
        return false;
    }
    common_agent_scope scope;
    scope.namespace_id = host_namespace_;
    scope.session_id = session_id_;
    return common_semi_oracle_make_dpo_candidate(ruleset, chosen, rejected, payload, scope,
        approved_prompt, redaction_policy_id, redaction_method, curator_approved,
        candidate, error);
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
