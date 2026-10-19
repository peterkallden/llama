#include "agent/adaptation/semi-oracle-outcome-guidance.h"
#include "agent/prediction/outcome-guidance.h"
#include "agent/adaptation/semi-oracle-target.h"
#include "hash/hash.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <map>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace {
using json = nlohmann::ordered_json;
using counts = std::array<double, 3>; // proven goal suffix, proven dead end, unknown
std::string fingerprint(const std::string & text) {
    return "sha256:" + hash_sha256_hex(text.data(), text.size());
}
const std::set<std::string> actions = {"simplify lhs", "simplify rhs",
    "move variable terms to the left", "move constants to the right",
    "divide both sides by the non-zero coefficient"};
struct projection_cache_entry { std::string key; common_outcome_guidance_projection value; };
std::mutex projection_cache_mutex;
std::list<projection_cache_entry> projection_cache_lru;
std::map<std::string,std::list<projection_cache_entry>::iterator> projection_cache_index;
constexpr size_t projection_cache_capacity=16;
}

bool common_semi_oracle_outcome_guidance_config_validate(
        const common_semi_oracle_outcome_guidance_config & config, std::string & error) {
    error.clear();
    if ((config.mode != "off" && config.mode != "shadow" && config.mode != "experimental") ||
            config.max_history_evaluations == 0 || config.max_history_evaluations > 1024 ||
            (config.search_algorithm != "exact" && config.search_algorithm != "focal") ||
            (config.sampling_policy != "posterior_sampling" && config.sampling_policy != "posterior_mean") ||
            !std::isfinite(config.suboptimality_bound) || config.suboptimality_bound < 1.0 ||
            config.suboptimality_bound > 2.0 ||
            (config.search_algorithm == "exact" && config.suboptimality_bound != 1.0)) {
        error = "SemiOracle requires valid guidance mode/history limit, exact or focal search with bound 1..2, and posterior_sampling or posterior_mean";
        return false;
    }
    return true;
}

bool common_semi_oracle_evaluate_with_history(common_semi_oracle_store & store,
        const common_semi_oracle_outcome_guidance_config & config,
        const std::string & host, const std::string & session,
        const common_semi_oracle_ruleset & ruleset, const common_semi_oracle_query & query,
        common_semi_oracle_result & result, std::string & error, const std::string & project) {
    if (!common_semi_oracle_outcome_guidance_config_validate(config, error)) return false;
    const auto started=std::chrono::steady_clock::now();
    const bool target=ruleset.domain==common_semi_oracle_domain::target_hunt;
    auto allowed=[&](const std::string & key){
        if(!target)return actions.count(key)!=0;
        return key.rfind("target_hunt_features_v3:fine:",0)==0 ||
            key.rfind("target_hunt_features_v3:coarse:",0)==0;
    };
    auto category=[&](const std::string & state,const std::string & action){
        return target?common_semi_oracle_target_category(state,action):action;
    };
    const bool trusted = !ruleset.assertions.empty() && std::all_of(
        ruleset.assertions.begin(), ruleset.assertions.end(), [](const auto & assertion) {
            return assertion.authority != common_semi_oracle_rule_authority::inferred;
        });
    if (config.mode == "off" || (!target && ruleset.domain != common_semi_oracle_domain::linear_algebra) ||
            query.level == common_semi_oracle_level::local || query.goal != (target?"reach:target":"solve:x") || !trusted)
        return common_semi_oracle_evaluate(ruleset, query, result, error);

    // Scope, rule revision, proof contract and budget all partition this family.
    const json context = {{"family", target?"target_hunt_search_suffix_v3":"algebra_search_suffix_v1"},
        {"scope", fingerprint(json::array({host, project, session}).dump())},
        {"ruleset_id", ruleset.id}, {"revision", ruleset.revision},
        {"rule_fingerprint", ruleset.semantic_fingerprint},
        {"action_contract", ruleset.action_contract_fingerprint},
        {"evaluator_revision", target?"target_hunt_astar_v3":"linear_astar_v1"}, {"goal", query.goal},
        {"host_action_order", query.level == common_semi_oracle_level::guided
            ? query.guidance : std::vector<std::string>{}},
        {"max_expansions", query.budget.max_expansions}, {"max_depth", query.budget.max_depth}};
    const auto state = json::parse(query.state_json, nullptr, false);
    const std::string fixture = target?common_semi_oracle_target_group(query.state_json):fingerprint(state.is_object() && state.contains("lhs") && state.contains("rhs")
        ? json{{"lhs", state["lhs"]}, {"rhs", state["rhs"]}}.dump() : query.state_json);
    std::vector<common_semi_oracle_search_history_record> records;
    std::string fallback;
    std::string history_error;
    try {
        if (!store.list_search_history(ruleset.id, ruleset.revision,
                config.max_history_evaluations, records, history_error))
            fallback = "search_history_unavailable";
    } catch (...) { fallback = "search_history_exception"; }
    std::map<std::string, std::map<std::string, counts>> grouped;
    json accepted = json::array();
    size_t rejected = 0;
    size_t duplicates = 0;
    std::map<std::string, std::string> unique;
    std::set<std::string> conflicts;
    if (fallback.empty()) for (const auto & record : records) {
        if (record.evaluation_id.empty()) { ++rejected; continue; }
        const auto inserted = unique.emplace(record.evaluation_id, record.guidance_json);
        if (!inserted.second) {
            ++duplicates;
            if (inserted.first->second != record.guidance_json) conflicts.insert(record.evaluation_id);
        }
    }
    for (const auto & record : unique) {
        if (conflicts.count(record.first)) { ++rejected; continue; }
        try {
            const auto row = json::parse(record.second);
            if (!row.at("schema_version").is_number_integer() ||
                    row.at("schema_version") != (target?4:1) || row.at("context") != context ||
                    !row.at("fixture_group").is_string() ||
                    !row.at("observations").is_object()) { ++rejected; continue; }
            const auto group = row.at("fixture_group").get<std::string>();
            // Keep the complete compatible history in the immutable cached
            // projection. The query fixture is excluded only while aggregating
            // that projection below, so the same snapshot can serve other
            // queries without rebuilding a fixture-specific cache entry.
            if (group.empty()) { ++rejected; continue; }
            std::map<std::string, counts> parsed;
            for (auto it = row.at("observations").begin(); it != row.at("observations").end(); ++it) {
                if (!allowed(it.key()) || !it.value().is_array() || it.value().size() != 3)
                    throw std::invalid_argument("invalid search action counts");
                counts values{};
                for (size_t i = 0; i < 3; ++i) {
                    if (!it.value().at(i).is_number()) throw std::invalid_argument("non-numeric search outcome weight");
                    values[i] = it.value().at(i).get<double>();
                    if (!std::isfinite(values[i]) || values[i] < 0.0 || values[i] > 100000.0)
                        throw std::invalid_argument("invalid search outcome weight");
                }
                parsed.emplace(it.key(), values);
            }
            for (const auto & item : parsed) for (size_t i = 0; i < 3; ++i)
                grouped[group][item.first][i] += item.second[i];
            accepted.push_back({{"evaluation_id", record.first},
                {"fixture_group", group}, {"observations", row.at("observations")}});
        } catch (...) { ++rejected; }
    }
    std::vector<common_outcome_guidance_projection_observation> projection_rows;
    for(const auto & group:grouped)for(const auto & item:group.second)
        projection_rows.push_back({group.first,item.first,{item.second[0],item.second[1],item.second[2]}});
    const std::string source_fingerprint=fingerprint(json{{"context",context},{"records",accepted}}.dump());
    common_outcome_guidance_projection projection;
    bool projection_hit=false;
    {
        std::lock_guard<std::mutex> lock(projection_cache_mutex);
        const auto cached=projection_cache_index.find(source_fingerprint);
        if(cached!=projection_cache_index.end()) {
            projection=cached->second->value;
            projection_cache_lru.splice(projection_cache_lru.begin(),projection_cache_lru,cached->second);
            projection_hit=true;
        }
    }
    if(!projection_hit) {
        if(!common_outcome_guidance_build_projection(projection_rows,"semi-oracle-outcome-projection-v2",
                source_fingerprint,0,projection,error)) {
            fallback="search_history_projection_invalid";
            history_error=error;
            error.clear();
            projection={};
        } else {
            std::lock_guard<std::mutex> lock(projection_cache_mutex);
            const auto concurrent=projection_cache_index.find(source_fingerprint);
            if(concurrent!=projection_cache_index.end()) {
                projection=concurrent->second->value;
                projection_cache_lru.splice(projection_cache_lru.begin(),projection_cache_lru,concurrent->second);
            } else {
                projection_cache_lru.push_front({source_fingerprint,projection});
                projection_cache_index[source_fingerprint]=projection_cache_lru.begin();
            }
            while(projection_cache_lru.size()>projection_cache_capacity){
                projection_cache_index.erase(projection_cache_lru.back().key);
                projection_cache_lru.pop_back();
            }
        }
    }
    std::map<std::string,std::vector<double>> projected_weights;
    std::map<std::string,size_t> group_counts;
    projection.aggregate_excluding(fixture,projected_weights,group_counts);
    size_t eligible_fixture_groups=0;
    for(const auto & group:projection.fixture_categories) {
        if(group.first==fixture)continue;
        const bool contributes=std::any_of(group.second.begin(),group.second.end(),[](const auto & item){
            return std::any_of(item.second.begin(),item.second.end(),[](double value){return value>0.0;});
        });
        if(contributes)++eligible_fixture_groups;
    }
    std::map<std::string,counts> weights;
    for(const auto & item:projected_weights)if(item.second.size()==3)
        weights[item.first]={item.second[0],item.second[1],item.second[2]};
    const std::string history_fingerprint=projection.fingerprint;
    json decisions = json::array();
    size_t omitted_decisions = 0;
    double adviser_ms=0.0;
    auto guided = query;
    guided.record_search_history = true;
    guided.use_historical_order = config.mode == "experimental";
    std::map<std::pair<std::string,std::string>,double> transition_scores;
    if (!query.transition_adviser && !query.transition_score && fallback.empty()) {
        guided.transition_score = [&](const std::string & before, const std::string &,
                const std::string & action) {
            const auto score=transition_scores.find({before,action});
            if(score==transition_scores.end())throw std::runtime_error("missing transition posterior score");
            return score->second;
        };
        guided.transition_adviser = [&](const std::string & current, const std::vector<std::string> & candidates,
                std::vector<size_t> & order, std::string & advice_error) {
            const auto advice_started=std::chrono::steady_clock::now();
            advice_error.clear();
            order.resize(candidates.size()); std::iota(order.begin(), order.end(), size_t{0});
            auto proposed = order;
            std::vector<double> utilities;
            json predictions = json::array();
            bool matching_history = false;
            // Target sampling depends on canonical geometry/posterior counts,
            // not fresh problem/evaluation IDs. Provenance remains in the history fingerprint.
            const auto sampling_input=target
                ?json::array({config.seed,fixture,json::parse(current).at("position"),weights,ruleset.semantic_fingerprint})
                :json::array({config.seed,current,history_fingerprint});
            const auto seed_base = sampling_input.dump();
            std::map<std::string, double> sampled_by_category;
            for (const auto & action : candidates) {
                const auto fine_key=category(current,action);
                const auto coarse_key=target?common_semi_oracle_target_coarse_category(current,action):fine_key;
                const auto key=target&&group_counts[fine_key]<8?coarse_key:fine_key;
                const auto found = weights.find(key);
                const counts w = found == weights.end() ? counts{} : found->second;
                matching_history = matching_history || w[0] + w[1] + w[2] > 0.0;
                auto sample = sampled_by_category.find(key);
                if (sample == sampled_by_category.end()) {
                    common_outcome_guidance_posterior p;
                    if (!common_outcome_guidance_make_posterior(w[0] + w[1], w[2], 0.5,
                            {w[0], w[1]}, {0.5, 0.5}, 0.0, 0.0, p)) {
                        advice_error = "invalid search posterior"; return false;
                    }
                    const auto seed_hex = fingerprint(json::array({seed_base,key}).dump()).substr(7, 16);
                    std::mt19937_64 category_rng(std::stoull(seed_hex, nullptr, 16));
                    const double score=config.sampling_policy=="posterior_mean"
                        ? p.p_known*p.category_probabilities[0]
                        : common_outcome_guidance_sample_expected_utility(p, {1.0, 0.0}, category_rng);
                    sample = sampled_by_category.emplace(key,score).first;
                }
                utilities.push_back(sample->second);
                common_outcome_guidance_posterior p;
                if (!common_outcome_guidance_make_posterior(w[0] + w[1], w[2], 0.5,
                        {w[0], w[1]}, {0.5, 0.5}, 0.0, 0.0, p)) {
                    advice_error = "invalid search posterior"; return false;
                }
                predictions.push_back({{"action", action}, {"category", key}, {"p_known", p.p_known},
                    {"p_known_interval", {p.p_known_low, p.p_known_high}},
                    {"p_goal_given_known", p.category_probabilities[0]},
                    {"p_dead_end_given_known", p.category_probabilities[1]},
                    {"p_goal_interval", {p.category_lows[0], p.category_highs[0]}},
                    {"p_dead_end_interval", {p.category_lows[1], p.category_highs[1]}},
                    {"effective_group_weight", w[0] + w[1] + w[2]},
                    {"sampled_utility", utilities.back()}});
            }
            if (matching_history) std::stable_sort(proposed.begin(), proposed.end(),
                [&](size_t a, size_t b) { return utilities[a] > utilities[b]; });
            // Absolute utilities are comparable across different parents.
            // Cold start preserves deterministic frontier order for all siblings.
            for(size_t i=0;i<candidates.size();++i)
                transition_scores[{current,candidates[i]}]=matching_history?-utilities[i]:0.0;
            if (config.mode == "experimental") order = proposed;
            if (decisions.size() < 128) decisions.push_back({{"state_fingerprint", fingerprint(current)},
                {"selection_reason", matching_history ? config.sampling_policy : "cold_start_original_order"},
                {"predictions", predictions}, {"original_actions", candidates}, {"proposed_order", proposed}, {"executed_order", order}});
            else ++omitted_decisions;
            adviser_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-advice_started).count();
            return true;
        };
    } else if (query.transition_adviser || query.transition_score) fallback = "existing host transition guidance retained";
    const auto search_started=std::chrono::steady_clock::now();
    if (!common_semi_oracle_evaluate(ruleset, guided, result, error)) return false;
    const double search_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-search_started).count();
    // No trace means the domain did not run this search (e.g. all-real solutions).
    auto trace = json::parse(result.search_guidance_json, nullptr, false);
    if (!trace.is_object()) return true;
    std::map<std::string, counts> observed;
    for (const auto & observation : trace.at("observations")) {
        const auto before=observation.value("state_before",std::string{});
        const auto action_text=observation.at("action").get<std::string>();
        const auto action = category(before,action_text);
        const auto outcome = observation.at("outcome").get<std::string>();
        const size_t outcome_index=outcome == "goal_reached" ? 0 : outcome == "dead_end" ? 1 : 2;
        if(allowed(action))++observed[action][outcome_index];
        if(target) {
            const auto coarse=common_semi_oracle_target_coarse_category(before,action_text);
            if(allowed(coarse))++observed[coarse][outcome_index];
        }
    }
    json observations = json::object();
    for (const auto & item : observed) observations[item.first] = item.second;
    json history_refs = json::array();
    for (const auto & item : accepted) history_refs.push_back(item["evaluation_id"]);
    result.search_guidance_json = json{{"schema_version", target?4:1}, {"context", context},
        {"fixture_group", fixture}, {"observations", observations},
        {"diagnostics", {{"mode", config.mode}, {"seed", config.seed},
            {"sampling_policy",config.sampling_policy},
            {"search_algorithm",result.applied_search_policy.algorithm==common_semi_oracle_search_algorithm::focal?"focal":"exact"},
            {"suboptimality_bound",result.applied_search_policy.suboptimality_bound},
            {"family",context["family"]},{"evaluator_revision",context["evaluator_revision"]},
            {"ruleset_id", ruleset.id}, {"ruleset_revision", ruleset.revision},
            {"fixture_group", fixture}, {"scope_fingerprint", context["scope"]},
            {"history_fingerprint", history_fingerprint}, {"history_records", records.size()},
            {"accepted_records", accepted.size()}, {"rejected_records", rejected},
            {"duplicate_records", duplicates}, {"conflicting_records", conflicts.size()},
            {"history_evaluation_refs", history_refs},
            {"fixture_groups", eligible_fixture_groups}, {"fallback_reason", fallback},
            {"projection_ms",std::chrono::duration<double,std::milli>(search_started-started).count()},
            {"projection_cache_hit",projection_hit},{"projection_builder_revision",projection.builder_revision},
            {"search_with_adviser_ms",search_ms},
            {"adviser_ms",adviser_ms},{"search_ms",std::max(0.0,search_ms-adviser_ms)},
            {"history_diagnostic", history_error},
            {"callback_fallbacks", trace.value("callback_fallbacks", size_t{0})},
            {"callback_error", trace.value("callback_error", std::string{})},
            {"decisions", decisions}, {"decision_count", decisions.size() + omitted_decisions},
            {"omitted_decisions", omitted_decisions}}}}.dump();
    if (result.guidance_ref.empty() && !history_fingerprint.empty())
        result.guidance_ref = "semi-oracle-history:" + history_fingerprint;
    if(target) {
        auto persisted=json::parse(result.search_guidance_json);
        persisted["problem_state"]=state;
        persisted["search_observations"]=json::array();
        for(const auto & observation:trace.at("observations")) {
            const auto before=json::parse(observation.at("state_before").get<std::string>());
            const auto after=json::parse(observation.at("state_after").get<std::string>());
            const auto before_text=observation.at("state_before").get<std::string>();
            const auto action=observation.at("action").get<std::string>();
            persisted["search_observations"].push_back({{"action",observation["action"]},
                {"outcome",observation["outcome"]},{"position_before",before["position"]},
                {"position_after",after["position"]},
                {"category",category(before_text,action)},
                {"coarse_category",common_semi_oracle_target_coarse_category(before_text,action)}});
        }
        result.search_guidance_json=persisted.dump();
    }
    return true;
}
