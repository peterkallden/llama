#include "agent/adaptation/semi-oracle-outcome-guidance.h"
#include "agent/prediction/outcome-guidance.h"
#include "agent/adaptation/semi-oracle-target.h"
#include "hash/hash.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
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
size_t projection_bytes(const common_outcome_guidance_projection & projection) {
    size_t bytes = sizeof(projection);
    bytes += projection.builder_revision.capacity() + projection.source_fingerprint.capacity() +
        projection.fingerprint.capacity();
    for (const auto & fixture : projection.fixture_categories) {
        bytes += sizeof(fixture) + 4 * sizeof(void *) + fixture.first.capacity();
        for (const auto & category : fixture.second)
            bytes += sizeof(category) + 4 * sizeof(void *) + category.first.capacity() +
                category.second.capacity() * sizeof(double);
    }
    return bytes;
}
}

bool common_semi_oracle_outcome_guidance_config_validate(
        const common_semi_oracle_outcome_guidance_config & config, std::string & error) {
    error.clear();
    if ((config.mode != "off" && config.mode != "shadow" && config.mode != "experimental") ||
            config.max_history_evaluations == 0 || config.max_history_evaluations > 1024 ||
            config.projection_cache_bytes > 256 * 1024 * 1024 ||
            (config.search_algorithm != "exact" && config.search_algorithm != "focal") ||
            (config.sampling_policy != "posterior_sampling" && config.sampling_policy != "posterior_mean") ||
            (config.exact_history_order != common_semi_oracle_history_order::before_heuristic &&
             config.exact_history_order != common_semi_oracle_history_order::after_heuristic) ||
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
    std::map<std::pair<std::string,std::string>,
        std::pair<std::string,std::string>> target_feature_cache;
    auto target_features = [&](const std::string & state, const std::string & action)
            -> const std::pair<std::string,std::string> & {
        const auto key = std::make_pair(state, action);
        const auto found = target_feature_cache.find(key);
        if (found != target_feature_cache.end()) return found->second;
        // History can span many maps; retaining every serialized state here
        // would bypass the projection cache's memory limit.
        if (target_feature_cache.size() >= 4096) target_feature_cache.clear();
        return target_feature_cache.emplace(key, std::make_pair(
            common_semi_oracle_target_category(state, action),
            common_semi_oracle_target_coarse_category(state, action))).first->second;
    };
    auto allowed=[&](const std::string & key){
        if(!target)return actions.count(key)!=0;
        return key.rfind("target_hunt_features_v3:fine:",0)==0 ||
            key.rfind("target_hunt_features_v3:coarse:",0)==0;
    };
    auto category=[&](const std::string & state,const std::string & action){
        return target?target_features(state,action).first:action;
    };
    auto coarse_category=[&](const std::string & state,const std::string & action){
        return target?target_features(state,action).second:category(state,action);
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
    common_outcome_guidance_projection projection;
    common_outcome_guidance_projection branch_projection;
    std::map<std::string, std::map<std::string, counts>> branch_grouped;
    bool projection_hit = false;
    uint64_t source_watermark = 0;
    bool watermark_supported = false;
    size_t rejected = 0, duplicates = 0, conflicting_records = 0, read_rows = 0;
    json accepted = json::array();
    std::string source_fingerprint;
    std::string cache_key;
    const auto store_identity = std::to_string(store.cache_identity());
    std::shared_ptr<const common_semi_oracle_history_projection_cache_entry> cached_projection;
    try {
        if (!store.get_search_history_revision(ruleset.id, ruleset.revision,
                source_watermark, watermark_supported, history_error) && history_error.empty())
            history_error = "search_history_revision_unavailable";
    } catch (...) { history_error = "search_history_revision_exception"; }
    if (watermark_supported && history_error.empty()) {
        cache_key = fingerprint(json{{"store",store_identity},{"context",context},
            {"watermark",source_watermark},{"limit",config.max_history_evaluations}}.dump());
        if (store.get_history_projection_cache(cache_key, cached_projection)) {
            projection = cached_projection->suffix_projection;
            source_fingerprint = projection.source_fingerprint;
            read_rows = cached_projection->read_rows;
            rejected = cached_projection->rejected_rows;
            duplicates = cached_projection->duplicate_rows;
            conflicting_records = cached_projection->conflicting_rows;
            branch_projection = cached_projection->branch_projection;
            for (const auto & ref : cached_projection->evaluation_refs)
                accepted.push_back({{"evaluation_id",ref}});
            projection_hit = true;
        }
    }
    if (!projection_hit) {
        try {
            if (!store.read_search_history_snapshot(ruleset.id, ruleset.revision,
                    config.max_history_evaluations, records, source_watermark,
                    watermark_supported, history_error))
                fallback = "search_history_unavailable";
        } catch (...) { fallback = "search_history_exception"; }
        if (fallback.empty()) {
            cache_key = watermark_supported ? fingerprint(json{{"store",store_identity},{"context",context},
                {"watermark",source_watermark},{"limit",config.max_history_evaluations}}.dump()) : std::string{};
        }
        read_rows = records.size();
        std::map<std::string, std::map<std::string, counts>> grouped;
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
                // Stage a complete record before changing either projection.
                // A malformed later branch must not leave partial contributions.
                std::map<std::string, counts> parsed_branch;
                json accepted_record = {{"evaluation_id", record.first},
                    {"fixture_group", group}, {"observations", row.at("observations")}};
                if (target && row.contains("branch_quality")) {
                    const auto & branch = row.at("branch_quality");
                    if (branch.is_object() && branch.value("schema_version", 0) == 1 &&
                            branch.value("family", std::string{}) == "target_hunt_branch_quality_v1" &&
                            branch.value("scope_fingerprint", std::string{}) == context["scope"] &&
                            branch.value("ruleset_id", std::string{}) == ruleset.id &&
                            branch.value("ruleset_revision", uint64_t{0}) == ruleset.revision &&
                            branch.value("ruleset_fingerprint", std::string{}) == ruleset.semantic_fingerprint &&
                            branch.value("fixture_group", std::string{}) == group &&
                            branch.value("feature_version", std::string{}) == "target_hunt_features_v3" &&
                            branch.value("truth_pass_version", std::string{}) == "reverse_bfs_regret_v1" &&
                            branch.contains("observations") && branch["observations"].is_array()) {
                        for (const auto & branch_row : branch["observations"]) {
                            if (!branch_row.is_object() || !branch_row.contains("outcome") ||
                                    !branch_row["outcome"].is_string() ||
                                    !branch_row.contains("state_before") || !branch_row["state_before"].is_string() ||
                                    !branch_row.contains("action") || !branch_row["action"].is_string() ||
                                    !branch_row.contains("fine_category") || !branch_row["fine_category"].is_string() ||
                                    !branch_row.contains("coarse_category") || !branch_row["coarse_category"].is_string())
                                throw std::invalid_argument("invalid branch observation");
                            const auto state_before = branch_row["state_before"].get<std::string>();
                            const auto action = branch_row["action"].get<std::string>();
                            std::string state_error;
                            if (!common_semi_oracle_target_state(ruleset, state_before, state_error) ||
                                    common_semi_oracle_target_group(state_before) != group ||
                                    category(state_before, action).empty() ||
                                    branch_row["fine_category"] != category(state_before, action) ||
                                    branch_row["coarse_category"] != coarse_category(state_before, action))
                                throw std::invalid_argument("branch fixture or category mismatch");
                            const auto outcome = branch_row["outcome"].get<std::string>();
                            const size_t ix = outcome == "optimal" ? 0 : outcome == "detour" ? 1 :
                                outcome == "unreachable" ? 2 : 3;
                            if (ix > 2) throw std::invalid_argument("invalid branch outcome");
                            for (const auto * feature : {"fine_category", "coarse_category"}) {
                                if (!branch_row.contains(feature) || !branch_row[feature].is_string()) continue;
                                const auto category_key = branch_row[feature].get<std::string>();
                                if (allowed(category_key)) parsed_branch[category_key][ix] += 1.0;
                            }
                        }
                        accepted_record["branch_quality"] = branch;
                    } else {
                        throw std::invalid_argument("branch provenance mismatch");
                    }
                }
                for (const auto & item : parsed) for (size_t i = 0; i < 3; ++i)
                    grouped[group][item.first][i] += item.second[i];
                for (const auto & item : parsed_branch) for (size_t i = 0; i < 3; ++i)
                    branch_grouped[group][item.first][i] += item.second[i];
                accepted.push_back(std::move(accepted_record));
            } catch (...) { ++rejected; }
        }
        conflicting_records = conflicts.size();
        std::vector<common_outcome_guidance_projection_observation> projection_rows;
        for (const auto & group : grouped) for (const auto & item : group.second)
            projection_rows.push_back({group.first,item.first,
                {item.second[0],item.second[1],item.second[2]}});
        std::vector<common_outcome_guidance_projection_observation> branch_rows;
        for (const auto & group : branch_grouped) for (const auto & item : group.second)
            branch_rows.push_back({group.first,item.first,
                {item.second[0],item.second[1],item.second[2]}});
        // Cache lookup identity is local to a live store; provenance is a
        // reproducible digest of validated content across close/reopen.
        const nlohmann::json canonical_source = json{{"context",context},{"records",accepted}};
        source_fingerprint = fingerprint(canonical_source.dump());
        if (cache_key.empty()) cache_key = source_fingerprint;
        if (!watermark_supported && fallback.empty()) {
            if (store.get_history_projection_cache(cache_key, cached_projection)) {
                projection = cached_projection->suffix_projection;
                branch_projection = cached_projection->branch_projection;
                projection_hit = true;
            }
        }
        if (!projection_hit && fallback.empty()) {
            if (!common_outcome_guidance_build_projection(projection_rows,
                    "semi-oracle-outcome-projection-v2", source_fingerprint, source_watermark,
                    projection, error)) {
                fallback = "search_history_projection_invalid";
                history_error = error;
                error.clear();
                projection = {};
            } else if (!common_outcome_guidance_build_projection(branch_rows,
                    "target-hunt-branch-quality-projection-v1", source_fingerprint,
                    source_watermark, branch_projection, error)) {
                fallback = "branch_quality_projection_invalid";
                history_error = error;
                error.clear();
                projection = {};
                branch_projection = {};
            }
        }
        if (!projection_hit && fallback.empty()) {
            try {
                const size_t bytes = projection_bytes(projection) + projection_bytes(branch_projection);
                common_semi_oracle_history_projection_cache_entry entry;
                entry.key = cache_key;
                entry.suffix_projection = projection;
                entry.branch_projection = branch_projection;
                entry.bytes = bytes + sizeof(entry) + entry.key.capacity();
                entry.read_rows = read_rows;
                entry.rejected_rows = rejected;
                entry.duplicate_rows = duplicates;
                entry.conflicting_rows = conflicting_records;
                for (const auto & row : accepted)
                    if (row.is_object() && row.contains("evaluation_id") && row["evaluation_id"].is_string()) {
                        entry.evaluation_refs.push_back(row["evaluation_id"].get<std::string>());
                        entry.bytes += entry.evaluation_refs.back().capacity();
                    }
                entry.bytes += entry.evaluation_refs.capacity() * sizeof(std::string);
                store.put_history_projection_cache(std::move(entry));
            } catch (...) {
                // Caching is optional: keep the valid projection and actual policy
                // even if retaining it fails. No second search is needed.
                history_error = "projection_cache_write_exception";
            }
        }
    }
    std::map<std::string,std::vector<double>> projected_weights;
    std::map<std::string,size_t> group_counts;
    projection.aggregate_excluding(fixture,projected_weights,group_counts);
    std::map<std::string,counts> weights;
    for(const auto & item:projected_weights)if(item.second.size()==3)
        weights[item.first]={item.second[0],item.second[1],item.second[2]};
    std::map<std::string,std::vector<double>> branch_projected_weights;
    std::map<std::string,size_t> branch_group_counts;
    if (target) branch_projection.aggregate_excluding(fixture,
        branch_projected_weights, branch_group_counts);
    const bool branch_labels_used = target && !branch_projected_weights.empty();
    if (branch_labels_used) {
        weights.clear();
        group_counts.clear();
        for (const auto & item : branch_projected_weights) {
            if (item.second.size() == 3)
                weights[item.first] = {item.second[0], item.second[1], item.second[2]};
        }
        group_counts = branch_group_counts;
    }
    const auto & active_projection = branch_labels_used ? branch_projection : projection;
    size_t eligible_fixture_groups = 0;
    for (const auto & group : active_projection.fixture_categories) {
        if (group.first == fixture) continue;
        const bool contributes = std::any_of(group.second.begin(), group.second.end(), [](const auto & item) {
            return std::any_of(item.second.begin(), item.second.end(), [](double value) { return value > 0.0; });
        });
        if (contributes) ++eligible_fixture_groups;
    }
    const std::string history_fingerprint = branch_labels_used
        ? fingerprint(json::array({projection.fingerprint, branch_projection.fingerprint}).dump())
        : projection.fingerprint;
    json decisions = json::array();
    size_t omitted_decisions = 0;
    double adviser_ms=0.0;
    auto guided = query;
    guided.record_search_history = true;
    guided.use_historical_order = config.mode == "experimental";
    std::map<std::pair<std::string,std::string>,double> transition_scores;
    std::map<std::pair<std::string,std::string>,std::string> transition_identities;
    if (!query.transition_adviser && !query.transition_score && fallback.empty()) {
        guided.transition_score = [&](const std::string & before, const std::string &,
                const std::string & action) {
            const auto score=transition_scores.find({before,action});
            if(score==transition_scores.end())throw std::runtime_error("missing transition posterior score");
            return score->second;
        };
        guided.historical_identity = [&](const std::string & before, const std::string &,
                const std::string & action) {
            const auto identity = transition_identities.find({before, action});
            return identity == transition_identities.end() ? std::string{} : identity->second;
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
                const auto coarse_key=target?coarse_category(current,action):fine_key;
                const auto key=target&&group_counts[fine_key]<8?coarse_key:fine_key;
                transition_identities[{current,action}]=key;
                const auto found = weights.find(key);
                const counts w = found == weights.end() ? counts{} : found->second;
                matching_history = matching_history || w[0] + w[1] + w[2] > 0.0;
                auto sample = sampled_by_category.find(key);
                if (sample == sampled_by_category.end()) {
                    common_outcome_guidance_posterior p;
                    const bool branch = branch_labels_used;
                    const std::vector<double> categorical = branch
                        ? std::vector<double>{w[0], w[1], w[2]}
                        : std::vector<double>{w[0], w[1]};
                    const std::vector<double> priors = branch
                        ? std::vector<double>{1.0/3.0, 1.0/3.0, 1.0/3.0}
                        : std::vector<double>{0.5, 0.5};
                    const double known = branch ? w[0] + w[1] + w[2] : w[0] + w[1];
                    const double unknown = branch ? 0.0 : w[2];
                    if (!common_outcome_guidance_make_posterior(known, unknown, 0.5,
                            categorical, priors, 0.0, 0.0, p)) {
                        advice_error = "invalid search posterior"; return false;
                    }
                    const auto seed_hex = fingerprint(json::array({seed_base,key}).dump()).substr(7, 16);
                    std::mt19937_64 category_rng(std::stoull(seed_hex, nullptr, 16));
                    const std::vector<double> utility = branch
                        ? std::vector<double>{1.0, 0.0, -1.0}
                        : std::vector<double>{1.0, 0.0};
                    const double score=config.sampling_policy=="posterior_mean"
                        ? p.p_known*std::inner_product(p.category_probabilities.begin(),
                            p.category_probabilities.end(), utility.begin(), 0.0)
                        : common_outcome_guidance_sample_expected_utility(p, utility, category_rng);
                    sample = sampled_by_category.emplace(key,score).first;
                }
                utilities.push_back(sample->second);
                common_outcome_guidance_posterior p;
                const double known = branch_labels_used ? w[0] + w[1] + w[2] : w[0] + w[1];
                const double unknown = branch_labels_used ? 0.0 : w[2];
                const std::vector<double> categorical = branch_labels_used
                    ? std::vector<double>{w[0], w[1], w[2]}
                    : std::vector<double>{w[0], w[1]};
                const std::vector<double> priors = branch_labels_used
                    ? std::vector<double>{1.0/3.0, 1.0/3.0, 1.0/3.0}
                    : std::vector<double>{0.5, 0.5};
                if (!common_outcome_guidance_make_posterior(known, unknown, 0.5,
                        categorical, priors, 0.0, 0.0, p)) {
                    advice_error = "invalid search posterior"; return false;
                }
                predictions.push_back({{"action", action}, {"category", key},
                    {"prediction_family",branch_labels_used?"target_hunt_branch_quality_v1":context["family"]},
                    {"branch_labels_used",branch_labels_used}, {"p_known", p.p_known},
                    {"p_known_interval", {p.p_known_low, p.p_known_high}},
                    {branch_labels_used?"p_optimal_given_known":"p_goal_given_known", p.category_probabilities[0]},
                    {branch_labels_used?"p_detour_given_known":"p_dead_end_given_known", p.category_probabilities[1]},
                    {branch_labels_used?"p_unreachable_given_known":"p_unknown_given_known",
                        branch_labels_used?p.category_probabilities[2]:1.0-p.p_known},
                    {"p_primary_interval", {p.category_lows[0], p.category_highs[0]}},
                    {"p_secondary_interval", {p.category_lows[1], p.category_highs[1]}},
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
            const auto coarse=coarse_category(before,action_text);
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
            {"exact_history_order",config.exact_history_order==common_semi_oracle_history_order::after_heuristic?"after_h":"before_h"},
            {"search_algorithm",result.applied_search_policy.algorithm==common_semi_oracle_search_algorithm::focal?"focal":"exact"},
            {"suboptimality_bound",result.applied_search_policy.suboptimality_bound},
            {"family",context["family"]},{"evaluator_revision",context["evaluator_revision"]},
            {"ruleset_id", ruleset.id}, {"ruleset_revision", ruleset.revision},
            {"fixture_group", fixture}, {"scope_fingerprint", context["scope"]},
            {"history_fingerprint", history_fingerprint}, {"history_records", read_rows},
            {"accepted_records", accepted.size()}, {"rejected_records", rejected},
            {"duplicate_records", duplicates}, {"conflicting_records", conflicting_records},
            {"history_evaluation_refs", history_refs},
            {"fixture_groups", eligible_fixture_groups}, {"fallback_reason", fallback},
            {"branch_quality_labels_used",branch_labels_used},
            {"branch_quality_projection_fingerprint",branch_projection.fingerprint},
            {"projection_ms",std::chrono::duration<double,std::milli>(search_started-started).count()},
            {"projection_cache_hit",projection_hit},{"projection_builder_revision",active_projection.builder_revision},
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
                {"coarse_category",coarse_category(before_text,action)}});
        }
        if (!result.search_branch_quality_json.empty()) {
            auto branch = json::parse(result.search_branch_quality_json, nullptr, false);
            if (branch.is_object() && branch.value("family", std::string{}) ==
                    "target_hunt_branch_quality_v1") {
                branch["scope_fingerprint"] = context["scope"];
                persisted["branch_quality"] = branch;
                result.search_branch_quality_json = branch.dump();
            }
        }
        result.search_guidance_json=persisted.dump();
    }
    return true;
}
