#include "agent/adaptation/semi-oracle-outcome-guidance.h"
#include "agent/prediction/outcome-guidance.h"
#include "hash/hash.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
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
}

bool common_semi_oracle_outcome_guidance_config_validate(
        const common_semi_oracle_outcome_guidance_config & config, std::string & error) {
    error.clear();
    if ((config.mode != "off" && config.mode != "shadow" && config.mode != "experimental") ||
            config.max_history_evaluations == 0 || config.max_history_evaluations > 1024) {
        error = "SemiOracle guidance requires mode off/shadow/experimental and max_history_evaluations 1..1024";
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
    const bool trusted = !ruleset.assertions.empty() && std::all_of(
        ruleset.assertions.begin(), ruleset.assertions.end(), [](const auto & assertion) {
            return assertion.authority != common_semi_oracle_rule_authority::inferred;
        });
    if (config.mode == "off" || ruleset.domain != common_semi_oracle_domain::linear_algebra ||
            query.level == common_semi_oracle_level::local || query.goal != "solve:x" || !trusted)
        return common_semi_oracle_evaluate(ruleset, query, result, error);

    // Scope, rule revision, proof contract and budget all partition this family.
    const json context = {{"family", "algebra_search_suffix_v1"},
        {"scope", fingerprint(json::array({host, project, session}).dump())},
        {"ruleset_id", ruleset.id}, {"revision", ruleset.revision},
        {"rule_fingerprint", ruleset.semantic_fingerprint},
        {"action_contract", ruleset.action_contract_fingerprint},
        {"evaluator_revision", "linear_astar_v1"}, {"goal", query.goal},
        {"host_action_order", query.level == common_semi_oracle_level::guided
            ? query.guidance : std::vector<std::string>{}},
        {"max_expansions", query.budget.max_expansions}, {"max_depth", query.budget.max_depth}};
    const auto state = json::parse(query.state_json, nullptr, false);
    const std::string fixture = fingerprint(state.is_object() && state.contains("lhs") && state.contains("rhs")
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
                    row.at("schema_version") != 1 || row.at("context") != context ||
                    !row.at("fixture_group").is_string() ||
                    !row.at("observations").is_object()) { ++rejected; continue; }
            const auto group = row.at("fixture_group").get<std::string>();
            if (group.empty() || group == fixture) { ++rejected; continue; }
            std::map<std::string, counts> parsed;
            for (auto it = row.at("observations").begin(); it != row.at("observations").end(); ++it) {
                if (!actions.count(it.key()) || !it.value().is_array() || it.value().size() != 3)
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
    std::map<std::string, counts> weights;
    for (const auto & group : grouped) for (const auto & item : group.second) {
        const double total = item.second[0] + item.second[1] + item.second[2];
        if (total > 0.0) for (size_t i = 0; i < 3; ++i) weights[item.first][i] += item.second[i] / total;
    }
    const std::string history_fingerprint = fingerprint(json{{"builder", "semi-oracle-history-v1"},
        {"context", context}, {"records", accepted}}.dump());
    json decisions = json::array();
    size_t omitted_decisions = 0;
    auto guided = query;
    guided.record_search_history = true;
    if (!query.transition_adviser && fallback.empty()) {
        guided.transition_adviser = [&](const std::string & current, const std::vector<std::string> & candidates,
                std::vector<size_t> & order, std::string & advice_error) {
            advice_error.clear();
            order.resize(candidates.size()); std::iota(order.begin(), order.end(), size_t{0});
            auto proposed = order;
            std::vector<double> utilities;
            json predictions = json::array();
            bool matching_history = false;
            const auto seed_hex = fingerprint(json::array({config.seed, current, history_fingerprint}).dump()).substr(7, 16);
            std::mt19937_64 rng(std::stoull(seed_hex, nullptr, 16));
            for (const auto & action : candidates) {
                const auto found = weights.find(action);
                const counts w = found == weights.end() ? counts{} : found->second;
                matching_history = matching_history || w[0] + w[1] + w[2] > 0.0;
                common_outcome_guidance_posterior p;
                if (!common_outcome_guidance_make_posterior(w[0] + w[1], w[2], 0.5,
                        {w[0], w[1]}, {0.5, 0.5}, 0.0, 0.0, p)) {
                    advice_error = "invalid search posterior"; return false;
                }
                utilities.push_back(common_outcome_guidance_sample_expected_utility(p, {1.0, 0.0}, rng));
                predictions.push_back({{"action", action}, {"p_known", p.p_known},
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
            if (config.mode == "experimental") order = proposed;
            if (decisions.size() < 128) decisions.push_back({{"state_fingerprint", fingerprint(current)},
                {"selection_reason", matching_history ? "posterior_sampling" : "cold_start_original_order"},
                {"predictions", predictions}, {"proposed_order", proposed}, {"executed_order", order}});
            else ++omitted_decisions;
            return true;
        };
    } else if (query.transition_adviser) fallback = "existing host transition adviser retained";
    if (!common_semi_oracle_evaluate(ruleset, guided, result, error)) return false;
    // No trace means the domain did not run this search (e.g. all-real solutions).
    auto trace = json::parse(result.search_guidance_json, nullptr, false);
    if (!trace.is_object()) return true;
    std::map<std::string, counts> observed;
    for (const auto & observation : trace.at("observations")) {
        const auto action = observation.at("action").get<std::string>();
        if (!actions.count(action)) continue;
        const auto outcome = observation.at("outcome").get<std::string>();
        ++observed[action][outcome == "goal_reached" ? 0 : outcome == "dead_end" ? 1 : 2];
    }
    json observations = json::object();
    for (const auto & item : observed) observations[item.first] = item.second;
    json history_refs = json::array();
    for (const auto & item : accepted) history_refs.push_back(item["evaluation_id"]);
    result.search_guidance_json = json{{"schema_version", 1}, {"context", context},
        {"fixture_group", fixture}, {"observations", observations},
        {"diagnostics", {{"mode", config.mode}, {"seed", config.seed},
            {"ruleset_id", ruleset.id}, {"ruleset_revision", ruleset.revision},
            {"fixture_group", fixture}, {"scope_fingerprint", context["scope"]},
            {"history_fingerprint", history_fingerprint}, {"history_records", records.size()},
            {"accepted_records", accepted.size()}, {"rejected_records", rejected},
            {"duplicate_records", duplicates}, {"conflicting_records", conflicts.size()},
            {"history_evaluation_refs", history_refs},
            {"fixture_groups", grouped.size()}, {"fallback_reason", fallback},
            {"history_diagnostic", history_error},
            {"callback_fallbacks", trace.value("callback_fallbacks", size_t{0})},
            {"callback_error", trace.value("callback_error", std::string{})},
            {"decisions", decisions}, {"decision_count", decisions.size() + omitted_decisions},
            {"omitted_decisions", omitted_decisions}}}}.dump();
    if (result.guidance_ref.empty()) result.guidance_ref = "semi-oracle-history:" + history_fingerprint;
    return true;
}
