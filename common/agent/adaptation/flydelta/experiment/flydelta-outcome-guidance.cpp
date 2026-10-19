#include "agent/adaptation/flydelta/experiment/flydelta-outcome-guidance.h"
#include "agent/prediction/outcome-guidance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include "hash/hash.h"

namespace {

using context_key = std::tuple<std::string, std::string, std::string, std::string,
    std::string, std::string, std::string, std::string>;
using bucket_key = std::tuple<std::string, std::string, std::string, size_t,
    int, int, int, int, std::string>;

context_key hard_key(const common_flydelta_outcome_guidance_context & c) {
    return {c.scope_fingerprint, c.model_profile_fingerprint,
        c.tokenizer_fingerprint, c.template_fingerprint,
        c.baseline_deployment_fingerprint, c.verifier_revision,
        c.ruleset_revision, c.action_contract_fingerprint};
}

int layer_bucket(double v) { return v < 0.25 ? 0 : (v < 0.5 ? 1 : (v < 0.75 ? 2 : 3)); }
int dose_bucket(double v) { return v < 0.25 ? 0 : (v < 0.5 ? 1 : (v < 0.75 ? 2 : 3)); }

nlohmann::ordered_json canonical_candidate_pool(
        const std::vector<common_flydelta_outcome_guidance_candidate> & candidates) {
    auto pool = nlohmann::ordered_json::array();
    for (const auto & candidate : candidates) pool.push_back({
        {"arm_id", candidate.arm_id}, {"basis_identity", candidate.basis_identity},
        {"rank", candidate.rank}, {"relative_layer", candidate.relative_layer},
        {"relative_dose", candidate.relative_dose},
        {"mixed_coefficients", candidate.mixed_coefficients},
        {"diagnostic_margin_sign", candidate.diagnostic_margin_sign},
    });
    return pool;
}

bucket_key key_for(const common_flydelta_outcome_guidance_context & c,
        const common_flydelta_outcome_guidance_candidate & a) {
    return {c.semantic_domain, c.strategy, c.estimator, a.rank,
        layer_bucket(a.relative_layer), dose_bucket(a.relative_dose),
        a.mixed_coefficients ? 1 : 0, a.diagnostic_margin_sign < 0 ? -1 :
            (a.diagnostic_margin_sign > 0 ? 1 : 0), a.basis_identity};
}

bucket_key key_for(const common_flydelta_outcome_guidance_observation & o) {
    common_flydelta_outcome_guidance_candidate candidate;
    candidate.arm_id = o.stable_observation_id;
    candidate.basis_identity = o.basis_identity;
    candidate.rank = o.rank;
    candidate.relative_layer = o.relative_layer;
    candidate.relative_dose = o.relative_dose;
    candidate.mixed_coefficients = o.mixed_coefficients;
    candidate.diagnostic_margin_sign = o.diagnostic_margin_sign;
    return key_for(o.context, candidate);
}

struct counts {
    double known = 0.0, unknown = 0.0, helped = 0.0, neutral = 0.0, harmed = 0.0;
    size_t fixture_groups = 0;
    size_t categorical_groups = 0;
};

template<class Predicate>
counts aggregate(const std::vector<const common_flydelta_outcome_guidance_observation*> & rows,
        Predicate pred, const std::set<std::string> & exclude = {}) {
    std::map<std::string, std::vector<const common_flydelta_outcome_guidance_observation*>> groups;
    for (const auto * o : rows) if (!exclude.count(o->fixture_group_id) && pred(*o))
        groups[o->fixture_group_id].push_back(o);
    counts c;
    for (const auto & group : groups) {
        ++c.fixture_groups;
        size_t known_rows = 0;
        for (const auto * o : group.second) {
            known_rows += o->outcome != common_flydelta_counterfactual_outcome::unknown;
        }
        const double verifiability_weight = 1.0 / static_cast<double>(group.second.size());
        const double categorical_weight = known_rows > 0
            ? 1.0 / static_cast<double>(known_rows) : 0.0;
        bool group_has_category = false;
        for (const auto * o : group.second) {
            if (o->outcome == common_flydelta_counterfactual_outcome::unknown) {
                c.unknown += verifiability_weight;
                continue;
            }
            c.known += verifiability_weight;
            switch (o->outcome) {
                case common_flydelta_counterfactual_outcome::helped: c.helped += categorical_weight; group_has_category = true; break;
                case common_flydelta_counterfactual_outcome::neutral: c.neutral += categorical_weight; group_has_category = true; break;
                case common_flydelta_counterfactual_outcome::harmed: c.harmed += categorical_weight; group_has_category = true; break;
                case common_flydelta_counterfactual_outcome::unknown: break;
            }
        }
        if (group_has_category) ++c.categorical_groups;
    }
    return c;
}

std::string history_hash(const std::vector<const common_flydelta_outcome_guidance_observation*> & rows) {
    std::vector<const common_flydelta_outcome_guidance_observation*> sorted = rows;
    std::sort(sorted.begin(), sorted.end(), [](const auto * a, const auto * b) {
        return a->stable_observation_id < b->stable_observation_id;
    });
    nlohmann::ordered_json canonical = {
        {"builder_revision", "flydelta-outcome-projection-v3"},
        {"observations", nlohmann::ordered_json::array()},
    };
    for (const auto * o : sorted) {
        canonical["observations"].push_back({
            {"id", o->stable_observation_id}, {"fixture_group", o->fixture_group_id},
            {"basis", o->basis_identity}, {"outcome", static_cast<int>(o->outcome)},
            {"outcome_known", o->outcome_known}, {"verifier_known", o->verifier_known},
            {"comparison_completed", o->comparison_completed}, {"overlay_applied", o->overlay_applied},
            {"rank", o->rank}, {"relative_layer", o->relative_layer},
            {"relative_dose", o->relative_dose}, {"mixed", o->mixed_coefficients},
            {"margin_sign", o->diagnostic_margin_sign},
            {"context", {
                {"scope", o->context.scope_fingerprint},
                {"model", o->context.model_profile_fingerprint},
                {"tokenizer", o->context.tokenizer_fingerprint},
                {"template", o->context.template_fingerprint},
                {"deployment", o->context.baseline_deployment_fingerprint},
                {"verifier", o->context.verifier_revision},
                {"ruleset", o->context.ruleset_revision},
                {"action_contract", o->context.action_contract_fingerprint},
                {"domain", o->context.semantic_domain}, {"strategy", o->context.strategy},
                {"estimator", o->context.estimator},
            }},
        });
    }
    const std::string material = canonical.dump();
    return "sha256:" + hash_sha256_hex(material.data(), material.size());
}

} // namespace

common_flydelta_outcome_guidance_candidate common_flydelta_outcome_guidance_layer_candidate(
        uint32_t layer, uint32_t model_n_layers, double scale) {
    common_flydelta_outcome_guidance_candidate candidate;
    candidate.arm_id = "layer:" + std::to_string(layer);
    candidate.rank = 1;
    candidate.relative_layer = model_n_layers > 1
        ? static_cast<double>(layer) / static_cast<double>(model_n_layers - 1) : 0.0;
    candidate.relative_dose = std::max(0.0, std::min(1.0, scale));
    // WHERE categories carry no raw coefficient coordinates or post-run margin.
    return candidate;
}

std::string common_flydelta_outcome_guidance_pool_fingerprint(
        const std::vector<common_flydelta_outcome_guidance_candidate> & candidates) {
    const std::string bytes = canonical_candidate_pool(candidates).dump();
    return "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
}

const char * common_flydelta_outcome_guidance_mode_name(common_flydelta_outcome_guidance_mode mode) {
    switch (mode) {
        case common_flydelta_outcome_guidance_mode::off: return "off";
        case common_flydelta_outcome_guidance_mode::shadow: return "shadow";
        case common_flydelta_outcome_guidance_mode::experimental: return "experimental";
    }
    return "shadow";
}

bool common_flydelta_parse_outcome_guidance_mode(const std::string & value,
        common_flydelta_outcome_guidance_mode & mode, std::string & error) {
    error.clear();
    if (value == "off") mode = common_flydelta_outcome_guidance_mode::off;
    else if (value == "shadow") mode = common_flydelta_outcome_guidance_mode::shadow;
    else if (value == "experimental") mode = common_flydelta_outcome_guidance_mode::experimental;
    else { error = "FlyDelta outcome guidance mode must be off, shadow or experimental"; return false; }
    return true;
}

bool common_flydelta_outcome_guidance_config_validate(
        const common_flydelta_outcome_guidance_config & c, std::string & error) {
    error.clear();
    if (c.schema_version != 1 || !std::isfinite(c.harm_penalty) || c.harm_penalty < 0 ||
            c.min_backoff_groups == 0 || !std::isfinite(c.max_prior_strength) ||
            c.max_prior_strength < 0 || c.max_prior_strength > 4) {
        error = "FlyDelta outcome guidance configuration is invalid"; return false;
    }
    return true;
}

bool common_flydelta_outcome_guidance_build_decision(
        const common_flydelta_outcome_guidance_config & config,
        const common_flydelta_outcome_guidance_context & context,
        const std::vector<common_flydelta_outcome_guidance_observation> & history,
        const std::vector<common_flydelta_outcome_guidance_candidate> & candidates,
        const size_t selection_limit, const uint64_t job_seed,
        common_flydelta_outcome_guidance_decision & d, std::string & error,
        const common_flydelta_outcome_guidance_policy policy,
        const size_t policy_sequence_start) {
    d = {}; error.clear();
    if (!common_flydelta_outcome_guidance_config_validate(config, error) ||
            context.scope_fingerprint.empty() || context.model_profile_fingerprint.empty() ||
            context.tokenizer_fingerprint.empty() || context.template_fingerprint.empty() ||
            context.baseline_deployment_fingerprint.empty() || context.verifier_revision.empty() ||
            context.semantic_domain.empty() || context.strategy.empty() || context.estimator.empty() ||
            selection_limit > candidates.size()) {
        if (error.empty()) error = "FlyDelta outcome guidance request lacks compatible context";
        return false;
    }
    for (size_t i=0; i<candidates.size(); ++i) {
        if (candidates[i].arm_id.empty() || !std::isfinite(candidates[i].relative_layer) ||
                !std::isfinite(candidates[i].relative_dose) || candidates[i].relative_layer < 0 ||
                candidates[i].relative_layer > 1 || candidates[i].relative_dose < 0 ||
                candidates[i].relative_dose > 1) {
            error = "FlyDelta outcome guidance candidate is invalid"; return false;
        }
        for (size_t j=0; j<i; ++j) if (candidates[i].arm_id == candidates[j].arm_id) {
            error = "FlyDelta outcome guidance arm ids must be unique"; return false;
        }
        d.original_order.push_back(i);
    }
    std::vector<const common_flydelta_outcome_guidance_observation*> rows;
    d.read_history_rows = history.size();
    std::map<std::string, std::string> signatures;
    std::set<std::string> conflicted;
    auto observation_signature = [](const auto & o) {
        return nlohmann::ordered_json{
            {"fixture_group", o.fixture_group_id}, {"basis", o.basis_identity},
            {"scope", o.context.scope_fingerprint}, {"model", o.context.model_profile_fingerprint},
            {"tokenizer", o.context.tokenizer_fingerprint}, {"template", o.context.template_fingerprint},
            {"deployment", o.context.baseline_deployment_fingerprint}, {"verifier", o.context.verifier_revision},
            {"ruleset", o.context.ruleset_revision}, {"action", o.context.action_contract_fingerprint},
            {"domain", o.context.semantic_domain}, {"strategy", o.context.strategy},
            {"estimator", o.context.estimator}, {"rank", o.rank},
            {"layer", o.relative_layer}, {"dose", o.relative_dose},
            {"mixed", o.mixed_coefficients}, {"margin", o.diagnostic_margin_sign},
            {"completed", o.comparison_completed}, {"overlay", o.overlay_applied},
            {"outcome_known", o.outcome != common_flydelta_counterfactual_outcome::unknown},
            {"verifier_known", o.verifier_known},
            {"outcome", static_cast<int>(o.outcome)},
        }.dump();
    };
    for (const auto & o : history) {
        if (o.stable_observation_id.empty() || o.fixture_group_id.empty() ||
                !o.comparison_completed || !o.overlay_applied ||
                hard_key(o.context) != hard_key(context) ||
                !std::isfinite(o.relative_layer) || !std::isfinite(o.relative_dose) ||
                o.relative_layer < 0 || o.relative_layer > 1 || o.relative_dose < 0 || o.relative_dose > 1 ||
                o.outcome_known != (o.outcome != common_flydelta_counterfactual_outcome::unknown) ||
                (o.outcome_known && !o.verifier_known)) {
            ++d.rejected_history_rows;
            continue;
        }
        const std::string signature = observation_signature(o);
        const auto inserted = signatures.emplace(o.stable_observation_id, signature);
        if (!inserted.second) {
            if (inserted.first->second != signature) conflicted.insert(o.stable_observation_id);
            else ++d.deduplicated_history_rows;
        }
    }
    for (const auto & o : history) {
        const auto found = signatures.find(o.stable_observation_id);
        if (found == signatures.end()) continue;
        if (conflicted.count(o.stable_observation_id)) { ++d.rejected_history_rows; continue; }
        if (observation_signature(o) != found->second) continue;
        // Identical retries are accepted once; a conflicting id is excluded
        // entirely so append order cannot choose the winning observation.
        if (std::find_if(rows.begin(), rows.end(), [&](const auto * prior) {
                return prior->stable_observation_id == o.stable_observation_id;
            }) != rows.end()) continue;
        if (!o.comparison_completed || !o.overlay_applied ||
                hard_key(o.context) != hard_key(context) || o.fixture_group_id.empty()) continue;
        rows.push_back(&o);
    }
    d.compatible_history_rows = rows.size();
    d.history_fingerprint = history_hash(rows);
    d.shadow_order = d.original_order;
    d.mixed_policy_order = d.original_order;
    d.selected_order = d.original_order;
    d.selection_reasons.assign(candidates.size(), config.mode == common_flydelta_outcome_guidance_mode::off
        ? "guidance_off_existing_order" : "shadow_preserved_existing_order");
    d.mixed_policy_reasons = d.selection_reasons;
    if (config.mode == common_flydelta_outcome_guidance_mode::off || candidates.empty()) return true;
    d.selected_order.clear();
    d.selection_reasons.clear();
    d.mixed_policy_reasons.clear();

    std::mt19937_64 rng(config.seed ^ job_seed);
    struct scored { size_t index; double utility; };
    std::vector<scored> sampled;
    std::vector<scored> by_mean;
    struct cached_bucket {
        counts leaf;
        std::set<std::string> leaf_groups;
        counts strategy_backoff;
        counts domain_backoff;
        size_t raw_observations = 0;
    };
    std::map<bucket_key, cached_bucket> bucket_cache;
    std::map<bucket_key, double> sampled_by_posterior;
    for (const auto * row : rows) {
        const auto key = key_for(*row);
        auto & cached = bucket_cache[key];
        cached.leaf_groups.insert(row->fixture_group_id);
        ++cached.raw_observations;
    }
    for (auto & [key, cached] : bucket_cache) {
        const auto & [domain, strategy_name, estimator_name, rank,
            layer, dose, mixed, margin, basis_identity] = key;
        (void) rank; (void) layer; (void) dose; (void) mixed; (void) margin;
        cached.leaf = aggregate(rows, [&](const auto & o) { return key_for(o) == key; });
        cached.strategy_backoff = aggregate(rows, [&](const auto & o) {
            return o.context.semantic_domain == domain &&
                o.context.strategy == strategy_name && o.context.estimator == estimator_name &&
                (strategy_name != "whirlpool" || layer_bucket(o.relative_layer) == layer);
        }, cached.leaf_groups);
        cached.domain_backoff = aggregate(rows, [&](const auto & o) {
            return o.context.semantic_domain == domain &&
                (strategy_name != "whirlpool" || layer_bucket(o.relative_layer) == layer);
        }, cached.leaf_groups);
        (void) basis_identity;
    }
    for (size_t i=0; i<candidates.size(); ++i) {
        const auto exact = key_for(context, candidates[i]);
        auto cached_it = bucket_cache.find(exact);
        if (cached_it == bucket_cache.end()) {
            cached_it = bucket_cache.emplace(exact, cached_bucket{}).first;
            auto & cached = cached_it->second;
            cached.strategy_backoff = aggregate(rows, [&](const auto & o) {
                return o.context.semantic_domain == context.semantic_domain &&
                    o.context.strategy == context.strategy && o.context.estimator == context.estimator &&
                    (context.strategy != "whirlpool" || layer_bucket(o.relative_layer) ==
                        layer_bucket(candidates[i].relative_layer));
            });
            cached.domain_backoff = aggregate(rows, [&](const auto & o) {
                return o.context.semantic_domain == context.semantic_domain &&
                    (context.strategy != "whirlpool" || layer_bucket(o.relative_layer) ==
                        layer_bucket(candidates[i].relative_layer));
            });
        }
        const auto & cached = cached_it->second;
        const counts & leaf = cached.leaf;
        const size_t leaf_groups = cached.leaf_groups.size();
        const counts & strategy = cached.strategy_backoff;
        const counts & domain = cached.domain_backoff;
        const counts & known_base = strategy.fixture_groups >= config.min_backoff_groups ? strategy : domain;
        const size_t base_groups = known_base.fixture_groups;
        const double prior = base_groups >= config.min_backoff_groups ?
            std::min(config.max_prior_strength, 4.0) : 0.0;
        const double known_total = known_base.known + known_base.unknown;
        const double known_mean = known_total > 0 ? known_base.known / known_total : 0.5;
        const counts & category_base = strategy.categorical_groups >= config.min_backoff_groups
            ? strategy : domain;
        const double category_prior = category_base.categorical_groups >= config.min_backoff_groups
            ? std::min(config.max_prior_strength, 4.0) : 0.0;
        const double outcome_total = category_base.helped + category_base.neutral + category_base.harmed;
        const double helped_mean = outcome_total > 0 ? category_base.helped / outcome_total : 1.0/3.0;
        const double neutral_mean = outcome_total > 0 ? category_base.neutral / outcome_total : 1.0/3.0;
        const double harmed_mean = outcome_total > 0 ? category_base.harmed / outcome_total : 1.0/3.0;
        common_outcome_guidance_posterior posterior;
        if (!common_outcome_guidance_make_posterior(
                leaf.known, leaf.unknown, known_mean,
                {leaf.helped, leaf.neutral, leaf.harmed},
                {helped_mean, neutral_mean, harmed_mean}, prior, category_prior, posterior)) {
            error = "FlyDelta outcome posterior inputs are invalid";
            return false;
        }
        common_flydelta_outcome_guidance_prediction p;
        p.arm_id = candidates[i].arm_id;
        p.level = leaf_groups ? "arm_category" : (prior > 0 ? "backoff" : "global_prior");
        p.p_known = posterior.p_known;
        p.p_helped_given_known = posterior.category_probabilities[0];
        p.p_neutral_given_known = posterior.category_probabilities[1];
        p.p_harmed_given_known = posterior.category_probabilities[2];
        p.p_known_low = posterior.p_known_low; p.p_known_high = posterior.p_known_high;
        p.p_helped_low = posterior.category_lows[0]; p.p_helped_high = posterior.category_highs[0];
        p.p_neutral_low = posterior.category_lows[1]; p.p_neutral_high = posterior.category_highs[1];
        p.p_harmed_low = posterior.category_lows[2]; p.p_harmed_high = posterior.category_highs[2];
        p.expected_utility = p.p_known*(p.p_helped_given_known-config.harm_penalty*p.p_harmed_given_known);
        p.known_weight = leaf.known; p.helped_weight = leaf.helped;
        p.neutral_weight = leaf.neutral; p.harmed_weight = leaf.harmed;
        p.effective_known_observations = leaf.known;
        p.effective_unknown_observations = leaf.unknown;
        p.backoff_fixture_groups = base_groups;
        p.backoff_prior_strength = prior;
        p.backoff_categorical_prior_strength = category_prior;
        p.raw_observations = cached.raw_observations;
        p.fixture_groups = leaf_groups;
        const std::string identity_material=nlohmann::ordered_json{
            {"context",hard_key(context)},{"category",exact}}.dump();
        p.posterior_identity="sha256:"+hash_sha256_hex(identity_material.data(),identity_material.size());
        auto draw=sampled_by_posterior.find(exact);
        if(draw==sampled_by_posterior.end()) {
            const std::string seed_material=nlohmann::ordered_json{
                {"seed",config.seed},{"decision_seed",job_seed},
                {"posterior_identity",p.posterior_identity}}.dump();
            const auto seed_hex=hash_sha256_hex(seed_material.data(),seed_material.size()).substr(0,16);
            std::mt19937_64 posterior_rng(std::stoull(seed_hex,nullptr,16));
            draw=sampled_by_posterior.emplace(exact,common_outcome_guidance_sample_expected_utility(
                posterior,{1.0,0.0,-config.harm_penalty},posterior_rng)).first;
        }
        p.sampled_utility=draw->second;
        d.predictions.push_back(p);
        by_mean.push_back({i, p.expected_utility});
        sampled.push_back({i,p.sampled_utility});
    }
    std::stable_sort(sampled.begin(),sampled.end(),[](const auto & a,const auto & b){return a.utility>b.utility;});
    std::stable_sort(by_mean.begin(), by_mean.end(), [](const auto & a, const auto & b) {
        return a.utility > b.utility;
    });
    d.shadow_order.clear();
    for (const auto & s:sampled) d.shadow_order.push_back(s.index);
    std::set<size_t> selected;
    size_t existing_cursor = 0;
    for (size_t slot=0; slot<selection_limit; ++slot) {
        const size_t policy_slot = (policy_sequence_start + slot) % 4;
        size_t choice = candidates.size();
        const char * reason = "existing_order";
        if (policy_slot < 2) {
            while (existing_cursor < d.original_order.size() &&
                    selected.count(d.original_order[existing_cursor])) ++existing_cursor;
            if (existing_cursor < d.original_order.size()) choice = d.original_order[existing_cursor++];
        } else if (policy_slot == 2) {
            reason = policy == common_flydelta_outcome_guidance_policy::posterior_mean
                ? "posterior_mean" : "posterior_sample";
            const auto & posterior_order =
                policy == common_flydelta_outcome_guidance_policy::posterior_mean
                    ? by_mean : sampled;
            for (const auto & s : posterior_order) if (!selected.count(s.index)) {
                choice = s.index;
                break;
            }
        } else {
            reason = "uniform_exploration";
            std::vector<size_t> remaining;
            for (size_t i=0; i<candidates.size(); ++i) if (!selected.count(i)) remaining.push_back(i);
            if (!remaining.empty()) {
                std::uniform_int_distribution<size_t> pick(0, remaining.size()-1);
                choice = remaining[pick(rng)];
            }
        }
        // Empty policy classes are filled from the existing ordering without
        // duplication; the bounded pool is still fully ordered for callers.
        if (choice == candidates.size()) {
            for (const size_t ix : d.original_order) if (!selected.count(ix)) {
                choice = ix;
                reason = "existing_order_policy_fallback";
                break;
            }
        }
        if (choice == candidates.size()) break;
        d.selected_order.push_back(choice);
        d.selection_reasons.push_back(reason);
        selected.insert(choice);
    }
    for (size_t ix : d.original_order) if (!selected.count(ix)) {
        d.selected_order.push_back(ix); d.selection_reasons.push_back("remaining_existing_order");
    }
    d.mixed_policy_order = d.selected_order;
    d.mixed_policy_reasons = d.selection_reasons;
    if (config.mode == common_flydelta_outcome_guidance_mode::shadow) {
        d.selected_order = d.original_order;
        d.selection_reasons.assign(candidates.size(), "shadow_preserved_existing_order");
    }
    return true;
}

bool common_flydelta_outcome_guidance_replay(
        const common_flydelta_outcome_guidance_config & config,
        const std::vector<common_flydelta_outcome_guidance_observation> & chronological,
        const std::vector<std::string> & holdout_fixture_groups,
        common_flydelta_outcome_guidance_replay_metrics & metrics,
        std::string & error) {
    metrics = {};
    error.clear();
    if (!common_flydelta_outcome_guidance_config_validate(config, error)) return false;
    const std::set<std::string> holdout(holdout_fixture_groups.begin(), holdout_fixture_groups.end());
    std::vector<common_flydelta_outcome_guidance_observation> available;
    double model_known_brier = 0.0, fixed_known_brier = 0.0, empirical_known_brier = 0.0;
    double model_outcome_brier = 0.0, fixed_outcome_brier = 0.0, empirical_outcome_brier = 0.0;
    std::array<size_t, 10> calibration_count{};
    std::array<double, 10> calibration_prediction{}, calibration_actual{};
    uint64_t replay_seed = config.seed;
    std::map<std::string, std::string> replay_seen;
    std::set<std::string> replay_conflicts;
    const auto replay_eligible = [](const auto & row) {
        const bool outcome_known =
            row.outcome != common_flydelta_counterfactual_outcome::unknown;
        return row.comparison_completed && row.overlay_applied &&
            !row.fixture_group_id.empty() && !row.stable_observation_id.empty() &&
            row.outcome_known == outcome_known && (!outcome_known || row.verifier_known);
    };
    const auto replay_signature = [](const auto & row) {
        return nlohmann::ordered_json{
            {"group", row.fixture_group_id},
            {"context", {row.context.scope_fingerprint, row.context.model_profile_fingerprint,
                row.context.tokenizer_fingerprint, row.context.template_fingerprint,
                row.context.verifier_revision, row.context.baseline_deployment_fingerprint,
                row.context.ruleset_revision, row.context.action_contract_fingerprint,
                row.context.semantic_domain, row.context.strategy, row.context.estimator}},
            {"rank", row.rank}, {"layer", row.relative_layer}, {"dose", row.relative_dose},
            {"basis", row.basis_identity}, {"outcome", static_cast<int>(row.outcome)},
            {"outcome_known", row.outcome_known}, {"verifier_known", row.verifier_known},
            {"margin_sign", row.diagnostic_margin_sign},
            {"completed", row.comparison_completed}, {"overlay", row.overlay_applied},
        }.dump();
    };
    for (const auto & row : chronological) {
        if (!replay_eligible(row)) continue;
        const auto signature = replay_signature(row);
        const auto found = replay_seen.emplace(row.stable_observation_id, signature);
        if (!found.second && found.first->second != signature)
            replay_conflicts.insert(row.stable_observation_id);
    }
    replay_seen.clear();
    for (const auto & observed : chronological) {
        if (!replay_eligible(observed) ||
                replay_conflicts.count(observed.stable_observation_id)) continue;
        if (!replay_seen.emplace(observed.stable_observation_id, replay_signature(observed)).second) continue;
        if (holdout.count(observed.fixture_group_id)) {
            common_flydelta_outcome_guidance_config scoring_config = config;
            scoring_config.mode = common_flydelta_outcome_guidance_mode::shadow;
            common_flydelta_outcome_guidance_candidate candidate;
            candidate.arm_id = observed.stable_observation_id;
            candidate.rank = observed.rank;
            candidate.relative_layer = observed.relative_layer;
            candidate.relative_dose = observed.relative_dose;
            candidate.mixed_coefficients = observed.mixed_coefficients;
            candidate.diagnostic_margin_sign = observed.diagnostic_margin_sign;
            candidate.basis_identity = observed.basis_identity;
            common_flydelta_outcome_guidance_decision decision;
            if (!common_flydelta_outcome_guidance_build_decision(
                    scoring_config, observed.context, available, {candidate}, 1,
                    replay_seed++, decision, error)) return false;
            if (decision.predictions.size() != 1) {
                error = "FlyDelta outcome replay did not produce a prediction";
                return false;
            }
            const auto & prediction = decision.predictions.front();
            const bool outcome_known = observed.outcome != common_flydelta_counterfactual_outcome::unknown;
            const double known_target = outcome_known ? 1.0 : 0.0;
            model_known_brier += std::pow(prediction.p_known - known_target, 2);
            const size_t calibration_bin = std::min<size_t>(9,
                static_cast<size_t>(prediction.p_known * 10.0));
            ++calibration_count[calibration_bin];
            calibration_prediction[calibration_bin] += prediction.p_known;
            calibration_actual[calibration_bin] += known_target;
            fixed_known_brier += std::pow(0.5 - known_target, 2);
            std::vector<const common_flydelta_outcome_guidance_observation*> compatible_past;
            for (const auto & row : available) {
                if (hard_key(row.context) == hard_key(observed.context))
                    compatible_past.push_back(&row);
            }
            const counts past = aggregate(
                compatible_past,
                [](const auto &) { return true; });
            const double past_total = past.known + past.unknown;
            const double historical_known = past_total > 0 ? past.known / past_total : 0.5;
            empirical_known_brier += std::pow(historical_known - known_target, 2);
            ++metrics.eligible_predictions;
            if (outcome_known) {
                const double helped_target = observed.outcome == common_flydelta_counterfactual_outcome::helped ? 1.0 : 0.0;
                const double neutral_target = observed.outcome == common_flydelta_counterfactual_outcome::neutral ? 1.0 : 0.0;
                const double harmed_target = observed.outcome == common_flydelta_counterfactual_outcome::harmed ? 1.0 : 0.0;
                model_outcome_brier += std::pow(prediction.p_helped_given_known-helped_target, 2) +
                    std::pow(prediction.p_neutral_given_known-neutral_target, 2) +
                    std::pow(prediction.p_harmed_given_known-harmed_target, 2);
                fixed_outcome_brier += std::pow(1.0/3.0-helped_target, 2) +
                    std::pow(1.0/3.0-neutral_target, 2) + std::pow(1.0/3.0-harmed_target, 2);
                const double outcome_total = past.helped + past.neutral + past.harmed;
                const double historical_helped = outcome_total > 0 ? past.helped/outcome_total : 1.0/3.0;
                const double historical_neutral = outcome_total > 0 ? past.neutral/outcome_total : 1.0/3.0;
                const double historical_harmed = outcome_total > 0 ? past.harmed/outcome_total : 1.0/3.0;
                empirical_outcome_brier += std::pow(historical_helped-helped_target,2) +
                    std::pow(historical_neutral-neutral_target,2) +
                    std::pow(historical_harmed-harmed_target,2);
                ++metrics.known_outcomes;
            } else {
                ++metrics.unknown_outcomes;
            }
        } else {
            available.push_back(observed);
        }
    }
    if (metrics.eligible_predictions > 0) {
        const double n = static_cast<double>(metrics.eligible_predictions);
        metrics.model_verifiability_brier = model_known_brier/n;
        metrics.fixed_prior_verifiability_brier = fixed_known_brier/n;
        metrics.historical_mean_verifiability_brier = empirical_known_brier/n;
        double calibration_error = 0.0;
        for (size_t i = 0; i < calibration_count.size(); ++i) if (calibration_count[i] > 0) {
            const double count = static_cast<double>(calibration_count[i]);
            calibration_error += count/n * std::abs(
                calibration_prediction[i]/count - calibration_actual[i]/count);
        }
        metrics.model_verifiability_ece = calibration_error;
    }
    if (metrics.known_outcomes > 0) {
        const double n = static_cast<double>(metrics.known_outcomes);
        metrics.model_outcome_brier = model_outcome_brier/n;
        metrics.fixed_prior_outcome_brier = fixed_outcome_brier/n;
        metrics.historical_mean_outcome_brier = empirical_outcome_brier/n;
    }
    return true;
}

bool common_flydelta_append_outcome_guidance_decision_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & event_context,
        const std::string & job_id,
        const common_flydelta_outcome_guidance_mode mode,
        const uint64_t seed,
        const int phase_id,
        const size_t decision_sequence,
        const size_t selection_limit,
        const size_t policy_sequence_start,
        const size_t policy_cursor_before,
        const size_t policy_slots_consumed,
        const common_flydelta_outcome_guidance_context & context,
        const std::vector<common_flydelta_outcome_guidance_candidate> & candidates,
        const common_flydelta_outcome_guidance_decision & decision,
        std::string & error) {
    using json = nlohmann::ordered_json;
    error.clear();
    if (event_context.event_id.empty() || event_context.idempotency_key.empty() ||
            event_context.source_id.empty() || job_id.empty() ||
            selection_limit > candidates.size() || policy_sequence_start >= 4 ||
            policy_slots_consumed != std::min(selection_limit, candidates.size()) ||
            decision.original_order.size() != candidates.size() ||
            decision.selected_order.size() != candidates.size() ||
            decision.selection_reasons.size() != candidates.size() ||
            decision.shadow_order.size() != candidates.size() ||
            decision.mixed_policy_order.size() != candidates.size() ||
            decision.mixed_policy_reasons.size() != candidates.size() ||
            (decision.predictions.size() != candidates.size() &&
             !(decision.predictions.empty() && !decision.fallback_reason.empty()))) {
        error = "FlyDelta outcome guidance lifecycle decision is incomplete";
        return false;
    }
    const auto candidate_pool = canonical_candidate_pool(candidates);
    const std::string candidate_pool_fingerprint =
        common_flydelta_outcome_guidance_pool_fingerprint(candidates);
    json payload = {
        {"record_type", "outcome_guidance_decision"},
        {"schema_version", 4}, {"job_id", job_id}, {"seed", seed},
        {"sampling_revision", "posterior-category-sampling-v2"},
        {"phase_id", phase_id}, {"decision_sequence", decision_sequence},
        {"stage", decision.stage}, {"round", decision.round},
        {"decision_id", decision.decision_id},
        {"selection_limit", selection_limit},
        {"policy_sequence_start", policy_sequence_start % 4},
        {"policy_cursor_before", policy_cursor_before},
        {"policy_slots_consumed", policy_slots_consumed},
        {"mode", common_flydelta_outcome_guidance_mode_name(mode)},
        {"history_fingerprint", decision.history_fingerprint},
        {"candidate_pool_fingerprint", candidate_pool_fingerprint},
        {"context", {
            {"scope_fingerprint", context.scope_fingerprint},
            {"model_profile_fingerprint", context.model_profile_fingerprint},
            {"tokenizer_fingerprint", context.tokenizer_fingerprint},
            {"template_fingerprint", context.template_fingerprint},
            {"baseline_deployment_fingerprint", context.baseline_deployment_fingerprint},
            {"verifier_revision", context.verifier_revision},
            {"ruleset_revision", context.ruleset_revision},
            {"action_contract_fingerprint", context.action_contract_fingerprint},
            {"semantic_domain", context.semantic_domain},
            {"strategy", context.strategy}, {"estimator", context.estimator},
        }},
        {"candidate_pool", candidate_pool},
        {"original_order", decision.original_order},
        {"shadow_order", decision.shadow_order},
        {"mixed_policy_order", decision.mixed_policy_order},
        {"selected_order", decision.selected_order},
        {"mixed_policy_reasons", decision.mixed_policy_reasons},
        {"selection_reasons", json::array()},
        {"fallback_reason", decision.fallback_reason},
        {"predictions", json::array()},
    };
    for (size_t pos = 0; pos < decision.selected_order.size(); ++pos) {
        const size_t candidate_index = decision.selected_order[pos];
        if (candidate_index >= candidates.size()) {
            error = "FlyDelta outcome guidance selected index is invalid";
            return false;
        }
        payload["selection_reasons"].push_back({
            {"arm_id", candidates[candidate_index].arm_id},
            {"reason", decision.selection_reasons[pos]},
        });
    }
    for (const auto & p : decision.predictions) payload["predictions"].push_back({
        {"arm_id", p.arm_id}, {"level", p.level}, {"p_known", p.p_known},
        {"posterior_identity",p.posterior_identity},{"sampled_utility",p.sampled_utility},
        {"p_known_interval_95_approx", {p.p_known_low, p.p_known_high}},
        {"p_helped_given_known", p.p_helped_given_known},
        {"p_neutral_given_known", p.p_neutral_given_known},
        {"p_harmed_given_known", p.p_harmed_given_known},
        {"p_helped_interval_95_approx", {p.p_helped_low, p.p_helped_high}},
        {"p_neutral_interval_95_approx", {p.p_neutral_low, p.p_neutral_high}},
        {"p_harmed_interval_95_approx", {p.p_harmed_low, p.p_harmed_high}},
        {"expected_utility", p.expected_utility}, {"raw_observations", p.raw_observations},
        {"fixture_groups", p.fixture_groups},
        {"effective_known_observations", p.effective_known_observations},
        {"effective_unknown_observations", p.effective_unknown_observations},
        {"backoff_fixture_groups", p.backoff_fixture_groups},
        {"backoff_prior_strength", p.backoff_prior_strength},
    });
    common_learning_lifecycle_record record;
    record.event_id = event_context.event_id;
    record.subject_id = job_id;
    record.kind = common_learning_lifecycle_kind::flydelta_experiment;
    record.status = common_learning_lifecycle_status::observed;
    record.idempotency_key = event_context.idempotency_key;
    record.source_id = event_context.source_id;
    record.namespace_id = event_context.scope.namespace_id;
    record.project_id = event_context.scope.project_id;
    record.session_id = event_context.scope.session_id;
    record.content_hash = event_context.content_hash;
    record.created_at = event_context.created_at;
    record.payload_json = payload.dump();
    return store.append(record, error);
}
