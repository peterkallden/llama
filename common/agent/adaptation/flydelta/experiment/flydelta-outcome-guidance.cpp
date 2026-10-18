#include "agent/adaptation/flydelta/experiment/flydelta-outcome-guidance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <sstream>
#include <tuple>

namespace {

using context_key = std::tuple<std::string, std::string, std::string, std::string,
    std::string, std::string, std::string, std::string>;
using bucket_key = std::tuple<std::string, std::string, std::string, size_t,
    int, int, int, int>;

context_key hard_key(const common_flydelta_outcome_guidance_context & c) {
    return {c.scope_fingerprint, c.model_profile_fingerprint,
        c.tokenizer_fingerprint, c.template_fingerprint,
        c.baseline_deployment_fingerprint, c.verifier_revision,
        c.ruleset_revision, c.action_contract_fingerprint};
}

int layer_bucket(double v) { return v < 0.25 ? 0 : (v < 0.5 ? 1 : (v < 0.75 ? 2 : 3)); }
int dose_bucket(double v) { return v < 0.25 ? 0 : (v < 0.5 ? 1 : (v < 0.75 ? 2 : 3)); }

bucket_key key_for(const common_flydelta_outcome_guidance_context & c,
        const common_flydelta_outcome_guidance_candidate & a) {
    return {c.semantic_domain, c.strategy, c.estimator, a.rank,
        layer_bucket(a.relative_layer), dose_bucket(a.relative_dose),
        a.mixed_coefficients ? 1 : 0, a.diagnostic_margin_sign < 0 ? -1 :
            (a.diagnostic_margin_sign > 0 ? 1 : 0)};
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

struct counts { double known=0, unknown=0, helped=0, neutral=0, harmed=0; };

template<class Predicate>
counts aggregate(const std::vector<const common_flydelta_outcome_guidance_observation*> & rows,
        Predicate pred, const std::set<std::string> & exclude = {}) {
    std::map<std::string, std::vector<const common_flydelta_outcome_guidance_observation*>> groups;
    for (const auto * o : rows) if (!exclude.count(o->fixture_group_id) && pred(*o))
        groups[o->fixture_group_id].push_back(o);
    counts c;
    for (const auto & group : groups) {
        const double w = 1.0 / static_cast<double>(group.second.size());
        for (const auto * o : group.second) {
            if (!o->verifier_known) { c.unknown += w; continue; }
            c.known += w;
            switch (o->outcome) {
                case common_flydelta_counterfactual_outcome::helped: c.helped += w; break;
                case common_flydelta_counterfactual_outcome::neutral: c.neutral += w; break;
                case common_flydelta_counterfactual_outcome::harmed: c.harmed += w; break;
                case common_flydelta_counterfactual_outcome::unknown: break;
            }
        }
    }
    return c;
}

std::string history_hash(const std::vector<const common_flydelta_outcome_guidance_observation*> & rows) {
    std::vector<const common_flydelta_outcome_guidance_observation*> sorted = rows;
    std::sort(sorted.begin(), sorted.end(), [](const auto * a, const auto * b) {
        return a->stable_observation_id < b->stable_observation_id;
    });
    uint64_t h = 1469598103934665603ULL;
    const auto add = [&h](const std::string & s) {
        for (const unsigned char ch : s) { h ^= ch; h *= 1099511628211ULL; }
        h ^= 0xff; h *= 1099511628211ULL;
    };
    const auto add_number = [&add](double v) {
        std::ostringstream value;
        value.precision(17);
        value << v;
        add(value.str());
    };
    for (const auto * o : sorted) {
        add(o->stable_observation_id);
        add(o->fixture_group_id);
        add(o->basis_identity);
        add(std::to_string(static_cast<int>(o->outcome)));
        add(o->verifier_known ? "known" : "unknown");
        add(std::to_string(o->rank));
        add_number(o->relative_layer);
        add_number(o->relative_dose);
        add(o->mixed_coefficients ? "mixed" : "single");
        add(std::to_string(o->diagnostic_margin_sign));
    }
    std::ostringstream out; out << std::hex << h;
    return out.str();
}

double sample_beta(double a, double b, std::mt19937_64 & rng) {
    std::gamma_distribution<double> ga(std::max(1e-6, a), 1.0), gb(std::max(1e-6, b), 1.0);
    const double x = ga(rng), y = gb(rng);
    return x + y > 0 ? x / (x + y) : 0.5;
}

std::pair<double, double> beta_interval(double a, double b) {
    const double total = a + b;
    const double mean = a / total;
    const double variance = a * b / (total * total * (total + 1.0));
    const double radius = 1.96 * std::sqrt(std::max(0.0, variance));
    return {std::max(0.0, mean-radius), std::min(1.0, mean+radius)};
}

} // namespace

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
        common_flydelta_outcome_guidance_decision & d, std::string & error) {
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
    std::set<std::string> seen;
    for (const auto & o : history) {
        if (o.stable_observation_id.empty() || o.fixture_group_id.empty() ||
                !o.comparison_completed || !o.overlay_applied ||
                hard_key(o.context) != hard_key(context) ||
                !std::isfinite(o.relative_layer) || !std::isfinite(o.relative_dose) ||
                o.relative_layer < 0 || o.relative_layer > 1 || o.relative_dose < 0 || o.relative_dose > 1) continue;
        if (o.verifier_known && o.outcome == common_flydelta_counterfactual_outcome::unknown) continue;
        if (!o.verifier_known && o.outcome != common_flydelta_counterfactual_outcome::unknown) continue;
        if (!seen.insert(o.stable_observation_id).second) continue;
        rows.push_back(&o);
    }
    d.history_fingerprint = history_hash(rows);
    d.shadow_order = d.original_order;
    d.selected_order = d.original_order;
    d.selection_reasons.assign(candidates.size(), config.mode == common_flydelta_outcome_guidance_mode::off
        ? "guidance_off_existing_order" : "shadow_preserved_existing_order");
    if (config.mode == common_flydelta_outcome_guidance_mode::off || candidates.empty()) return true;
    d.selected_order.clear();
    d.selection_reasons.clear();

    std::mt19937_64 rng(config.seed ^ job_seed);
    struct scored { size_t index; double utility; };
    std::vector<scored> sampled;
    for (size_t i=0; i<candidates.size(); ++i) {
        const auto exact = key_for(context, candidates[i]);
        const counts leaf = aggregate(rows, [&](const auto & o) { return key_for(o) == exact; });
        std::set<std::string> leaf_group_ids;
        for (auto * o : rows) if (key_for(*o) == exact) leaf_group_ids.insert(o->fixture_group_id);
        const counts strategy = aggregate(rows, [&](const auto & o) {
            return o.context.semantic_domain == context.semantic_domain &&
                o.context.strategy == context.strategy && o.context.estimator == context.estimator;
        }, leaf_group_ids);
        const counts domain = aggregate(rows, [&](const auto & o) {
            return o.context.semantic_domain == context.semantic_domain;
        }, leaf_group_ids);
        const size_t leaf_groups = leaf_group_ids.size();
        const size_t strategy_groups = static_cast<size_t>(std::ceil(strategy.known + strategy.unknown));
        const counts & base = strategy_groups >= config.min_backoff_groups ? strategy : domain;
        const size_t base_groups = base.known + base.unknown > 0
            ? static_cast<size_t>(std::ceil(base.known + base.unknown)) : 0;
        const double prior = base_groups >= config.min_backoff_groups ?
            std::min(config.max_prior_strength, 4.0) : 0.0;
        const double total = base.known + base.unknown;
        const double known_mean = total > 0 ? base.known / total : 0.5;
        const double outcome_total = base.helped + base.neutral + base.harmed;
        const double helped_mean = outcome_total > 0 ? base.helped / outcome_total : 1.0/3.0;
        const double neutral_mean = outcome_total > 0 ? base.neutral / outcome_total : 1.0/3.0;
        const double harmed_mean = outcome_total > 0 ? base.harmed / outcome_total : 1.0/3.0;
        const double pka = 1.0 + leaf.known + prior * known_mean;
        const double pkb = 1.0 + leaf.unknown + prior * (1.0-known_mean);
        const double pha = 1.0 + leaf.helped + prior * helped_mean;
        const double pna = 1.0 + leaf.neutral + prior * neutral_mean;
        const double pda = 1.0 + leaf.harmed + prior * harmed_mean;
        common_flydelta_outcome_guidance_prediction p;
        p.arm_id = candidates[i].arm_id;
        p.level = leaf_groups ? "arm_category" : (prior > 0 ? "backoff" : "global_prior");
        p.p_known = pka/(pka+pkb);
        p.p_helped_given_known = pha/(pha+pna+pda);
        p.p_neutral_given_known = pna/(pha+pna+pda);
        p.p_harmed_given_known = pda/(pha+pna+pda);
        const auto known_interval = beta_interval(pka, pkb);
        const auto helped_interval = beta_interval(pha, pna + pda);
        const auto neutral_interval = beta_interval(pna, pha + pda);
        const auto harm_interval = beta_interval(pda, pha+pna);
        p.p_known_low = known_interval.first; p.p_known_high = known_interval.second;
        p.p_helped_low = helped_interval.first; p.p_helped_high = helped_interval.second;
        p.p_neutral_low = neutral_interval.first; p.p_neutral_high = neutral_interval.second;
        p.p_harmed_low = harm_interval.first; p.p_harmed_high = harm_interval.second;
        p.expected_utility = p.p_known*(p.p_helped_given_known-config.harm_penalty*p.p_harmed_given_known);
        p.known_weight = leaf.known; p.helped_weight = leaf.helped;
        p.neutral_weight = leaf.neutral; p.harmed_weight = leaf.harmed;
        p.effective_known_observations = leaf.known;
        p.effective_unknown_observations = leaf.unknown;
        p.backoff_fixture_groups = base_groups;
        p.backoff_prior_strength = prior;
        p.raw_observations = 0;
        for (auto * o:rows) if (key_for(*o)==exact) ++p.raw_observations;
        p.fixture_groups = leaf_groups;
        d.predictions.push_back(p);
        const double sk = sample_beta(pka, pkb, rng);
        std::gamma_distribution<double> g1(pha,1.0),g2(pna,1.0),g3(pda,1.0);
        const double x=g1(rng),y=g2(rng),z=g3(rng),sum=x+y+z;
        const double utility = sum > 0 ? sk*(x/sum-config.harm_penalty*z/sum) : 0.0;
        sampled.push_back({i,utility});
    }
    std::stable_sort(sampled.begin(),sampled.end(),[](const auto & a,const auto & b){return a.utility>b.utility;});
    d.shadow_order.clear();
    for (const auto & s:sampled) d.shadow_order.push_back(s.index);
    std::set<size_t> selected;
    for (size_t slot=0; slot<selection_limit;) {
        // Keep two existing choices, then draw one posterior choice and one
        // uniform exploration choice when the candidate/budget pool permits.
        for (int k=0;k<2 && slot<selection_limit;++k) {
            for (size_t ix:d.original_order) if (!selected.count(ix)) {
                d.selected_order.push_back(ix); d.selection_reasons.push_back("existing_order");
                selected.insert(ix); ++slot; break;
            }
        }
        if (slot<selection_limit) {
            for (const auto & s:sampled) if (!selected.count(s.index)) {
                d.selected_order.push_back(s.index); d.selection_reasons.push_back("posterior_sample");
                selected.insert(s.index); ++slot; break;
            }
        }
        if (slot<selection_limit) {
            std::vector<size_t> remaining; for (size_t i=0;i<candidates.size();++i) if (!selected.count(i)) remaining.push_back(i);
            if (!remaining.empty()) { std::uniform_int_distribution<size_t> pick(0,remaining.size()-1); const auto ix=remaining[pick(rng)]; d.selected_order.push_back(ix); d.selection_reasons.push_back("uniform_exploration"); selected.insert(ix); ++slot; }
        }
    }
    for (size_t ix : d.original_order) if (!selected.count(ix)) {
        d.selected_order.push_back(ix); d.selection_reasons.push_back("remaining_existing_order");
    }
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
    for (const auto & observed : chronological) {
        const bool eligible = observed.comparison_completed && observed.overlay_applied &&
            !observed.fixture_group_id.empty() && !observed.stable_observation_id.empty() &&
            (!observed.verifier_known
                ? observed.outcome == common_flydelta_counterfactual_outcome::unknown
                : observed.outcome != common_flydelta_counterfactual_outcome::unknown);
        if (!eligible) continue;
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
            const double known_target = observed.verifier_known ? 1.0 : 0.0;
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
            if (observed.verifier_known) {
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
        const common_flydelta_outcome_guidance_context & context,
        const std::vector<common_flydelta_outcome_guidance_candidate> & candidates,
        const common_flydelta_outcome_guidance_decision & decision,
        std::string & error) {
    using json = nlohmann::ordered_json;
    error.clear();
    if (event_context.event_id.empty() || event_context.idempotency_key.empty() ||
            event_context.source_id.empty() || job_id.empty() ||
            decision.original_order.size() != candidates.size() ||
            decision.selected_order.size() != candidates.size() ||
            decision.selection_reasons.size() != candidates.size() ||
            decision.shadow_order.size() != candidates.size() ||
            (decision.predictions.size() != candidates.size() &&
             !(decision.predictions.empty() && !decision.fallback_reason.empty()))) {
        error = "FlyDelta outcome guidance lifecycle decision is incomplete";
        return false;
    }
    json payload = {
        {"record_type", "outcome_guidance_decision"},
        {"schema_version", 1}, {"job_id", job_id}, {"seed", seed},
        {"mode", common_flydelta_outcome_guidance_mode_name(mode)},
        {"history_fingerprint", decision.history_fingerprint},
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
        {"candidate_pool", json::array()},
        {"original_order", decision.original_order},
        {"shadow_order", decision.shadow_order},
        {"selected_order", decision.selected_order},
        {"selection_reasons", json::array()},
        {"fallback_reason", decision.fallback_reason},
        {"predictions", json::array()},
    };
    for (const auto & candidate : candidates) payload["candidate_pool"].push_back({
        {"arm_id", candidate.arm_id}, {"basis_identity", candidate.basis_identity},
        {"rank", candidate.rank},
        {"relative_layer", candidate.relative_layer}, {"relative_dose", candidate.relative_dose},
        {"mixed_coefficients", candidate.mixed_coefficients},
        {"diagnostic_margin_sign", candidate.diagnostic_margin_sign},
    });
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
