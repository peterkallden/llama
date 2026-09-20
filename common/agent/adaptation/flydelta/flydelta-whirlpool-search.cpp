#include "agent/adaptation/flydelta/flydelta-whirlpool-search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace {

float diagnostic_score(const common_flydelta_layer_diagnostic & value) {
    return value.progress * std::max(0.0f, value.cosine) /
        (1.0f + std::max(0.0f, value.leakage));
}

bool contains(const std::vector<uint32_t> & values, uint32_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool safe_geometry(const common_flydelta_representation_diagnostics & geometry,
        bool available, const common_flydelta_whirlpool_search_config & config) {
    // A margin-only Whirlpool is allowed when no dose controller is active.
    // Once dose regulation is enabled, missing geometry is unknown rather than
    // safe: the arm cannot be compared or admitted to the safety envelope.
    return available ? (geometry.cosine >= config.min_cosine &&
        geometry.leakage <= config.max_leakage &&
        geometry.shift_norm <= config.max_shift_norm) : !config.use_dose_controller;
}

float objective(const common_flydelta_counterfactual_trial & baseline,
        const common_flydelta_decision_margin & baseline_margin,
        const common_flydelta_counterfactual_trial & trial,
        const common_flydelta_decision_margin & margin,
        const common_flydelta_representation_diagnostics & geometry,
        bool geometry_available,
        const common_flydelta_whirlpool_search_config & config) {
    float value = trial.quality - baseline.quality;
    if (baseline_margin.available && margin.available) {
        value += config.margin_weight *
            (margin.normalized_delta() - baseline_margin.normalized_delta());
    }
    if (geometry_available) {
        value += config.geometry_weight * geometry.progress *
            std::max(0.0f, geometry.cosine);
        value -= config.leakage_penalty * geometry.leakage;
    }
    return std::isfinite(value) ? value : -std::numeric_limits<float>::infinity();
}

uint32_t initial_centre(const common_flydelta_whirlpool_search_config & config) {
    float best_score = -std::numeric_limits<float>::infinity();
    uint32_t best_layer = 0;
    for (const auto & diagnostic : config.layer_diagnostics) {
        if (diagnostic.layer_index == 0 || !contains(config.available_layers,
                diagnostic.layer_index)) continue;
        const float score = diagnostic_score(diagnostic);
        if (score > best_score) {
            best_score = score;
            best_layer = diagnostic.layer_index;
        }
    }
    if (best_layer != 0) return best_layer;
    if (!config.seed_layers.empty()) return config.seed_layers[config.seed_layers.size() / 2];
    return config.available_layers[config.available_layers.size() / 2];
}

std::vector<uint32_t> probes(uint32_t centre, uint32_t radius,
        const std::vector<uint32_t> & available, size_t limit) {
    std::vector<uint32_t> result;
    const uint32_t half = std::max<uint32_t>(1, radius / 2);
    const std::vector<int64_t> offsets = {
        0, -static_cast<int64_t>(radius), static_cast<int64_t>(radius),
        -static_cast<int64_t>(half), static_cast<int64_t>(half)};
    for (const int64_t offset : offsets) {
        const int64_t candidate = static_cast<int64_t>(centre) + offset;
        if (candidate <= 0 || candidate > std::numeric_limits<uint32_t>::max()) continue;
        const uint32_t layer = static_cast<uint32_t>(candidate);
        if (!contains(available, layer) || contains(result, layer)) continue;
        result.push_back(layer);
        if (result.size() >= limit) break;
    }
    return result;
}

} // namespace

bool common_flydelta_whirlpool_search_config_validate(
        const common_flydelta_whirlpool_search_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.available_layers.empty() ||
            config.available_layers.size() > 64 ||
            !std::is_sorted(config.available_layers.begin(), config.available_layers.end()) ||
            config.available_layers.front() == 0 ||
            std::adjacent_find(config.available_layers.begin(), config.available_layers.end()) !=
                config.available_layers.end() || config.max_rounds == 0 ||
            config.max_rounds > 16 || config.probes_per_round < 2 ||
            config.probes_per_round > 8 || config.max_trials == 0 || config.max_trials > 64 ||
            config.initial_radius == 0 || !std::isfinite(config.total_scale) ||
            config.total_scale <= 0.0f || config.total_scale > 1.0f ||
            !std::isfinite(config.shrink_factor) || config.shrink_factor <= 0.0f ||
            config.shrink_factor >= 1.0f || !std::isfinite(config.margin_weight) ||
            config.margin_weight < 0.0f || !std::isfinite(config.geometry_weight) ||
            config.geometry_weight < 0.0f || !std::isfinite(config.leakage_penalty) ||
            config.leakage_penalty < 0.0f || !std::isfinite(config.max_leakage) ||
            config.max_leakage < 0.0f || !std::isfinite(config.max_shift_norm) ||
            config.max_shift_norm <= 0.0f || !std::isfinite(config.min_cosine) ||
            config.min_cosine < -1.0f || config.min_cosine > 1.0f ||
            config.max_dose_retries > 1 ||
            (config.use_dose_controller &&
             !common_flydelta_dose_policy_validate(config.dose_policy, error))) {
        error = "FlyDelta Whirlpool search configuration is invalid";
        return false;
    }
    for (const uint32_t layer : config.seed_layers) {
        if (layer == 0 || !contains(config.available_layers, layer)) {
            error = "FlyDelta Whirlpool seed layer is unavailable";
            return false;
        }
    }
    for (const auto & diagnostic : config.layer_diagnostics) {
        if (diagnostic.layer_index == 0 || !contains(config.available_layers,
                diagnostic.layer_index) || !std::isfinite(diagnostic.cosine) ||
                !std::isfinite(diagnostic.progress) || !std::isfinite(diagnostic.leakage) ||
                !std::isfinite(diagnostic.shift_norm)) {
            error = "FlyDelta Whirlpool layer diagnostic is invalid";
            return false;
        }
    }
    return true;
}

bool common_flydelta_whirlpool_trace_validate(
        const common_flydelta_whirlpool_trace & trace,
        const common_flydelta_whirlpool_search_config & config,
        size_t trial_count,
        std::string & error) {
    error.clear();
    if (trace.schema_version != 1 || trace.model_evaluations < trial_count + 1 ||
            trace.rounds.size() > config.max_rounds ||
            (trace.best_trial_index != static_cast<size_t>(-1) &&
                trace.best_trial_index >= trial_count) ||
            !std::isfinite(trace.best_search_score) || trace.final_centre == 0 ||
            trace.final_radius == 0) {
        error = "FlyDelta Whirlpool trace is invalid";
        return false;
    }
    for (const auto & round : trace.rounds) {
        if (round.round >= config.max_rounds || round.centre_before == 0 ||
                round.radius_before == 0 || round.probed_layers.empty() ||
                round.probed_layers.size() > config.probes_per_round ||
                round.best_probe_layer == 0 ||
                !std::isfinite(round.best_probe_score) || round.centre_after == 0 ||
                round.radius_after == 0) {
            error = "FlyDelta Whirlpool round trace is invalid";
            return false;
        }
    }
    return true;
}

bool common_flydelta_run_whirlpool_search_batched(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_whirlpool_search_config & config,
        const common_flydelta_whirlpool_search_runner & baseline_runner,
        const common_flydelta_whirlpool_search_batch_runner & batch_runner,
        std::vector<common_flydelta_intervention_region_trial> & trials,
        common_flydelta_intervention_region_selection & selection,
        common_flydelta_whirlpool_trace & trace,
        std::string & error) {
    error.clear();
    trials.clear();
    selection = {};
    trace = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_whirlpool_search_config_validate(config, error) ||
            !baseline_runner || !batch_runner) {
        if (error.empty()) error = "FlyDelta Whirlpool runner is invalid";
        return false;
    }

    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    common_flydelta_representation_diagnostics baseline_geometry;
    bool baseline_geometry_available = false;
    if (!baseline_runner(fixture, nullptr, baseline, baseline_margin,
                baseline_geometry, baseline_geometry_available, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error)) {
        return false;
    }
    trace.model_evaluations = 1;

    uint32_t centre = initial_centre(config);
    uint32_t radius = config.initial_radius;
    std::vector<uint32_t> visited;
    float best_objective = -std::numeric_limits<float>::infinity();
    size_t best_trial = 0;
    std::unordered_map<uint32_t, common_flydelta_dose_state> dose_states;

    for (size_t round = 0; round < config.max_rounds && trials.size() < config.max_trials;
            ++round) {
        common_flydelta_whirlpool_round_trace round_trace;
        round_trace.round = round;
        round_trace.centre_before = centre;
        round_trace.radius_before = radius;
        const auto round_probes = probes(centre, radius, config.available_layers,
            std::min(config.probes_per_round, config.max_trials - trials.size()));
        std::vector<common_flydelta_intervention_region_candidate> candidates;
        for (const uint32_t layer : round_probes) {
            if (contains(visited, layer) || trials.size() + candidates.size() >= config.max_trials) continue;
            visited.push_back(layer);
            round_trace.probed_layers.push_back(layer);
            common_flydelta_intervention_region_candidate candidate;
            candidate.layer_indices = {layer};
            candidate.anchor_layer_index = layer;
            candidate.total_scale = config.total_scale;
            candidate.per_layer_scale = config.total_scale;
            candidate.source = common_flydelta_layer_search_candidate_source::diagnostic_singleton;

            candidates.push_back(std::move(candidate));
        }

        std::vector<common_flydelta_counterfactual_trial> batch_trials;
        std::vector<common_flydelta_decision_margin> batch_margins;
        std::vector<common_flydelta_representation_diagnostics> batch_geometries;
        std::vector<bool> batch_geometry_available;
        std::vector<float> requested_scales;
        requested_scales.reserve(candidates.size());
        for (const auto & candidate : candidates) {
            requested_scales.push_back(candidate.total_scale);
        }
        if (!candidates.empty() && !batch_runner(fixture, candidates, batch_trials,
                batch_margins, batch_geometries, batch_geometry_available, error)) return false;
        if (batch_trials.size() != candidates.size() || batch_margins.size() != candidates.size() ||
                batch_geometries.size() != candidates.size() ||
                batch_geometry_available.size() != candidates.size()) {
            error = "FlyDelta Whirlpool batch returned an incomplete probe set";
            return false;
        }

        std::vector<size_t> retry_indices;
        std::vector<common_flydelta_intervention_region_candidate> retry_candidates;
        std::vector<common_flydelta_dose_decision> dose_decisions(candidates.size());
        std::vector<bool> dose_retry(candidates.size(), false);
        for (size_t index = 0; index < candidates.size(); ++index) {
            auto & candidate = candidates[index];
            auto & counterfactual = batch_trials[index];
            auto & margin = batch_margins[index];
            auto & geometry = batch_geometries[index];
            const bool geometry_available = batch_geometry_available[index];
            if (!common_flydelta_counterfactual_trial_validate(counterfactual, error) ||
                    !common_flydelta_decision_margin_validate(margin, error)) return false;
            if (geometry_available &&
                    !common_flydelta_representation_diagnostics_validate(geometry, error)) return false;
            if (!config.use_dose_controller) continue;
            common_flydelta_dose_observation observation{
                geometry_available, candidate.total_scale,
                geometry_available ? geometry.shift_norm : 0.0f,
                geometry_available ? geometry.progress : 0.0f,
                geometry_available ? geometry.leakage : 0.0f,
            };
            if (!common_flydelta_dose_observe(config.dose_policy,
                    dose_states[candidate.anchor_layer_index], observation,
                    dose_decisions[index], error)) return false;
            if (dose_decisions[index].action == common_flydelta_dose_action::retry_lower &&
                    dose_decisions[index].proposed_safe_strength && config.max_dose_retries > 0) {
                dose_retry[index] = true;
                candidate.total_scale = *dose_decisions[index].proposed_safe_strength;
                candidate.per_layer_scale = candidate.total_scale;
                retry_indices.push_back(index);
                retry_candidates.push_back(candidate);
            }
        }
        if (!retry_candidates.empty()) {
            std::vector<common_flydelta_counterfactual_trial> retry_trials;
            std::vector<common_flydelta_decision_margin> retry_margins;
            std::vector<common_flydelta_representation_diagnostics> retry_geometries;
            std::vector<bool> retry_geometry_available;
            if (!batch_runner(fixture, retry_candidates, retry_trials, retry_margins,
                    retry_geometries, retry_geometry_available, error) ||
                    retry_trials.size() != retry_candidates.size() ||
                    retry_margins.size() != retry_candidates.size() ||
                    retry_geometries.size() != retry_candidates.size() ||
                    retry_geometry_available.size() != retry_candidates.size()) {
                if (error.empty()) error = "FlyDelta Whirlpool retry batch was incomplete";
                return false;
            }
            for (size_t retry = 0; retry < retry_indices.size(); ++retry) {
                const size_t index = retry_indices[retry];
                batch_trials[index] = std::move(retry_trials[retry]);
                batch_margins[index] = std::move(retry_margins[retry]);
                batch_geometries[index] = std::move(retry_geometries[retry]);
                batch_geometry_available[index] = retry_geometry_available[retry];
                if (!common_flydelta_counterfactual_trial_validate(batch_trials[index], error) ||
                        !common_flydelta_decision_margin_validate(batch_margins[index], error)) return false;
                if (batch_geometry_available[index] &&
                        !common_flydelta_representation_diagnostics_validate(batch_geometries[index], error)) return false;
                const auto & candidate = candidates[index];
                common_flydelta_dose_observation observation{
                    batch_geometry_available[index], candidate.total_scale,
                    batch_geometry_available[index] ? batch_geometries[index].shift_norm : 0.0f,
                    batch_geometry_available[index] ? batch_geometries[index].progress : 0.0f,
                    batch_geometry_available[index] ? batch_geometries[index].leakage : 0.0f,
                };
                if (!common_flydelta_dose_observe(config.dose_policy,
                        dose_states[candidate.anchor_layer_index], observation,
                        dose_decisions[index], error)) return false;
            }
        }

        bool found_round_candidate = false;
        float round_best = -std::numeric_limits<float>::infinity();
        uint32_t round_centre = centre;
        for (size_t index = 0; index < candidates.size(); ++index) {
            const auto & candidate = candidates[index];
            const auto & counterfactual = batch_trials[index];
            const auto & margin = batch_margins[index];
            const auto & geometry = batch_geometries[index];
            const bool geometry_available = batch_geometry_available[index];
            common_flydelta_dose_decision dose_decision = dose_decisions[index];
            const bool dose_retry_performed = dose_retry[index];

            common_flydelta_intervention_region_trial region_trial;
            region_trial.candidate = candidate;
            region_trial.requested_total_scale = requested_scales[index];
            region_trial.executed_total_scale = candidate.total_scale;
            region_trial.dose_evaluated = config.use_dose_controller;
            region_trial.dose_action = dose_decision.action;
            region_trial.relative_dose = dose_decision.relative_dose;
            region_trial.dose_comparable = dose_decision.comparable;
            region_trial.dose_safety_limited =
                dose_retry_performed || dose_decision.safety_limited;
            region_trial.dose_reason = dose_retry_performed
                ? "retry_lower: " + dose_decision.reason : dose_decision.reason;
            region_trial.outcome = common_flydelta_classify_counterfactual(
                baseline, counterfactual);
            region_trial.quality_delta = counterfactual.quality - baseline.quality;
            region_trial.margin = margin;
            region_trial.margin_comparison.available = baseline_margin.available && margin.available;
            region_trial.margin_comparison.baseline = baseline_margin;
            region_trial.margin_comparison.candidate = margin;
            region_trial.executed = counterfactual.executed;
            region_trial.verifier_known = baseline.verifier_known && counterfactual.verifier_known;
            region_trial.geometry_available = geometry_available;
            region_trial.geometry = geometry;
            region_trial.safe_to_continue = safe_geometry(
                geometry, geometry_available, config);
            region_trial.promising = region_trial.outcome ==
                common_flydelta_counterfactual_outcome::helped ||
                (region_trial.safe_to_continue && geometry_available && geometry.progress > 0.0f) ||
                (baseline_margin.available && margin.available &&
                    margin.normalized_delta() > baseline_margin.normalized_delta());
            region_trial.search_score = objective(baseline, baseline_margin, counterfactual,
                margin, geometry, geometry_available, config);
            region_trial.evidence_ref = counterfactual.evidence_ref;
            trials.push_back(std::move(region_trial));
            const auto & stored = trials.back();
            if (stored.search_score > best_objective) {
                best_objective = stored.search_score;
                best_trial = trials.size() - 1;
            }
            if (stored.search_score > round_best) {
                round_best = stored.search_score;
                round_centre = candidate.anchor_layer_index;
                found_round_candidate = true;
            }
        }
        trace.model_evaluations += candidates.size() + retry_candidates.size();
        // Never recede from the best observed point. A later local round may
        // be noisier than the previous one; it can still shrink the trust
        // region without moving the centre to a worse probe.
        if (found_round_candidate && round_best >= best_objective) centre = round_centre;
        if (radius > 1) {
            radius = std::max<uint32_t>(1, static_cast<uint32_t>(
                std::floor(static_cast<float>(radius) * config.shrink_factor)));
        }
        round_trace.best_probe_layer = found_round_candidate ? round_centre : 0;
        round_trace.best_probe_score = found_round_candidate ? round_best : 0.0f;
        round_trace.centre_after = centre;
        round_trace.radius_after = radius;
        if (!round_trace.probed_layers.empty()) trace.rounds.push_back(std::move(round_trace));
        if (radius == 1 && round + 1 >= config.max_rounds) break;
        if (best_trial >= trials.size()) break;
    }

    float best_helped_score = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (trial.outcome != common_flydelta_counterfactual_outcome::helped ||
                !trial.executed || !trial.verifier_known) continue;
        if (!selection.selected || trial.quality_delta > best_helped_score ||
                (trial.quality_delta == best_helped_score &&
                 trial.candidate.total_scale < trials[selection.trial_index].candidate.total_scale)) {
            selection.selected = true;
            selection.trial_index = index;
            selection.score = trial.quality_delta;
            best_helped_score = trial.quality_delta;
        }
    }
    trace.best_trial_index = best_trial;
    trace.best_search_score = best_objective;
    trace.final_centre = centre;
    trace.final_radius = radius;
    if (!common_flydelta_whirlpool_trace_validate(trace, config, trials.size(), error)) {
        return false;
    }
    return true;
}

bool common_flydelta_run_whirlpool_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_whirlpool_search_config & config,
        const common_flydelta_whirlpool_search_runner & runner,
        std::vector<common_flydelta_intervention_region_trial> & trials,
        common_flydelta_intervention_region_selection & selection,
        common_flydelta_whirlpool_trace & trace,
        std::string & error) {
    if (!runner) {
        error = "FlyDelta Whirlpool runner is invalid";
        return false;
    }
    const common_flydelta_whirlpool_search_batch_runner batch_runner =
        [&](const common_flydelta_experiment_fixture & current_fixture,
            const std::vector<common_flydelta_intervention_region_candidate> & candidates,
            std::vector<common_flydelta_counterfactual_trial> & batch_trials,
            std::vector<common_flydelta_decision_margin> & batch_margins,
            std::vector<common_flydelta_representation_diagnostics> & batch_geometries,
            std::vector<bool> & batch_geometry_available,
            std::string & batch_error) {
            batch_trials.clear();
            batch_margins.clear();
            batch_geometries.clear();
            batch_geometry_available.clear();
            batch_trials.reserve(candidates.size());
            batch_margins.reserve(candidates.size());
            batch_geometries.reserve(candidates.size());
            batch_geometry_available.reserve(candidates.size());
            for (const auto & candidate : candidates) {
                common_flydelta_counterfactual_trial trial;
                common_flydelta_decision_margin margin;
                common_flydelta_representation_diagnostics geometry;
                bool geometry_available = false;
                if (!runner(current_fixture, &candidate, trial, margin, geometry,
                        geometry_available, batch_error)) return false;
                batch_trials.push_back(std::move(trial));
                batch_margins.push_back(std::move(margin));
                batch_geometries.push_back(std::move(geometry));
                batch_geometry_available.push_back(geometry_available);
            }
            return true;
        };
    return common_flydelta_run_whirlpool_search_batched(
        fixture, config, runner, batch_runner, trials, selection, trace, error);
}
