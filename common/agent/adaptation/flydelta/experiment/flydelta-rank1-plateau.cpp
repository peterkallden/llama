#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
bool finite_value(float value) { return std::isfinite(value); }
}

bool common_flydelta_rank1_plateau_config_validate(
        const common_flydelta_rank1_plateau_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.minimum_bootstrap_arms < 2 ||
            config.minimum_bootstrap_arms > 64 || config.required_plateau_rounds == 0 ||
            config.required_plateau_rounds > 8 || !finite_value(config.minimum_useful_margin) ||
            !finite_value(config.maximum_recent_gain_ratio) || config.maximum_recent_gain_ratio < 0.0f ||
            config.maximum_recent_gain_ratio > 1.0f || config.maximum_region_span == 0 ||
            config.maximum_region_span > 64 || !finite_value(config.shallow_rank_threshold) ||
            config.shallow_rank_threshold <= 1.0f) {
        error = "FlyDelta rank-one plateau configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_evaluate_rank1_plateau(
        const common_flydelta_rank1_plateau_config & config,
        float effective_rank,
        const std::vector<common_flydelta_rank1_plateau_round> & rounds,
        common_flydelta_rank1_plateau_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_rank1_plateau_config_validate(config, error) ||
            !finite_value(effective_rank) || effective_rank < 0.0f || rounds.empty() ||
            rounds.size() > 32) {
        if (error.empty()) error = "FlyDelta rank-one plateau input is invalid";
        return false;
    }
    result.minimum_anchor_layer = std::numeric_limits<uint32_t>::max();
    for (const auto & round : rounds) {
        if (!round.safe_to_continue || round.evaluated_arms == 0 ||
                !finite_value(round.best_margin_delta) || round.anchor_layer == 0) {
            result = {};
            result.action = common_flydelta_rank1_plateau_action::continue_bootstrap;
            return true;
        }
        result.safe_arm_count += round.evaluated_arms;
        result.best_margin_delta = std::max(result.best_margin_delta, round.best_margin_delta);
        result.minimum_anchor_layer = std::min(result.minimum_anchor_layer, round.anchor_layer);
        result.maximum_anchor_layer = std::max(result.maximum_anchor_layer, round.anchor_layer);
    }
    if (result.safe_arm_count < config.minimum_bootstrap_arms ||
            effective_rank >= config.shallow_rank_threshold ||
            result.best_margin_delta <= config.minimum_useful_margin ||
            result.maximum_anchor_layer - result.minimum_anchor_layer > config.maximum_region_span) {
        result.minimum_anchor_layer = result.minimum_anchor_layer ==
            std::numeric_limits<uint32_t>::max() ? 0 : result.minimum_anchor_layer;
        result.action = common_flydelta_rank1_plateau_action::continue_bootstrap;
        return true;
    }
    result.eligible = true;
    if (rounds.size() < 2) {
        result.action = common_flydelta_rank1_plateau_action::refine_bootstrap;
        return true;
    }
    for (size_t index = rounds.size(); index > 1; --index) {
        const float previous = rounds[index - 2].best_margin_delta;
        const float current = rounds[index - 1].best_margin_delta;
        if (current < previous) break;
        const float recent_gain = current - previous;
        const float ratio = recent_gain /
            std::max(std::fabs(current), std::numeric_limits<float>::epsilon());
        if (result.plateau_streak == 0) result.recent_gain_ratio = ratio;
        if (ratio > config.maximum_recent_gain_ratio) break;
        ++result.plateau_streak;
    }
    result.plateau = result.plateau_streak >= config.required_plateau_rounds;
    result.action = result.plateau
        ? common_flydelta_rank1_plateau_action::orthogonal_search
        : common_flydelta_rank1_plateau_action::refine_bootstrap;
    return true;
}

bool common_flydelta_decide_rank1_plateau_utility(
        const common_flydelta_rank1_plateau_result & plateau,
        common_flydelta_utility_gate_decision & decision,
        std::string & error) {
    error.clear();
    decision = {};
    if (!finite_value(plateau.best_margin_delta) || !finite_value(plateau.recent_gain_ratio)) {
        error = "FlyDelta rank-one plateau result is invalid";
        return false;
    }
    decision.history = {plateau.plateau_streak, 0};
    if (!plateau.eligible) {
        decision.action = common_flydelta_utility_gate_action::retain;
        return true;
    }
    decision.utility_qualified = true;
    decision.action = plateau.action == common_flydelta_rank1_plateau_action::orthogonal_search
        ? common_flydelta_utility_gate_action::orthogonal_search
        : plateau.action == common_flydelta_rank1_plateau_action::refine_bootstrap
            ? common_flydelta_utility_gate_action::refine_bootstrap
            : common_flydelta_utility_gate_action::retain;
    return true;
}
