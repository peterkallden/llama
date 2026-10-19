#include "agent/prediction/outcome-guidance.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

namespace {

std::pair<double, double> interval(double a, double b) {
    const double total = a + b;
    const double mean = total > 0.0 ? a / total : 0.5;
    const double variance = total > 0.0
        ? a * b / (total * total * (total + 1.0)) : 0.25;
    const double radius = 1.96 * std::sqrt(std::max(0.0, variance));
    return {std::max(0.0, mean - radius), std::min(1.0, mean + radius)};
}

} // namespace

bool common_outcome_guidance_make_posterior(
        const double known_weight,
        const double unknown_weight,
        const double prior_known_mean,
        const std::vector<double> & category_weights,
        const std::vector<double> & prior_category_means,
        const double verifiability_prior_strength,
        const double categorical_prior_strength,
        common_outcome_guidance_posterior & posterior) {
    posterior = {};
    if (!std::isfinite(known_weight) || known_weight < 0.0 ||
            !std::isfinite(unknown_weight) || unknown_weight < 0.0 ||
            !std::isfinite(prior_known_mean) || prior_known_mean < 0.0 || prior_known_mean > 1.0 ||
            !std::isfinite(verifiability_prior_strength) || verifiability_prior_strength < 0.0 ||
            !std::isfinite(categorical_prior_strength) || categorical_prior_strength < 0.0 ||
            category_weights.empty() || category_weights.size() != prior_category_means.size()) return false;
    double prior_sum = 0.0;
    for (size_t i = 0; i < category_weights.size(); ++i) {
        if (!std::isfinite(category_weights[i]) || category_weights[i] < 0.0 ||
                !std::isfinite(prior_category_means[i]) || prior_category_means[i] < 0.0 ||
                prior_category_means[i] > 1.0) return false;
        prior_sum += prior_category_means[i];
    }
    if (!(prior_sum > 0.0) || !std::isfinite(prior_sum)) return false;

    posterior.known_alpha = 1.0 + known_weight +
        verifiability_prior_strength * prior_known_mean;
    posterior.known_beta = 1.0 + unknown_weight +
        verifiability_prior_strength * (1.0 - prior_known_mean);
    const double known_total = posterior.known_alpha + posterior.known_beta;
    posterior.p_known = posterior.known_alpha / known_total;
    const auto known_bounds = interval(posterior.known_alpha, posterior.known_beta);
    posterior.p_known_low = known_bounds.first;
    posterior.p_known_high = known_bounds.second;

    posterior.category_alpha.reserve(category_weights.size());
    posterior.category_probabilities.reserve(category_weights.size());
    posterior.category_lows.reserve(category_weights.size());
    posterior.category_highs.reserve(category_weights.size());
    const double prior_normalizer = prior_sum;
    double alpha_sum = 0.0;
    for (size_t i = 0; i < category_weights.size(); ++i) {
        const double alpha = 1.0 + category_weights[i] +
            categorical_prior_strength * prior_category_means[i] / prior_normalizer;
        posterior.category_alpha.push_back(alpha);
        alpha_sum += alpha;
    }
    for (size_t i = 0; i < posterior.category_alpha.size(); ++i) {
        const double alpha = posterior.category_alpha[i];
        posterior.category_probabilities.push_back(alpha / alpha_sum);
        double other = alpha_sum - alpha;
        const auto bounds = interval(alpha, other);
        posterior.category_lows.push_back(bounds.first);
        posterior.category_highs.push_back(bounds.second);
    }
    return true;
}

double common_outcome_guidance_sample_expected_utility(
        const common_outcome_guidance_posterior & posterior,
        const std::vector<double> & category_utilities,
        std::mt19937_64 & rng) {
    if (category_utilities.size() != posterior.category_alpha.size()) return 0.0;
    std::gamma_distribution<double> known(posterior.known_alpha, 1.0);
    std::gamma_distribution<double> unknown(posterior.known_beta, 1.0);
    const double k = known(rng), u = unknown(rng);
    const double p_known = k + u > 0.0 ? k / (k + u) : 0.5;
    std::vector<double> draws;
    draws.reserve(posterior.category_alpha.size());
    double total = 0.0;
    for (double alpha : posterior.category_alpha) {
        std::gamma_distribution<double> gamma(std::max(1e-6, alpha), 1.0);
        draws.push_back(gamma(rng));
        total += draws.back();
    }
    if (!(total > 0.0)) return 0.0;
    double utility = 0.0;
    for (size_t i = 0; i < draws.size(); ++i) {
        if (!std::isfinite(category_utilities[i])) return 0.0;
        utility += draws[i] / total * category_utilities[i];
    }
    return p_known * utility;
}
