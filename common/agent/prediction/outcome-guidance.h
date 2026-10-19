#pragma once

#include <cstddef>
#include <random>
#include <vector>

// Domain-neutral conjugate categorical outcome prediction. Callers own the
// meaning and provenance of categories and the construction of weighted counts.
struct common_outcome_guidance_posterior {
    double p_known = 0.5;
    double p_known_low = 0.0;
    double p_known_high = 1.0;
    std::vector<double> category_probabilities;
    std::vector<double> category_lows;
    std::vector<double> category_highs;
    double known_alpha = 1.0;
    double known_beta = 1.0;
    std::vector<double> category_alpha;
};

bool common_outcome_guidance_make_posterior(
        double known_weight,
        double unknown_weight,
        double prior_known_mean,
        const std::vector<double> & category_weights,
        const std::vector<double> & prior_category_means,
        double verifiability_prior_strength,
        double categorical_prior_strength,
        common_outcome_guidance_posterior & posterior);

double common_outcome_guidance_sample_expected_utility(
        const common_outcome_guidance_posterior & posterior,
        const std::vector<double> & category_utilities,
        std::mt19937_64 & rng);
