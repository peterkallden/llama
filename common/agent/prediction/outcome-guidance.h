#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

// Explicit opt-ins for retrospective guidance to reuse evidence beyond one
// behavior or session. Family membership is host configuration, never a model
// supplied feature. Project history remains within the same namespace/project.
struct common_outcome_guidance_generalization_config {
    bool tool_family_backoff = false;
    bool project_history = false;
    std::map<std::string, std::vector<std::string>> tool_families;
};

bool common_outcome_guidance_generalization_config_validate(
        const common_outcome_guidance_generalization_config & config,
        std::string & error);

struct common_outcome_guidance_projection_observation {
    std::string fixture_group;
    std::string category;
    std::vector<double> outcome_weights;
};

// Immutable-by-convention, domain-neutral grouped projection. Each fixture
// contributes at most one normalized unit of weight per category; consumers
// may exclude the query fixture without rebuilding or mutating the snapshot.
struct common_outcome_guidance_projection {
    std::string builder_revision;
    std::string source_fingerprint;
    std::string fingerprint;
    uint64_t source_watermark = 0;
    size_t accepted_rows = 0;
    std::map<std::string, std::map<std::string, std::vector<double>>> fixture_categories;

    void aggregate_excluding(const std::string & fixture_group,
            std::map<std::string, std::vector<double>> & category_weights,
            std::map<std::string, size_t> & fixture_counts) const;
};

bool common_outcome_guidance_build_projection(
        const std::vector<common_outcome_guidance_projection_observation> & observations,
        const std::string & builder_revision,
        const std::string & source_fingerprint,
        uint64_t source_watermark,
        common_outcome_guidance_projection & projection,
        std::string & error);

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
