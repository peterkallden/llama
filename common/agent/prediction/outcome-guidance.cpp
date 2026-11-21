#include "agent/prediction/outcome-guidance.h"
#include "hash/hash.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <utility>
#include <set>

bool common_outcome_guidance_generalization_config_validate(
        const common_outcome_guidance_generalization_config & config,
        std::string & error) {
    error.clear();
    if (config.tool_families.size() > 64) {
        error = "outcome guidance supports at most 64 configured tool families";
        return false;
    }
    std::set<std::string> domains;
    for (const auto & family : config.tool_families) {
        if (family.first.empty() || family.first.size() > 128 ||
                family.second.size() < 2 || family.second.size() > 64) {
            error = "each outcome tool family requires an id and 2..64 behavior domains";
            return false;
        }
        std::set<std::string> local;
        for (const auto & domain : family.second) {
            if (domain.empty() || domain.size() > 128 || !local.insert(domain).second ||
                    !domains.insert(domain).second) {
                error = "outcome tool-family behavior domains must be unique non-empty strings";
                return false;
            }
        }
    }
    return true;
}

namespace {

std::pair<double, double> interval(double a, double b) {
    const double total = a + b;
    const double mean = total > 0.0 ? a / total : 0.5;
    const double variance = total > 0.0
        ? mean * (1.0 - mean) / (total + 1.0) : 0.25;
    const double radius = 1.96 * std::sqrt(std::max(0.0, variance));
    return {std::max(0.0, mean - radius), std::min(1.0, mean + radius)};
}

} // namespace

void common_outcome_guidance_projection::aggregate_excluding(const std::string & excluded,
        std::map<std::string,std::vector<double>> & category_weights,
        std::map<std::string,size_t> & fixture_counts) const {
    category_weights.clear(); fixture_counts.clear();
    for(const auto & fixture:fixture_categories) {
        if(!excluded.empty()&&fixture.first==excluded)continue;
        for(const auto & item:fixture.second) {
            auto & target=category_weights[item.first];
            if(target.empty())target.assign(item.second.size(),0.0);
            double total=0.0;for(double value:item.second)total+=value;
            if(total<=0.0)continue;
            ++fixture_counts[item.first];
            for(size_t i=0;i<target.size();++i)target[i]+=item.second[i]/total;
        }
    }
}

bool common_outcome_guidance_build_projection(
        const std::vector<common_outcome_guidance_projection_observation> & observations,
        const std::string & builder_revision,const std::string & source_fingerprint,
        const uint64_t source_watermark,common_outcome_guidance_projection & projection,
        std::string & error) {
    using json=nlohmann::ordered_json;
    projection={};error.clear();
    if(builder_revision.empty()||source_fingerprint.empty()){
        error="outcome projection requires builder and source fingerprints";return false;
    }
    std::map<std::string,std::map<std::string,std::vector<double>>> grouped;
    std::map<std::string,size_t> category_dimensions;
    json canonical=json::array();
    // Canonicalize accumulation as well as serialization: floating-point sums
    // must not differ for two inputs carrying the same projection fingerprint.
    std::vector<const common_outcome_guidance_projection_observation *> ordered;
    ordered.reserve(observations.size());
    for(const auto & row:observations){
        for(double value:row.outcome_weights)if(!std::isfinite(value)||value<0.0){
            error="outcome projection weights must be finite and non-negative";return false;
        }
        ordered.push_back(&row);
    }
    std::sort(ordered.begin(),ordered.end(),[](const auto * lhs,const auto * rhs){
        if(lhs->fixture_group!=rhs->fixture_group)return lhs->fixture_group<rhs->fixture_group;
        if(lhs->category!=rhs->category)return lhs->category<rhs->category;
        return lhs->outcome_weights<rhs->outcome_weights;
    });
    for(const auto * entry:ordered){
        const auto & row=*entry;
        if(row.fixture_group.empty()||row.category.empty()||row.outcome_weights.empty()){
            error="outcome projection row has an empty identity or category vector";return false;
        }
        const auto dimension=category_dimensions.emplace(row.category,row.outcome_weights.size());
        if(!dimension.second&&dimension.first->second!=row.outcome_weights.size()){
            error="outcome projection category dimension changed across fixtures";return false;
        }
        auto & target=grouped[row.fixture_group][row.category];
        if(!target.empty()&&target.size()!=row.outcome_weights.size()){
            error="outcome projection category dimension changed";return false;
        }
        if(target.empty())target.assign(row.outcome_weights.size(),0.0);
        json weights=json::array();
        for(size_t i=0;i<row.outcome_weights.size();++i){
            const double value=row.outcome_weights[i];
            if(!std::isfinite(value)||value<0.0){error="outcome projection weights must be finite and non-negative";return false;}
            target[i]+=value;
            if(!std::isfinite(target[i])){error="outcome projection weight sum overflowed";return false;}
            weights.push_back(value);
        }
        canonical.push_back({{"fixture_group",row.fixture_group},{"category",row.category},{"weights",weights}});
    }
    for(auto & fixture:grouped)for(auto & category:fixture.second){
        // Scale first so even a finite vector whose total exceeds DBL_MAX
        // has a finite, normalized group contribution.
        const double scale=*std::max_element(category.second.begin(),category.second.end());
        if(scale>0.0){
            double total=0.0;for(double value:category.second)total+=value/scale;
            for(auto & value:category.second)value=(value/scale)/total;
        }
    }
    projection.builder_revision=builder_revision;
    projection.source_fingerprint=source_fingerprint;
    projection.source_watermark=source_watermark;
    projection.accepted_rows=observations.size();
    projection.fixture_categories=std::move(grouped);
    std::sort(canonical.begin(),canonical.end(),[](const auto & lhs,const auto & rhs){
        return lhs.dump()<rhs.dump();
    });
    const auto fingerprint_input=json{{"projection_format_version",2},
        {"builder_revision",builder_revision},{"source_fingerprint",source_fingerprint},
        {"source_watermark",source_watermark},{"observations",canonical}}.dump();
    projection.fingerprint="sha256:"+hash_sha256_hex(fingerprint_input.data(),fingerprint_input.size());
    return true;
}

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
    if(!std::isfinite(known_total)){posterior={};return false;}
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
    if(!std::isfinite(alpha_sum)){posterior={};return false;}
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
