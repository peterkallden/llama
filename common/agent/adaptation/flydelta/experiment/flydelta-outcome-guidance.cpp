#include "agent/adaptation/flydelta/experiment/flydelta-outcome-guidance.h"
#include "agent/prediction/outcome-guidance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <list>
#include <mutex>
#include <numeric>
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
        {"direction_identity", candidate.direction_identity}, {"dose_identity", candidate.dose_identity},
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

struct cached_bucket {
    counts leaf;
    std::set<std::string> leaf_groups;
    counts strategy_backoff;
    counts domain_backoff;
    size_t raw_observations = 0;
};
using strategy_key = std::tuple<std::string, std::string, std::string, int>;

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

// Each independent fixture contributes at most one unit. Local repetitions
// remain a single fixture, never independent generalization evidence.
struct progress_counts {
    double known = 0.0, unknown = 0.0;
    std::array<double, 4> categories{}; // improved, unchanged, regressed, solved
    size_t groups = 0;
    size_t known_groups = 0;
    size_t observations = 0;
};
struct progress_bucket { progress_counts leaf, backoff; };
using progress_key = std::tuple<bucket_key, std::string, std::string, std::string, std::string>;
using progress_backoff_key = std::tuple<std::string, std::string, std::string, std::string, std::string, std::string>;

using progress_label_cache = std::map<const common_flydelta_outcome_guidance_observation *, common_flydelta_progress_transition>;

template<class Predicate>
progress_counts aggregate_progress(
        const std::vector<const common_flydelta_outcome_guidance_observation *> & rows,
        Predicate predicate, const common_flydelta_outcome_guidance_context * local = nullptr,
        progress_label_cache * labels = nullptr) {
    using transition = common_flydelta_progress_transition;
    std::map<std::string, std::vector<transition>> groups;
    progress_counts counts;
    for (const auto * row : rows) {
        if (!row->progress || !predicate(*row)) continue;
        auto value = transition::unknown;
        const auto cached = labels ? labels->find(row) : progress_label_cache::iterator{};
        if (labels && cached != labels->end()) value = cached->second;
        else {
            value = common_flydelta_progress_evidence_classify(*row->progress);
            if (local && local->local_parent) {
                auto evidence = *row->progress;
                evidence.parent = local->local_parent;
                evidence.parent_ref = local->local_parent_ref;
                value = common_flydelta_progress_evidence_classify(evidence);
            }
            if (row->outcome == common_flydelta_counterfactual_outcome::harmed) value = transition::regressed;
            // Never give an already HELPED arm a second partial-success reward.
            if (row->outcome == common_flydelta_counterfactual_outcome::helped && value == transition::improved)
                value = transition::unknown;
            if (labels) labels->emplace(row, value);
        }
        groups[row->fixture_group_id].push_back(value);
        ++counts.observations;
    }
    counts.groups = groups.size();
    for (const auto & group : groups) {
        const auto & values = group.second;
        const auto known = std::count_if(values.begin(), values.end(), [](auto value) { return value != transition::unknown; });
        counts.known_groups += known > 0;
        for (const auto value : values) {
            if (value == transition::unknown) { counts.unknown += 1.0 / values.size(); continue; }
            counts.known += 1.0 / values.size();
            const size_t category = value == transition::improved ? 0 : value == transition::unchanged ? 1 :
                value == transition::regressed ? 2 : 3;
            counts.categories[category] += 1.0 / known;
        }
    }
    return counts;
}

std::string progress_signature(const common_flydelta_outcome_guidance_observation & row) {
    return nlohmann::ordered_json{{"task", row.task_fingerprint}, {"lineage", row.lineage_ref},
        {"direction", row.direction_identity}, {"dose", row.dose_identity},
        {"progress", row.progress ? nlohmann::ordered_json::parse(common_flydelta_progress_evidence_to_json(*row.progress))
            : nlohmann::ordered_json(nullptr)}}.dump();
}

std::string history_hash(const std::vector<const common_flydelta_outcome_guidance_observation*> & rows) {
    std::vector<const common_flydelta_outcome_guidance_observation*> sorted = rows;
    std::sort(sorted.begin(), sorted.end(), [](const auto * a, const auto * b) {
        return a->stable_observation_id < b->stable_observation_id;
    });
    nlohmann::ordered_json canonical = {
        {"builder_revision", "flydelta-outcome-projection-v5"},
        {"observations", nlohmann::ordered_json::array()},
    };
    for (const auto * o : sorted) {
        canonical["observations"].push_back({
            {"id", o->stable_observation_id}, {"fixture_group", o->fixture_group_id},
            {"progress_signature", progress_signature(*o)},
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

struct common_flydelta_outcome_guidance_projection {
    context_key compatibility;
    common_flydelta_outcome_guidance_decision metadata;
    std::map<bucket_key, cached_bucket> buckets;
    std::map<progress_key, progress_bucket> progress_buckets;
    std::map<progress_backoff_key, progress_counts> progress_backoffs;
    std::vector<common_flydelta_outcome_guidance_observation> local_rows;
    std::map<strategy_key, counts> strategies;
    std::map<std::pair<std::string, int>, counts> domains;
    size_t estimated_bytes = 0;
};

struct common_flydelta_outcome_guidance_projection_cache::impl {
    std::mutex mutex;
    std::list<std::pair<std::string, common_flydelta_outcome_guidance_projection_ptr>> entries;
    size_t bytes = 0;
    size_t budget = 16 * 1024 * 1024;
};

common_flydelta_outcome_guidance_projection_cache::common_flydelta_outcome_guidance_projection_cache()
    : impl_(new impl) {}
common_flydelta_outcome_guidance_projection_cache::~common_flydelta_outcome_guidance_projection_cache() = default;
void common_flydelta_outcome_guidance_projection_cache::set_budget(size_t bytes) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->budget = bytes;
    while (!impl_->entries.empty() && impl_->bytes > bytes) {
        const auto & last = impl_->entries.back();
        impl_->bytes -= last.second->estimated_bytes + last.first.capacity() + 4 * sizeof(void *);
        impl_->entries.pop_back();
    }
}

bool common_flydelta_outcome_guidance_projection_cache::get_or_build(
        const std::string & frozen_snapshot_identity,
        const common_flydelta_outcome_guidance_context & context,
        const std::vector<common_flydelta_outcome_guidance_observation> & history,
        const std::string & excluded_fixture_group,
        common_flydelta_outcome_guidance_projection_ptr & projection, std::string & error,
        bool * cache_hit) {
    error.clear(); projection.reset();
    if (cache_hit) *cache_hit = false;
    if (frozen_snapshot_identity.empty()) { error = "missing frozen history identity"; return false; }
    const auto key = nlohmann::ordered_json::array({frozen_snapshot_identity,
        hard_key(context), excluded_fixture_group}).dump();
    std::unique_lock<std::mutex> lock(impl_->mutex);
    for (auto it = impl_->entries.begin(); it != impl_->entries.end(); ++it)
        if (it->first == key) {
            if (cache_hit) *cache_hit = true;
            projection = it->second;
            impl_->entries.splice(impl_->entries.begin(), impl_->entries, it);
            return true;
        }
    lock.unlock();
    if (!common_flydelta_outcome_guidance_build_projection(context, history, projection,
            error, excluded_fixture_group)) return false;
    lock.lock();
    for (auto it = impl_->entries.begin(); it != impl_->entries.end(); ++it)
        if (it->first == key) {
            projection = it->second;
            // This call already paid for a build; do not report it as a warm hit.
            impl_->entries.splice(impl_->entries.begin(), impl_->entries, it);
            return true;
        }
    const size_t bytes = projection->estimated_bytes + key.capacity() + 4 * sizeof(void *);
    if (impl_->budget == 0 || bytes > impl_->budget) return true;
    try {
        impl_->entries.emplace_front(key, projection);
        impl_->bytes += projection->estimated_bytes + impl_->entries.front().first.capacity() + 4 * sizeof(void *);
        while (impl_->entries.size() > 16 || impl_->bytes > impl_->budget) {
            const auto & last = impl_->entries.back();
            impl_->bytes -= last.second->estimated_bytes + last.first.capacity() + 4 * sizeof(void *);
            impl_->entries.pop_back();
        }
    } catch (...) { /* Retaining a valid projection is optional. */ }
    return true;
}

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
    if (c.schema_version != 1 || !std::isfinite(c.progress_reward) || c.progress_reward < 0 || c.progress_reward > 1 ||
            !std::isfinite(c.harm_penalty) || c.harm_penalty < 0 ||
            c.min_backoff_groups == 0 || !std::isfinite(c.max_prior_strength) ||
            c.max_prior_strength < 0 || c.max_prior_strength > 4 ||
            c.projection_cache_bytes > 256 * 1024 * 1024 ||
            c.matched_max_arms == 0 || c.matched_max_arms > 48) {
        error = "FlyDelta outcome guidance configuration is invalid"; return false;
    }
    return true;
}


bool common_flydelta_outcome_guidance_build_projection(
        const common_flydelta_outcome_guidance_context & context,
        const std::vector<common_flydelta_outcome_guidance_observation> & history,
        common_flydelta_outcome_guidance_projection_ptr & projection,
        std::string & error, const std::string & excluded_fixture_group) {
    projection.reset(); error.clear();
    if (context.scope_fingerprint.empty() || context.model_profile_fingerprint.empty() ||
            context.tokenizer_fingerprint.empty() || context.template_fingerprint.empty() ||
            context.baseline_deployment_fingerprint.empty() || context.verifier_revision.empty()) {
        error = "outcome projection requires authoritative compatibility identities";
        return false;
    }
    common_flydelta_outcome_guidance_decision d;
    std::vector<const common_flydelta_outcome_guidance_observation*> rows, all_rows;
    std::vector<common_flydelta_outcome_guidance_observation> local_rows;
    d.read_history_rows = history.size();
    std::map<std::string, std::string> signatures;
    std::set<std::string> conflicted;
    auto observation_signature = [](const auto & o) {
        return nlohmann::ordered_json{
            {"fixture_group", o.fixture_group_id}, {"basis", o.basis_identity},
            {"progress_signature", progress_signature(o)},
            {"scope", o.context.scope_fingerprint}, {"model", o.context.model_profile_fingerprint},
            {"tokenizer", o.context.tokenizer_fingerprint}, {"template", o.context.template_fingerprint},
            {"deployment", o.context.baseline_deployment_fingerprint}, {"verifier", o.context.verifier_revision},
            {"ruleset", o.context.ruleset_revision}, {"action", o.context.action_contract_fingerprint},
            {"domain", o.context.semantic_domain}, {"strategy", o.context.strategy},
            {"estimator", o.context.estimator}, {"rank", o.rank},
            {"layer", o.relative_layer}, {"dose", o.relative_dose},
            {"mixed", o.mixed_coefficients}, {"margin", o.diagnostic_margin_sign},
            {"completed", o.comparison_completed}, {"overlay", o.overlay_applied},
            {"outcome_known", o.outcome_known},
            {"verifier_known", o.verifier_known},
            {"outcome", static_cast<int>(o.outcome)},
        }.dump();
    };
    for (const auto & o : history) {
        std::string progress_error;
        if (o.task_fingerprint.size() > 512 || o.lineage_ref.size() > 2048 ||
                o.direction_identity.size() > 512 || o.dose_identity.size() > 4096 ||
                (o.progress && !common_flydelta_progress_evidence_validate(*o.progress, progress_error))) {
            ++d.rejected_history_rows; continue;
        }
        if (o.stable_observation_id.empty() || o.fixture_group_id.empty() ||
                o.context.semantic_domain.empty() || o.context.strategy.empty() || o.context.estimator.empty() ||
                (o.outcome != common_flydelta_counterfactual_outcome::helped &&
                 o.outcome != common_flydelta_counterfactual_outcome::neutral &&
                 o.outcome != common_flydelta_counterfactual_outcome::harmed &&
                 o.outcome != common_flydelta_counterfactual_outcome::unknown) ||
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
    std::set<std::string> accepted_ids;
    for (const auto & o : history) {
        const auto found = signatures.find(o.stable_observation_id);
        if (found == signatures.end()) continue;
        if (conflicted.count(o.stable_observation_id)) { ++d.rejected_history_rows; continue; }
        if (observation_signature(o) != found->second) continue;
        // Identical retries are accepted once; a conflicting id is excluded
        // entirely so append order cannot choose the winning observation.
        if (!o.comparison_completed || !o.overlay_applied ||
                hard_key(o.context) != hard_key(context) || o.fixture_group_id.empty()) continue;
        if (!accepted_ids.insert(o.stable_observation_id).second) continue;
        all_rows.push_back(&o);
        if (o.fixture_group_id == excluded_fixture_group) {
            if (o.progress) local_rows.push_back(o);
            continue;
        }
        rows.push_back(&o);
    }
    d.compatible_history_rows = rows.size();
    d.history_fingerprint = history_hash(all_rows);
    std::map<bucket_key, cached_bucket> bucket_cache;
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

    auto built = std::make_shared<common_flydelta_outcome_guidance_projection>();
    built->compatibility = hard_key(context);
    built->metadata = std::move(d);
    built->buckets = std::move(bucket_cache);
    built->local_rows = std::move(local_rows);
    progress_label_cache progress_labels;
    const auto project_progress = [&](const auto & predicate) {
        return aggregate_progress(rows, predicate, nullptr, &progress_labels);
    };
    // Progress schemas are separate partitions. Goal identifiers from different
    // objectives are never directly compared or counted as independent trials.
    for (const auto * row : rows) {
        if (!row->progress) continue;
        const auto & objective = row->progress->baseline;
        const progress_key key{key_for(*row), objective.objective_revision, objective.verifier_ref, row->dose_identity, row->direction_identity};
        if (built->progress_buckets.count(key)) continue;
        auto compatible = [&](const auto & other) {
            return other.progress && other.progress->baseline.objective_revision == objective.objective_revision &&
                other.progress->baseline.verifier_ref == objective.verifier_ref && other.dose_identity == row->dose_identity;
        };
        const progress_backoff_key backoff_key{row->context.semantic_domain, row->context.strategy,
            row->context.estimator, objective.objective_revision, objective.verifier_ref, row->dose_identity};
        if (!built->progress_backoffs.count(backoff_key))
            built->progress_backoffs[backoff_key] = project_progress([&](const auto & other) {
                return compatible(other) && other.context.semantic_domain == row->context.semantic_domain &&
                    other.context.strategy == row->context.strategy && other.context.estimator == row->context.estimator;
            });
        auto & bucket = built->progress_buckets[key];
        bucket.leaf = project_progress([&](const auto & other) { return compatible(other) && key_for(other) == key_for(*row) && other.direction_identity == row->direction_identity; });
        std::set<std::string> leaf_groups;
        for (const auto * other : rows) if (compatible(*other) && key_for(*other) == key_for(*row) && other->direction_identity == row->direction_identity) leaf_groups.insert(other->fixture_group_id);
        bucket.backoff = project_progress([&](const auto & other) {
            return compatible(other) && !leaf_groups.count(other.fixture_group_id) &&
                other.context.semantic_domain == row->context.semantic_domain && other.context.strategy == row->context.strategy &&
                other.context.estimator == row->context.estimator;
        });
    }
    std::set<std::string> domains;
    std::set<strategy_key> strategies;
    for (const auto * row : rows) {
        domains.insert(row->context.semantic_domain);
        strategies.emplace(row->context.semantic_domain, row->context.strategy,
            row->context.estimator, row->context.strategy == "whirlpool" ? layer_bucket(row->relative_layer) : -1);
    }
    for (const auto & domain : domains) for (int region = -1; region < 4; ++region) {
        built->domains[{domain, region}] = aggregate(rows, [&](const auto & o) {
            return o.context.semantic_domain == domain && (region < 0 || layer_bucket(o.relative_layer) == region);
        });
    }
    for (const auto & strategy : strategies) {
        const auto & [domain, name, estimator, region] = strategy;
        built->strategies[strategy] = aggregate(rows, [&](const auto & o) {
            return o.context.semantic_domain == domain && o.context.strategy == name &&
                o.context.estimator == estimator && (region < 0 || layer_bucket(o.relative_layer) == region);
        });
    }
    built->estimated_bytes = sizeof(*built) + built->metadata.history_fingerprint.capacity();
    std::apply([&](const auto &... field) { ((built->estimated_bytes += field.capacity()), ...); }, built->compatibility);
    for (const auto & entry : built->buckets) {
        built->estimated_bytes += sizeof(entry) + 4 * sizeof(void *);
        built->estimated_bytes += std::get<0>(entry.first).capacity() + std::get<1>(entry.first).capacity() +
            std::get<2>(entry.first).capacity() + std::get<8>(entry.first).capacity();
        for (const auto & group : entry.second.leaf_groups)
            built->estimated_bytes += sizeof(group) + group.capacity() + 4 * sizeof(void *);
    }
    for (const auto & entry : built->strategies)
        built->estimated_bytes += sizeof(entry) + 4 * sizeof(void *) +
            std::get<0>(entry.first).capacity() + std::get<1>(entry.first).capacity() + std::get<2>(entry.first).capacity();
    for (const auto & entry : built->domains)
        built->estimated_bytes += sizeof(entry) + 4 * sizeof(void *) + entry.first.first.capacity();
    for (const auto & entry : built->progress_buckets) {
        built->estimated_bytes += sizeof(entry) + 4 * sizeof(void *) +
            std::get<1>(entry.first).capacity() + std::get<2>(entry.first).capacity() + std::get<3>(entry.first).capacity() +
            std::get<4>(entry.first).capacity();
        const auto & key = std::get<0>(entry.first);
        built->estimated_bytes += std::get<0>(key).capacity() + std::get<1>(key).capacity() +
            std::get<2>(key).capacity() + std::get<8>(key).capacity();
    }
    for (const auto & entry : built->progress_backoffs) {
        built->estimated_bytes += sizeof(entry) + 4 * sizeof(void *);
        std::apply([&](const auto &... field) { ((built->estimated_bytes += field.capacity()), ...); }, entry.first);
    }
    for (const auto & row : built->local_rows)
        built->estimated_bytes += sizeof(row) + progress_signature(row).size() + row.stable_observation_id.capacity() + row.basis_identity.capacity();
    projection = std::move(built);
    return true;
}

bool common_flydelta_outcome_guidance_build_decision(
        const common_flydelta_outcome_guidance_config & config,
        const common_flydelta_outcome_guidance_context & context,
        const std::vector<common_flydelta_outcome_guidance_observation> & history,
        const std::vector<common_flydelta_outcome_guidance_candidate> & candidates,
        size_t selection_limit, uint64_t job_seed,
        common_flydelta_outcome_guidance_decision & decision, std::string & error,
        common_flydelta_outcome_guidance_policy policy, size_t policy_sequence_start) {
    common_flydelta_outcome_guidance_projection_ptr projection;
    // off does no history validation, hashing or aggregation.
    const std::vector<common_flydelta_outcome_guidance_observation> empty_history;
    if (!common_flydelta_outcome_guidance_build_projection(context,
            config.mode == common_flydelta_outcome_guidance_mode::off ? empty_history : history,
            projection, error)) return false;
    return common_flydelta_outcome_guidance_decide_from_projection(config, context, projection,
        candidates, selection_limit, job_seed, decision, error, policy, policy_sequence_start);
}

bool common_flydelta_outcome_guidance_decide_from_projection(
        const common_flydelta_outcome_guidance_config & config,
        const common_flydelta_outcome_guidance_context & context,
        const common_flydelta_outcome_guidance_projection_ptr & projection,
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
    if (!projection || projection->compatibility != hard_key(context)) {
        error = "outcome projection compatibility mismatch"; return false;
    }
    d.read_history_rows = projection->metadata.read_history_rows;
    d.rejected_history_rows = projection->metadata.rejected_history_rows;
    d.deduplicated_history_rows = projection->metadata.deduplicated_history_rows;
    d.compatible_history_rows = projection->metadata.compatible_history_rows;
    d.history_fingerprint = projection->metadata.history_fingerprint;
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

    if ((context.progress_baseline && !common_flydelta_progress_assessment_validate(*context.progress_baseline, error)) ||
            (context.local_parent && (context.local_parent_ref.empty() || !context.progress_baseline ||
                !common_flydelta_progress_comparable(*context.progress_baseline, *context.local_parent)))) {
        if (error.empty()) error = "FlyDelta progress query has an incompatible baseline or parent";
        return false;
    }
    std::mt19937_64 rng(config.seed ^ job_seed);
    struct scored { size_t index; double utility; };
    std::vector<scored> sampled;
    std::vector<scored> by_mean;
    const auto & bucket_cache = projection->buckets;
    std::map<bucket_key, double> sampled_by_posterior;
    std::map<progress_key, progress_counts> local_counts;
    progress_label_cache local_labels;
    std::vector<const common_flydelta_outcome_guidance_observation *> local_rows;
    for (const auto & row : projection->local_rows) local_rows.push_back(&row);
    for (size_t i=0; i<candidates.size(); ++i) {
        const auto exact = key_for(context, candidates[i]);
        const auto cached_it = bucket_cache.find(exact);
        cached_bucket unseen;
        if (cached_it == bucket_cache.end()) {
            const int region = context.strategy == "whirlpool" ? layer_bucket(candidates[i].relative_layer) : -1;
            const auto strategy = projection->strategies.find(
                {context.semantic_domain, context.strategy, context.estimator, region});
            if (strategy != projection->strategies.end()) unseen.strategy_backoff = strategy->second;
            const auto domain = projection->domains.find({context.semantic_domain, region});
            if (domain != projection->domains.end()) unseen.domain_backoff = domain->second;
        }
        const auto & cached = cached_it == bucket_cache.end() ? unseen : cached_it->second;
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
        if (context.progress_baseline && context.progress_baseline->assessment_known && !candidates[i].dose_identity.empty()) {
            const auto & objective = *context.progress_baseline;
            const progress_key progress_partition{exact, objective.objective_revision, objective.verifier_ref,
                candidates[i].dose_identity, candidates[i].direction_identity};
            progress_bucket statistics;
            const auto found = projection->progress_buckets.find(progress_partition);
            if (found != projection->progress_buckets.end()) statistics = found->second;
            else {
                const auto backoff = projection->progress_backoffs.find({context.semantic_domain, context.strategy,
                    context.estimator, objective.objective_revision, objective.verifier_ref, candidates[i].dose_identity});
                if (backoff != projection->progress_backoffs.end()) statistics.backoff = backoff->second;
            }
            auto local = local_counts.find(progress_partition);
            if (local == local_counts.end()) {
                const auto counts = aggregate_progress(local_rows, [&](const auto & row) {
                    return !context.local_task_fingerprint.empty() && !context.local_lineage_ref.empty() &&
                        row.task_fingerprint == context.local_task_fingerprint && row.lineage_ref == context.local_lineage_ref &&
                        row.direction_identity == candidates[i].direction_identity && row.dose_identity == candidates[i].dose_identity &&
                        key_for(row) == exact && common_flydelta_progress_comparable(row.progress->baseline, objective);
                }, &context, &local_labels);
                local = local_counts.emplace(progress_partition, counts).first;
            }
            const auto & nearby = local->second;
            auto progress_leaf = statistics.leaf;
            progress_leaf.known += nearby.known; progress_leaf.unknown += nearby.unknown;
            progress_leaf.observations += nearby.observations;
            for (size_t category = 0; category < 4; ++category) progress_leaf.categories[category] += nearby.categories[category];
            const auto & backoff = statistics.backoff;
            const double progress_prior = backoff.groups >= config.min_backoff_groups
                ? config.max_prior_strength : 0.0;
            p.progress_evidence_available = progress_leaf.observations > 0 || progress_prior > 0;
            if (p.progress_evidence_available) {
                const double total = backoff.known + backoff.unknown;
                const double category_total = std::accumulate(backoff.categories.begin(), backoff.categories.end(), 0.0);
                std::vector<double> means(4, 0.25);
                if (category_total > 0) for (size_t category = 0; category < 4; ++category)
                    means[category] = backoff.categories[category] / category_total;
                common_outcome_guidance_posterior progress_posterior;
                if (!common_outcome_guidance_make_posterior(progress_leaf.known, progress_leaf.unknown,
                        total > 0 ? backoff.known / total : 0.5,
                        std::vector<double>(progress_leaf.categories.begin(), progress_leaf.categories.end()), means,
                        progress_prior, backoff.known_groups >= config.min_backoff_groups ? progress_prior : 0.0, progress_posterior)) {
                    error = "FlyDelta progress posterior inputs are invalid"; return false;
                }
                p.p_progress_known = progress_posterior.p_known;
                p.p_improved_given_progress_known = progress_posterior.category_probabilities[0];
                p.p_regressed_given_progress_known = progress_posterior.category_probabilities[2];
                p.p_solved_given_progress_known = progress_posterior.category_probabilities[3];
                p.p_safe_partial_improved = p.p_progress_known * p.p_improved_given_progress_known;
                p.p_safe_partial_low = progress_posterior.p_known_low * progress_posterior.category_lows[0];
                p.p_safe_partial_high = progress_posterior.p_known_high * progress_posterior.category_highs[0];
                // Full outcomes retain their existing harm cost. Partial
                // regressions receive no reward and remain explicit diagnostics.
                p.progress_utility_available = progress_leaf.known > 0 ||
                    (progress_prior > 0 && backoff.known_groups >= config.min_backoff_groups);
                p.progress_utility = p.progress_utility_available ? config.progress_reward * p.p_safe_partial_improved : 0.0;
                p.expected_utility += p.progress_utility;
                p.effective_progress_known = progress_leaf.known;
                p.effective_progress_unknown = progress_leaf.unknown;
                p.local_observations = nearby.observations;
                const auto progress_identity = nlohmann::ordered_json{{"outcome_posterior", p.posterior_identity},
                    {"partition", progress_partition}, {"task", context.local_task_fingerprint},
                    {"lineage", context.local_lineage_ref}, {"history", d.history_fingerprint},
                    {"parent", context.local_parent ? nlohmann::ordered_json::parse(
                        common_flydelta_progress_assessment_to_json(*context.local_parent)) : nlohmann::ordered_json(nullptr)}}.dump();
                const auto fingerprint = hash_sha256_hex(progress_identity.data(), progress_identity.size());
                p.posterior_identity = "sha256:" + fingerprint;
                std::mt19937_64 progress_rng(config.seed ^ job_seed ^ std::stoull(fingerprint.substr(0, 16), nullptr, 16));
                if (p.progress_utility_available)
                    p.sampled_utility += common_outcome_guidance_sample_expected_utility(progress_posterior,
                        {config.progress_reward, 0.0, 0.0, 0.0}, progress_rng);
            }
        }
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
            {"progress_signature", progress_signature(row)},
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
            candidate.direction_identity = observed.direction_identity;
            candidate.dose_identity = observed.dose_identity;
            auto scoring_context = observed.context;
            if (observed.progress) {
                scoring_context.progress_baseline = observed.progress->baseline;
                scoring_context.local_parent = observed.progress->parent;
                scoring_context.local_parent_ref = observed.progress->parent_ref;
            }
            common_flydelta_outcome_guidance_decision decision;
            if (!common_flydelta_outcome_guidance_build_decision(
                    scoring_config, scoring_context, available, {candidate}, 1,
                    replay_seed++, decision, error)) return false;
            if (decision.predictions.size() != 1) {
                error = "FlyDelta outcome replay did not produce a prediction";
                return false;
            }
            const auto & prediction = decision.predictions.front();
            if (observed.progress) {
                const auto progress = common_flydelta_progress_evidence_classify(*observed.progress);
                const bool known = progress != common_flydelta_progress_transition::unknown;
                metrics.model_progress_coverage_brier += std::pow(prediction.p_progress_known - (known ? 1.0 : 0.0), 2);
                ++metrics.progress_predictions;
                if (known) {
                    const bool improved = progress == common_flydelta_progress_transition::improved &&
                        observed.outcome != common_flydelta_counterfactual_outcome::helped &&
                        observed.outcome != common_flydelta_counterfactual_outcome::harmed;
                    metrics.model_progress_brier += std::pow(prediction.p_improved_given_progress_known - (improved ? 1.0 : 0.0), 2);
                    ++metrics.known_progress_predictions;
                }
            }
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
    if (metrics.progress_predictions) metrics.model_progress_coverage_brier /= metrics.progress_predictions;
    if (metrics.known_progress_predictions) metrics.model_progress_brier /= metrics.known_progress_predictions;
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
        {"sampling_revision", "outcome-and-progress-sampling-v3"},
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
    payload["progress_query"] = {{"task_fingerprint", context.local_task_fingerprint},
        {"lineage_ref", context.local_lineage_ref}, {"parent_ref", context.local_parent_ref},
        {"baseline", context.progress_baseline ? json::parse(common_flydelta_progress_assessment_to_json(*context.progress_baseline)) : json(nullptr)},
        {"parent", context.local_parent ? json::parse(common_flydelta_progress_assessment_to_json(*context.local_parent)) : json(nullptr)}};
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
    for (const auto & p : decision.predictions) {
        json prediction = {
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
        };
        prediction["progress"] = json::parse(common_flydelta_outcome_guidance_progress_prediction_to_json(p));
        payload["predictions"].push_back(std::move(prediction));
    }

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

bool common_flydelta_outcome_guidance_observation_from_json(const std::string & text,
        common_flydelta_outcome_guidance_observation & o, std::string & error) {
    o = {}; error.clear();
    if (text.size() > 5U * 1024U * 1024U) { error = "FlyDelta guidance observation exceeds its byte bound"; return false; }
    try {
        const auto row = nlohmann::ordered_json::parse(text);
        const auto & g = row.at("outcome_guidance");
        const int schema = g.value("schema_version", 0);
        if (!g.is_object() || (schema != 1 && schema != 2)) { error = "Unsupported FlyDelta guidance observation"; return false; }
        o.stable_observation_id = g.value("observation_id", "");
        o.fixture_group_id = g.value("fixture_group_id", "");
        o.context.scope_fingerprint = g.value("scope_fingerprint", "");
        o.context.model_profile_fingerprint = g.value("model_profile_fingerprint", "");
        o.context.tokenizer_fingerprint = g.value("tokenizer_fingerprint", "");
        o.context.template_fingerprint = g.value("template_fingerprint", "");
        o.context.baseline_deployment_fingerprint = g.value("baseline_deployment_fingerprint", "");
        o.context.verifier_revision = g.value("verifier_revision", "");
        o.context.ruleset_revision = g.value("ruleset_revision", "");
        o.context.action_contract_fingerprint = g.value("action_contract_fingerprint", "");
        o.context.semantic_domain = g.value("domain", "");
        o.context.strategy = g.value("strategy", "");
        o.context.estimator = g.value("estimator", "");
        o.basis_identity = g.value("basis_id", "");
        o.rank = g.value("rank", size_t{0});
        o.relative_layer = g.value("relative_layer", 0.0);
        o.relative_dose = g.value("relative_dose", 0.0);
        o.mixed_coefficients = g.value("mixed_coefficients", false);
        o.diagnostic_margin_sign = g.value("diagnostic_margin_sign", 0);
        o.comparison_completed = g.value("comparison_completed", false);
        o.overlay_applied = g.value("overlay_applied", false);
        o.verifier_known = row.value("verifier_known", false);
        const auto outcome = row.value("outcome", std::string("unknown"));
        if (outcome == "helped") o.outcome = common_flydelta_counterfactual_outcome::helped;
        else if (outcome == "neutral") o.outcome = common_flydelta_counterfactual_outcome::neutral;
        else if (outcome == "harmed") o.outcome = common_flydelta_counterfactual_outcome::harmed;
        else if (outcome != "unknown") { error = "Invalid FlyDelta guidance outcome"; return false; }
        o.outcome_known = g.value("outcome_known", o.outcome != common_flydelta_counterfactual_outcome::unknown);
        if (schema == 2) {
            o.task_fingerprint = g.value("task_fingerprint", "");
            o.lineage_ref = g.value("lineage_ref", "");
            o.direction_identity = g.value("direction_identity", "");
            o.dose_identity = g.value("dose_identity", "");
            if (g.contains("progress")) {
                o.progress.emplace();
                if (!common_flydelta_progress_evidence_from_json(g.at("progress").dump(), *o.progress, error)) return false;
            }
        }
        if (o.stable_observation_id.empty() || o.outcome_known !=
                (o.outcome != common_flydelta_counterfactual_outcome::unknown)) {
            error = "Inconsistent FlyDelta guidance observation identity or outcome"; return false;
        }
    } catch (const std::exception & exception) {
        error = std::string("Malformed FlyDelta guidance observation: ") + exception.what();
        o = {}; return false;
    }
    return true;
}

std::string common_flydelta_outcome_guidance_progress_prediction_to_json(
        const common_flydelta_outcome_guidance_prediction & p) {
    return nlohmann::ordered_json{{"evidence_available", p.progress_evidence_available},
        {"utility_available", p.progress_utility_available},
        {"p_known", p.p_progress_known}, {"p_improved_given_known", p.p_improved_given_progress_known},
        {"p_regressed_given_known", p.p_regressed_given_progress_known}, {"p_solved_given_known", p.p_solved_given_progress_known},
        {"p_safe_partial_improved", p.p_safe_partial_improved},
        {"safe_partial_interval_product_bounds", {p.p_safe_partial_low, p.p_safe_partial_high}},
        {"utility", p.progress_utility}, {"effective_known", p.effective_progress_known},
        {"effective_unknown", p.effective_progress_unknown}, {"local_observations", p.local_observations},
        {"authority", "search_only"}}.dump();
}
