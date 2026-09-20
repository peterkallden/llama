#include "agent/adaptation/adaptation-evidence-routing.h"
#include "agent/adaptation/learning-transaction.h"
#include "agent/adaptation/training-candidate.h"
#include "agent/adaptation/flydelta/flydelta-aggregation.h"
#include "agent/adaptation/flydelta/flydelta-deep-search.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "check failed at line " << __LINE__ \
                  << ": " << #condition << " error=" << error << '\n'; \
        return 1; \
    } \
} while (false)

namespace {

constexpr size_t kDimension = 8;
constexpr int32_t kLayer = 25;

bool read_jsonl(const std::filesystem::path & path, std::vector<json> & cases, std::string & error) {
    std::ifstream input(path);
    if (!input) { error = "could not open replay fixture"; return false; }
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto value = json::parse(line, nullptr, false);
        if (value.is_discarded() || !value.is_object()) {
            error = "invalid replay fixture JSON at line " + std::to_string(line_number);
            return false;
        }
        cases.push_back(value);
    }
    if (cases.empty()) { error = "replay fixture is empty"; return false; }
    return true;
}

common_learning_transaction echo_transaction(const json & entry, std::string & error) {
    const auto id = entry.value("id", "");
    const auto expected_tool = entry.value("expected_tool", "");
    common_learning_transaction transaction;
    transaction.id = "learning://observation/repair-echo-replay/" + id;
    transaction.created_at = "2026-09-17T00:00:00Z";
    transaction.observation.id = transaction.id;
    transaction.observation.scope.namespace_id = "local";
    transaction.observation.scope.project_id = "flydelta-repair-echo-replay";
    transaction.observation.scope.session_id = "replay-smoke";
    transaction.observation.scope.turn_id = "turn:" + id;
    transaction.observation.source_turn_id = transaction.observation.scope.turn_id;
    transaction.observation.source_plan_id = "plan:" + id;
    const json context = {
        {"kind", "repair_echo_failure"},
        {"expected_tool", expected_tool},
        {"canonical_repair", entry.value("canonical_repair", json::object())},
        {"failed_output", entry.value("failed_output", json::object())},
        {"repair_output", entry.value("repair_output", json::object())},
        {"diagnostic", entry.value("diagnostic", "model failed to replay the host-constructed canonical repair")},
        {"host_evaluated", true},
        {"verifier_known", true},
        {"host_outcome", "UNKNOWN"},
        {"learning_credit", false},
    };
    common_learning_signal signal(
        common_learning_signal_type::repair_echo_failure,
        transaction.observation.source_plan_id,
        "repair-echo:" + id,
        expected_tool,
        "evidence:dataset-repair:" + id + ":repair-echo-failure",
        "model failed to replay the host-constructed canonical repair",
        "data",
        "native");
    signal.repair_context_json = context.dump();
    transaction.observation.signals.push_back(std::move(signal));
    transaction.observation.evidence_ids = {transaction.observation.signals.front().evidence_id};
    transaction.observation.cause = common_learning_cause::model_behavior;
    transaction.observation.verification = common_learning_verification::unverified;
    transaction.observation.idempotency_key = "repair-echo-replay:" + id;
    transaction.observation.collection_allowed = true;
    transaction.observation.content_hash = common_learning_observation_hash(transaction.observation);
    if (!common_learning_transaction_validate(transaction, 16, error)) return {};
    return transaction;
}

std::vector<float> spread_vector(size_t group, size_t variant) {
    // The logged question/failed/canonical outputs are real model-smoke
    // material. Hidden states are controlled synthetic spread so this smoke
    // can exercise rank-2/deep search without claiming to be Qwen capture.
    const size_t mode = variant % 2;
    const float orbit = static_cast<float>(variant / 2);
    std::vector<float> values(kDimension, 0.0f);
    const size_t first = (group * 2) % 6;
    const size_t second = first + 1;
    values[first] = mode == 0 ? 1.00f + 0.04f * orbit : 0.30f + 0.03f * orbit;
    values[second] = mode == 0 ? 0.28f + 0.02f * orbit : 1.00f - 0.03f * orbit;
    values[6] = 0.04f * static_cast<float>(group + 1);
    values[7] = 0.015f * (orbit + 1.0f);
    return values;
}

common_flydelta_behavior_delta make_delta(const json & entry, size_t group, size_t variant) {
    const auto id = entry.value("id", "");
    common_flydelta_behavior_delta delta;
    delta.id = "delta://replay/" + id + "/spread-" + std::to_string(variant);
    delta.source = common_adaptation_evidence_source::tool_repair;
    delta.behavior_key = entry.value("behavior_key", "");
    delta.capture_manifest_id = "manifest://replay/" + id;
    delta.host_evidence_ref = "evidence://replay/" + id;
    delta.model_profile_fingerprint = "profile:synthetic-spread-qwen";
    delta.execution_context_fingerprint = "context:dataset-sales-replay";
    delta.capture_layout_revision = "layout:synthetic-spread-v1";
    delta.scope_fingerprint = "scope:local";
    delta.tokenizer_fingerprint = "tokenizer:qwen-replay";
    delta.template_fingerprint = "template:qwen-replay";
    delta.generation_semantics_fingerprint = "generation:replay-v1";
    delta.layer_index = kLayer;
    delta.values = spread_vector(group, variant);
    return delta;
}

common_flydelta_intervention_credit experimental_credit(
        const std::string & fixture_id, const std::string & id) {
    common_flydelta_intervention_credit credit;
    credit.experiment_id = "experiment://replay/" + id;
    credit.candidate_id = "candidate://replay/" + id;
    credit.fixture_id = fixture_id;
    credit.outcome = common_flydelta_counterfactual_outcome::unknown;
    credit.eligible_for_learning = false;
    return credit;
}

common_flydelta_experiment_fixture fixture(const std::string & behavior_key) {
    common_flydelta_experiment_fixture value;
    value.id = "fixture://replay/" + behavior_key;
    value.task_fingerprint = "task://dataset-repair-replay/" + behavior_key;
    value.model_profile_fingerprint = "profile:synthetic-spread-qwen";
    value.tokenizer_fingerprint = "tokenizer:qwen-replay";
    value.template_fingerprint = "template:qwen-replay";
    value.execution_context_fingerprint = "context:dataset-sales-replay";
    value.verifier_revision = "verifier:dataset-replay-v1";
    return value;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0] << " REPLAY_JSONL [LEDGER_JSONL]\n";
        return 2;
    }
    std::string error;
    std::vector<json> cases;
    if (!read_jsonl(argv[1], cases, error)) { std::cerr << error << '\n'; return 1; }

    const bool remove_ledger = argc == 2;
    const auto ledger_path = remove_ledger
        ? std::filesystem::temp_directory_path() / "llama-agent-flydelta-repair-echo-replay.jsonl"
        : std::filesystem::path(argv[2]);
    if (remove_ledger) {
        std::error_code ignored;
        std::filesystem::remove(ledger_path, ignored);
    }
    common_learning_jsonl_transaction_store ledger;
    CHECK(ledger.open(ledger_path, error));

    size_t echo_persisted = 0;
    for (const auto & entry : cases) {
        CHECK(entry.value("canonical_repair", json::object()).is_object());
        CHECK(entry.value("failed_output", json::object()).is_object());
        CHECK(entry.value("repair_output", json::object()).is_object());
        CHECK(entry.value("repair_output", json::object()) == entry.value("canonical_repair", json::object()));
        const auto id = entry.value("id", "");
        const auto observed_outcome = entry.value(
            "observed_outcome", id == "repair-echo-query-many-fields"
                ? "repair_echo_failure" : "host_certified_repair");
        if (observed_outcome != "repair_echo_failure") continue;
        auto transaction = echo_transaction(entry, error);
        CHECK(!transaction.id.empty());
        CHECK(common_learning_destination_for(transaction.observation) == common_learning_destination::retain);
        CHECK(ledger.append(transaction, error));
        CHECK(ledger.append(transaction, error));
        ++echo_persisted;
    }

    std::map<std::string, std::vector<json>> grouped;
    for (const auto & entry : cases) grouped[entry.value("behavior_key", "")].push_back(entry);
    CHECK(grouped.size() == 3);

    size_t natural_shallow_groups = 0;
    size_t natural_deep_groups = 0;
    size_t experimental_rank2_groups = 0;
    size_t forced_deep_groups = 0;
    size_t whirlpool_groups = 0;
    size_t tfo_groups = 0;
    size_t total_search_trials = 0;
    size_t group_index = 0;
    for (const auto & group_entry : grouped) {
        const auto & behavior_key = group_entry.first;
        const auto & group_cases = group_entry.second;
        // The replay corpus may grow within a behavior family. Keep at
        // least two independent source cases for the synthetic spread, but
        // let the replay use all available cases before cycling if fewer
        // than six are present.
        CHECK(group_cases.size() >= 2);
        const auto experiment_fixture = fixture(behavior_key);
        common_flydelta_direction_search_config identity;
        identity.dimension = kDimension;
        identity.layer_index = kLayer;
        identity.min_samples = 2;
        identity.max_samples = 32;
        identity.mode = common_flydelta_direction_search_mode::experimental;
        identity.source = common_adaptation_evidence_source::tool_repair;
        identity.behavior_key = behavior_key;
        identity.model_profile_fingerprint = experiment_fixture.model_profile_fingerprint;
        identity.execution_context_fingerprint = experiment_fixture.execution_context_fingerprint;
        identity.capture_layout_revision = "layout:synthetic-spread-v1";

        std::vector<common_flydelta_contrast_sample> samples;
        for (size_t variant = 0; variant < 6; ++variant) {
            const auto & entry = group_cases[variant % group_cases.size()];
            const auto delta = make_delta(entry, group_index, variant);
            samples.push_back({delta, experimental_credit(experiment_fixture.id, delta.id)});
        }

        common_flydelta_aggregation_config aggregation_config;
        aggregation_config.identity = identity;
        aggregation_config.depth.min_shallow_samples = 2;
        aggregation_config.depth.min_deep_samples = 6;
        aggregation_config.depth.max_samples = 32;
        aggregation_config.scope_fingerprint = "scope:local";
        aggregation_config.tokenizer_fingerprint = experiment_fixture.tokenizer_fingerprint;
        aggregation_config.template_fingerprint = experiment_fixture.template_fingerprint;
        aggregation_config.generation_semantics_fingerprint = "generation:replay-v1";
        aggregation_config.max_retained_samples = 32;
        common_flydelta_incremental_aggregation aggregation(aggregation_config);
        for (const auto & sample : samples) CHECK(aggregation.ingest(sample, error));
        common_flydelta_evidence_depth_result depth;
        CHECK(aggregation.assess_depth(depth, error));
        // Replay observations are deliberately UNKNOWN/NEUTRAL experimental
        // material. They may exercise the rank-2/deep primitives below, but
        // they must not manufacture natural evidence capacity.
        CHECK(depth.compatible_samples == 0 && depth.experimental_samples == 6);
        CHECK(depth.effective_rank == 0);
        CHECK(!depth.shallow_ready && !depth.deep_ready);
        CHECK(depth.depth == common_flydelta_search_depth::bootstrap);

        std::vector<common_flydelta_direction_candidate> candidates;
        CHECK(common_flydelta_build_direction_candidates(identity, samples, candidates, error));
        CHECK(candidates.size() >= 3);
        for (const auto & candidate : candidates) CHECK(candidate.experimental_only);
        common_flydelta_low_rank_basis basis;
        CHECK(common_flydelta_build_low_rank_basis(kDimension, 2, candidates, basis, error));
        CHECK(basis.vectors.size() == 2);
        ++experimental_rank2_groups;

        common_flydelta_coefficient_search_config coefficient_config;
        coefficient_config.step = 0.05f;
        coefficient_config.max_candidates = 8;
        coefficient_config.max_l2_norm = 0.32f;
        std::vector<std::vector<float>> shallow_controls;
        CHECK(common_flydelta_propose_shallow_rank_two_controls(
            coefficient_config, true, shallow_controls, error));
        CHECK(shallow_controls.size() == 4);

        auto coefficient_runner = [&](const common_flydelta_experiment_fixture & current_fixture,
                const common_flydelta_low_rank_basis & current_basis,
                const std::vector<float> & coefficients, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial,
                common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry,
                bool & geometry_available, std::string &) {
            (void) current_basis;
            trial = {};
            margin = {};
            geometry = {};
            geometry_available = false;
            trial.executed = true;
            trial.verifier_known = true;
            trial.overlay_applied = apply_overlay;
            trial.intervention_count = apply_overlay ? coefficients.size() : 0;
            trial.evidence_ref = current_fixture.id + (apply_overlay ? ":candidate" : ":baseline");
            float effect = 0.0f;
            if (apply_overlay && coefficients.size() >= 2) {
                effect = 2.8f * coefficients[0] + 2.2f * coefficients[1];
                effect -= 0.25f * std::fabs(coefficients[0] - coefficients[1]);
            }
            margin.available = true;
            margin.positive_total_logprob = -0.40f + effect;
            margin.negative_total_logprob = 0.0f;
            margin.positive_token_count = 1;
            margin.negative_token_count = 1;
            float norm = 0.0f;
            for (const float value : coefficients) norm += value * value;
            norm = std::sqrt(norm);
            trial.passed = apply_overlay && effect >= 0.10f;
            trial.quality = trial.passed ? std::min(1.0f, 0.5f + effect) : 0.0f;
            if (apply_overlay) {
                geometry_available = true;
                geometry.schema_version = 1;
                geometry.layer_index = kLayer;
                geometry.cosine = 0.72f;
                geometry.progress = std::max(0.01f, 0.20f + effect * 0.1f);
                geometry.leakage = 0.08f + norm * 0.1f;
                geometry.shift_norm = norm;
            }
            return true;
        };

        size_t shallow_helped = 0;
        for (const auto & coefficients : shallow_controls) {
            common_flydelta_counterfactual_trial trial;
            common_flydelta_decision_margin margin;
            common_flydelta_representation_diagnostics geometry;
            bool available = false;
            CHECK(coefficient_runner(experiment_fixture, basis, coefficients, true,
                trial, margin, geometry, available, error));
            CHECK(common_flydelta_decision_margin_validate(margin, error));
            if (trial.passed) ++shallow_helped;
        }
        CHECK(shallow_helped > 0);

        common_flydelta_search_pipeline_config pipeline_config;
        pipeline_config.dimension = kDimension;
        pipeline_config.max_directions = 1;
        pipeline_config.scale.initial_scale = 0.04f;
        pipeline_config.scale.growth_factor = 2.0f;
        pipeline_config.scale.max_scale = 0.16f;
        pipeline_config.scale.max_geometric_trials = 3;
        pipeline_config.scale.min_cosine = 0.3f;
        pipeline_config.scale.max_leakage = 1.0f;
        pipeline_config.scale.max_shift_norm = 1.0f;
        pipeline_config.region_max_singleton_layers = 3;
        pipeline_config.region_max_neighborhoods = 2;
        pipeline_config.region_max_trials = 8;
        pipeline_config.region_max_stalled_scales = 2;
        common_flydelta_search_pipeline_direction pipeline_direction;
        pipeline_direction.direction = candidates.front();
        pipeline_direction.available_layers = {22, 24, 25};
        pipeline_direction.layer_anchors = {25};
        common_flydelta_search_pipeline_result pipeline_result;
        CHECK(common_flydelta_run_search_pipeline(
            experiment_fixture, pipeline_config, {pipeline_direction},
            [&](const common_flydelta_experiment_fixture & current_fixture,
                    const common_flydelta_direction_candidate &, const common_flydelta_layer_candidate * layer,
                    float scale, bool apply_overlay, common_flydelta_counterfactual_trial & trial,
                    common_flydelta_decision_margin & margin, common_flydelta_scale_geometry & geometry,
                    std::string & runner_error) {
                trial = {};
                margin = {};
                geometry = {};
                trial.executed = true;
                trial.verifier_known = true;
                trial.overlay_applied = apply_overlay;
                trial.intervention_count = layer == nullptr ? 0 : layer->layer_indices.size();
                trial.evidence_ref = current_fixture.id + (apply_overlay ? ":candidate" : ":baseline");
                const bool target = layer != nullptr && std::find(layer->layer_indices.begin(),
                    layer->layer_indices.end(), static_cast<uint32_t>(kLayer)) != layer->layer_indices.end();
                const float effect = apply_overlay && target ? scale * 3.0f : 0.0f;
                trial.passed = apply_overlay && effect >= 0.10f;
                trial.quality = trial.passed ? std::min(1.0f, effect) : 0.0f;
                margin.available = true;
                margin.positive_total_logprob = -0.40f + effect;
                margin.negative_total_logprob = 0.0f;
                margin.positive_token_count = 1;
                margin.negative_token_count = 1;
                if (apply_overlay) {
                    geometry.available = true;
                    geometry.cosine = target ? 0.70f : 0.40f;
                    geometry.progress = target ? effect : 0.01f;
                    geometry.leakage = target ? 0.10f : 0.35f;
                    geometry.shift_norm = scale;
                }
                return common_flydelta_decision_margin_validate(margin, runner_error);
            }, pipeline_result, error));
        total_search_trials += pipeline_result.directions.front().region_trials.size();
        if (!pipeline_result.directions.front().whirlpool_trace.rounds.empty()) ++whirlpool_groups;

        common_flydelta_deep_search_config deep_config;
        deep_config.max_rank = 2;
        deep_config.max_directions = 3;
        deep_config.full_generation_top_k = 2;
        deep_config.coefficients.strategy = common_flydelta_coefficient_search_strategy::tfo_lite;
        deep_config.coefficients.step = 0.05f;
        deep_config.coefficients.max_candidates = 8;
        deep_config.coefficients.max_l2_norm = 0.32f;
        deep_config.coefficients.population_size = 4;
        deep_config.coefficients.iterations = 2;
        deep_config.coefficients.leakage_penalty = 0.10f;
        std::vector<common_flydelta_deep_search_direction> deep_directions;
        for (const auto & candidate : candidates) deep_directions.push_back({candidate, true, candidate.median_alignment});
        common_flydelta_deep_search_result deep_result;
        CHECK(common_flydelta_run_deep_search(
            experiment_fixture, deep_config, deep_directions,
            coefficient_runner, coefficient_runner, deep_result, error));
        CHECK(deep_result.basis.vectors.size() == 2 && !deep_result.coefficient_trials.empty());
        ++forced_deep_groups;
        ++tfo_groups;
        std::cout << "flydelta_spread_group behavior_key=" << behavior_key
                  << " samples=" << depth.compatible_samples
                  << " experimental_samples=" << depth.experimental_samples
                  << " effective_rank=" << depth.effective_rank
                  << " depth=" << common_flydelta_search_depth_name(depth.depth)
                  << " directions=" << candidates.size()
                  << " rank2=" << basis.vectors.size()
                  << " whirlpool_trials=" << pipeline_result.directions.front().region_trials.size()
                  << " tfo_trials=" << deep_result.coefficient_trials.size() << '\n';
        ++group_index;
    }

    const auto persisted = ledger.list(error);
    CHECK(persisted.size() == echo_persisted && echo_persisted == 1);
    for (const auto & transaction : persisted) {
        CHECK(transaction.observation.signals.size() == 1);
        const auto & signal = transaction.observation.signals.front();
        CHECK(signal.type == common_learning_signal_type::repair_echo_failure);
        CHECK(common_learning_destination_for(transaction.observation) == common_learning_destination::retain);
        const auto context = json::parse(signal.repair_context_json, nullptr, false);
        CHECK(!context.is_discarded() && context.value("learning_credit", true) == false);
    }
    CHECK(natural_shallow_groups == 0 && natural_deep_groups == 0);
    CHECK(experimental_rank2_groups == 3 && forced_deep_groups == 3);
    CHECK(whirlpool_groups == 3 && tfo_groups == 3);
    CHECK(total_search_trials > 0);
    std::cout << "flydelta_repair_echo_replay=passed"
              << " cases=" << cases.size()
              << " synthetic_spread=yes"
              << " natural_shallow_groups=" << natural_shallow_groups
              << " natural_deep_groups=" << natural_deep_groups
              << " experimental_rank2_groups=" << experimental_rank2_groups
              << " forced_deep_groups=" << forced_deep_groups
              << " whirlpool_groups=" << whirlpool_groups
              << " tfo_groups=" << tfo_groups
              << " echo_persisted=" << echo_persisted
              << " ledger=\"" << ledger_path.string() << "\"\n";
    if (remove_ledger) {
        std::error_code ignored;
        std::filesystem::remove(ledger_path, ignored);
    }
    return 0;
}
