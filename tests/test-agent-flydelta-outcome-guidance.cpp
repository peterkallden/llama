#include "agent/adaptation/flydelta/experiment/flydelta-outcome-guidance.h"
#include "agent/prediction/outcome-guidance.h"
#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/tooling/registry/tool-registry.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <limits>
#include <nlohmann/json.hpp>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_outcome_guidance_context context() {
    common_flydelta_outcome_guidance_context c;
    c.scope_fingerprint = "scope-A";
    c.model_profile_fingerprint = "model-A";
    c.tokenizer_fingerprint = "tok-A";
    c.template_fingerprint = "template-A";
    c.baseline_deployment_fingerprint = "base-A";
    c.verifier_revision = "oracle-A";
    c.semantic_domain = "dataset.aggregate";
    c.strategy = "coordinate";
    c.estimator = "margin-geometry-v1";
    return c;
}

static common_flydelta_outcome_guidance_observation observation(
        const std::string & id, const std::string & group,
        common_flydelta_counterfactual_outcome outcome) {
    common_flydelta_outcome_guidance_observation o;
    o.stable_observation_id = id;
    o.fixture_group_id = group;
    o.context = context();
    o.rank = 2;
    o.relative_layer = .5;
    o.relative_dose = .5;
    o.comparison_completed = true;
    o.overlay_applied = true;
    o.verifier_known = outcome != common_flydelta_counterfactual_outcome::unknown;
    o.outcome = outcome;
    o.outcome_known = outcome != common_flydelta_counterfactual_outcome::unknown;
    return o;
}

static common_flydelta_progress_evidence progress_evidence() {
    common_flydelta_progress_evidence e;
    e.baseline_ref = "baseline-arm"; e.candidate_ref = "candidate-arm";
    e.baseline_evidence_ref = "evidence:baseline"; e.candidate_evidence_ref = "evidence:candidate";
    e.baseline.objective_id = "objective"; e.baseline.objective_revision = "goals:v1";
    e.baseline.verifier_ref = "verifier:v1";
    e.baseline.assessment_known = e.baseline.regression_known = true;
    e.baseline.satisfied_goals = {"A"}; e.baseline.remaining_goals = {"B", "C", "D"};
    e.candidate = e.baseline;
    e.candidate.satisfied_goals = {"A", "B"}; e.candidate.remaining_goals = {"C", "D"};
    return e;
}

static int progress_contracts() {
    using transition = common_flydelta_progress_transition;
    std::string error;
    auto e = progress_evidence();
    CHECK(common_flydelta_progress_evidence_classify(e) == transition::improved);
    common_flydelta_progress_evidence restored;
    CHECK(common_flydelta_progress_evidence_from_json(common_flydelta_progress_evidence_to_json(e), restored, error));
    CHECK(common_flydelta_progress_evidence_to_json(restored) == common_flydelta_progress_evidence_to_json(e));
    e.candidate.regression_known = false;
    CHECK(common_flydelta_progress_evidence_classify(e) == transition::unknown);
    e = progress_evidence(); e.parent = e.candidate; e.parent_ref = "anchor:1";
    e.parent->satisfied_goals = {"A", "C"}; e.parent->remaining_goals = {"B", "D"};
    e.candidate.satisfied_goals = {"A", "B", "D"}; e.candidate.remaining_goals = {"C"};
    CHECK(common_flydelta_progress_evidence_classify(e) == transition::regressed);
    e = progress_evidence(); e.candidate.objective_revision = "goals:v2";
    CHECK(common_flydelta_progress_evidence_classify(e) == transition::unknown);
    e = progress_evidence(); e.candidate_evidence_ref.clear();
    CHECK(!common_flydelta_progress_evidence_validate(e, error));

    auto c = context(); c.progress_baseline = progress_evidence().baseline;
    common_flydelta_outcome_guidance_candidate a;
    a.arm_id = "arm"; a.rank = 2; a.relative_layer = a.relative_dose = 0.5;
    a.basis_identity = "basis"; a.direction_identity = "positive"; a.dose_identity = "dose:v1";
    auto row = observation("progress:1", "fixture:1", common_flydelta_counterfactual_outcome::unknown);
    row.basis_identity = a.basis_identity; row.direction_identity = a.direction_identity; row.dose_identity = a.dose_identity;
    row.task_fingerprint = "task"; row.lineage_ref = "search"; row.progress = progress_evidence();
    auto negative = row; negative.stable_observation_id = "progress:2";
    negative.direction_identity = "negative";
    negative.progress->candidate = negative.progress->baseline;
    common_flydelta_outcome_guidance_config config;
    common_flydelta_outcome_guidance_decision decision;
    auto reverse = a; reverse.arm_id = "opposite"; reverse.direction_identity = "negative";
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, {row, negative}, {a, reverse}, 2, 7, decision, error));
    CHECK(decision.predictions[0].p_safe_partial_improved > decision.predictions[1].p_safe_partial_improved);
    CHECK(decision.predictions[0].effective_unknown_observations > 0);
    CHECK(decision.predictions[0].effective_progress_known > 0);
    CHECK(decision.selected_order == decision.original_order); // shadow cannot execute its recommendation
    config.mode = common_flydelta_outcome_guidance_mode::experimental;
    common_flydelta_outcome_guidance_decision changed;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, {row, negative},
        {reverse, a}, 1, 7, changed, error, common_flydelta_outcome_guidance_policy::posterior_mean, 2));
    CHECK(changed.selection_reasons.front() == "posterior_mean");
    CHECK(changed.original_order.front() == 0 && changed.selected_order.front() == 1);
    config.mode = common_flydelta_outcome_guidance_mode::shadow;
    const auto original = decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, {row, row, negative}, {a, reverse}, 2, 7, decision, error));
    CHECK(decision.predictions[0].p_safe_partial_improved == original.predictions[0].p_safe_partial_improved);
    auto conflict = row; conflict.progress->candidate = conflict.progress->baseline;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, {row, conflict}, {a}, 1, 7, decision, error));
    CHECK(!decision.predictions[0].progress_evidence_available);
    auto uncovered = row; uncovered.progress->candidate.regression_known = false;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, {uncovered}, {a}, 1, 7, decision, error));
    CHECK(decision.predictions[0].effective_progress_unknown > 0);
    CHECK(!decision.predictions[0].progress_utility_available && decision.predictions[0].progress_utility == 0);
    auto diagnostic_only = row; diagnostic_only.comparison_completed = false;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, {diagnostic_only}, {a}, 1, 7, decision, error));
    CHECK(decision.rejected_history_rows == 1 && !decision.predictions[0].progress_evidence_available);
    auto wrong_scope = row; wrong_scope.context.scope_fingerprint = "other-scope";
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, {wrong_scope}, {a}, 1, 7, decision, error));
    CHECK(decision.compatible_history_rows == 0 && !decision.predictions[0].progress_evidence_available);
    auto legacy = row; legacy.progress.reset();
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, {legacy}, {a}, 1, 7, decision, error));
    CHECK(!decision.predictions[0].progress_evidence_available);

    common_flydelta_outcome_guidance_projection_ptr projection;
    CHECK(common_flydelta_outcome_guidance_build_projection(c, {row}, projection, error, row.fixture_group_id));
    c.local_task_fingerprint = row.task_fingerprint; c.local_lineage_ref = row.lineage_ref;
    CHECK(common_flydelta_outcome_guidance_decide_from_projection(config, c, projection, {a}, 1, 7, decision, error));
    CHECK(decision.compatible_history_rows == 0 && decision.predictions[0].fixture_groups == 0);
    CHECK(decision.predictions[0].local_observations == 1);
    CHECK(decision.predictions[0].progress_evidence_available);
    c.local_parent = row.progress->candidate; c.local_parent_ref = "anchor:1";
    CHECK(common_flydelta_outcome_guidance_decide_from_projection(config, c, projection, {a}, 1, 7, decision, error));
    CHECK(decision.predictions[0].p_safe_partial_improved < original.predictions[0].p_safe_partial_improved);
    c.local_task_fingerprint = "other-task";
    CHECK(common_flydelta_outcome_guidance_decide_from_projection(config, c, projection, {a}, 1, 7, decision, error));
    CHECK(decision.predictions[0].local_observations == 0 && !decision.predictions[0].progress_evidence_available);
    config.progress_reward = 0;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, {row}, {a}, 1, 7, decision, error));
    CHECK(decision.predictions[0].progress_utility == 0);
    config.progress_reward = -0.1;
    CHECK(!common_flydelta_outcome_guidance_config_validate(config, error));

    nlohmann::ordered_json record = {{"outcome", "unknown"}, {"verifier_known", false}, {"outcome_guidance", {
        {"schema_version", 2}, {"observation_id", "stable"}, {"outcome_known", false},
        {"progress", nlohmann::ordered_json::parse(common_flydelta_progress_evidence_to_json(progress_evidence()))}}}};
    common_flydelta_outcome_guidance_observation decoded;
    CHECK(common_flydelta_outcome_guidance_observation_from_json(record.dump(), decoded, error));
    CHECK(decoded.progress && decoded.outcome == common_flydelta_counterfactual_outcome::unknown);
    record["outcome_guidance"]["progress"]["candidate_evidence_ref"] = "";
    CHECK(!common_flydelta_outcome_guidance_observation_from_json(record.dump(), decoded, error));
    record["outcome_guidance"].erase("progress"); record["outcome_guidance"]["schema_version"] = 1;
    CHECK(common_flydelta_outcome_guidance_observation_from_json(record.dump(), decoded, error) && !decoded.progress);

    common_flydelta_search_observation persisted;
    persisted.experiment_id = "experiment"; persisted.candidate_id = "experimental:arm"; persisted.host_evaluated = true;
    persisted.guidance_observation_id = row.stable_observation_id;
    persisted.guidance_fixture_group_id = row.fixture_group_id;
    persisted.guidance_scope_fingerprint = row.context.scope_fingerprint;
    persisted.guidance_model_profile_fingerprint = row.context.model_profile_fingerprint;
    persisted.guidance_tokenizer_fingerprint = row.context.tokenizer_fingerprint;
    persisted.guidance_template_fingerprint = row.context.template_fingerprint;
    persisted.guidance_baseline_deployment_fingerprint = row.context.baseline_deployment_fingerprint;
    persisted.guidance_verifier_revision = row.context.verifier_revision;
    persisted.guidance_domain = row.context.semantic_domain; persisted.guidance_strategy = row.context.strategy;
    persisted.guidance_estimator = row.context.estimator; persisted.guidance_basis_id = row.basis_identity;
    persisted.guidance_rank = row.rank; persisted.guidance_relative_layer = row.relative_layer;
    persisted.guidance_relative_dose = row.relative_dose;
    persisted.guidance_comparison_completed = persisted.guidance_overlay_applied = true;
    persisted.guidance_task_fingerprint = row.task_fingerprint; persisted.guidance_lineage_ref = row.lineage_ref;
    persisted.guidance_direction_id = row.direction_identity; persisted.guidance_dose_identity = row.dose_identity;
    persisted.guidance_progress = row.progress;
    common_flydelta_search_decision disposition;
    CHECK(common_flydelta_decide_search_disposition(persisted, disposition, error));
    common_flydelta_lifecycle_event_context event;
    event.event_id = event.idempotency_key = "progress:event"; event.source_id = "contract";
    event.scope.namespace_id = "namespace"; event.scope.session_id = "session";
    event.content_hash = "hash:progress"; event.created_at = "2026-10-27T00:00:00Z";
    common_learning_in_memory_lifecycle_store journal;
    CHECK(common_flydelta_append_search_lifecycle(journal, event, persisted, disposition, nullptr, error));
    CHECK(common_flydelta_append_search_lifecycle(journal, event, persisted, disposition, nullptr, error));
    const auto saved = journal.list(error);
    CHECK(error.empty() && saved.size() == 1);
    common_learning_lifecycle_record reloaded;
    CHECK(common_learning_lifecycle_from_json(common_learning_lifecycle_to_json(saved.front()), reloaded, error));
    CHECK(common_flydelta_outcome_guidance_observation_from_json(reloaded.payload_json, decoded, error));
    CHECK(decoded.progress && common_flydelta_progress_evidence_classify(*decoded.progress) == transition::improved);
    CHECK(decoded.outcome == common_flydelta_counterfactual_outcome::unknown);
    auto holdout = row; holdout.stable_observation_id = "holdout:1"; holdout.fixture_group_id = "holdout";
    common_flydelta_outcome_guidance_replay_metrics metrics;
    config = {};
    CHECK(common_flydelta_outcome_guidance_replay(config, {row, holdout}, {"holdout"}, metrics, error));
    CHECK(metrics.progress_predictions == 1 && metrics.known_progress_predictions == 1);
    CHECK(metrics.known_outcomes == 0 && metrics.unknown_outcomes == 1);
    return 0;
}

int main() {
    CHECK(common_flydelta_guidance_policy_start(0, 0, size_t(-1)) == 0);
    CHECK(common_flydelta_guidance_policy_start(0, 2, size_t(-1)) == 2);
    CHECK(common_flydelta_guidance_policy_start(0, 4, size_t(-1)) == 0);
    CHECK(common_flydelta_guidance_policy_start(0, 2, size_t(-1), 1) == 1);
    CHECK(common_flydelta_guidance_policy_start(0, 0, 3) == 3);

    if (const auto failure = progress_contracts()) return failure;
    std::string error;
    common_outcome_guidance_generalization_config generalization;
    CHECK(!generalization.tool_family_backoff && !generalization.project_history &&
        generalization.tool_families.empty());
    CHECK(common_outcome_guidance_generalization_config_validate(generalization,error));
    generalization.tool_families["data-analysis"]={"dataset.aggregate","dataset.join"};
    CHECK(common_outcome_guidance_generalization_config_validate(generalization,error));
    generalization.tool_family_backoff=true;
    generalization.project_history=true;
    CHECK(common_outcome_guidance_generalization_config_validate(generalization,error));
    generalization.tool_families["duplicate"]={"dataset.join","repository.search"};
    CHECK(!common_outcome_guidance_generalization_config_validate(generalization,error));
    generalization.tool_families.erase("duplicate");
    common_outcome_guidance_posterior shared_posterior;
    CHECK(common_outcome_guidance_make_posterior(
        3.0, 1.0, .75, {2.0, 1.0, 0.0}, {2.0/3.0, 1.0/3.0, 0.0},
        2.0, 1.0, shared_posterior));
    CHECK(shared_posterior.p_known > .5 && shared_posterior.p_known < 1.0);
    CHECK(shared_posterior.category_probabilities.size() == 3);
    CHECK(std::abs(shared_posterior.category_probabilities[0] +
        shared_posterior.category_probabilities[1] +
        shared_posterior.category_probabilities[2] - 1.0) < 1e-12);
    CHECK(!common_outcome_guidance_make_posterior(
        -1.0, 0.0, .5, {1.0}, {1.0}, 0.0, 0.0, shared_posterior));
    common_outcome_guidance_projection projection;
    const std::vector<common_outcome_guidance_projection_observation> projection_rows={
        {"fixture-a","shared-category",{1.0,0.0}},
        {"fixture-a","shared-category",{0.0,1.0}},
        {"fixture-b","shared-category",{1.0,0.0}},
        {"fixture-b","other-category",{0.0,1.0}},
    };
    CHECK(common_outcome_guidance_build_projection(projection_rows,"test-builder-v1","source-fingerprint",7,projection,error));
    CHECK(projection.accepted_rows==4&&projection.source_watermark==7&&!projection.fingerprint.empty());
    auto reordered_projection_rows=projection_rows;
    std::reverse(reordered_projection_rows.begin(),reordered_projection_rows.end());
    common_outcome_guidance_projection reordered_projection;
    CHECK(common_outcome_guidance_build_projection(reordered_projection_rows,"test-builder-v1",
        "source-fingerprint",7,reordered_projection,error));
    CHECK(reordered_projection.fingerprint==projection.fingerprint);
    std::map<std::string,std::vector<double>> projected_weights;std::map<std::string,size_t> projected_groups;
    projection.aggregate_excluding("fixture-a",projected_weights,projected_groups);
    CHECK(projected_groups["shared-category"]==1&&projected_weights["shared-category"].size()==2);
    CHECK(std::abs(projected_weights["shared-category"][0]-1.0)<1e-12&&
        std::abs(projected_weights["shared-category"][1])<1e-12);
    const std::vector<common_outcome_guidance_projection_observation> invalid_projection_rows={
        {"fixture-a","bad",{1.0}}, {"fixture-a","bad",{1.0,0.0}}};
    CHECK(!common_outcome_guidance_build_projection(invalid_projection_rows,
        "test-builder-v1","bad-source",0,projection,error));
    const std::vector<common_outcome_guidance_projection_observation> inconsistent_fixtures={
        {"fixture-a","same-category",{1.0}}, {"fixture-b","same-category",{1.0,0.0}}};
    CHECK(!common_outcome_guidance_build_projection(inconsistent_fixtures,
        "test-builder-v1","bad-source",0,projection,error));
    const double huge=std::numeric_limits<double>::max();
    CHECK(!common_outcome_guidance_build_projection({{"fixture-a","overflow",{huge}},
        {"fixture-a","overflow",{huge}}},"test-builder-v1","bad-source",0,projection,error));
    CHECK(!common_outcome_guidance_make_posterior(huge,huge,.5,{1.0},{1.0},0.0,0.0,shared_posterior));
    std::vector<common_outcome_guidance_projection_observation> fractional_rows={
        {"fixture-a","float-sum",{1e16,1e16}},
        {"fixture-a","float-sum",{1.0,0.0}},
        {"fixture-a","float-sum",{1.0,0.0}}};
    CHECK(common_outcome_guidance_build_projection(fractional_rows,"test-builder-v1","same-source",0,projection,error));
    std::reverse(fractional_rows.begin(),fractional_rows.end());
    CHECK(common_outcome_guidance_build_projection(fractional_rows,"test-builder-v1","same-source",0,reordered_projection,error));
    CHECK(projection.fingerprint==reordered_projection.fingerprint&&
        projection.fixture_categories==reordered_projection.fixture_categories);
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    CHECK(catalog.bootstrap("analysis", bootstrap, error));
    const auto * predict_definition = catalog.find_definition("prediction.predict");
    CHECK(predict_definition != nullptr);
    CHECK(catalog.find_definition("outcome.predict") == nullptr);
    const auto input_schema = nlohmann::json::parse(predict_definition->input_schema_json);
    CHECK(input_schema["properties"]["family"]["enum"] ==
        nlohmann::json({"flydelta","target_hunt"}));
    const auto output_schema = nlohmann::json::parse(predict_definition->result_schema_json);
    CHECK(output_schema["properties"]["family"]["enum"].size() == 2);
    CHECK(predict_definition != nullptr && predict_definition->risk_class == common_tool_risk_class::local_read);
    const auto all_configured = catalog.load_profile("all-configured", error);
    CHECK(error.empty() && std::any_of(all_configured.begin(), all_configured.end(),
        [](const auto & definition) { return definition.name == "prediction.predict"; }));
    const auto result_schema = nlohmann::json::parse(predict_definition->result_schema_json, nullptr, false);
    CHECK(result_schema.is_object() && result_schema.contains("properties") &&
        result_schema["properties"].contains("predictions") &&
        result_schema["properties"].contains("unresolved_refs") &&
        result_schema["properties"].contains("scope_fingerprint"));
    const std::string sample_result = nlohmann::ordered_json{
        {"family", "flydelta_intervention_v1"}, {"status", "available"},
        {"scope_fingerprint", "sha256:scope"},
        {"history_watermark", 12}, {"history_fingerprint", "sha256:history"},
        {"unresolved_refs", nlohmann::ordered_json::array()},
        {"predictions", nlohmann::ordered_json::array({nlohmann::ordered_json{
            {"family", "flydelta_intervention_v1"},
            {"observation_ref", "obs-1"},
            {"prediction_kind", "chronological_leave_fixture_group_out"},
            {"p_known", 0.8}, {"p_known_interval", {0.5, 1.0}},
            {"p_helped_given_known", 0.7}, {"p_neutral_given_known", 0.2},
            {"p_harmed_given_known", 0.1}, {"expected_utility", 0.2},
            {"evidence_groups", 8}, {"backoff_level", "backoff"},
            {"history_fingerprint", "sha256:prior"}, {"domain", "dataset"},
            {"strategy", "deep"}, {"estimator", "coefficient-search"},
            {"observed_outcome", "helped"},
        }})},
    }.dump();
    std::string normalized_result;
    CHECK(common_schema_normalize_and_validate_object(
        sample_result, predict_definition->result_schema_json, normalized_result, error));
    const std::string target_hunt_result = nlohmann::ordered_json{
        {"family", "target_hunt_branch_quality_v1"}, {"status", "available"},
        {"scope_mode", "session"}, {"scope_fingerprint", "sha256:scope"},
        {"history_watermark", 12}, {"history_fingerprint", "sha256:history"},
        {"unresolved_refs", nlohmann::ordered_json::array()},
        {"predictions", nlohmann::ordered_json::array({nlohmann::ordered_json{
            {"candidate_ref", "target-hunt:opaque"},
            {"family", "target_hunt_branch_quality_v1"},
            {"prediction_kind", "retrospective_leave_fixture_group_out"},
            {"p_known", 0.9}, {"p_optimal_given_known", 0.7},
            {"p_detour_given_known", 0.2}, {"p_unreachable_given_known", 0.1},
            {"expected_utility", 0.6}, {"evidence_groups", 8},
            {"backoff_level", "fine_feature"},
            {"history_fingerprint", "sha256:history"}, {"observed_outcome", "optimal"},
        }})},
    }.dump();
    CHECK(common_schema_normalize_and_validate_object(
        target_hunt_result, predict_definition->result_schema_json, normalized_result, error));
    common_native_tool_bindings bindings;
    bool prediction_called = false;
    bindings.prediction_predict = [&prediction_called, &sample_result](const std::string & input) {
        prediction_called = input == R"({"candidate_refs":["obs-1"]})";
        return common_tool_execution_result::success(sample_result);
    };
    common_tool_registry registry;
    common_tool_adapter_result adapter_result;
    CHECK(common_register_native_tool_adapters(catalog, "analysis", bindings,
        registry, adapter_result, error));
    CHECK(registry.contains("prediction.predict") && registry.is_read_only("prediction.predict"));
    CHECK(!registry.contains("outcome.predict"));
    CHECK(registry.execute({"prediction.predict", R"({"candidate_refs":["obs-1"]})"}).ok);
    CHECK(prediction_called);
    common_flydelta_outcome_guidance_config config;
    CHECK(config.mode == common_flydelta_outcome_guidance_mode::shadow);
    CHECK(common_flydelta_outcome_guidance_config_validate(config, error));

    const auto c = context();
    std::vector<common_flydelta_outcome_guidance_candidate> candidates(4);
    candidates[0].arm_id = "arm-A"; candidates[0].rank = 1;
    candidates[0].relative_layer = .5; candidates[0].relative_dose = .5;
    candidates[1].arm_id = "arm-B"; candidates[1].rank = 2;
    candidates[1].relative_layer = .5; candidates[1].relative_dose = .5;
    candidates[1].mixed_coefficients = true; candidates[1].diagnostic_margin_sign = 1;
    candidates[2].arm_id = "arm-C"; candidates[2].rank = 1;
    candidates[2].relative_layer = .5; candidates[2].relative_dose = .9;
    candidates[2].mixed_coefficients = true; candidates[2].diagnostic_margin_sign = -1;
    candidates[3].arm_id = "arm-D"; candidates[3].rank = 1;
    candidates[3].relative_layer = .5; candidates[3].relative_dose = .2;
    std::vector<common_flydelta_outcome_guidance_observation> history;
    for (int i=0;i<8;++i) history.push_back(observation(
        "obs-"+std::to_string(i), "fixture-"+std::to_string(i),
        common_flydelta_counterfactual_outcome::helped));
    // Repeated measurements from one fixture share total weight one.
    history.push_back(observation("same-fixture-help", "fixture-shared",
        common_flydelta_counterfactual_outcome::helped));
    history.push_back(observation("same-fixture-harm", "fixture-shared",
        common_flydelta_counterfactual_outcome::harmed));
    // Duplicate retries, incomplete comparisons, and unapplied overlays are
    // rejected from the projection.
    history.push_back(history.front());
    auto diagnostic = observation("diagnostic", "fixture-diagnostic",
        common_flydelta_counterfactual_outcome::helped);
    diagnostic.comparison_completed = false;
    history.push_back(diagnostic);
    auto no_overlay = observation("no-overlay", "fixture-no-overlay",
        common_flydelta_counterfactual_outcome::helped);
    no_overlay.overlay_applied = false;
    history.push_back(no_overlay);
    auto unknown = observation("unknown", "fixture-unknown",
        common_flydelta_counterfactual_outcome::unknown);
    history.push_back(unknown);
    auto wrong_model = observation("wrong-model", "fixture-wrong-model",
        common_flydelta_counterfactual_outcome::harmed);
    wrong_model.context.model_profile_fingerprint = "model-B";
    history.push_back(wrong_model);

    common_flydelta_outcome_guidance_decision shadow;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 55, shadow, error));
    CHECK(shadow.original_order == std::vector<size_t>({0,1,2,3}));
    CHECK(shadow.selected_order == shadow.original_order);
    CHECK(shadow.selection_reasons.size() == candidates.size());
    CHECK(shadow.selection_reasons[0] == "shadow_preserved_existing_order");
    CHECK(shadow.predictions.size() == candidates.size());
    CHECK(shadow.predictions[0].level == "backoff");
    CHECK(shadow.predictions[0].fixture_groups == 0);
    CHECK(shadow.predictions[0].p_helped_given_known > .60);
    CHECK(shadow.predictions[0].p_harmed_high > shadow.predictions[0].p_harmed_low);
    CHECK(shadow.predictions[0].backoff_fixture_groups >= 8);
    CHECK(shadow.predictions[0].backoff_prior_strength == 4.0);
    auto shared_candidates=candidates;
    auto same_posterior=candidates[0];same_posterior.arm_id="same-posterior";
    shared_candidates.push_back(same_posterior);
    common_flydelta_outcome_guidance_decision shared_draw,reordered_draw;
    CHECK(common_flydelta_outcome_guidance_build_decision(config,c,history,shared_candidates,
        shared_candidates.size(),55,shared_draw,error));
    CHECK(shared_draw.predictions.front().posterior_identity==shared_draw.predictions.back().posterior_identity);
    CHECK(shared_draw.predictions.front().sampled_utility==shared_draw.predictions.back().sampled_utility);
    CHECK(shared_draw.predictions[0].posterior_identity!=shared_draw.predictions[1].posterior_identity);
    std::reverse(shared_candidates.begin(),shared_candidates.end());
    CHECK(common_flydelta_outcome_guidance_build_decision(config,c,history,shared_candidates,
        shared_candidates.size(),55,reordered_draw,error));
    for(const auto & expected:shared_draw.predictions){
        const auto actual=std::find_if(reordered_draw.predictions.begin(),reordered_draw.predictions.end(),
            [&](const auto & value){return value.arm_id==expected.arm_id;});
        CHECK(actual!=reordered_draw.predictions.end()&&actual->sampled_utility==expected.sampled_utility);
    }

    // Layer search can borrow only the deliberately weak same-domain prior
    // from Deep/TFO strategies; the exact deployment/verifier partition stays.
    auto whirlpool_context = c;
    whirlpool_context.strategy = "whirlpool";
    whirlpool_context.estimator = "layer-region-prior-v2";
    common_flydelta_outcome_guidance_candidate layer_candidate;
    layer_candidate.arm_id = "layer:12";
    layer_candidate.rank = 1;
    layer_candidate.relative_layer = .5;
    layer_candidate.relative_dose = .2;
    common_flydelta_outcome_guidance_decision cross_stage;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, whirlpool_context, history, {layer_candidate}, 1, 56, cross_stage, error));
    CHECK(cross_stage.predictions.size() == 1);
    CHECK(cross_stage.predictions.front().level == "backoff");
    CHECK(cross_stage.predictions.front().backoff_fixture_groups >= 8);
    CHECK(cross_stage.predictions.front().p_helped_given_known > .60);

    // Transfer keeps WHERE: equal-sized early/harmed and late/helped histories
    // must produce different priors for the two proposed layer regions.
    std::vector<common_flydelta_outcome_guidance_observation> regional_history;
    for (int i = 0; i < 8; ++i) {
        auto early = observation("early-" + std::to_string(i), "early-group-" + std::to_string(i),
            common_flydelta_counterfactual_outcome::harmed);
        early.relative_layer = .125;
        auto late = observation("late-" + std::to_string(i), "late-group-" + std::to_string(i),
            common_flydelta_counterfactual_outcome::helped);
        late.relative_layer = .875;
        regional_history.push_back(early);
        regional_history.push_back(late);
    }
    const std::vector<common_flydelta_outcome_guidance_candidate> regional_candidates = {
        common_flydelta_outcome_guidance_layer_candidate(2, 17, .2),
        common_flydelta_outcome_guidance_layer_candidate(14, 17, .2),
    };
    common_flydelta_outcome_guidance_decision regional;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, whirlpool_context,
        regional_history, regional_candidates, 2, 56, regional, error));
    CHECK(regional.predictions[1].p_helped_given_known > regional.predictions[0].p_helped_given_known);
    auto direct_layer = regional_history.back();
    direct_layer.context = whirlpool_context;
    direct_layer.rank = regional_candidates.back().rank;
    direct_layer.basis_identity = regional_candidates.back().basis_identity;
    direct_layer.relative_layer = regional_candidates.back().relative_layer;
    direct_layer.relative_dose = regional_candidates.back().relative_dose;
    direct_layer.diagnostic_margin_sign = regional_candidates.back().diagnostic_margin_sign;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, whirlpool_context,
        {direct_layer}, {regional_candidates.back()}, 1, 56, regional, error));
    CHECK(regional.predictions.front().level == "arm_category");

    common_flydelta_outcome_guidance_decision direct;
    common_flydelta_outcome_guidance_candidate direct_candidate;
    direct_candidate.arm_id = "direct";
    direct_candidate.rank = 2;
    direct_candidate.relative_layer = .5;
    direct_candidate.relative_dose = .5;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, {direct_candidate}, 1, 55, direct, error));
    CHECK(direct.predictions[0].fixture_groups == 10);
    CHECK(direct.predictions[0].raw_observations == 11);
    CHECK(direct.predictions[0].effective_known_observations == 9.0);
    CHECK(direct.predictions[0].effective_unknown_observations == 1.0);

    auto mixed_group_history = std::vector{observation("mixed-help", "mixed-group",
        common_flydelta_counterfactual_outcome::helped)};
    common_flydelta_outcome_guidance_decision known_only_group;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, mixed_group_history, {direct_candidate}, 1, 9, known_only_group, error));
    mixed_group_history.push_back(observation("mixed-unknown", "mixed-group",
        common_flydelta_counterfactual_outcome::unknown));
    common_flydelta_outcome_guidance_decision mixed_known_unknown_group;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, mixed_group_history, {direct_candidate}, 1, 9,
        mixed_known_unknown_group, error));
    CHECK(mixed_known_unknown_group.predictions[0].p_known <
        known_only_group.predictions[0].p_known);
    CHECK(std::fabs(mixed_known_unknown_group.predictions[0].p_helped_given_known -
        known_only_group.predictions[0].p_helped_given_known) < 1e-12);

    config.mode = common_flydelta_outcome_guidance_mode::experimental;
    common_flydelta_outcome_guidance_decision experimental_a, experimental_b;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 0, experimental_a, error));
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 0, experimental_b, error));
    CHECK(experimental_a.selected_order == experimental_b.selected_order);
    CHECK(experimental_a.selected_order.size() == candidates.size());
    CHECK(experimental_a.history_fingerprint == experimental_b.history_fingerprint);
    CHECK(experimental_a.selection_reasons.size() == candidates.size());
    CHECK(experimental_a.selection_reasons[0] == "existing_order");
    CHECK(experimental_a.selection_reasons[1] == "existing_order");
    CHECK(experimental_a.selection_reasons[2] == "posterior_sample");
    CHECK(experimental_a.selection_reasons[3] == "uniform_exploration");
    CHECK(experimental_a.mixed_policy_reasons == experimental_a.selection_reasons);
    CHECK(experimental_a.mixed_policy_order.size() == candidates.size());
    {
        auto sorted_order = experimental_a.mixed_policy_order;
        std::sort(sorted_order.begin(), sorted_order.end());
        CHECK(sorted_order == experimental_a.original_order);
    }
    common_flydelta_outcome_guidance_decision continued_policy;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 0, continued_policy, error,
        common_flydelta_outcome_guidance_policy::posterior_sampling, 1));
    CHECK(continued_policy.selection_reasons[0] == "existing_order");
    CHECK(continued_policy.selection_reasons[1] == "posterior_sample");
    CHECK(continued_policy.selection_reasons[2] == "uniform_exploration");
    CHECK(continued_policy.selection_reasons[3] == "existing_order");
    common_flydelta_outcome_guidance_decision posterior_mean;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 0, posterior_mean, error,
        common_flydelta_outcome_guidance_policy::posterior_mean));
    CHECK(posterior_mean.selection_reasons[2] == "posterior_mean");
    CHECK(posterior_mean.selection_reasons[3] == "uniform_exploration");
    CHECK(posterior_mean.selected_order == posterior_mean.mixed_policy_order);
    {
        auto sorted_order = posterior_mean.selected_order;
        std::sort(sorted_order.begin(), sorted_order.end());
        CHECK(sorted_order == posterior_mean.original_order);
    }

    // Exact categories are basis-local; a same-rank arm on a different basis
    // must not inherit leaf evidence (it may still receive coarse backoff).
    auto other_basis = candidates[0];
    other_basis.arm_id = "other-basis";
    other_basis.rank = 2;
    other_basis.basis_identity = "basis-C";
    auto basis_a = observation("basis-a", "basis-fixture-a",
        common_flydelta_counterfactual_outcome::helped);
    basis_a.basis_identity = "basis-A";
    auto basis_b = observation("basis-b", "basis-fixture-b",
        common_flydelta_counterfactual_outcome::harmed);
    basis_b.basis_identity = "basis-B";
    common_flydelta_outcome_guidance_decision basis_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, {basis_a, basis_b}, {other_basis}, 1, 0, basis_decision, error));
    CHECK(basis_decision.predictions[0].fixture_groups == 0);
    CHECK(std::abs(basis_decision.predictions[0].p_helped_given_known - 1.0/3.0) < 1e-9);

    common_learning_in_memory_lifecycle_store guidance_store;
    common_flydelta_lifecycle_event_context guidance_event;
    guidance_event.event_id = "guidance:test:1";
    guidance_event.idempotency_key = guidance_event.event_id;
    guidance_event.source_id = "test-outcome-guidance";
    guidance_event.scope.namespace_id = "local";
    guidance_event.scope.project_id = "agent-tests";
    guidance_event.scope.session_id = "guidance-session";
    guidance_event.content_hash = "sha256:guidance-test";
    guidance_event.created_at = "2026-10-18T00:00:00Z";
    experimental_a.stage = "tfo_population";
    experimental_a.phase_id = 2;
    experimental_a.round = 1;
    experimental_a.decision_id = "tfo-population-1";
    CHECK(common_flydelta_append_outcome_guidance_decision_lifecycle(
        guidance_store, guidance_event, "job-guidance", config.mode, config.seed, 0, 0,
        candidates.size(), 0, 0, candidates.size(), c, candidates, experimental_a, error));
    const auto guidance_records = guidance_store.list(error);
    CHECK(error.empty() && guidance_records.size() == 1);
    const auto guidance_payload = nlohmann::ordered_json::parse(
        guidance_records.front().payload_json);
    CHECK(guidance_payload["record_type"] == "outcome_guidance_decision");
    CHECK(guidance_payload["schema_version"] == 4);
    CHECK(guidance_payload["stage"] == "tfo_population");
    CHECK(guidance_payload["round"] == 1);
    CHECK(guidance_payload["decision_id"] == "tfo-population-1");
    CHECK(guidance_payload["decision_sequence"] == 0);
    CHECK(guidance_payload["policy_sequence_start"] == 0);
    CHECK(guidance_payload["policy_slots_consumed"] == candidates.size());
    CHECK(guidance_payload["candidate_pool_fingerprint"].get<std::string>().rfind(
        "sha256:", 0) == 0);
    CHECK(guidance_payload["candidate_pool_fingerprint"] ==
        common_flydelta_outcome_guidance_pool_fingerprint(candidates));
    CHECK(guidance_payload["selection_reasons"].size() == candidates.size());
    CHECK(guidance_payload["predictions"][0].contains("p_harmed_interval_95_approx"));

    // The OFF contract is identity and does not return an empty permutation.
    config.mode = common_flydelta_outcome_guidance_mode::off;
    common_flydelta_outcome_guidance_decision disabled;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 55, disabled, error));
    CHECK(disabled.selected_order == disabled.original_order);
    CHECK(disabled.selection_reasons.size() == candidates.size());
    CHECK(disabled.predictions.empty());

    // An UNKNOWN host comparison only lowers P(known); it adds no outcome
    // mass to the helped/neutral/harmed Dirichlet.
    const auto before = shadow.predictions[0];
    std::vector<common_flydelta_outcome_guidance_observation> unknown_only = {unknown};
    unknown_only[0].rank = candidates[0].rank;
    config.mode = common_flydelta_outcome_guidance_mode::shadow;
    common_flydelta_outcome_guidance_decision unknown_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, unknown_only, {candidates[0]}, 1, 2, unknown_decision, error));
    CHECK(unknown_decision.predictions[0].p_known < .5);
    CHECK(std::abs(unknown_decision.predictions[0].p_helped_given_known - 1.0/3.0) < 1e-9);

    // A changed observation under the same stable id changes the projection
    // fingerprint; a duplicate retry contributes only once.
    auto changed = history.front();
    changed.outcome = common_flydelta_counterfactual_outcome::harmed;
    std::vector<common_flydelta_outcome_guidance_observation> changed_history = {changed};
    auto original_history = history;
    original_history.resize(1);
    common_flydelta_outcome_guidance_decision changed_decision, original_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, changed_history, {candidates[0]}, 1, 2, changed_decision, error));
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, original_history, {candidates[0]}, 1, 2, original_decision, error));
    CHECK(changed_decision.history_fingerprint != original_decision.history_fingerprint);

    auto conflict = original_history.front();
    conflict.outcome = common_flydelta_counterfactual_outcome::harmed;
    common_flydelta_outcome_guidance_decision conflict_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, {original_history.front(), conflict}, {candidates[0]}, 1, 2,
        conflict_decision, error));
    CHECK(conflict_decision.rejected_history_rows == 2);
    CHECK(conflict_decision.predictions[0].fixture_groups == 0);

    // Chronological replay holds out whole fixture groups and compares the
    // posterior with fixed and simple historical-share baselines.
    config.mode = common_flydelta_outcome_guidance_mode::shadow;
    std::vector<common_flydelta_outcome_guidance_observation> chronological;
    for (int i = 0; i < 8; ++i) chronological.push_back(observation(
        "train-" + std::to_string(i), "train-fixture-" + std::to_string(i),
        i < 6 ? common_flydelta_counterfactual_outcome::helped :
            common_flydelta_counterfactual_outcome::neutral));
    chronological.push_back(observation("holdout-a", "holdout-fixture-a",
        common_flydelta_counterfactual_outcome::helped));
    chronological.push_back(observation("holdout-b", "holdout-fixture-b",
        common_flydelta_counterfactual_outcome::unknown));
    auto malformed_unknown = observation("holdout-malformed", "holdout-fixture-malformed",
        common_flydelta_counterfactual_outcome::unknown);
    malformed_unknown.outcome_known = true;
    chronological.push_back(malformed_unknown);
    common_flydelta_outcome_guidance_replay_metrics replay;
    CHECK(common_flydelta_outcome_guidance_replay(config, chronological,
        {"holdout-fixture-a", "holdout-fixture-b", "holdout-fixture-malformed"}, replay, error));
    CHECK(replay.eligible_predictions == 2 && replay.known_outcomes == 1 &&
        replay.unknown_outcomes == 1);
    CHECK(replay.model_verifiability_brier >= 0 && replay.model_verifiability_brier <= 1);
    CHECK(replay.model_verifiability_ece >= 0 && replay.model_verifiability_ece <= 1);
    CHECK(replay.model_outcome_brier >= 0 && replay.model_outcome_brier <= 2);
    CHECK(replay.fixed_prior_verifiability_brier >= 0 &&
        replay.historical_mean_verifiability_brier >= 0);

    common_flydelta_outcome_guidance_context incompatible = c;
    incompatible.baseline_deployment_fingerprint = "base-B";
    common_flydelta_outcome_guidance_decision incompatible_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, incompatible, history, {candidates[0]}, 1, 2, incompatible_decision, error));
    CHECK(incompatible_decision.predictions[0].p_helped_given_known < before.p_helped_given_known);
    common_flydelta_outcome_guidance_context wrong_ruleset = c;
    wrong_ruleset.ruleset_revision = "rules-v2";
    common_flydelta_outcome_guidance_decision wrong_ruleset_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, wrong_ruleset, history, {candidates[0]}, 1, 2, wrong_ruleset_decision, error));
    CHECK(wrong_ruleset_decision.predictions[0].fixture_groups == 0);

    // A frozen projection has identical decisions to the compatibility path,
    // survives caller history changes, and is shared across algorithmic stages.
    common_flydelta_outcome_guidance_projection_ptr frozen_projection;
    CHECK(common_flydelta_outcome_guidance_build_projection(c, chronological, frozen_projection, error));
    common_flydelta_outcome_guidance_decision projected_decision, raw_decision;
    CHECK(common_flydelta_outcome_guidance_decide_from_projection(config, c, frozen_projection,
        candidates, 2, 17, projected_decision, error));
    CHECK(common_flydelta_outcome_guidance_build_decision(config, c, chronological,
        candidates, 2, 17, raw_decision, error));
    CHECK(projected_decision.history_fingerprint == raw_decision.history_fingerprint);
    CHECK(projected_decision.selected_order == raw_decision.selected_order);
    CHECK(projected_decision.predictions.size() == raw_decision.predictions.size());
    for (size_t i = 0; i < raw_decision.predictions.size(); ++i) {
        CHECK(projected_decision.predictions[i].sampled_utility == raw_decision.predictions[i].sampled_utility);
        CHECK(projected_decision.predictions[i].known_weight == raw_decision.predictions[i].known_weight);
    }
    CHECK(!common_flydelta_outcome_guidance_decide_from_projection(config, incompatible,
        frozen_projection, candidates, 2, 17, projected_decision, error));
    common_flydelta_outcome_guidance_projection_cache projection_cache;
    common_flydelta_outcome_guidance_projection_ptr cached_projection, next_projection;
    CHECK(projection_cache.get_or_build("job:frozen:1", c, chronological, {}, cached_projection, error));
    auto where_context = c;
    where_context.strategy = "whirlpool";
    where_context.estimator = "layer-region-prior-v2";
    CHECK(projection_cache.get_or_build("job:frozen:1", where_context, chronological, {}, next_projection, error));
    CHECK(cached_projection == next_projection);
    CHECK(projection_cache.get_or_build("job:frozen:1", c, chronological,
        "train-fixture-0", next_projection, error));
    CHECK(cached_projection != next_projection);
    CHECK(common_flydelta_outcome_guidance_decide_from_projection(config, c, next_projection,
        candidates, 2, 17, projected_decision, error));
    CHECK(projected_decision.compatible_history_rows + 1 == raw_decision.compatible_history_rows);
    auto updated_history = chronological;
    updated_history.push_back(observation("next-job", "next-group", common_flydelta_counterfactual_outcome::harmed));
    CHECK(projection_cache.get_or_build("job:frozen:2", c, updated_history, {}, next_projection, error));
    CHECK(cached_projection != next_projection);
    CHECK(common_flydelta_outcome_guidance_decide_from_projection(config, c, cached_projection,
        candidates, 2, 17, projected_decision, error));
    CHECK(projected_decision.history_fingerprint == raw_decision.history_fingerprint);
    projection_cache.set_budget(0);
    CHECK(projection_cache.get_or_build("job:frozen:1", c, chronological, {}, cached_projection, error));
    CHECK(projection_cache.get_or_build("job:frozen:1", c, chronological, {}, next_projection, error));
    CHECK(cached_projection != next_projection);
    config.harm_penalty = -1;
    CHECK(!common_flydelta_outcome_guidance_config_validate(config, error));
    std::cout << "FlyDelta outcome guidance checks passed\n";
    return 0;
}
