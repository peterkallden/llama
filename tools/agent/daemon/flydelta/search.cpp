#include "../agent-daemon-flydelta-internal.h"
#include "agent/adaptation/flydelta/flydelta-search-budget.h"

#include "agent/agent-residual-patch.h"
#include "agent/adaptation/flydelta/experiment/flydelta-guidance-qualification.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "state.h"

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_run_search_pipeline(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_search_pipeline_config & config,
        common_flydelta_search_pipeline_result & result,
        std::string & error) {
    error.clear();
    if (job.seed.evidence_ref.empty() || job.seed.evidence_ref.size() > 512) {
        error = "FlyDelta WHERE diagnostics require a bounded source evidence reference";
        return false;
    }
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    if (!provider) {
        error = "FlyDelta WHERE search requires a scoped resource provider";
        return false;
    }
    common_flydelta_experiment_fixture fixture;
    fixture.id = job.seed.verifier_ref;
    fixture.task_fingerprint = job.seed.task_fingerprint;
    fixture.model_profile_fingerprint = job.seed.model_profile_fingerprint;
    fixture.tokenizer_fingerprint = job.seed.tokenizer_fingerprint;
    fixture.template_fingerprint = job.seed.template_fingerprint;
    fixture.execution_context_fingerprint = job.seed.execution_context_fingerprint;
    fixture.verifier_revision = job.seed.verifier_ref;
    fixture.scope_fingerprint = daemon_flydelta_scope_fingerprint(job.seed.scope);
    fixture.guidance_scope_fingerprint = common_flydelta_outcome_guidance_scope_fingerprint(job.seed.scope);
    fixture.semantic_domain = job.seed.behavior_key.empty()
        ? common_adaptation_evidence_source_name(job.seed.source) : job.seed.behavior_key;
    fixture.model_n_layers = provider->model_n_layers;
    const std::string baseline_identity = "flydelta-base-deployment-v1\n" +
        job.seed.model_profile_fingerprint + "\n" + job.seed.tokenizer_fingerprint + "\n" +
        job.seed.template_fingerprint + "\n" + std::to_string(provider->model_n_layers);
    fixture.baseline_deployment_fingerprint = "sha256:" +
        hash_sha256_hex(baseline_identity.data(), baseline_identity.size());
    fixture.outcome_guidance_verifier_revision = job.seed.verifier_contract_revision.empty()
        ? daemon_flydelta_verifier_contract_revision(provider->oracle_registry, job.seed.verifier_ref)
        : job.seed.verifier_contract_revision;
    fixture.guidance_fixture_group_id = job.seed.fixture_group_id;
    fixture.ruleset_revision = job.seed.ruleset_revision;
    fixture.action_contract_fingerprint = job.seed.action_contract_fingerprint;
    if (!common_flydelta_experiment_fixture_validate(fixture, error)) return false;

    std::vector<common_flydelta_basis_direction> basis;
    uint32_t inherited_layer = 0;
    if (!job.concept_localization.layers.empty()) {
        if (!config.use_whirlpool_search || !config.use_intervention_region_search) {
            error = "FlyDelta concept relocalization requires the existing Whirlpool WHERE path";
            return false;
        }
        if (!daemon_flydelta_resolve_concept_layer_family(provider, job, basis, inherited_layer, error)) return false;
    } else if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, basis, error)) return false;
    common_flydelta_search_pipeline_direction input;
    input.direction.kind = common_flydelta_direction_kind::raw_repair;
    input.direction.layer_index = basis.front().layer_index;
    input.direction.values = basis.front().values;
    input.direction.origin = "runtime_resource";
    input.direction.extraction_id = job.seed.candidate_ref;
    input.direction.source_samples = 1;
    input.direction.retained_samples = 1;
    input.direction.experimental_only = true;
    for (const auto & direction : basis) {
        input.available_layers.push_back(static_cast<uint32_t>(direction.layer_index));
        input.layer_anchors.push_back(static_cast<uint32_t>(direction.layer_index));
    }
    std::sort(input.available_layers.begin(), input.available_layers.end());
    input.available_layers.erase(std::unique(input.available_layers.begin(),
        input.available_layers.end()), input.available_layers.end());
    std::sort(input.layer_anchors.begin(), input.layer_anchors.end());
    input.layer_anchors.erase(std::unique(input.layer_anchors.begin(), input.layer_anchors.end()),
        input.layer_anchors.end());
    if (inherited_layer != 0) {
        // The origin is only a prior. If its signal was filtered out, use the
        // nearest available realization without moving a vector across layers.
        const auto prior = std::min_element(input.available_layers.begin(), input.available_layers.end(),
            [&](uint32_t left, uint32_t right) {
                return std::abs(static_cast<int64_t>(left) - inherited_layer) <
                    std::abs(static_cast<int64_t>(right) - inherited_layer);
            });
        input.layer_anchors = {*prior};
    }
    if (!common_flydelta_search_pipeline_direction_validate(input, config, error)) return false;

    size_t arm_index = 0;
    const auto capture_layers = input.available_layers;
    std::string baseline_capture_arm_id;
    const auto make_arm = [provider, &job, &arm_index](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_layer_candidate * layer,
            const float scale,
            const bool apply_overlay,
            const std::string & wave_id,
            const size_t proposal_index,
            const bool full_execution,
            common_flydelta_arm_request & arm,
            std::string & runner_error) {
        arm = {};
        arm.job_id = job.id;
        arm.wave_id = wave_id;
        arm.proposal_index = proposal_index;
        arm.arm_id = "flydelta://runtime/" + job.id + "/" + std::to_string(arm_index++);
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = current_fixture.id;
        arm.intervention_ref = job.seed.candidate_ref;
        arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
        arm.apply_overlay = apply_overlay;
        arm.fresh_context = true;
        arm.alpha = apply_overlay ? scale : 0.0f;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = full_execution;
        arm.request_host_verification = full_execution;
        // Diagnostics are first-class search observations.  The production
        // host captures the small requested layer set and keeps it only for
        // this bounded comparison wave.
        arm.request_capture = true;
        arm.max_generated_tokens = 0;
        if (layer != nullptr && apply_overlay) {
            arm.layer_indices = layer->layer_indices;
            arm.coefficients.assign(layer->layer_indices.size(), 1.0f);
        }
        return common_flydelta_arm_request_validate(arm, runner_error);
    };
    common_flydelta_search_pipeline_runner runner = [provider, &job, &make_arm,
            &capture_layers, &baseline_capture_arm_id](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_direction_candidate &,
            const common_flydelta_layer_candidate * layer,
            float scale,
            bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_scale_geometry & geometry,
            std::string & runner_error) {
        common_flydelta_arm_request arm;
        if (!make_arm(current_fixture, layer, scale, apply_overlay,
                apply_overlay ? "region-diagnostic" : "baseline-diagnostic", 0, false,
                arm, runner_error)) return false;
        if (!apply_overlay) {
            // A baseline has no overlay layers of its own, but it must capture
            // the same searchable layers as later arms for aligned geometry.
            arm.layer_indices = capture_layers;
            arm.coefficients.assign(capture_layers.size(), 0.0f);
        }
        common_flydelta_arm_batch_request batch;
        batch.batch_id = job.id + ":runtime";
        batch.wave_id = arm.wave_id;
        batch.arms.push_back(arm);
        common_flydelta_arm_batch_result batch_result;
        if (!daemon_flydelta_execute_batch(provider, batch, batch_result, runner_error) ||
                batch_result.arms.size() != 1) return false;
        const auto & arm_result = batch_result.arms.front();
        if (!apply_overlay) baseline_capture_arm_id = arm.arm_id;
        trial.executed = arm_result.executed;
        trial.host_evaluated = arm_result.host_evaluated;
        trial.verifier_known = arm_result.verifier_known;
        trial.passed = daemon_flydelta_arm_semantic_passed(arm_result);
        trial.quality = arm_result.quality;
        trial.overlay_applied = arm_result.overlay_applied;
        trial.intervention_count = arm.layer_indices.size();
        trial.semantic_progress_observation = arm_result.semantic_progress_observation;
        trial.observed_decision_summary = arm_result.observed_decision_summary;
        trial.expected_decision_summary = arm_result.expected_decision_summary;
        trial.verifier_reason = arm_result.verifier_reason;
        trial.evidence_ref = arm_result.generation_ref;
        margin = arm_result.margin;
        geometry = {};
        if (apply_overlay && !baseline_capture_arm_id.empty() &&
                !arm_result.capture_ref.empty()) {
            std::shared_ptr<const common_flydelta_hidden_state_capture> baseline_capture;
            std::shared_ptr<const common_flydelta_hidden_state_capture> overlay_capture;
            {
                std::lock_guard<std::mutex> lock(provider->capture_mutex);
                const auto baseline_it = provider->captures.find(baseline_capture_arm_id);
                const auto overlay_it = provider->captures.find(arm.arm_id);
                if (baseline_it != provider->captures.end()) baseline_capture = baseline_it->second;
                if (overlay_it != provider->captures.end()) overlay_capture = overlay_it->second;
            }
            std::vector<common_flydelta_basis_direction> available;
            if (baseline_capture && overlay_capture &&
                    daemon_flydelta_parse_directions(*provider, arm.intervention_ref,
                        available, runner_error)) {
                const uint32_t diagnostic_layer = layer == nullptr || layer->layer_indices.empty()
                    ? 0U : *std::max_element(layer->layer_indices.begin(), layer->layer_indices.end());
                const auto direction = std::find_if(available.begin(), available.end(),
                    [&](const auto & value) { return value.layer_index == static_cast<int32_t>(diagnostic_layer); });
                if (direction != available.end() && direction->layer_index + 1 < static_cast<int32_t>(provider->model_n_layers)) {
                    common_flydelta_behavior_delta delta;
                    delta.id = arm.arm_id + ":diagnostic";
                    delta.behavior_key = job.seed.behavior_key;
                    delta.capture_manifest_id = arm.context_ref;
                    delta.host_evidence_ref = job.seed.evidence_ref;
                    delta.model_profile_fingerprint = provider->model_profile_fingerprint;
                    delta.execution_context_fingerprint = job.seed.execution_context_fingerprint;
                    delta.capture_layout_revision = provider->capture_layout_revision;
                    delta.layer_index = direction->layer_index + 1;
                    delta.values = direction->values;
                    common_flydelta_representation_diagnostics diagnostics;
                    if (!common_flydelta_representation_diagnostics_from_captures(
                            *baseline_capture, *overlay_capture, delta,
                            4U * 1024U * 1024U, diagnostics, runner_error)) return false;
                    geometry.available = true;
                    geometry.measurement_layer = diagnostics.layer_index;
                    geometry.cosine = diagnostics.cosine;
                    geometry.progress = diagnostics.progress;
                    geometry.leakage = diagnostics.leakage;
                    geometry.shift_norm = diagnostics.shift_norm;
                }
            }
        }
        return arm_result.executed;
    };
    const common_flydelta_search_pipeline_batch_runner batch_runner = [provider, &job, &make_arm,
            &baseline_capture_arm_id](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_direction_candidate &,
            const std::vector<common_flydelta_intervention_region_candidate> & candidates,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_scale_geometry> & geometries,
            std::vector<bool> & geometry_available,
            std::string & runner_error) {
        common_flydelta_arm_batch_request batch;
        batch.batch_id = job.id + ":region";
        batch.wave_id = "whirlpool-region";
        batch.arms.reserve(candidates.size());
        for (size_t index = 0; index < candidates.size(); ++index) {
            common_flydelta_layer_candidate layer;
            layer.layer_indices = candidates[index].layer_indices;
            layer.total_scale = candidates[index].total_scale;
            common_flydelta_arm_request arm;
            if (!make_arm(current_fixture, &layer, candidates[index].total_scale, true,
                    batch.wave_id, index, false, arm, runner_error)) return false;
            batch.arms.push_back(std::move(arm));
        }
        common_flydelta_arm_batch_result batch_result;
        if (!daemon_flydelta_execute_batch(provider, batch, batch_result, runner_error) ||
                batch_result.arms.size() != candidates.size()) {
            if (runner_error.empty()) runner_error = "FlyDelta daemon region batch returned incomplete arms";
            return false;
        }
        trials.clear();
        margins.clear();
        geometries.clear();
        geometry_available.clear();
        trials.reserve(candidates.size());
        margins.reserve(candidates.size());
        geometries.reserve(candidates.size());
        geometry_available.reserve(candidates.size());
        for (size_t index = 0; index < candidates.size(); ++index) {
            const auto & arm = batch_result.arms[index];
            if (!arm.executed || arm.arm_id != batch.arms[index].arm_id) {
                runner_error = "FlyDelta daemon region batch returned an invalid arm";
                return false;
            }
            common_flydelta_counterfactual_trial trial;
            trial.executed = arm.executed;
            trial.host_evaluated = arm.host_evaluated;
            trial.verifier_known = arm.verifier_known;
            trial.passed = daemon_flydelta_arm_semantic_passed(arm);
            trial.quality = arm.quality;
            trial.overlay_applied = arm.overlay_applied;
            trial.intervention_count = batch.arms[index].layer_indices.size();
            trial.semantic_progress_observation = arm.semantic_progress_observation;
            trial.observed_decision_summary = arm.observed_decision_summary;
            trial.expected_decision_summary = arm.expected_decision_summary;
            trial.verifier_reason = arm.verifier_reason;
            trial.evidence_ref = arm.generation_ref;
            trials.push_back(std::move(trial));
            margins.push_back(arm.margin);
            common_flydelta_scale_geometry geometry;
            if (!baseline_capture_arm_id.empty() && !arm.capture_ref.empty()) {
                std::shared_ptr<const common_flydelta_hidden_state_capture> baseline_capture;
                std::shared_ptr<const common_flydelta_hidden_state_capture> overlay_capture;
                {
                    std::lock_guard<std::mutex> lock(provider->capture_mutex);
                    const auto baseline_it = provider->captures.find(baseline_capture_arm_id);
                    const auto overlay_it = provider->captures.find(arm.arm_id);
                    if (baseline_it != provider->captures.end()) baseline_capture = baseline_it->second;
                    if (overlay_it != provider->captures.end()) overlay_capture = overlay_it->second;
                }
                std::vector<common_flydelta_basis_direction> available;
                if (baseline_capture && overlay_capture &&
                        daemon_flydelta_parse_directions(*provider, batch.arms[index].intervention_ref,
                            available, runner_error)) {
                    const auto direction = std::find_if(available.begin(), available.end(),
                        [&](const auto & value) {
                            return value.layer_index == static_cast<int32_t>(
                                *std::max_element(candidates[index].layer_indices.begin(), candidates[index].layer_indices.end()));
                        });
                    if (direction != available.end() && direction->layer_index + 1 < static_cast<int32_t>(provider->model_n_layers)) {
                        common_flydelta_behavior_delta delta;
                        delta.id = arm.arm_id + ":diagnostic";
                        delta.behavior_key = job.seed.behavior_key;
                        delta.capture_manifest_id = batch.arms[index].context_ref;
                        delta.host_evidence_ref = job.seed.evidence_ref;
                        delta.model_profile_fingerprint = provider->model_profile_fingerprint;
                        delta.execution_context_fingerprint = job.seed.execution_context_fingerprint;
                        delta.capture_layout_revision = provider->capture_layout_revision;
                        delta.layer_index = direction->layer_index + 1;
                        delta.values = direction->values;
                        common_flydelta_representation_diagnostics diagnostics;
                        if (!common_flydelta_representation_diagnostics_from_captures(
                                *baseline_capture, *overlay_capture, delta,
                                4U * 1024U * 1024U, diagnostics, runner_error)) return false;
                        geometry.available = true;
                        geometry.measurement_layer = diagnostics.layer_index;
                        geometry.cosine = diagnostics.cosine;
                        geometry.progress = diagnostics.progress;
                        geometry.leakage = diagnostics.leakage;
                        geometry.shift_norm = diagnostics.shift_norm;
                    }
                }
            }
            geometries.push_back(std::move(geometry));
            geometry_available.push_back(geometries.back().available);
        }
        return true;
    };
    const auto guidance_config = provider->outcome_guidance_config;
    common_flydelta_outcome_guidance_adviser whirlpool_adviser;
    if (guidance_config.mode != common_flydelta_outcome_guidance_mode::off &&
            provider->lifecycle_store && !fixture.scope_fingerprint.empty() &&
            !fixture.model_profile_fingerprint.empty() &&
            !fixture.tokenizer_fingerprint.empty() && !fixture.template_fingerprint.empty() &&
            !fixture.baseline_deployment_fingerprint.empty() &&
            !fixture.outcome_guidance_verifier_revision.empty()) {
        common_flydelta_outcome_guidance_context base_context;
        base_context.scope_fingerprint = fixture.guidance_scope_fingerprint.empty()
            ? fixture.scope_fingerprint : fixture.guidance_scope_fingerprint;
        base_context.model_profile_fingerprint = fixture.model_profile_fingerprint;
        base_context.tokenizer_fingerprint = fixture.tokenizer_fingerprint;
        base_context.template_fingerprint = fixture.template_fingerprint;
        base_context.baseline_deployment_fingerprint = fixture.baseline_deployment_fingerprint;
        base_context.verifier_revision = fixture.outcome_guidance_verifier_revision;
        base_context.ruleset_revision = fixture.ruleset_revision;
        base_context.action_contract_fingerprint = fixture.action_contract_fingerprint;
        base_context.semantic_domain = fixture.semantic_domain;
        base_context.strategy = "whirlpool";
        base_context.estimator = "layer-region-prior-v2";
        const nlohmann::ordered_json context_identity = {
            {"scope", base_context.scope_fingerprint},
            {"model", base_context.model_profile_fingerprint},
            {"tokenizer", base_context.tokenizer_fingerprint},
            {"template", base_context.template_fingerprint},
            {"deployment", base_context.baseline_deployment_fingerprint},
            {"verifier", base_context.verifier_revision},
            {"ruleset", base_context.ruleset_revision},
            {"action_contract", base_context.action_contract_fingerprint},
        };
        const std::string context_bytes = context_identity.dump();
        const std::string context_fingerprint = "sha256:" +
            hash_sha256_hex(context_bytes.data(), context_bytes.size());
        const auto history_projection_started = std::chrono::steady_clock::now();
        std::vector<common_flydelta_outcome_guidance_observation> whirlpool_history;
        uint64_t history_watermark = 0;
        size_t history_rows_read = 0;
        size_t history_rows_rejected = 0;
        std::string history_failure;
        std::vector<common_learning_lifecycle_record> history_records;
        if (!provider->lifecycle_store->list_scope_with_watermark(
                job.seed.scope.namespace_id, job.seed.scope.project_id,
                job.seed.scope.session_id, history_records, history_watermark, error)) {
            history_failure = "lifecycle_history_read_failed:" + error;
            error.clear();
        } else {
            history_rows_read = history_records.size();
            nlohmann::ordered_json frozen_rows = nlohmann::ordered_json::array();
            bool snapshot_found = false;
            bool snapshot_invalid = false;
            for (const auto & record : history_records) {
                if (record.kind != common_learning_lifecycle_kind::flydelta_experiment ||
                        record.registration_order == 0) continue;
                try {
                    const auto payload = nlohmann::ordered_json::parse(
                        record.payload_json, nullptr, false);
                    if (!payload.is_object() || payload.value("record_type", "") !=
                            "outcome_guidance_history_snapshot" ||
                            payload.value("job_id", "") != job.id ||
                            payload.value("context_fingerprint", "") != context_fingerprint) continue;
                    if (payload.value("schema_version", 0) != 2 || !payload.contains("rows") ||
                            !payload["rows"].is_array()) { snapshot_invalid = true; break; }
                    const auto canonical = nlohmann::ordered_json{
                        {"builder_revision", "flydelta-outcome-history-snapshot-v2"},
                        {"context_fingerprint", context_fingerprint},
                        {"source_watermark", payload.value("source_watermark", uint64_t{0})},
                        {"rows", payload["rows"]},
                    };
                    const std::string bytes = canonical.dump();
                    const std::string expected = "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
                    if (payload.value("snapshot_fingerprint", std::string()) != expected ||
                            record.content_hash != expected) { snapshot_invalid = true; break; }
                    frozen_rows = payload["rows"];
                    history_watermark = payload.value("source_watermark", uint64_t{0});
                    snapshot_found = true;
                    break;
                } catch (const std::exception &) {
                    ++history_rows_rejected;
                }
            }
            if (!snapshot_found && !snapshot_invalid) {
                for (const auto & record : history_records) {
                    if (record.kind != common_learning_lifecycle_kind::flydelta_result ||
                            record.registration_order == 0) continue;
                    try {
                        const auto payload = nlohmann::ordered_json::parse(
                            record.payload_json, nullptr, false);
                        if (!payload.is_object() || payload.value("record_type", "") !=
                                "candidate_search_decision" || !payload.contains("outcome_guidance") ||
                                !payload["outcome_guidance"].is_object()) continue;
                        frozen_rows.push_back({
                            {"outcome_guidance", payload["outcome_guidance"]},
                            {"outcome", payload.value("outcome", "unknown")},
                            {"verifier_known", payload.value("verifier_known", false)},
                            {"source_registration_order", record.registration_order},
                        });
                    } catch (const std::exception &) {
                        ++history_rows_rejected;
                    }
                }
                const auto canonical = nlohmann::ordered_json{
                    {"builder_revision", "flydelta-outcome-history-snapshot-v2"},
                    {"context_fingerprint", context_fingerprint},
                    {"source_watermark", history_watermark}, {"rows", frozen_rows},
                };
                const std::string bytes = canonical.dump();
                const std::string fingerprint = "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
                common_learning_lifecycle_record snapshot;
                snapshot.event_id = job.id + ":outcome-guidance-snapshot:" + context_fingerprint;
                snapshot.idempotency_key = snapshot.event_id;
                snapshot.subject_id = job.id;
                snapshot.kind = common_learning_lifecycle_kind::flydelta_experiment;
                snapshot.status = common_learning_lifecycle_status::observed;
                snapshot.source_id = "daemon-flydelta-outcome-guidance-snapshot";
                snapshot.namespace_id = job.seed.scope.namespace_id;
                snapshot.project_id = job.seed.scope.project_id;
                snapshot.session_id = job.seed.scope.session_id;
                snapshot.content_hash = fingerprint;
                snapshot.created_at = std::to_string(std::time(nullptr));
                snapshot.payload_json = nlohmann::ordered_json{
                    {"record_type", "outcome_guidance_history_snapshot"},
                    {"schema_version", 2}, {"job_id", job.id},
                    {"context_fingerprint", context_fingerprint},
                    {"source_watermark", history_watermark},
                    {"snapshot_fingerprint", fingerprint}, {"rows", frozen_rows},
                }.dump();
                if (!provider->lifecycle_store->append(snapshot, error)) {
                    history_failure = "history_snapshot_persist_failed:" + error;
                    error.clear();
                }
            }
            if (snapshot_invalid) history_failure = "frozen_history_snapshot_fingerprint_invalid";
            if (history_failure.empty()) {
                for (const auto & row : frozen_rows) {
                    try {
                        common_flydelta_outcome_guidance_observation observation;
                        std::string parse_error;
                        if (!common_flydelta_outcome_guidance_observation_from_json(row.dump(), observation, parse_error) ||
                                observation.context.scope_fingerprint != base_context.scope_fingerprint) {
                            ++history_rows_rejected; continue;
                        }
                        whirlpool_history.push_back(std::move(observation));
                    } catch (const std::exception &) {
                        ++history_rows_rejected;
                    }
                }
            }
        }
        common_flydelta_outcome_guidance_projection_ptr history_projection;
        bool history_projection_cache_hit = false;
        if (history_failure.empty() && job.seed.fixture_group_id.empty())
            history_failure = "fixture_group_identity_unavailable";
        if (history_failure.empty()) {
            try {
                if (!provider->outcome_projection_cache->get_or_build(
                        job.id + ":" + std::to_string(history_watermark),
                        base_context, whirlpool_history, job.seed.fixture_group_id,
                        history_projection, error, &history_projection_cache_hit)) history_failure = "history_projection_invalid:" + error;
            } catch (...) { history_failure = "history_projection_exception"; }
            error.clear();
        }
        const double projection_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - history_projection_started).count();
        auto whirlpool_policy_cursor = std::make_shared<size_t>(0);
        auto whirlpool_persisted_policy = std::make_shared<
            std::map<std::string, std::pair<size_t, size_t>>>();
        for (const auto & record : history_records) {
            if (record.kind != common_learning_lifecycle_kind::flydelta_experiment) continue;
            try {
                const auto payload = nlohmann::ordered_json::parse(record.payload_json);
                if (payload.value("record_type", "") != "outcome_guidance_decision" ||
                        payload.value("job_id", "") != job.id ||
                        payload.value("schema_version", 0) != 4) continue;
                const auto stage = payload.value("stage", std::string());
                if (stage != "whirlpool_initial_centre" && stage != "whirlpool_probe") continue;
                const auto start = payload.at("policy_sequence_start").get<size_t>();
                if (start >= 4) continue;
                (*whirlpool_persisted_policy)[record.event_id] = {
                    start, payload.at("policy_cursor_before").get<size_t>()};
            } catch (const std::exception &) { ++history_rows_rejected; }
        }
        uint64_t whirlpool_policy_origin = guidance_config.seed;
        for (const unsigned char ch : job.id) {
            whirlpool_policy_origin ^= ch;
            whirlpool_policy_origin *= 1099511628211ULL;
        }
        whirlpool_adviser = [provider, job, base_context, history_projection,
                guidance_config, context_fingerprint, history_watermark, history_rows_read,
                history_rows_rejected, whirlpool_policy_cursor, whirlpool_persisted_policy,
                whirlpool_policy_origin,
                history_failure, projection_ms, history_projection_cache_hit](const common_flydelta_outcome_guidance_request & request,
                common_flydelta_outcome_guidance_decision & decision, std::string & adviser_error) {
            if (request.schema_version != 1 || request.stage.empty() || request.decision_id.empty() ||
                    request.selection_limit > request.candidates.size()) {
                adviser_error = "invalid_typed_guidance_request";
                return false;
            }
            auto context = base_context;
            if (request.stage == "whirlpool_initial_centre" || request.stage == "whirlpool_probe") {
                context.strategy = "whirlpool";
                context.estimator = "layer-region-prior-v2";
            }
            uint64_t request_hash = 1469598103934665603ULL;
            for (const unsigned char ch : job.id + ":" + request.stage + ":" +
                    request.decision_id + ":" + std::to_string(request.round)) {
                request_hash ^= ch;
                request_hash *= 1099511628211ULL;
            }
            const uint64_t seed = guidance_config.seed ^ request_hash;
            const std::string candidate_fingerprint =
                common_flydelta_outcome_guidance_pool_fingerprint(request.candidates);
            const std::string stable_id = job.id + ":" + request.stage + ":" +
                std::to_string(request.phase_id) + ":" + std::to_string(request.round) + ":" +
                request.decision_id;
            const std::string event_id = stable_id + ":" + candidate_fingerprint + ":limit:" +
                std::to_string(request.selection_limit);
            size_t policy_cursor_before = *whirlpool_policy_cursor;
            size_t policy_start = common_flydelta_guidance_policy_start(
                whirlpool_policy_origin, policy_cursor_before, request.policy_sequence_start);
            const auto persisted_policy = whirlpool_persisted_policy->find(event_id);
            if (persisted_policy != whirlpool_persisted_policy->end()) {
                policy_cursor_before = persisted_policy->second.second;
                policy_start = common_flydelta_guidance_policy_start(
                    whirlpool_policy_origin, policy_cursor_before, request.policy_sequence_start,
                    persisted_policy->second.first);
            }
            if (!history_failure.empty()) adviser_error = history_failure;
            const auto started = std::chrono::steady_clock::now();
            const bool built = history_failure.empty() && common_flydelta_outcome_guidance_decide_from_projection(
                guidance_config, context, history_projection, request.candidates, request.selection_limit,
                seed, decision, adviser_error,
                common_flydelta_outcome_guidance_policy::posterior_sampling, policy_start);
            if (!built) {
                decision = {};
                decision.fallback_reason = adviser_error.empty() ? "guidance_unavailable" : adviser_error;
                for (size_t i = 0; i < request.candidates.size(); ++i) {
                    decision.original_order.push_back(i);
                    decision.shadow_order.push_back(i);
                    decision.mixed_policy_order.push_back(i);
                    decision.selected_order.push_back(i);
                    decision.selection_reasons.push_back("fallback_existing_order");
                    decision.mixed_policy_reasons.push_back("fallback_existing_order");
                }
                adviser_error.clear();
            }
            decision.stage = request.stage;
            decision.phase_id = request.phase_id;
            decision.round = request.round;
            decision.decision_id = request.decision_id;
            decision.shadow_mode = guidance_config.mode == common_flydelta_outcome_guidance_mode::shadow;
            nlohmann::ordered_json decision_identity = {
                {"history", decision.history_fingerprint},
                {"context", context_fingerprint},
                {"candidate_pool", candidate_fingerprint},
                {"selected", decision.selected_order},
                {"posterior", decision.shadow_order},
                {"mixed", decision.mixed_policy_order},
                {"seed", seed}, {"mode", common_flydelta_outcome_guidance_mode_name(
                    guidance_config.mode)},
                {"harm_penalty", guidance_config.harm_penalty},
                {"policy_sequence_start", policy_start},
            };
            nlohmann::ordered_json prediction_identity = nlohmann::ordered_json::array();
            for (const auto & prediction : decision.predictions) prediction_identity.push_back({
                {"arm_id", prediction.arm_id}, {"level", prediction.level},
                {"p_known", prediction.p_known},
                {"p_helped", prediction.p_helped_given_known},
                {"p_neutral", prediction.p_neutral_given_known},
                {"p_harmed", prediction.p_harmed_given_known},
                {"utility", prediction.expected_utility},
            });
            decision_identity["predictions"] = std::move(prediction_identity);
            const std::string decision_bytes = decision_identity.dump();
            const std::string decision_fingerprint = "sha256:" +
                hash_sha256_hex(decision_bytes.data(), decision_bytes.size());
            if (provider->lifecycle_store) {
                common_flydelta_lifecycle_event_context event;
                event.event_id = event_id;
                event.idempotency_key = event.event_id;
                event.source_id = "daemon-flydelta-outcome-guidance";
                event.scope = job.seed.scope;
                event.content_hash = decision_fingerprint;
                event.created_at = std::to_string(std::time(nullptr));
                std::string persist_error;
                if (!common_flydelta_append_outcome_guidance_decision_lifecycle(
                        *provider->lifecycle_store, event, job.id, guidance_config.mode, seed,
                        request.phase_id, request.round, request.selection_limit, policy_start,
                        policy_cursor_before,
                        std::min(request.selection_limit, request.candidates.size()),
                        context, request.candidates, decision, persist_error)) {
                    decision.fallback_reason = "lifecycle_write_failed:" + persist_error;
                    decision.selected_order = decision.original_order;
                } else {
                    (*whirlpool_persisted_policy)[event_id] = {policy_start, policy_cursor_before};
                    *whirlpool_policy_cursor = std::max(*whirlpool_policy_cursor,
                        policy_cursor_before + std::min(request.selection_limit, request.candidates.size()));
                }
            }
            nlohmann::ordered_json trace = {
                {"event", "flydelta_outcome_guidance"}, {"schema_version", 3},
                {"stage", request.stage}, {"decision_id", request.decision_id},
                {"phase_id", request.phase_id}, {"round", request.round}, {"job_id", job.id},
                {"mode", common_flydelta_outcome_guidance_mode_name(guidance_config.mode)},
                {"history_snapshot", context_fingerprint}, {"history_watermark", history_watermark},
                {"read_history_rows", history_rows_read},
                {"compatible_history_rows", decision.compatible_history_rows},
                {"deduplicated_history_rows", decision.deduplicated_history_rows},
                {"rejected_history_rows", history_rows_rejected +
                    decision.rejected_history_rows},
                {"history_projection_ms", projection_ms},
                {"history_projection_cache_hit", history_projection_cache_hit},
                {"history_fallback_reason", history_failure},
                {"selection_limit", request.selection_limit}, {"candidate_count", request.candidates.size()},
                {"seed", seed}, {"policy_sequence_start", policy_start},
                {"history_fingerprint", decision.history_fingerprint},
                {"original_order", decision.original_order}, {"posterior_order", decision.shadow_order},
                {"hypothetical_order", decision.mixed_policy_order},
                {"selected_order", decision.selected_order}, {"fallback_reason", decision.fallback_reason},
                {"predictions", nlohmann::ordered_json::array()},
                {"candidate_pool", nlohmann::ordered_json::array()},
            };
            for (const auto & candidate : request.candidates) trace["candidate_pool"].push_back({
                {"arm_id", candidate.arm_id}, {"rank", candidate.rank},
                {"relative_layer", candidate.relative_layer}, {"relative_dose", candidate.relative_dose},
                {"basis_identity", candidate.basis_identity},
            });
            trace["execution_frontier"] = nlohmann::ordered_json::array();
            for (size_t rank = 0; rank < decision.selected_order.size(); ++rank) {
                const size_t index = decision.selected_order[rank];
                if (index >= request.candidates.size()) continue;
                trace["execution_frontier"].push_back({
                    {"arm_id", request.candidates[index].arm_id},
                    {"selected", rank < request.selection_limit},
                    {"reason", rank < request.selection_limit ? "within_selection_limit" : "outside_selection_limit"},
                });
            }
            for (const auto & prediction : decision.predictions) trace["predictions"].push_back({
                {"arm_id", prediction.arm_id}, {"level", prediction.level},
                {"p_known", prediction.p_known}, {"p_helped", prediction.p_helped_given_known},
                {"p_neutral", prediction.p_neutral_given_known}, {"p_harmed", prediction.p_harmed_given_known},
                {"expected_utility", prediction.expected_utility},
                {"progress", nlohmann::ordered_json::parse(common_flydelta_outcome_guidance_progress_prediction_to_json(prediction))},
                {"fixture_groups", prediction.fixture_groups},
                {"backoff_groups", prediction.backoff_fixture_groups},
            });
            trace["prediction_and_ordering_ms"] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            std::clog << "[flydelta] outcome_guidance " << trace.dump() << '\n';
            return true;
        };
    }
    auto frozen_pipeline = config;
    frozen_pipeline.whirlpool_global_probes_per_round = job.whirlpool_global_probes_per_round;
    if (!common_flydelta_run_search_pipeline_batched(
            fixture, frozen_pipeline, {input}, runner, batch_runner, result, error,
            whirlpool_adviser)) return false;

    // Whirlpool/region search ranks compact diagnostic observations first.
    // Only the bounded frontier is re-run with generation and host
    // verification. The bounded selector may reserve one slot for a
    // diagnostically admissible global exploration candidate.
    constexpr size_t full_generation_top_k = 3;
    const auto frontier = common_flydelta_select_region_frontier(result, full_generation_top_k);
    for (const auto & location : frontier) {
        result.directions[location.direction_index].region_trials[location.trial_index].frontier_selection_reason = location.reason;
    }
    if (frontier.empty()) {
        result.search_status = common_flydelta_search_status::no_useful_utility;
        return true;
    }

    common_flydelta_arm_batch_request full_batch;
    full_batch.batch_id = job.id + ":region-full-generation";
    full_batch.wave_id = "whirlpool-region-full-generation";
    common_flydelta_arm_request baseline;
    if (!make_arm(fixture, nullptr, 0.0f, false, full_batch.wave_id, 0, true,
            baseline, error)) return false;
    baseline.layer_indices = capture_layers;
    baseline.coefficients.assign(capture_layers.size(), 0.0f);
    if (!common_flydelta_arm_request_validate(baseline, error)) return false;
    full_batch.arms.push_back(baseline);
    for (size_t index = 0; index < frontier.size(); ++index) {
        const auto & selected = result.directions[frontier[index].direction_index].region_trials[
            frontier[index].trial_index];
        common_flydelta_layer_candidate layer;
        layer.layer_indices = selected.candidate.layer_indices;
        layer.total_scale = selected.candidate.total_scale;
        common_flydelta_arm_request arm;
        if (!make_arm(fixture, &layer, selected.candidate.total_scale, true,
                full_batch.wave_id, index + 1, true, arm, error)) return false;
        full_batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result full_result;
    if (!daemon_flydelta_execute_batch(provider, full_batch, full_result, error) ||
            full_result.arms.size() != full_batch.arms.size()) {
        if (error.empty()) error = "FlyDelta region full-generation batch returned incomplete arms";
        return false;
    }
    const auto & baseline_arm = full_result.arms.front();
    common_flydelta_counterfactual_trial baseline_trial;
    baseline_trial.executed = baseline_arm.executed;
    baseline_trial.host_evaluated = baseline_arm.host_evaluated;
    baseline_trial.overlay_applied = baseline_arm.overlay_applied;
    baseline_trial.passed = daemon_flydelta_arm_semantic_passed(baseline_arm);
    baseline_trial.quality = baseline_arm.quality;
    baseline_trial.verifier_known = baseline_arm.verifier_known;
    baseline_trial.semantic_progress_observation = baseline_arm.semantic_progress_observation;
    baseline_trial.observed_decision_summary = baseline_arm.observed_decision_summary;
    baseline_trial.expected_decision_summary = baseline_arm.expected_decision_summary;
    baseline_trial.verifier_reason = baseline_arm.verifier_reason;
    baseline_trial.evidence_ref = baseline_arm.generation_ref;
    if (!common_flydelta_counterfactual_trial_validate(baseline_trial, error)) return false;
    for (size_t index = 0; index < frontier.size(); ++index) {
        const auto location = frontier[index];
        auto & trial = result.directions[location.direction_index].region_trials[location.trial_index];
        const auto & arm = full_result.arms[index + 1];
        common_flydelta_counterfactual_trial candidate_trial;
        candidate_trial.executed = arm.executed;
        candidate_trial.host_evaluated = arm.host_evaluated;
        candidate_trial.overlay_applied = arm.overlay_applied;
        candidate_trial.passed = daemon_flydelta_arm_semantic_passed(arm);
        candidate_trial.quality = arm.quality;
        candidate_trial.verifier_known = arm.verifier_known;
        candidate_trial.semantic_progress_observation = arm.semantic_progress_observation;
        candidate_trial.observed_decision_summary = arm.observed_decision_summary;
        candidate_trial.expected_decision_summary = arm.expected_decision_summary;
        candidate_trial.verifier_reason = arm.verifier_reason;
        candidate_trial.evidence_ref = arm.generation_ref;
        if (!common_flydelta_counterfactual_trial_validate(candidate_trial, error)) return false;
        trial.outcome = common_flydelta_classify_counterfactual(baseline_trial, candidate_trial);
        trial.semantic_progress = common_flydelta_compare_semantic_progress(
            baseline_trial.semantic_progress_observation,
            candidate_trial.semantic_progress_observation,
            baseline_trial.passed, candidate_trial.passed);
        trial.quality_delta = candidate_trial.quality - baseline_trial.quality;
        trial.margin = arm.margin;
        trial.margin_comparison.available = baseline_arm.margin.available && arm.margin.available;
        trial.margin_comparison.baseline = baseline_arm.margin;
        trial.margin_comparison.candidate = arm.margin;
        trial.executed = candidate_trial.executed;
        trial.overlay_applied = candidate_trial.overlay_applied;
        trial.full_generation_completed = baseline_trial.executed &&
            candidate_trial.executed && baseline_trial.host_evaluated &&
            candidate_trial.host_evaluated;
        // This loop iterates only over the bounded full-generation frontier,
        // but the result still comes from the arm's actual execution metadata.
        trial.host_evaluated = candidate_trial.host_evaluated;
        trial.verifier_known = baseline_trial.verifier_known && candidate_trial.verifier_known;
        trial.candidate_observation = candidate_trial;
        trial.baseline_observation = baseline_trial;
        trial.semantic_progress_observation = candidate_trial.semantic_progress_observation;
        trial.observed_decision_summary = candidate_trial.observed_decision_summary;
        trial.expected_decision_summary = candidate_trial.expected_decision_summary;
        trial.verifier_reason = candidate_trial.verifier_reason;
        trial.evidence_ref = candidate_trial.evidence_ref;
    }
    result.selection = {};
    for (size_t direction_index = 0; direction_index < result.directions.size(); ++direction_index) {
        auto & direction = result.directions[direction_index];
        direction.region_selection = {};
        for (size_t trial_index = 0; trial_index < direction.region_trials.size(); ++trial_index) {
            const auto & trial = direction.region_trials[trial_index];
            if (trial.outcome != common_flydelta_counterfactual_outcome::helped ||
                    !trial.executed || !trial.verifier_known) continue;
            if (!direction.region_selection.selected ||
                    trial.quality_delta > direction.region_selection.score) {
                direction.region_selection = {true, trial_index, trial.quality_delta};
            }
            const bool better = !result.selection.selected ||
                trial.quality_delta > result.selection.score ||
                (trial.quality_delta == result.selection.score &&
                 trial.candidate.layer_indices.size() <
                    result.directions[result.selection.direction_index].region_trials[
                        result.selection.region_trial_index].candidate.layer_indices.size());
            if (better) {
                result.selection.selected = true;
                result.selection.intervention_region = true;
                result.selection.direction_index = direction_index;
                result.selection.region_trial_index = trial_index;
                result.selection.layer_result_index = 0;
                result.selection.scale = trial.candidate.total_scale;
                result.selection.score = trial.quality_delta;
            }
        }
    }
    result.search_status = result.selection.selected
        ? common_flydelta_search_status::candidate_available
        : common_flydelta_search_status::no_useful_utility;
    if (provider->lifecycle_store) {
        common_flydelta_lifecycle_event_context lifecycle_context;
        lifecycle_context.event_id = job.id + ":whirlpool-verified-search-results";
        lifecycle_context.idempotency_key = lifecycle_context.event_id;
        lifecycle_context.source_id = "daemon-flydelta-whirlpool-search-results";
        lifecycle_context.scope = job.seed.scope;
        lifecycle_context.content_hash = job.seed.candidate_ref + ":" +
            std::to_string(result.selection.selected ? result.selection.region_trial_index : 0);
        lifecycle_context.created_at = std::to_string(std::time(nullptr));
        std::string lifecycle_error;
        if (!common_flydelta_append_search_pipeline_lifecycle(
                *provider->lifecycle_store, lifecycle_context, fixture, result,
                job.seed.candidate_ref, lifecycle_error)) {
            std::clog << "[flydelta] whirlpool_result_persistence_failed job="
                << job.id << " error=" << lifecycle_error << '\n';
        }
    }
    return true;
}

bool daemon_flydelta_run_post_bootstrap_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_plan & plan,
        common_flydelta_search_pipeline_result & output,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    fixture.scope_fingerprint = daemon_flydelta_scope_fingerprint(job.seed.scope);
    fixture.guidance_scope_fingerprint = common_flydelta_outcome_guidance_scope_fingerprint(job.seed.scope);
    fixture.semantic_domain = job.seed.behavior_key.empty()
        ? common_adaptation_evidence_source_name(job.seed.source)
        : job.seed.behavior_key;
    fixture.model_n_layers = provider->model_n_layers;
    const std::string baseline_identity = "flydelta-base-deployment-v1\n" +
        job.seed.model_profile_fingerprint + "\n" + job.seed.tokenizer_fingerprint + "\n" +
        job.seed.template_fingerprint + "\n" + std::to_string(provider->model_n_layers);
    fixture.baseline_deployment_fingerprint = "sha256:" +
        hash_sha256_hex(baseline_identity.data(), baseline_identity.size());
    if (fixture.outcome_guidance_verifier_revision.empty()) {
        fixture.outcome_guidance_verifier_revision = daemon_flydelta_verifier_contract_revision(
            provider->oracle_registry, job.seed.verifier_ref);
    }
    const int32_t layer_index = static_cast<int32_t>(plan.continuation.region.anchor_layer_index);
    std::vector<common_flydelta_direction_candidate> candidates;
    const size_t max_rank = plan.phase == common_flydelta_experiment_phase::shallow_controls
        ? 2 : 4;
    common_flydelta_low_rank_basis basis;
    json resume_anchor;
    if (!plan.search_anchor_ref.empty()) {
        if (plan.phase != common_flydelta_experiment_phase::deep_controls ||
                !daemon_flydelta_read_json_bounded(*provider, plan.search_anchor_ref,
                    16U * 1024U * 1024U, resume_anchor, error)) {
            if (error.empty()) error = "FlyDelta search anchor requires the Deep continuation phase";
            return false;
        }
        try {
            if (resume_anchor.at("kind") != "flydelta_search_anchor" || resume_anchor.at("schema_version") != 1) {
                error = "FlyDelta search anchor kind or version is invalid"; return false;
            }
            basis.layer_index = resume_anchor.at("layer_index").get<int32_t>();
            basis.dimension = resume_anchor.at("dimension").get<size_t>();
            basis.vectors = resume_anchor.at("basis_vectors").get<std::vector<std::vector<float>>>();
            if (basis.layer_index != layer_index || basis.dimension != provider->model_n_embd ||
                    !common_flydelta_low_rank_basis_validate(basis, max_rank, error) ||
                    resume_anchor.at("basis_fingerprint") != common_flydelta_coefficient_basis_identity(basis)) {
                if (error.empty()) error = "FlyDelta search anchor basis is incompatible";
                return false;
            }
            for (const auto & values : basis.vectors) {
                common_flydelta_direction_candidate candidate;
                candidate.kind = common_flydelta_direction_kind::raw_repair;
                candidate.layer_index = basis.layer_index;
                candidate.values = values;
                candidate.origin = "persisted_experimental_anchor";
                candidate.extraction_id = plan.search_anchor_ref;
                candidate.source_samples = candidate.retained_samples = 1;
                // This is restored basis material, not a new alignment measurement.
                candidate.median_alignment = 0.0f;
                candidate.experimental_only = true;
                if (!common_flydelta_direction_candidate_validate(candidate, basis.dimension, error)) return false;
                candidates.push_back(std::move(candidate));
            }
        } catch (const std::exception & exception) {
            error = std::string("FlyDelta search anchor basis is malformed: ") + exception.what();
            return false;
        }
    } else {
        if (!daemon_flydelta_resolve_evidence_direction_candidates(
                provider, job, layer_index, 2, candidates, error)) return false;
        if (candidates.size() > max_rank) candidates.resize(max_rank);

        // Existing post-Bootstrap directions are already HELPED and learning-
        // eligible evidence. Use the semantic resolver as an admission filter so
        // only directions from the same concept/behavior/surface/Oracle revision
        // can enter the ordinary low-rank basis. This does not admit experimental
        // concept candidates and does not change the existing search budget.
        common_flydelta_semantic_basis_query basis_query;
        basis_query.source = job.seed.source;
        basis_query.concept_key = job.seed.teaching_key.empty()
            ? job.seed.behavior_key : job.seed.teaching_key;
        basis_query.behavior_key = job.seed.behavior_key;
        basis_query.model_profile_fingerprint = job.seed.model_profile_fingerprint;
        basis_query.tokenizer_fingerprint = job.seed.tokenizer_fingerprint;
        basis_query.template_fingerprint = job.seed.template_fingerprint;
        basis_query.capture_layout_revision = provider->capture_layout_revision;
        basis_query.scope_fingerprint = daemon_flydelta_scope_fingerprint(job.seed.scope);
        basis_query.oracle_ref = job.seed.verifier_ref;
        basis_query.oracle_revision = job.seed.verifier_ref;
        basis_query.max_rank = max_rank;
        std::vector<common_flydelta_semantic_direction_candidate> semantic_candidates;
        semantic_candidates.reserve(candidates.size());
        for (const auto & candidate : candidates) {
            common_flydelta_synthesis_candidate_descriptor identity;
            identity.candidate_ref = candidate.extraction_id;
            identity.concept_key = basis_query.concept_key;
            identity.behavior_key = basis_query.behavior_key;
            identity.source_ref = job.seed.evidence_ref;
            identity.oracle_ref = basis_query.oracle_ref;
            identity.oracle_revision = basis_query.oracle_revision;
            identity.source = basis_query.source;
            identity.model_profile_fingerprint = basis_query.model_profile_fingerprint;
            identity.tokenizer_fingerprint = basis_query.tokenizer_fingerprint;
            identity.template_fingerprint = basis_query.template_fingerprint;
            identity.capture_layout_revision = basis_query.capture_layout_revision;
            identity.scope_fingerprint = basis_query.scope_fingerprint;
            identity.host_verified = true;
            identity.experimental_only = false;
            common_flydelta_synthesis_candidate_descriptor descriptor;
            if (!common_flydelta_synthesis_candidate_descriptor_from_direction(
                        candidate, identity, descriptor, error)) return false;
            semantic_candidates.push_back({candidate, std::move(descriptor)});
        }
        std::vector<size_t> selected_candidate_indices;
        if (!common_flydelta_resolve_semantic_basis(
                    basis_query, semantic_candidates, basis,
                    selected_candidate_indices, error)) return false;
        std::vector<common_flydelta_direction_candidate> resolved_candidates;
        resolved_candidates.reserve(selected_candidate_indices.size());
        for (const size_t index : selected_candidate_indices) {
            resolved_candidates.push_back(candidates[index]);
        }
        candidates = std::move(resolved_candidates);
    }

    common_flydelta_coefficient_search_config coefficient_config;
    std::string effective_dose_mode = provider->dose_mode;
    std::string effective_calibration_ref = provider->dose_calibration_ref;
    std::string calibration_fingerprint;
    if (!plan.search_anchor_ref.empty()) {
        try {
            effective_dose_mode = resume_anchor.at("dose_mode").get<std::string>();
            effective_calibration_ref = resume_anchor.at("calibration_ref").get<std::string>();
        } catch (const std::exception & exception) {
            error = std::string("FlyDelta anchor dose identity is malformed: ") + exception.what();
            return false;
        }
    }
    coefficient_config.strategy = plan.run_tfo_lite
        ? common_flydelta_coefficient_search_strategy::tfo_lite
        : common_flydelta_coefficient_search_strategy::coordinate;
    if (effective_dose_mode == "direction_dose" && plan.run_tfo_lite) {
        error = "FlyDelta TFO-lite does not yet support direction_dose";
        return false;
    }
    if (effective_dose_mode == "direction_dose") {
        coefficient_config.experimental_dose_mode = common_flydelta_coefficient_dose_mode::direction_dose;
        if (!daemon_flydelta_load_dose_calibration(
                provider, fixture, layer_index, effective_calibration_ref, coefficient_config,
                calibration_fingerprint, error)) return false;
    } else if (effective_dose_mode != "raw_coefficients") {
        error = "FlyDelta coefficient dose mode is invalid";
        return false;
    }
    coefficient_config.max_candidates = plan.budget.max_coefficient_trials == 0
        ? 16 : std::min<size_t>(16, plan.budget.max_coefficient_trials);
    // Proposal waves may contain up to the resident parallel capacity. The
    // paired diagnostic runner reserves one physical slot for its baseline
    // and partitions candidates accordingly below.
    const size_t resident_parallelism = provider->host
        ? provider->host->context_key().n_parallel : 0;
    if (resident_parallelism > 0) {
        coefficient_config.max_batch_arms = std::min<size_t>(64, resident_parallelism);
    }
    coefficient_config.guidance_layer_index = layer_index;
    coefficient_config.guidance_model_layer_count = provider->model_n_layers;
    common_flydelta_outcome_guidance_context guidance_context;
    guidance_context.scope_fingerprint = fixture.guidance_scope_fingerprint.empty()
        ? fixture.scope_fingerprint : fixture.guidance_scope_fingerprint;
    guidance_context.model_profile_fingerprint = fixture.model_profile_fingerprint;
    guidance_context.tokenizer_fingerprint = fixture.tokenizer_fingerprint;
    guidance_context.template_fingerprint = fixture.template_fingerprint;
    guidance_context.baseline_deployment_fingerprint = fixture.baseline_deployment_fingerprint;
    guidance_context.verifier_revision = fixture.outcome_guidance_verifier_revision;
    guidance_context.ruleset_revision = fixture.ruleset_revision;
    guidance_context.action_contract_fingerprint = fixture.action_contract_fingerprint;
    guidance_context.semantic_domain = fixture.semantic_domain;
    guidance_context.strategy = common_flydelta_coefficient_search_strategy_name(
        coefficient_config.strategy);
    guidance_context.estimator = "coefficient-margin-geometry-v1";
    guidance_context.local_task_fingerprint = fixture.task_fingerprint;
    guidance_context.local_lineage_ref = job.seed.candidate_ref;
    const std::string progress_wave_material = nlohmann::ordered_json{{"job", job.id},
        {"phase", static_cast<int>(plan.phase)}, {"anchor", plan.search_anchor_ref},
        {"logical_used", plan.search_model_calls_used}}.dump();
    const std::string progress_wave_identity = "sha256:" + hash_sha256_hex(
        progress_wave_material.data(), progress_wave_material.size());
    coefficient_config.guidance_basis_identity =
        common_flydelta_coefficient_basis_identity(basis);
    std::vector<common_flydelta_outcome_guidance_observation> guidance_history;
    bool guidance_history_load_failed = false;
    size_t guidance_ineligible_history_rows = 0;
    size_t guidance_read_history_rows = 0;
    uint64_t guidance_history_watermark = 0;
    std::string guidance_history_failure_reason;
    std::vector<common_learning_lifecycle_record> frozen_qualification_records;
    bool matched_evaluation_recorded = false;
    double guidance_projection_ms = 0.0;
    const auto guidance_preparation_started = std::chrono::steady_clock::now();
    auto persisted_guidance_sequence_by_pool = std::make_shared<std::map<std::string, size_t>>();
    auto persisted_guidance_policy_start_by_pool = std::make_shared<std::map<std::string, size_t>>();
    auto persisted_guidance_policy_cursor_before_by_pool =
        std::make_shared<std::map<std::string, size_t>>();
    auto policy_cursor_committed_pools = std::make_shared<std::set<std::string>>();
    auto next_guidance_sequence_by_phase = std::make_shared<std::map<int, size_t>>();
    auto next_guidance_policy_cursor_by_phase = std::make_shared<std::map<int, size_t>>();
    const auto guidance_config = provider->outcome_guidance_config;
    const bool guidance_runtime_eligible =
        plan.phase == common_flydelta_experiment_phase::deep_controls ||
        plan.run_tfo_lite || coefficient_config.strategy ==
            common_flydelta_coefficient_search_strategy::tfo_lite;
    const bool guidance_identity_available = !job.seed.fixture_group_id.empty() &&
        !guidance_context.scope_fingerprint.empty() &&
        !guidance_context.model_profile_fingerprint.empty() &&
        !guidance_context.tokenizer_fingerprint.empty() &&
        !guidance_context.template_fingerprint.empty() &&
        !guidance_context.baseline_deployment_fingerprint.empty() &&
        !guidance_context.verifier_revision.empty() &&
        ((guidance_context.semantic_domain.find("placement_game") == std::string::npos &&
          guidance_context.semantic_domain.find("linear_algebra") == std::string::npos) ||
         (!guidance_context.ruleset_revision.empty() &&
          !guidance_context.action_contract_fingerprint.empty()));
    if (guidance_config.mode != common_flydelta_outcome_guidance_mode::off &&
            guidance_runtime_eligible && guidance_identity_available &&
            provider->lifecycle_store) {
        std::string history_error;
        std::vector<common_learning_lifecycle_record> records;
        if (!provider->lifecycle_store->list_scope_with_watermark(
                job.seed.scope.namespace_id, job.seed.scope.project_id,
                job.seed.scope.session_id, records, guidance_history_watermark,
                history_error)) {
            records.clear();
        }
        guidance_read_history_rows = records.size();
        if (history_error.empty()) {
            for (const auto & record : records) {
                if (record.kind != common_learning_lifecycle_kind::flydelta_experiment) continue;
                try {
                    const auto payload = nlohmann::ordered_json::parse(record.payload_json, nullptr, false);
                    if (!payload.is_object() || payload.value("record_type", "") !=
                            "outcome_guidance_decision" || payload.value("job_id", "") != job.id ||
                            (payload.value("schema_version", 0) != 2 &&
                             payload.value("schema_version", 0) != 3 &&
                             payload.value("schema_version", 0) != 4) ||
                            !payload.contains("candidate_pool_fingerprint")) continue;
                    const auto saved_stage = payload.value("stage", std::string());
                    if (saved_stage == "whirlpool_initial_centre" || saved_stage == "whirlpool_probe") {
                        continue; // WHERE owns its cursor; do not shift coefficient phases.
                    }
                    const int phase_id = payload.value("phase_id", -1);
                    const size_t sequence = payload.value("decision_sequence", size_t{0});
                    const std::string pool_fingerprint = payload.value(
                        "candidate_pool_fingerprint", std::string());
                    if (phase_id < 0 || pool_fingerprint.empty()) continue;
                    const size_t selection_limit = payload.value("selection_limit", size_t{0});
                    const std::string key = payload.value("schema_version", 0) >= 4
                        ? payload.value("stage", std::string("")) + ":" +
                            std::to_string(phase_id) + ":" +
                            std::to_string(payload.value("round", size_t{0})) + ":" +
                            payload.value("decision_id", std::string("")) + ":" +
                            pool_fingerprint + ":" + std::to_string(selection_limit)
                        : std::to_string(phase_id) + ":" + pool_fingerprint + ":" +
                            std::to_string(selection_limit);
                    (*persisted_guidance_sequence_by_pool)[key] = sequence;
                    if (payload.value("schema_version", 0) >= 3) {
                        const size_t policy_start = payload.value("policy_sequence_start", size_t{0}) % 4;
                        const size_t cursor_before = payload.value("policy_cursor_before", size_t{0});
                        const size_t policy_count = payload.value("policy_slots_consumed", size_t{0});
                        (*persisted_guidance_policy_start_by_pool)[key] = policy_start;
                        (*persisted_guidance_policy_cursor_before_by_pool)[key] = cursor_before;
                        (*next_guidance_policy_cursor_by_phase)[phase_id] = std::max(
                            (*next_guidance_policy_cursor_by_phase)[phase_id],
                            cursor_before + policy_count);
                        policy_cursor_committed_pools->insert(key);
                    }
                    (*next_guidance_sequence_by_phase)[phase_id] = std::max(
                        (*next_guidance_sequence_by_phase)[phase_id], sequence + 1);
                } catch (...) {
                    ++guidance_ineligible_history_rows;
                }
            }
            const nlohmann::ordered_json context_identity = {
                {"scope", guidance_context.scope_fingerprint},
                {"model", guidance_context.model_profile_fingerprint},
                {"tokenizer", guidance_context.tokenizer_fingerprint},
                {"template", guidance_context.template_fingerprint},
                {"deployment", guidance_context.baseline_deployment_fingerprint},
                {"verifier", guidance_context.verifier_revision},
                {"ruleset", guidance_context.ruleset_revision},
                {"action_contract", guidance_context.action_contract_fingerprint},
                {"search_wave", progress_wave_identity},
            };
            const std::string context_bytes = context_identity.dump();
            const std::string context_fingerprint = "sha256:" +
                hash_sha256_hex(context_bytes.data(), context_bytes.size());
            nlohmann::ordered_json frozen_rows;
            bool snapshot_found = false;
            bool snapshot_invalid = false;
            for (const auto & record : records) {
                if (record.kind != common_learning_lifecycle_kind::flydelta_experiment ||
                        record.registration_order == 0) continue;
                try {
                const auto payload = nlohmann::ordered_json::parse(record.payload_json, nullptr, false);
                if (!payload.is_object() || payload.value("record_type", "") !=
                        "outcome_guidance_history_snapshot" ||
                        payload.value("job_id", "") != job.id ||
                        payload.value("context_fingerprint", "") != context_fingerprint) continue;
                if (!payload.contains("rows") || !payload["rows"].is_array() ||
                        payload.value("schema_version", 0) != 2) {
                    snapshot_invalid = true;
                    break;
                }
                const uint64_t source_watermark = payload.value("source_watermark", uint64_t{0});
                const nlohmann::ordered_json canonical = {
                    {"builder_revision", "flydelta-outcome-history-snapshot-v2"},
                    {"context_fingerprint", context_fingerprint},
                    {"source_watermark", source_watermark},
                    {"rows", payload["rows"]},
                };
                const std::string canonical_bytes = canonical.dump();
                const std::string expected_fingerprint = "sha256:" +
                    hash_sha256_hex(canonical_bytes.data(), canonical_bytes.size());
                if (payload.value("snapshot_fingerprint", std::string()) != expected_fingerprint ||
                        record.content_hash != expected_fingerprint) {
                    snapshot_invalid = true;
                    break;
                }
                frozen_rows = payload["rows"];
                guidance_history_watermark = source_watermark;
                snapshot_found = true;
                break;
                } catch (...) {
                    ++guidance_ineligible_history_rows;
                }
            }
            const auto parse_guidance_row = [&](const nlohmann::ordered_json & g,
                    const std::string & outcome_value, const bool verifier_known,
                    common_flydelta_outcome_guidance_observation & observation) {
                std::string parse_error;
                const nlohmann::ordered_json row{{"outcome_guidance", g}, {"outcome", outcome_value},
                    {"verifier_known", verifier_known}};
                return common_flydelta_outcome_guidance_observation_from_json(row.dump(), observation, parse_error) &&
                    observation.context.scope_fingerprint == guidance_context.scope_fingerprint;
            };
            if (snapshot_invalid) {
                guidance_history_load_failed = true;
                guidance_history_failure_reason = "frozen_history_snapshot_fingerprint_invalid";
            } else if (snapshot_found && frozen_rows.is_array()) {
                for (const auto & row : frozen_rows) {
                    try {
                        common_flydelta_outcome_guidance_observation observation;
                        if (parse_guidance_row(row.value("outcome_guidance", nlohmann::ordered_json{}),
                                row.value("outcome", "unknown"), row.value("verifier_known", false),
                                observation) && !observation.stable_observation_id.empty()) {
                            guidance_history.push_back(std::move(observation));
                        } else ++guidance_ineligible_history_rows;
                    } catch (...) { ++guidance_ineligible_history_rows; }
                }
            } else if (!snapshot_found) {
                frozen_rows = nlohmann::ordered_json::array();
                for (const auto & record : records) {
                    if (record.kind != common_learning_lifecycle_kind::flydelta_result ||
                            record.registration_order == 0) continue;
                    try {
                        const auto payload = nlohmann::ordered_json::parse(record.payload_json, nullptr, false);
                        if (!payload.is_object() || payload.value("record_type", "") != "candidate_search_decision") continue;
                        if (!payload.contains("outcome_guidance") || !payload["outcome_guidance"].is_object()) {
                            ++guidance_ineligible_history_rows;
                            continue;
                        }
                        common_flydelta_outcome_guidance_observation observation;
                        if (!parse_guidance_row(payload["outcome_guidance"],
                                payload.value("outcome", "unknown"),
                                payload.value("verifier_known", false), observation)) {
                            ++guidance_ineligible_history_rows;
                            continue;
                        }
                        if (!observation.stable_observation_id.empty()) {
                            guidance_history.push_back(observation);
                            frozen_rows.push_back({
                                {"outcome_guidance", payload["outcome_guidance"]},
                                {"outcome", payload.value("outcome", "unknown")},
                                {"verifier_known", payload.value("verifier_known", false)},
                                {"source_registration_order", record.registration_order},
                            });
                        } else ++guidance_ineligible_history_rows;
                    } catch (...) { ++guidance_ineligible_history_rows; }
                }
                const nlohmann::ordered_json canonical = {
                    {"builder_revision", "flydelta-outcome-history-snapshot-v2"},
                    {"context_fingerprint", context_fingerprint},
                    {"source_watermark", guidance_history_watermark},
                    {"rows", frozen_rows},
                };
                const std::string canonical_bytes = canonical.dump();
                const std::string snapshot_fingerprint = "sha256:" +
                    hash_sha256_hex(canonical_bytes.data(), canonical_bytes.size());
                nlohmann::ordered_json snapshot_payload = {
                    {"record_type", "outcome_guidance_history_snapshot"},
                    {"schema_version", 2}, {"job_id", job.id},
                    {"context_fingerprint", context_fingerprint},
                    {"source_watermark", guidance_history_watermark},
                    {"snapshot_fingerprint", snapshot_fingerprint},
                    {"rows", frozen_rows},
                };
                common_learning_lifecycle_record snapshot;
                snapshot.event_id = job.id + ":outcome-guidance-snapshot:" + context_fingerprint;
                snapshot.idempotency_key = snapshot.event_id;
                snapshot.subject_id = job.id;
                snapshot.kind = common_learning_lifecycle_kind::flydelta_experiment;
                snapshot.status = common_learning_lifecycle_status::observed;
                snapshot.source_id = "daemon-flydelta-outcome-guidance-snapshot";
                snapshot.namespace_id = job.seed.scope.namespace_id;
                snapshot.project_id = job.seed.scope.project_id;
                snapshot.session_id = job.seed.scope.session_id;
                snapshot.content_hash = snapshot_fingerprint;
                snapshot.created_at = std::to_string(std::time(nullptr));
                snapshot.payload_json = snapshot_payload.dump();
                std::string snapshot_error;
                if (!provider->lifecycle_store->append(snapshot, snapshot_error)) {
                    guidance_history_load_failed = true;
                    guidance_history_failure_reason = "history_snapshot_persist_failed:" + snapshot_error;
                    guidance_history.clear();
                }
            }
            // Resume must not gain authority from qualifications appended
            // after this job's frozen history watermark.
            if (!guidance_history_load_failed) {
                for (const auto & row : records) {
                    if (row.event_id.find(job.id + ":deep-matched:") == 0) matched_evaluation_recorded = true;
                    if (row.kind == common_learning_lifecycle_kind::flydelta_experiment &&
                            row.subject_id == "deep_full_generation_selection" &&
                            row.registration_order > 0 && row.registration_order <= guidance_history_watermark) {
                        frozen_qualification_records.push_back(row);
                    }
                }
            }
        } else {
            guidance_history_load_failed = true;
            guidance_history_failure_reason = "lifecycle_history_read_failed";
        }
    } else if (guidance_config.mode != common_flydelta_outcome_guidance_mode::off &&
            guidance_runtime_eligible && !guidance_identity_available) {
        guidance_history_load_failed = true;
        guidance_history_failure_reason = job.seed.fixture_group_id.empty()
            ? "fixture_group_identity_unavailable" : "authoritative_guidance_identity_unavailable";
    }
    common_flydelta_outcome_guidance_projection_ptr guidance_projection;
    bool guidance_projection_cache_hit = false;
    if (guidance_config.mode != common_flydelta_outcome_guidance_mode::off &&
            guidance_runtime_eligible && !guidance_history_load_failed) {
        const auto frozen_identity = job.id + ":" + std::to_string(guidance_history_watermark);
        try {
            if (!provider->outcome_projection_cache->get_or_build(frozen_identity, guidance_context,
                    guidance_history, job.seed.fixture_group_id, guidance_projection, error, &guidance_projection_cache_hit)) {
                guidance_history_load_failed = true;
                guidance_history_failure_reason = "history_projection_invalid:" + error;
            }
        } catch (...) {
            guidance_history_load_failed = true;
            guidance_history_failure_reason = "history_projection_exception";
        }
        error.clear();
    }
    if (guidance_config.mode != common_flydelta_outcome_guidance_mode::off && guidance_runtime_eligible)
        guidance_projection_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - guidance_preparation_started).count();
    const common_flydelta_outcome_guidance_adviser outcome_adviser =
        guidance_config.mode == common_flydelta_outcome_guidance_mode::off ||
            !guidance_runtime_eligible
            ? common_flydelta_outcome_guidance_adviser{}
            : common_flydelta_outcome_guidance_adviser(
            [guidance_context, guidance_projection, guidance_config, guidance_projection_cache_hit,
            persisted_guidance_sequence_by_pool, persisted_guidance_policy_start_by_pool,
            persisted_guidance_policy_cursor_before_by_pool,
            policy_cursor_committed_pools,
            next_guidance_sequence_by_phase, next_guidance_policy_cursor_by_phase,
            lifecycle_store = provider->lifecycle_store, scope = job.seed.scope,
            projection_ms = guidance_projection_ms,
            history_watermark = guidance_history_watermark,
            phase_id = static_cast<int>(plan.phase),
            job_id = job.id, progress_wave_identity, history_load_failed = guidance_history_load_failed,
            history_failure_reason = guidance_history_failure_reason,
            frozen_qualification_records,
            read_history_rows = guidance_read_history_rows,
            ineligible_rows = guidance_ineligible_history_rows](
            const common_flydelta_outcome_guidance_request & request,
                common_flydelta_outcome_guidance_decision & decision,
                std::string & adviser_error) {
            const auto & pool = request.candidates;
            auto decision_context = guidance_context;
            decision_context.progress_baseline = request.progress_baseline;
            decision_context.local_parent = request.progress_parent;
            decision_context.local_parent_ref = request.progress_parent_ref;
            const size_t limit = request.selection_limit;
            const std::string stage = request.stage.empty() ? "coefficient_search" : request.stage;
            const int decision_phase_id = request.phase_id >= 0 ? request.phase_id : phase_id;
            const size_t round = request.round;
            const std::string decision_id = (request.decision_id.empty()
                ? stage + ":" + std::to_string(decision_phase_id) + ":" + std::to_string(round)
                : request.decision_id) + ":" + progress_wave_identity;
            double prediction_ms = 0.0;
            const std::string pool_fingerprint =
                common_flydelta_outcome_guidance_pool_fingerprint(pool);
            const std::string sequence_key = stage + ":" + std::to_string(decision_phase_id) + ":" +
                std::to_string(round) + ":" + decision_id + ":" + pool_fingerprint + ":" +
                std::to_string(limit);
            size_t wave_index = 0;
            const auto persisted = persisted_guidance_sequence_by_pool->find(sequence_key);
            if (persisted != persisted_guidance_sequence_by_pool->end()) {
                wave_index = persisted->second;
            } else {
                wave_index = (*next_guidance_sequence_by_phase)[phase_id]++;
                (*persisted_guidance_sequence_by_pool)[sequence_key] = wave_index;
            }
            uint64_t policy_origin = guidance_config.seed;
            for (const unsigned char ch : job_id) {
                policy_origin ^= ch;
                policy_origin *= 1099511628211ULL;
            }
            for (const unsigned char ch : stage) {
                policy_origin ^= ch;
                policy_origin *= 1099511628211ULL;
            }
            policy_origin ^= static_cast<uint64_t>(decision_phase_id);
            size_t policy_cursor_before = (*next_guidance_policy_cursor_by_phase)[phase_id];
            size_t policy_sequence_start = common_flydelta_guidance_policy_start(
                policy_origin, policy_cursor_before, request.policy_sequence_start);
            const auto persisted_policy_start =
                persisted_guidance_policy_start_by_pool->find(sequence_key);
            if (request.policy_sequence_start < 4) {
                policy_sequence_start = request.policy_sequence_start;
            } else if (persisted_policy_start != persisted_guidance_policy_start_by_pool->end()) {
                policy_sequence_start = common_flydelta_guidance_policy_start(
                    policy_origin, policy_cursor_before, request.policy_sequence_start, persisted_policy_start->second);
                const auto persisted_cursor =
                    persisted_guidance_policy_cursor_before_by_pool->find(sequence_key);
                if (persisted_cursor != persisted_guidance_policy_cursor_before_by_pool->end()) {
                    policy_cursor_before = persisted_cursor->second;
                }
            } else {
                (*persisted_guidance_policy_start_by_pool)[sequence_key] = policy_sequence_start;
                (*persisted_guidance_policy_cursor_before_by_pool)[sequence_key] = policy_cursor_before;
            }
            const size_t policy_slots_consumed = std::min(limit, pool.size());
            const bool deep_selection_point = stage == "deep_full_generation_frontier" ||
                stage == "coefficient_full_generation_frontier";
            bool deep_selection_qualified = false;
            std::string deep_selection_qualification_reason = "not_applicable";
            if (deep_selection_point && !history_load_failed) {
                common_flydelta_guidance_qualification_context context;
                context.authorized_scope = scope;
                context.outcome_context = guidance_context;
                context.selection_limit = limit;
                context.policy_revision = common_flydelta_guidance_qualification_policy_revision(guidance_config);
                context.progress_reward = guidance_config.progress_reward;
                context.harm_penalty = guidance_config.harm_penalty;
                common_flydelta_guidance_qualification_result restored;
                std::string qualification_error;
                if (!common_flydelta_guidance_find_qualification(frozen_qualification_records, context,
                        history_watermark, restored, deep_selection_qualified,
                        deep_selection_qualification_reason, qualification_error)) {
                    deep_selection_qualified = false;
                    deep_selection_qualification_reason = "qualification_record_invalid:" + qualification_error;
                }
            }
            common_flydelta_outcome_guidance_config effective_guidance_config = guidance_config;
            effective_guidance_config.mode = common_flydelta_guidance_effective_mode(
                guidance_config.mode,
                common_flydelta_guidance_decision_point::deep_full_generation_selection,
                !deep_selection_point || deep_selection_qualified);
            uint64_t wave_seed = 1469598103934665603ULL;
            for (const unsigned char ch : job_id) { wave_seed ^= ch; wave_seed *= 1099511628211ULL; }
            wave_seed ^= static_cast<uint64_t>(decision_phase_id);
            for (const unsigned char ch : stage) { wave_seed ^= ch; wave_seed *= 1099511628211ULL; }
            wave_seed ^= static_cast<uint64_t>(round);
            wave_seed *= 1099511628211ULL;
            wave_seed ^= static_cast<uint64_t>(wave_index);
            if (history_load_failed) {
                adviser_error = history_failure_reason.empty()
                    ? "lifecycle_history_read_failed" : history_failure_reason;
            }
            const auto prediction_started = std::chrono::steady_clock::now();
            if (history_load_failed || !common_flydelta_outcome_guidance_decide_from_projection(
                    effective_guidance_config, decision_context, guidance_projection, pool, limit,
                    wave_seed, decision, adviser_error,
                    common_flydelta_outcome_guidance_policy::posterior_sampling,
                    policy_sequence_start)) {
                decision = {};
                decision.fallback_reason = adviser_error;
                for (size_t i = 0; i < pool.size(); ++i) {
                    decision.original_order.push_back(i);
                    decision.shadow_order.push_back(i);
                    decision.mixed_policy_order.push_back(i);
                    decision.selected_order.push_back(i);
                    decision.selection_reasons.push_back("fallback_existing_order");
                    decision.mixed_policy_reasons.push_back("fallback_existing_order");
                }
                adviser_error.clear();
            }
            decision.stage = stage;
            decision.phase_id = decision_phase_id;
            decision.round = round;
            decision.decision_id = decision_id;
            decision.shadow_mode = effective_guidance_config.mode == common_flydelta_outcome_guidance_mode::shadow;
            prediction_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prediction_started).count();
            if (lifecycle_store) {
                common_flydelta_lifecycle_event_context event_context;
                const auto event_identity = nlohmann::ordered_json{{"job", job_id}, {"stage", stage},
                    {"phase", decision_phase_id}, {"round", round}, {"decision", decision_id}, {"sequence", wave_index},
                    {"history", decision.history_fingerprint}, {"pool", pool_fingerprint}, {"limit", limit}}.dump();
                event_context.event_id = job_id + ":outcome-guidance-v5:" + hash_sha256_hex(event_identity.data(), event_identity.size());
                event_context.idempotency_key = event_context.event_id;
                event_context.source_id = "daemon-flydelta-outcome-guidance";
                event_context.scope = scope;
                event_context.content_hash = decision.history_fingerprint + ":" +
                    std::to_string(wave_seed);
                const auto now = std::chrono::system_clock::now();
                const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                std::tm utc{};
                gmtime_r(&now_time, &utc);
                std::ostringstream timestamp;
                timestamp << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
                event_context.created_at = timestamp.str();
                std::string persist_error;
                const bool decision_persisted = common_flydelta_append_outcome_guidance_decision_lifecycle(
                    *lifecycle_store, event_context, job_id, guidance_config.mode,
                        wave_seed, decision_phase_id, wave_index, limit, policy_sequence_start,
                        policy_cursor_before, policy_slots_consumed, decision_context, pool,
                        decision, persist_error);
                if (!decision_persisted) {
                    // This callback runs before the arm wave. If the decision
                    // cannot be made durable, do not execute an unrecorded
                    // experimental order. Shadow mode already preserves this
                    // order, but make the fallback explicit in both modes.
                    decision.selected_order = decision.original_order;
                    decision.selection_reasons.assign(
                        decision.original_order.size(),
                        "fallback_decision_persistence_failed");
                    decision.fallback_reason = "lifecycle_write_failed:" + persist_error;
                } else if (decision.fallback_reason.empty() &&
                        policy_cursor_committed_pools->insert(sequence_key).second) {
                    (*next_guidance_policy_cursor_by_phase)[phase_id] += policy_slots_consumed;
                }
            } else {
                decision.selected_order = decision.original_order;
                decision.selection_reasons.assign(
                    decision.original_order.size(), "fallback_lifecycle_store_unavailable");
                decision.fallback_reason = "lifecycle_store_unavailable";
            }
            nlohmann::ordered_json trace = {
                {"event", "flydelta_outcome_guidance"},
                {"schema_version", 2},
                {"stage", "decision_before_execution"},
                {"search_stage", stage}, {"search_round", round},
                {"decision_id", decision_id},
                {"job_id", job_id},
                {"phase_id", decision_phase_id},
                {"seed", wave_seed},
                {"mode", common_flydelta_outcome_guidance_mode_name(guidance_config.mode)},
                {"effective_mode", common_flydelta_outcome_guidance_mode_name(
                    common_flydelta_guidance_effective_mode(guidance_config.mode,
                        common_flydelta_guidance_decision_point::deep_full_generation_selection,
                        !deep_selection_point || deep_selection_qualified))},
                {"qualification_decision_point", deep_selection_point
                    ? "deep_full_generation_selection" : "not_applicable"},
                {"qualification_capability", deep_selection_point
                    ? "select_budgeted_subset" : "not_applicable"},
                {"qualification_usable", deep_selection_point && deep_selection_qualified},
                {"qualification_reason", deep_selection_point
                    ? deep_selection_qualification_reason : "not_applicable"},
                {"history_fingerprint", decision.history_fingerprint},
                {"decision_sequence", wave_index},
                {"candidate_pool_fingerprint", pool_fingerprint},
                {"projection_ms", projection_ms},
                {"projection_cache_hit", guidance_projection_cache_hit},
                {"prediction_and_ordering_ms", prediction_ms},
                {"read_history_rows", read_history_rows},
                {"history_watermark", history_watermark},
                {"compatible_history_rows", decision.compatible_history_rows},
                {"deduplicated_history_rows", decision.deduplicated_history_rows},
                {"rejected_history_rows", decision.rejected_history_rows + ineligible_rows},
                {"ineligible_history_rows", ineligible_rows},
                {"candidate_count", pool.size()},
                {"selection_limit", limit},
                {"policy_sequence_start", policy_sequence_start},
                {"policy_cursor_before", policy_cursor_before},
                {"policy_slots_consumed", policy_slots_consumed},
                {"original_order", decision.original_order},
                {"selected_order", decision.selected_order},
                {"mixed_policy_order", decision.mixed_policy_order},
                {"mixed_policy_reasons", decision.mixed_policy_reasons},
                {"selection_reasons", decision.selection_reasons},
                {"shadow_order", decision.shadow_order},
                {"fallback_reason", decision.fallback_reason},
                {"candidate_pool", nlohmann::ordered_json::array()},
                {"predictions", nlohmann::ordered_json::array()},
            };
            for (const auto & candidate : pool) trace["candidate_pool"].push_back({
                {"arm_id", candidate.arm_id},
                {"basis_identity", candidate.basis_identity},
                {"rank", candidate.rank},
                {"relative_layer", candidate.relative_layer},
                {"relative_dose", candidate.relative_dose},
                {"mixed_coefficients", candidate.mixed_coefficients},
                {"diagnostic_margin_sign", candidate.diagnostic_margin_sign},
            });
            const auto arm_ids = [&pool](const std::vector<size_t> & order, size_t count) {
                auto ids = nlohmann::ordered_json::array();
                for (size_t i = 0; i < std::min(count, order.size()); ++i) {
                    ids.push_back(pool.at(order[i]).arm_id);
                }
                return ids;
            };
            trace["original_arm_ids"] = arm_ids(decision.original_order, pool.size());
            trace["posterior_ranked_arm_ids"] = arm_ids(decision.shadow_order, pool.size());
            trace["hypothetical_selected_arm_ids"] = arm_ids(decision.mixed_policy_order, limit);
            trace["selected_arm_ids"] = arm_ids(decision.selected_order, limit);
            trace["execution_frontier"] = nlohmann::ordered_json::array();
            for (size_t rank = 0; rank < decision.selected_order.size(); ++rank) {
                const size_t index = decision.selected_order[rank];
                if (index >= pool.size()) continue;
                trace["execution_frontier"].push_back({
                    {"arm_id", pool[index].arm_id},
                    {"selected", rank < limit},
                    {"reason", rank < limit ? "within_selection_limit" : "outside_selection_limit"},
                });
            }
            trace["execution_order_changed"] =
                trace["selected_arm_ids"] != arm_ids(decision.original_order, limit);
            for (const auto & prediction : decision.predictions) trace["predictions"].push_back({
                {"arm_id", prediction.arm_id}, {"level", prediction.level},
                {"p_known", prediction.p_known},
                {"p_helped_given_known", prediction.p_helped_given_known},
                {"p_neutral_given_known", prediction.p_neutral_given_known},
                {"p_harmed_given_known", prediction.p_harmed_given_known},
                {"p_known_interval_95_approx", {prediction.p_known_low, prediction.p_known_high}},
                {"p_helped_interval_95_approx", {prediction.p_helped_low, prediction.p_helped_high}},
                {"p_neutral_interval_95_approx", {prediction.p_neutral_low, prediction.p_neutral_high}},
                {"p_harmed_interval_95_approx", {prediction.p_harmed_low, prediction.p_harmed_high}},
                {"expected_utility", prediction.expected_utility},
                {"progress", nlohmann::ordered_json::parse(common_flydelta_outcome_guidance_progress_prediction_to_json(prediction))},
                {"raw_observations", prediction.raw_observations},
                {"fixture_groups", prediction.fixture_groups},
                {"effective_known_observations", prediction.effective_known_observations},
                {"effective_unknown_observations", prediction.effective_unknown_observations},
                {"backoff_fixture_groups", prediction.backoff_fixture_groups},
                {"backoff_prior_strength", prediction.backoff_prior_strength},
            });
            std::clog << "[flydelta] outcome_guidance " << trace.dump() << '\n';
            return true;
        });
    common_flydelta_search_execution_budget execution_budget;
    execution_budget.limit = plan.search_model_calls_used == 0 && plan.search_anchor_ref.empty()
        ? provider->search_model_call_budget : plan.search_model_call_limit;
    execution_budget.used = plan.search_model_calls_used;
    if (!provider->lifecycle_store) {
        error = "FlyDelta coefficient search requires a durable execution ledger";
        return false;
    }
    const size_t base_used = plan.search_model_calls_used;
    const auto update_consumption = [&](const common_flydelta_search_consumption & consumption) {
        execution_budget.used = consumption.logical_used;
        execution_budget.physical_rows_started = consumption.physical_rows_started;
        execution_budget.physical_rows_completed = consumption.physical_rows_completed;
        execution_budget.physical_rows_unresolved = consumption.physical_rows_unresolved;
    };
    common_flydelta_search_consumption consumed;
    if (!common_flydelta_load_search_consumption(*provider->lifecycle_store, job,
            base_used, execution_budget.limit, consumed, error)) return false;
    update_consumption(consumed);
    execution_budget.begin_execution = [&](const std::vector<std::string> & logical_ids,
            const std::string & group, std::string & attempt, std::string & begin_error) {
        common_flydelta_search_consumption consumption;
        if (!common_flydelta_begin_search_execution(*provider->lifecycle_store, job, base_used,
                execution_budget.limit, logical_ids, group, execution_budget.max_physical_attempts,
                attempt, consumption, begin_error)) return false;
        update_consumption(consumption);
        return true;
    };
    execution_budget.finish_execution = [&](const std::string & attempt, size_t rows, bool succeeded,
            std::string & finish_error) {
        common_flydelta_search_consumption consumption;
        if (!common_flydelta_finish_search_execution(*provider->lifecycle_store, job, attempt, rows, succeeded, finish_error) ||
                !common_flydelta_load_search_consumption(*provider->lifecycle_store, job, base_used,
                    execution_budget.limit, consumption, finish_error)) return false;
        update_consumption(consumption);
        std::clog << "[flydelta] execution_consumption job=" << job.id
            << " logical_used=" << consumption.logical_used
            << " physical_rows_started=" << consumption.physical_rows_started
            << " physical_rows_completed=" << consumption.physical_rows_completed
            << " physical_rows_unresolved=" << consumption.physical_rows_unresolved << '\n';
        return true;
    };
    // Admit only slices whose complete worst-case work fits: diagnostic
    // pairs, one dose retry, capture fallback, full baseline and finalists.
    // This prevents a budget stop halfway through publishing an anchor.
    const size_t worst_case_calls = 2 + coefficient_config.max_candidates * 8 +
        std::max<size_t>(3, guidance_config.matched_evaluation ? guidance_config.matched_max_arms : 3) * 2;
    if (base_used > execution_budget.limit ||
            worst_case_calls > execution_budget.limit - base_used) {
        output = {};
        common_flydelta_search_pipeline_direction_result direction;
        direction.direction = candidates.front();
        output.directions.push_back(std::move(direction));
        output.search_status = common_flydelta_search_status::budget_limited;
        output.search_model_call_limit = execution_budget.limit;
        output.search_model_calls_used = execution_budget.used;
        output.physical_rows_started = execution_budget.physical_rows_started;
        output.physical_rows_completed = execution_budget.physical_rows_completed;
        output.physical_rows_unresolved = execution_budget.physical_rows_unresolved;
        return true;
    }
    const auto run_mode = [provider, &job, &fixture, &execution_budget, phase = plan.phase,
            strategy = coefficient_config.strategy](
            const common_flydelta_experiment_fixture &,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<float> & coefficients,
            const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & has_geometry, std::string & runner_error,
            const bool full_execution) {
        std::vector<common_flydelta_counterfactual_trial> trials;
        std::vector<common_flydelta_decision_margin> margins;
        std::vector<common_flydelta_representation_diagnostics> geometries;
        std::vector<bool> geometry_flags;
        if (!daemon_flydelta_run_coefficient_arm_batch(
                provider, job, fixture, current_basis, {coefficients}, apply_overlay,
                trials, margins, geometries, geometry_flags, runner_error,
                full_execution, full_execution ? "deep-full-single" : "deep-diagnostic-single",
                phase, strategy, &execution_budget)) {
            return false;
        }
        trial = std::move(trials.front());
        margin = std::move(margins.front());
        geometry = std::move(geometries.front());
        has_geometry = geometry_flags.front();
        return true;
    };
    const auto run_batch_mode = [provider, &job, &fixture, &execution_budget, phase = plan.phase,
            strategy = coefficient_config.strategy,
            resident_parallelism] (
            const common_flydelta_experiment_fixture &,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_representation_diagnostics> & geometries,
            std::vector<bool> & geometry_flags, std::string & runner_error,
            const bool full_execution) {
        trials.clear();
        margins.clear();
        geometries.clear();
        geometry_flags.clear();
        if (coefficients.empty()) {
            runner_error = "Deep/TFO coefficient batch is empty";
            return false;
        }
        // Full-generation finalists have no paired diagnostic baseline and
        // can use every resident sequence. Diagnostic arms each need a
        // simultaneous baseline, so leave one slot for it in every physical
        // batch. This keeps baseline IDs and geometry reductions local to
        // their corresponding arm even when a proposal wave is larger.
        const size_t candidate_capacity = full_execution
            ? std::max<size_t>(1, resident_parallelism)
            : (resident_parallelism > 1 ? resident_parallelism - 1 : 1);
        for (size_t begin = 0, part = 0; begin < coefficients.size(); ++part) {
            const size_t count = std::min(candidate_capacity, coefficients.size() - begin);
            std::vector<std::vector<float>> chunk(
                coefficients.begin() + static_cast<std::ptrdiff_t>(begin),
                coefficients.begin() + static_cast<std::ptrdiff_t>(begin + count));
            std::vector<common_flydelta_counterfactual_trial> part_trials;
            std::vector<common_flydelta_decision_margin> part_margins;
            std::vector<common_flydelta_representation_diagnostics> part_geometries;
            std::vector<bool> part_geometry_flags;
            const std::string suffix = full_execution
                ? "deep-full-batch-part-" + std::to_string(part)
                : "deep-diagnostic-batch-part-" + std::to_string(part);
            if (!daemon_flydelta_run_coefficient_arm_batch(
                    provider, job, fixture, current_basis, chunk, true,
                    part_trials, part_margins, part_geometries, part_geometry_flags,
                    runner_error, full_execution, suffix, phase, strategy, &execution_budget)) {
                return false;
            }
            trials.insert(trials.end(),
                std::make_move_iterator(part_trials.begin()),
                std::make_move_iterator(part_trials.end()));
            margins.insert(margins.end(),
                std::make_move_iterator(part_margins.begin()),
                std::make_move_iterator(part_margins.end()));
            geometries.insert(geometries.end(),
                std::make_move_iterator(part_geometries.begin()),
                std::make_move_iterator(part_geometries.end()));
            geometry_flags.insert(geometry_flags.end(),
                part_geometry_flags.begin(), part_geometry_flags.end());
            begin += count;
        }
        return trials.size() == coefficients.size() &&
            margins.size() == coefficients.size() &&
            geometries.size() == coefficients.size() &&
            geometry_flags.size() == coefficients.size();
    };
    const auto run_diagnostic = [run_mode](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<float> & coefficients,
            const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & has_geometry, std::string & runner_error) {
        return run_mode(current_fixture, current_basis, coefficients, apply_overlay,
            trial, margin, geometry, has_geometry, runner_error, false);
    };
    const auto run_diagnostic_batch = [run_batch_mode](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_representation_diagnostics> & geometries,
            std::vector<bool> & geometry_flags, std::string & runner_error) {
        return run_batch_mode(current_fixture, current_basis, coefficients, trials,
            margins, geometries, geometry_flags, runner_error, false);
    };
    const auto run_full = [run_mode](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<float> & coefficients,
            const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & has_geometry, std::string & runner_error) {
        return run_mode(current_fixture, current_basis, coefficients, apply_overlay,
            trial, margin, geometry, has_geometry, runner_error, true);
    };
    const auto run_full_batch = [run_batch_mode](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_representation_diagnostics> & geometries,
            std::vector<bool> & geometry_flags, std::string & runner_error) {
        return run_batch_mode(current_fixture, current_basis, coefficients, trials,
            margins, geometries, geometry_flags, runner_error, true);
    };
    std::vector<common_flydelta_coefficient_trial> trials;
    common_flydelta_coefficient_selection selection;
    std::string provisional_anchor_ref;
    size_t provisional_trial_index = static_cast<size_t>(-1);
    if (plan.phase == common_flydelta_experiment_phase::deep_controls) {
        common_flydelta_deep_search_config deep_config;
        deep_config.max_directions = std::min<size_t>(candidates.size(), 4);
        deep_config.max_rank = std::min(max_rank, deep_config.max_directions);
        if (!resume_anchor.is_null()) {
            try {
                deep_config.max_rank = resume_anchor.at("search_policy").at("max_rank").get<size_t>();
                deep_config.max_directions = resume_anchor.at("search_policy").at("max_directions").get<size_t>();
            } catch (const std::exception & exception) {
                error = std::string("FlyDelta search anchor policy is malformed: ") + exception.what();
                return false;
            }
        }
        deep_config.full_generation_top_k = plan.budget.full_generation_top_k == 0
            ? 3 : std::min<size_t>(3, plan.budget.full_generation_top_k);
        deep_config.coefficients = coefficient_config;
        const auto & dose_policy = coefficient_config.dose_policy;
        const json search_policy = {
            {"version", 1}, {"strategy", static_cast<int>(coefficient_config.strategy)},
            {"step", coefficient_config.step}, {"max_candidates", coefficient_config.max_candidates},
            {"max_l2_norm", coefficient_config.max_l2_norm}, {"seed", coefficient_config.seed},
            {"population_size", coefficient_config.population_size}, {"iterations", coefficient_config.iterations},
            {"exploration_scale", coefficient_config.exploration_scale}, {"norm_penalty", coefficient_config.norm_penalty},
            {"leakage_penalty", coefficient_config.leakage_penalty},
            {"residual_fraction", coefficient_config.residual_fraction},
            {"residual_fraction_candidates", coefficient_config.residual_fraction_candidates},
            {"use_dose_controller", coefficient_config.use_dose_controller},
            {"max_dose_retries", coefficient_config.max_dose_retries},
            {"max_physical_attempts", execution_budget.max_physical_attempts},
            {"dose_policy", {{"version", dose_policy.schema_version}, {"max_shift_norm", dose_policy.max_shift_norm},
                {"max_leakage", dose_policy.max_leakage}, {"target_relative_dose", dose_policy.target_relative_dose},
                {"target_absolute_fraction", dose_policy.target_absolute_fraction}, {"min_strength", dose_policy.min_strength},
                {"max_strength", dose_policy.max_strength}, {"backoff_safety_factor", dose_policy.backoff_safety_factor}}},
            {"max_rank", deep_config.max_rank}, {"max_directions", deep_config.max_directions},
            {"full_generation_top_k", deep_config.full_generation_top_k}};
        if (!plan.search_anchor_ref.empty()) {
            const auto & anchor = resume_anchor;
            try {
                if (anchor.at("kind") != "flydelta_search_anchor" || anchor.at("schema_version") != 1 ||
                        anchor.at("task_fingerprint") != fixture.task_fingerprint ||
                        anchor.at("model_profile_fingerprint") != fixture.model_profile_fingerprint ||
                        anchor.at("tokenizer_fingerprint") != fixture.tokenizer_fingerprint ||
                        anchor.at("template_fingerprint") != fixture.template_fingerprint ||
                        anchor.at("execution_context_fingerprint") != fixture.execution_context_fingerprint ||
                        anchor.at("verifier_revision") != fixture.verifier_revision ||
                        anchor.at("baseline_ref") != job.seed.baseline_ref ||
                        anchor.at("capture_layout_revision") != provider->capture_layout_revision ||
                        anchor.at("dose_mode") != effective_dose_mode ||
                        anchor.at("calibration_ref") != effective_calibration_ref ||
                        anchor.at("calibration_fingerprint") != calibration_fingerprint ||
                        anchor.at("search_policy") != search_policy) {
                    error = "FlyDelta search anchor provenance is incompatible"; return false;
                }
                deep_config.anchor_basis.layer_index = anchor.at("layer_index").get<int32_t>();
                deep_config.anchor_basis.dimension = anchor.at("dimension").get<size_t>();
                deep_config.anchor_basis.vectors = anchor.at("basis_vectors").get<std::vector<std::vector<float>>>();
                deep_config.coefficients.anchor_basis_identity = anchor.at("basis_fingerprint").get<std::string>();
                deep_config.coefficients.anchor_coefficients = anchor.at("applied_coefficients").get<std::vector<float>>();
                deep_config.coefficients.evaluated_coefficients = anchor.value("evaluated_coefficients",
                    std::vector<std::vector<float>>{});
                if (!daemon_flydelta_anchor_observation_from_json(
                        anchor.at("observation"), deep_config.anchor_observation, error)) return false;
                std::vector<float> physical;
                if (!common_flydelta_reconstruct_intervention(deep_config.anchor_basis,
                        deep_config.coefficients.anchor_basis_identity,
                        deep_config.coefficients.anchor_coefficients, physical, error)) return false;
                const auto expected = anchor.at("intervention_fingerprint").get<std::string>();
                const auto actual = json(physical).dump();
                if (expected != hash_sha256_hex(actual.data(), actual.size())) {
                    error = "FlyDelta search anchor physical intervention fingerprint differs"; return false;
                }
                deep_config.resume_anchor = true;
                deep_config.anchor_ref = plan.search_anchor_ref;
            } catch (const std::exception & exception) {
                error = std::string("FlyDelta search anchor is malformed: ") + exception.what();
                return false;
            }
        }
        deep_config.matched_evaluation = guidance_config.matched_evaluation &&
            guidance_config.mode != common_flydelta_outcome_guidance_mode::off &&
            !guidance_history_load_failed && !matched_evaluation_recorded && provider->lifecycle_store;
        deep_config.matched_max_arms = guidance_config.matched_max_arms;
        deep_config.matched_seed = guidance_config.seed;
        std::vector<common_flydelta_deep_search_direction> deep_directions;
        for (const auto & candidate : candidates) {
            common_flydelta_deep_search_direction value;
            value.direction = candidate;
            value.decision_score_available = resume_anchor.is_null();
            value.decision_score = candidate.median_alignment;
            deep_directions.push_back(std::move(value));
        }
        common_flydelta_deep_search_result deep_result;
        if (!common_flydelta_run_deep_search_batched(
                fixture, deep_config, deep_directions, run_diagnostic, run_diagnostic_batch,
                run_full, run_full_batch, deep_result, error, outcome_adviser)) return false;
        if (deep_result.provisional_selection.selected && !deep_result.coefficient_selection.selected) {
            provisional_trial_index = deep_result.provisional_selection.trial_index;
            const auto & selected = deep_result.coefficient_trials.at(provisional_trial_index);
            std::vector<float> physical;
            const auto basis_fingerprint = common_flydelta_coefficient_basis_identity(deep_result.basis);
            if (!common_flydelta_reconstruct_intervention(deep_result.basis, basis_fingerprint,
                    selected.coefficients, physical, error)) return false;
            const auto physical_json = json(physical).dump();
            auto evaluated = deep_config.coefficients.evaluated_coefficients;
            for (const auto & trial : deep_result.coefficient_trials) {
                if (!trial.executed || !trial.overlay_applied || trial.coefficients.empty()) continue;
                for (const auto & coefficients : {trial.requested_coefficients, trial.coefficients}) {
                    if (!coefficients.empty() && std::find(evaluated.begin(), evaluated.end(), coefficients) == evaluated.end()) {
                        evaluated.push_back(coefficients);
                    }
                }
            }
            if (evaluated.size() > 4096) { error = "FlyDelta anchor trial history exceeds its bound"; return false; }
            const json anchor = {{"kind", "flydelta_search_anchor"}, {"schema_version", 1},
                {"task_fingerprint", fixture.task_fingerprint},
                {"model_profile_fingerprint", fixture.model_profile_fingerprint},
                {"tokenizer_fingerprint", fixture.tokenizer_fingerprint},
                {"template_fingerprint", fixture.template_fingerprint},
                {"execution_context_fingerprint", fixture.execution_context_fingerprint},
                {"verifier_revision", fixture.verifier_revision}, {"baseline_ref", job.seed.baseline_ref},
                {"capture_layout_revision", provider->capture_layout_revision},
                {"dose_mode", effective_dose_mode}, {"calibration_ref", effective_calibration_ref},
                {"calibration_fingerprint", calibration_fingerprint},
                {"search_policy", search_policy},
                {"parent_anchor_ref", plan.search_anchor_ref}, {"arm_id", selected.arm_id},
                {"evidence_ref", selected.evidence_ref},
                // Runtime generation references are diagnostic correlation IDs;
                // the persisted assessment below is the checkpoint evidence.
                {"evidence_ref_is_runtime", selected.evidence_ref.rfind("flydelta://runtime/", 0) == 0},
                {"verifier_reason", selected.verifier_reason},
                {"observed_decision_summary", selected.observed_decision_summary},
                {"expected_decision_summary", selected.expected_decision_summary},
                {"search_model_call_limit", execution_budget.limit},
                {"search_model_calls_used", execution_budget.used},
                {"physical_rows_started", execution_budget.physical_rows_started},
                {"physical_rows_completed", execution_budget.physical_rows_completed},
                {"physical_rows_unresolved", execution_budget.physical_rows_unresolved},
                {"layer_index", deep_result.basis.layer_index}, {"dimension", deep_result.basis.dimension},
                {"basis_vectors", deep_result.basis.vectors}, {"basis_fingerprint", basis_fingerprint},
                {"requested_coefficients", selected.requested_coefficients},
                {"applied_coefficients", selected.coefficients},
                {"evaluated_coefficients", evaluated},
                {"requested_residual_fraction", selected.requested_residual_fraction},
                {"applied_residual_fraction", selected.applied_residual_fraction},
                {"intervention_fingerprint", hash_sha256_hex(physical_json.data(), physical_json.size())},
                {"observation", daemon_flydelta_anchor_observation_json(selected.semantic_progress_observation)}};
            const auto serialized = anchor.dump();
            const auto digest = hash_sha256_hex(serialized.data(), serialized.size());
            if (!daemon_flydelta_put_json_resource(provider,
                    "flydelta-search-anchor-" + digest.substr(0, 24) + ".json",
                    "flydelta_search_anchor", job.seed.candidate_ref, serialized,
                    provisional_anchor_ref, error)) return false;
            json stored_anchor;
            if (!daemon_flydelta_read_json_bounded(*provider, provisional_anchor_ref,
                    16U * 1024U * 1024U, stored_anchor, error)) return false;
            if (stored_anchor != anchor) {
                error = "FlyDelta search anchor could not be verified after persistence";
                return false;
            }
        }
        if ((deep_result.matched_evaluation_complete || matched_evaluation_recorded) &&
                guidance_config.matched_evaluation &&
                guidance_config.mode != common_flydelta_outcome_guidance_mode::off &&
                !guidance_history_load_failed && provider->lifecycle_store) {
            common_flydelta_guidance_qualification_context context;
            context.authorized_scope = job.seed.scope;
            context.outcome_context = guidance_context;
            context.selection_limit = deep_result.matched_evaluation_complete
                ? deep_result.matched_evaluation.baseline_selection.size()
                : static_cast<size_t>(std::count_if(deep_result.coefficient_trials.begin(),
                    deep_result.coefficient_trials.end(), [](const auto & trial) {
                        return trial.mutation_kind == "full_generation_top_arm";
                    }));
            context.policy_revision = common_flydelta_guidance_qualification_policy_revision(guidance_config);
            context.progress_reward = guidance_config.progress_reward;
            context.harm_penalty = guidance_config.harm_penalty;
            common_flydelta_lifecycle_event_context event;
            event.event_id = job.id + ":deep-matched:" + (deep_result.matched_evaluation_complete
                ? deep_result.matched_evaluation.evaluation_id : "refresh");
            event.idempotency_key = event.event_id;
            event.source_id = "daemon-flydelta-deep-matched-evaluation";
            event.scope = job.seed.scope;
            const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm utc{};
            gmtime_r(&now, &utc);
            std::ostringstream timestamp;
            timestamp << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
            event.created_at = timestamp.str();
            std::string qualification_error;
            common_flydelta_guidance_qualification_result qualification;
            const bool stored = deep_result.matched_evaluation_complete
                ? common_flydelta_guidance_record_matched_evaluation(*provider->lifecycle_store, event,
                    context, deep_result.matched_evaluation, qualification_error, &qualification)
                : common_flydelta_guidance_refresh_qualification(*provider->lifecycle_store, event,
                    context, qualification_error, &qualification);
            std::clog << "[flydelta] matched_evaluation " << nlohmann::ordered_json{
                {"job_id", job.id}, {"complete", deep_result.matched_evaluation_complete},
                {"restored_evaluation", matched_evaluation_recorded}, {"persisted", stored},
                {"fixture_group_id", deep_result.matched_evaluation.fixture_group_id},
                {"history_fingerprint", deep_result.matched_evaluation.history_fingerprint},
                {"candidate_pool_fingerprint", deep_result.matched_evaluation.candidate_pool_fingerprint},
                {"baseline_selection", deep_result.matched_evaluation.baseline_selection},
                {"guided_selection", deep_result.matched_evaluation.guided_selection},
                {"random_selection", deep_result.matched_evaluation.random_ablation_selection},
                {"union_arm_count", deep_result.matched_evaluation.union_arm_results.size()},
                {"qualification_status", stored
                    ? common_flydelta_guidance_qualification_status_name(qualification.status) : "unavailable"},
                {"fixture_groups", qualification.fixture_groups},
                {"baseline_helped", qualification.baseline_helped}, {"guided_helped", qualification.guided_helped},
                {"baseline_harmed", qualification.baseline_harmed}, {"guided_harmed", qualification.guided_harmed},
                {"baseline_unknown", qualification.baseline_unknown}, {"guided_unknown", qualification.guided_unknown},
                {"inference_costs_available", qualification.inference_costs_available},
                {"error", qualification_error}}.dump() << '\n';
        } else if (guidance_config.matched_evaluation) {
            std::clog << "[flydelta] matched_evaluation job=" << job.id << " complete=false reason="
                << (matched_evaluation_recorded ? "already_recorded" :
                    deep_result.matched_evaluation_reason.empty() ? "not_eligible_or_disabled" :
                    deep_result.matched_evaluation_reason) << '\n';
        }
        basis = deep_result.basis;
        trials = deep_result.coefficient_trials;
        selection = deep_result.coefficient_selection;
    } else if (!common_flydelta_run_low_rank_coefficient_search_batched_staged(
            fixture, basis, coefficient_config,
            run_diagnostic, run_diagnostic_batch,
            run_full, run_full_batch,
            plan.budget.full_generation_top_k == 0
                ? 3 : std::min<size_t>(3, plan.budget.full_generation_top_k),
            trials, selection, error, outcome_adviser)) {
        return false;
    }

    if (provider->lifecycle_store && !trials.empty() &&
            !fixture.scope_fingerprint.empty() &&
            !fixture.baseline_deployment_fingerprint.empty() &&
            !fixture.semantic_domain.empty() && fixture.model_n_layers > 1) {
        common_flydelta_lifecycle_event_context lifecycle_context;
        lifecycle_context.event_id = job.id + ":coefficient-search:" + progress_wave_identity;
        lifecycle_context.idempotency_key = lifecycle_context.event_id;
        lifecycle_context.source_id = "daemon-flydelta-outcome-guidance";
        lifecycle_context.scope = job.seed.scope;
        lifecycle_context.content_hash = job.seed.task_fingerprint + ":" +
            fixture.baseline_deployment_fingerprint;
        const auto now = std::chrono::system_clock::now();
        const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm utc{};
        gmtime_r(&now_time, &utc);
        std::ostringstream timestamp;
        timestamp << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
        lifecycle_context.created_at = timestamp.str();
        std::string lifecycle_error;
        if (!common_flydelta_append_coefficient_search_lifecycle(
                *provider->lifecycle_store, lifecycle_context, fixture, basis,
                coefficient_config, trials, job.seed.candidate_ref, lifecycle_error)) {
            std::clog << "[flydelta] outcome_guidance_result_persistence_failed job="
                << job.id << " phase=" << common_flydelta_experiment_phase_name(plan.phase)
                << " error=" << lifecycle_error << '\n';
        }
    }

    common_flydelta_search_pipeline_direction_result direction_result;
    direction_result.direction = candidates.front();
    direction_result.direction.values = basis.vectors.front();
    direction_result.direction.layer_index = basis.layer_index;
    direction_result.direction.origin = "runtime_post_bootstrap_basis";
    if (selection.selected) {
        const auto & selected = trials.at(selection.trial_index);
        if (!daemon_flydelta_register_composed_direction(provider, job.seed.candidate_ref,
                basis, selected.coefficients, direction_result.intervention_ref, error)) return false;
        std::vector<common_flydelta_basis_direction> composed;
        if (!daemon_flydelta_parse_directions(*provider, direction_result.intervention_ref,
                composed, error) || composed.size() != 1) return false;
        direction_result.direction.values = composed.front().values;
        direction_result.selected_coefficients = selected.coefficients;
    }
    direction_result.region_trials.reserve(trials.size());
    for (size_t trial_index = 0; trial_index < trials.size(); ++trial_index) {
        const auto & coefficient_trial = trials[trial_index];
        common_flydelta_intervention_region_trial trial;
        if (!common_flydelta_coefficient_trial_to_region(coefficient_trial, basis, trial, error)) return false;
        // Preserve the arm's actual host-evaluation reference. The job-level
        // fallback is useful for diagnostic arms that did not produce one,
        // but must not replace the evidence needed to resume a finalist.
        if (trial.evidence_ref.empty()) {
            trial.evidence_ref = "flydelta://runtime/coefficient-trial/" + job.id;
        }
        if (!provisional_anchor_ref.empty() && trial_index == provisional_trial_index) {
            trial.promising = true;
        }
        if (!common_flydelta_intervention_region_trial_validate(trial, error)) return false;
        direction_result.region_trials.push_back(std::move(trial));
    }
    direction_result.region_selection.selected = selection.selected;
    direction_result.region_selection.trial_index = selection.trial_index;
    direction_result.region_selection.score = selection.score;
    output = {};
    output.provisional_anchor_ref = provisional_anchor_ref;
    output.search_model_call_limit = execution_budget.limit;
    output.search_model_calls_used = execution_budget.used;
    output.physical_rows_started = execution_budget.physical_rows_started;
    output.physical_rows_completed = execution_budget.physical_rows_completed;
    output.physical_rows_unresolved = execution_budget.physical_rows_unresolved;
    output.directions.push_back(std::move(direction_result));
    const auto & regions = output.directions.front().region_trials;
    output.search_status = std::any_of(regions.begin(), regions.end(), [](const auto & trial) {
        return trial.executed && trial.safe_to_continue && trial.promising &&
            trial.outcome != common_flydelta_counterfactual_outcome::harmed;
    }) ? common_flydelta_search_status::candidate_available : common_flydelta_search_status::no_useful_utility;
    output.selection.selected = selection.selected;
    output.selection.intervention_region = true;
    output.selection.direction_index = 0;
    output.selection.region_trial_index = selection.trial_index;
    output.selection.score = selection.score;
    return true;
}

// Representation augmentation is a host-owned continuation over opaque
// donor resources.  The resource contains references to two already
// host-approved contexts: the target context and the target-plus-donor
// context.  No prompt or activation data is synthesized by this binding.

} // namespace agent_daemon_flydelta_internal
