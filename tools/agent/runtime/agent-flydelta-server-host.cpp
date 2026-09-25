#include "agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_prepare_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_arm_request & arm,
        common_agent_generation_request & request,
        std::string & error) {
    if (!daemon_flydelta_parse_context(*provider, arm.context_ref, request, error)) return false;
    if (arm.max_generated_tokens != 0) {
        request.options.n_predict = std::min(
            request.options.n_predict, static_cast<int>(arm.max_generated_tokens));
    }
    // Diagnostic arms still evaluate the prompt and may capture hidden state,
    // but they must not decode a token.  The arm contract already separates
    // diagnostic work from full generation; carry that distinction into the
    // resident server request instead of letting the context defaults decode.
    if (!arm.request_generation) {
        request.options.n_predict = 0;
    }
    if (arm.apply_overlay) {
        std::vector<common_flydelta_basis_direction> available;
        if (!daemon_flydelta_parse_directions(*provider, arm.intervention_ref, available, error)) {
            return false;
        }
        std::vector<common_flydelta_basis_direction> selected;
        std::vector<float> coefficients;
        selected.reserve(arm.layer_indices.size());
        coefficients.reserve(arm.layer_indices.size());
        for (size_t index = 0; index < arm.layer_indices.size(); ++index) {
            const auto it = std::find_if(available.begin(), available.end(), [&](const auto & direction) {
                return direction.layer_index == static_cast<int32_t>(arm.layer_indices[index]);
            });
            if (it == available.end()) {
                error = "FlyDelta arm requests a layer missing from its intervention resource";
                return false;
            }
            selected.push_back(*it);
            coefficients.push_back(arm.coefficients[index]);
        }
        common_flydelta_gate_config gate_config;
        gate_config.enabled = true;
        gate_config.max_scale = 1.0f;
        common_flydelta_activation_request activation_request;
        activation_request.candidate_id = arm.intervention_ref;
        activation_request.artifact_id = arm.arm_id;
        activation_request.model_profile_fingerprint = provider->model_profile_fingerprint;
        activation_request.capture_layout_revision = provider->capture_layout_revision;
        activation_request.model_n_embd = provider->model_n_embd;
        activation_request.model_n_layers = provider->model_n_layers;
        activation_request.il_start = 1;
        activation_request.il_end = static_cast<int32_t>(provider->model_n_layers - 1);
        activation_request.directions = std::move(selected);
        activation_request.coefficients = std::move(coefficients);
        activation_request.gate_request.explicit_opt_in = true;
        activation_request.gate_request.candidate_status = common_flydelta_candidate_status::approved;
        activation_request.gate_request.basis_available = true;
        activation_request.gate_request.familiarity = 1.0f;
        activation_request.gate_request.novelty = 0.0f;
        activation_request.gate_request.requested_scale = arm.alpha;
        common_flydelta_activation_result activation;
        if (!common_flydelta_prepare_activation(
                gate_config, activation_request, 64U * 1024U * 1024U, activation, error)) {
            return false;
        }
        request.flydelta_activation = std::make_shared<const common_flydelta_activation_result>(
            std::move(activation));
    }
    if (arm.request_capture) {
        auto capture = std::make_shared<common_flydelta_hidden_state_capture_request>();
        capture->enabled = true;
        capture->layer_indices = arm.layer_indices;
        capture->position = common_flydelta_capture_position::generation_boundary;
        capture->token_index = -1;
        capture->max_bytes = arm.max_capture_bytes == 0
            ? 4U * 1024U * 1024U : arm.max_capture_bytes;
        capture->model_profile_fingerprint = provider->model_profile_fingerprint;
        capture->capture_layout_revision = provider->capture_layout_revision;
        request.flydelta_capture = std::move(capture);
    }
    return true;
}

bool daemon_flydelta_score_teacher_forced_margin_batch(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::vector<common_flydelta_arm_request> & arms,
        const std::vector<common_agent_generation_request> & contexts,
        common_agent_inference & scorer,
        std::vector<common_flydelta_decision_margin> & margins,
        std::string & error) {
    error.clear();
    margins.assign(arms.size(), {});
    if (arms.size() != contexts.size()) {
        error = "FlyDelta teacher-score batch has mismatched arms and contexts";
        return false;
    }

    common_agent_teacher_forced_choice_batch_request score_batch;
    std::vector<size_t> score_indices;
    score_batch.choices.reserve(arms.size());
    score_indices.reserve(arms.size());
    for (size_t index = 0; index < arms.size(); ++index) {
        const auto & arm = arms[index];
        if (!arm.request_teacher_forced_margin) continue;
        json fixture;
        if (!daemon_flydelta_read_json(*provider, arm.fixture_ref, fixture, error)) return false;
        if (!fixture.contains("positive_continuation") ||
                !fixture.contains("negative_continuation")) {
            continue;
        }
        if (!fixture["positive_continuation"].is_string() ||
                !fixture["negative_continuation"].is_string()) {
            error = "FlyDelta fixture decision continuations must be strings";
            return false;
        }
        common_agent_teacher_forced_choice_request score_request;
        score_request.sequence_id = arm.arm_id;
        // Score from a fresh immutable arm context. The generation request
        // belongs to backend transport; semantic reference resolution stays
        // host-owned and must not depend on that transport object's lifetime.
        if (!daemon_flydelta_prepare_arm(provider, arm, score_request.context, error)) return false;
        score_request.context.flydelta_capture.reset();
        score_request.choice_prefix = fixture.value("choice_prefix", "");
        score_request.positive_continuation = fixture["positive_continuation"].get<std::string>();
        score_request.negative_continuation = fixture["negative_continuation"].get<std::string>();
        score_batch.choices.push_back(std::move(score_request));
        score_indices.push_back(index);
    }
    if (score_batch.choices.empty()) return true;

    common_agent_teacher_forced_choice_batch_result scores;
    if (!scorer.score_teacher_forced_choice_batch(score_batch, scores)) {
        error = scores.error_message.empty()
            ? "FlyDelta teacher-forced batch scoring failed" : scores.error_message;
        return false;
    }
    if (scores.choices.size() != score_indices.size()) {
        error = "FlyDelta teacher-forced batch scoring returned incomplete results";
        return false;
    }
    for (size_t index = 0; index < score_indices.size(); ++index) {
        const auto & score = scores.choices[index];
        auto & margin = margins[score_indices[index]];
        margin.available = score.available;
        margin.positive_total_logprob = score.positive_total_logprob;
        margin.negative_total_logprob = score.negative_total_logprob;
        margin.positive_token_count = score.positive_token_count;
        margin.negative_token_count = score.negative_token_count;
    }
    return true;
}

bool daemon_flydelta_verify_generation(
        const json & fixture,
        const std::string & generated,
        bool & verifier_known,
        bool & passed,
        std::string & error) {
    error.clear();
    verifier_known = false;
    passed = false;

    const std::string mode = fixture.value("verification_mode", "");
    if (mode == "normalized_call") {
        if (!fixture.contains("expected_decision")) {
            error = "FlyDelta normalized_call fixture requires expected_decision";
            return false;
        }
        common_flydelta_semantic_decision expected;
        common_flydelta_semantic_decision actual;
        common_flydelta_semantic_decision_status expected_status;
        common_flydelta_semantic_decision_status actual_status;
        std::string decision_error;
        const std::string expected_text = fixture.at("expected_decision").is_string()
            ? fixture.at("expected_decision").get<std::string>()
            : fixture.at("expected_decision").dump();
        const bool expected_valid = common_flydelta_parse_semantic_decision(
            expected_text, expected, expected_status, decision_error);
        if (!expected_valid) {
            error = "FlyDelta normalized_call fixture expected_decision is invalid: " +
                decision_error;
            return false;
        }
        decision_error.clear();
        const bool actual_valid = common_flydelta_parse_semantic_decision(
            generated, actual, actual_status, decision_error);
        verifier_known = true;
        passed = actual_valid && common_flydelta_semantic_decision_equal(expected, actual);
        return true;
    }

    if (mode == "tool_only" && fixture.contains("expected_tool")) {
        if (!fixture.at("expected_tool").is_string()) {
            error = "FlyDelta tool_only fixture expected_tool must be a string";
            return false;
        }
        const std::string expected_tool = fixture.at("expected_tool").get<std::string>();
        common_flydelta_semantic_decision actual;
        common_flydelta_semantic_decision_status status;
        std::string decision_error;
        const bool parsed = common_flydelta_parse_semantic_decision(
            generated, actual, status, decision_error);
        verifier_known = true;
        if (parsed) {
            const std::string expected_operation = expected_tool.rfind("data.", 0) == 0
                ? expected_tool.substr(5) : expected_tool;
            passed = actual.operation == expected_operation;
        } else {
            // Keep the existing tool-only fixture behavior for operations
            // that are intentionally outside the semantic-decision IR.
            passed = generated.find(expected_tool) != std::string::npos;
        }
        return true;
    }

    if (fixture.contains("expected_contains") || fixture.contains("expected_tool")) {
        std::string expected;
        if (fixture.contains("expected_tool")) {
            if (!fixture.at("expected_tool").is_string()) {
                error = "FlyDelta fixture expected_tool must be a string";
                return false;
            }
            expected = fixture.at("expected_tool").get<std::string>();
        } else if (!fixture.at("expected_contains").is_string()) {
            error = "FlyDelta fixture expected_contains must be a string";
            return false;
        } else {
            expected = fixture.at("expected_contains").get<std::string>();
        }
        verifier_known = true;
        passed = !expected.empty() && generated.find(expected) != std::string::npos;
    }
    return true;
}

// A single arm has no baseline/candidate relation, so a semantic verifier
// pass is represented as NEUTRAL in host_outcome. Search runners still need
// the individual predicate for the existing counterfactual classifiers; keep
// that translation local to the daemon instead of changing outcome semantics.
bool daemon_flydelta_arm_semantic_passed(const common_flydelta_arm_result & result) {
    return result.executed && result.host_evaluated && result.verifier_known &&
        result.host_outcome == common_flydelta_counterfactual_outcome::neutral;
}

bool daemon_flydelta_finalize_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_arm_request & arm,
        const common_agent_generation_result & generation,
        common_flydelta_arm_result & result,
        std::string & error) {
    error.clear();
    result = {};
    result.arm_id = arm.arm_id;
    result.executed = common_agent_generation_succeeded(generation);
    result.requested_alpha = arm.alpha;
    result.executed_alpha = arm.alpha;
    result.generation_available = arm.request_generation;
    result.quality = static_cast<float>(generation.decoded_tokens);
    if (!result.executed) {
        result.provenance_ref = generation.error_message;
        return false;
    }
    if (generation.flydelta_capture && generation.flydelta_capture->captured) {
        result.capture_ref = "flydelta://runtime/capture/" + arm.arm_id;
        {
            std::lock_guard<std::mutex> lock(provider->capture_mutex);
            provider->captures[arm.arm_id] = generation.flydelta_capture;
        }
        std::string persisted_capture_ref;
        if (!daemon_flydelta_persist_capture(
                provider, arm.arm_id, generation.flydelta_capture,
                persisted_capture_ref, error)) return false;
        result.capture_ref = persisted_capture_ref;
    }
    if (!arm.request_host_verification) {
        result.generation_ref = "flydelta://runtime/generation/" + arm.arm_id;
        return common_flydelta_arm_result_validate(result, error);
    }
    json fixture;
    if (!daemon_flydelta_read_json(*provider, arm.fixture_ref, fixture, error)) return false;
    if (arm.request_host_verification) {
        // The host attempted verification even when the fixture has no
        // applicable verifier. Preserve that distinction so UNKNOWN is not
        // mistaken for an unevaluated arm.
        result.host_evaluated = true;
        bool verifier_known = false;
        bool passed = false;
        if (!daemon_flydelta_verify_generation(
                fixture, generation.content, verifier_known, passed, error)) return false;
        if (verifier_known) {
            result.verifier_known = true;
            // An individual arm has no paired baseline in this callback. A
            // semantic pass is therefore NEUTRAL experimental truth here;
            // common_flydelta_run_counterfactual() derives HELPED only from
            // the baseline/candidate pair below.
            result.host_outcome = passed
                ? common_flydelta_counterfactual_outcome::neutral
                : common_flydelta_counterfactual_outcome::harmed;
            result.quality = passed ? 1.0f : 0.0f;
        }
    }
    result.generation_ref = "flydelta://runtime/generation/" + arm.arm_id;
    return common_flydelta_arm_result_validate(result, error);
}

bool daemon_flydelta_capture_for_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & arm_id,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture) {
    {
        std::lock_guard<std::mutex> lock(provider->capture_mutex);
        const auto it = provider->captures.find(arm_id);
        if (it != provider->captures.end()) {
            capture = it->second;
            return static_cast<bool>(capture);
        }
    }
    return daemon_flydelta_load_persisted_capture(provider, arm_id, capture);
}

bool daemon_flydelta_diagnostics_for_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const std::string & baseline_arm_id,
        const common_flydelta_arm_request & arm,
        uint32_t layer,
        common_flydelta_representation_diagnostics & diagnostics,
        std::string & error) {
    diagnostics = {};
    std::shared_ptr<const common_flydelta_hidden_state_capture> baseline;
    std::shared_ptr<const common_flydelta_hidden_state_capture> overlay;
    if (!daemon_flydelta_capture_for_arm(provider, baseline_arm_id, baseline) ||
            !daemon_flydelta_capture_for_arm(provider, arm.arm_id, overlay)) {
        return true;
    }
    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, arm.intervention_ref, directions, error)) return false;
    const auto direction = std::find_if(directions.begin(), directions.end(), [&](const auto & value) {
        return value.layer_index == static_cast<int32_t>(layer);
    });
    if (direction == directions.end()) {
        error = "FlyDelta diagnostic arm has no compatible direction";
        return false;
    }
    common_flydelta_behavior_delta delta;
    delta.id = arm.arm_id + ":diagnostic";
    delta.behavior_key = job.seed.behavior_key;
    delta.capture_manifest_id = arm.context_ref;
    delta.host_evidence_ref = job.seed.evidence_ref;
    delta.model_profile_fingerprint = provider->model_profile_fingerprint;
    delta.execution_context_fingerprint = job.seed.execution_context_fingerprint;
    delta.capture_layout_revision = provider->capture_layout_revision;
    delta.layer_index = direction->layer_index;
    delta.values = direction->values;
    return common_flydelta_representation_diagnostics_from_captures(
        *baseline, *overlay, delta, 4U * 1024U * 1024U, diagnostics, error);
}

bool daemon_flydelta_run_coefficient_arm_batch(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const std::vector<std::vector<float>> & coefficients,
        const bool apply_overlay,
        std::vector<common_flydelta_counterfactual_trial> & trials,
        std::vector<common_flydelta_decision_margin> & margins,
        std::vector<common_flydelta_representation_diagnostics> & geometries,
        std::vector<bool> & geometry_available,
        std::string & error,
        const bool full_execution,
        const std::string & wave_suffix) {
    error.clear();
    if (coefficients.empty()) {
        error = "FlyDelta coefficient batch is empty";
        return false;
    }
    common_flydelta_arm_batch_request request;
    request.batch_id = job.id + ":coefficient";
    request.wave_id = apply_overlay ? "coefficient-controls" : "coefficient-baseline";
    if (!wave_suffix.empty()) request.wave_id += ":" + wave_suffix;
    request.arms.reserve(coefficients.size());
    for (size_t index = 0; index < coefficients.size(); ++index) {
        const auto & values = coefficients[index];
        std::string intervention_ref = job.seed.candidate_ref;
        float strength = 0.0f;
        if (apply_overlay) {
            if (!daemon_flydelta_register_composed_direction(
                    provider, job.seed.candidate_ref, basis, values,
                    intervention_ref, error)) return false;
            for (const float value : values) strength += value * value;
            strength = std::sqrt(strength);
        }
        common_flydelta_arm_request arm;
        arm.job_id = job.id;
        arm.wave_id = request.wave_id;
        arm.proposal_index = index;
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = fixture.id;
        arm.intervention_ref = intervention_ref;
        arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
        // Baseline arms still capture the same layer as the overlay arms so
        // diagnostics and KV/capture alignment remain valid.  The zero
        // coefficient disables the overlay without creating an invalid empty
        // capture request.
        arm.layer_indices = {static_cast<uint32_t>(basis.layer_index)};
        arm.coefficients = {apply_overlay ? 1.0f : 0.0f};
        arm.alpha = apply_overlay ? strength : 0.0f;
        arm.apply_overlay = apply_overlay;
        arm.fresh_context = true;
        arm.request_capture = true;
        arm.request_teacher_forced_margin = true;
        // Exploratory callers can request compact diagnostics and teacher
        // scoring only.  Full generation/verification is reserved for the
        // selected frontier arm; ordinary Shallow/Deep/TFO callers retain
        // the historical full-execution default.
        arm.request_generation = full_execution;
        arm.request_host_verification = full_execution;
        arm.max_capture_bytes = 4U * 1024U * 1024U;
        arm.max_generated_tokens = 64;
        const std::string identity = job.id + "\n" + request.wave_id + "\n" +
            std::to_string(index) + "\n" + intervention_ref + "\n" +
            std::to_string(arm.alpha);
        arm.arm_id = "flydelta://runtime/" + job.id + "/coefficient/" +
            hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        request.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result batch;
    if (!daemon_flydelta_execute_batch(provider, request, batch, error) ||
            batch.arms.size() != request.arms.size()) {
        if (error.empty()) error = "FlyDelta coefficient batch returned incomplete arms";
        return false;
    }
    trials.clear();
    margins.clear();
    geometries.clear();
    geometry_available.clear();
    for (size_t index = 0; index < batch.arms.size(); ++index) {
        const auto & arm = batch.arms[index];
        if (!arm.executed || arm.arm_id != request.arms[index].arm_id) {
            error = "FlyDelta coefficient batch returned an invalid arm";
            return false;
        }
        common_flydelta_counterfactual_trial trial;
        trial.executed = arm.executed;
        trial.verifier_known = arm.verifier_known;
        trial.passed = daemon_flydelta_arm_semantic_passed(arm);
        trial.quality = arm.quality;
        trial.overlay_applied = apply_overlay;
        trial.intervention_count = request.arms[index].layer_indices.size();
        trial.evidence_ref = arm.generation_ref;
        trials.push_back(std::move(trial));
        margins.push_back(arm.margin);
        common_flydelta_representation_diagnostics geometry;
        geometry.cosine = arm.cosine;
        geometry.progress = arm.progress;
        geometry.leakage = arm.leakage;
        geometry.shift_norm = arm.shift_norm;
        geometries.push_back(geometry);
        geometry_available.push_back(arm.geometry_available);
    }
    return true;
}

// Resolves the existing host-certified behavior deltas into the low-rank
// basis material consumed by Shallow/Deep.  A direction resource is a
// rank-one intervention transport; it must not be reused as an invented
// rank-two basis merely because a later phase needs one.
bool daemon_flydelta_run_rank1_alpha_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_fixture & fixture,
        const uint32_t layer_index,
        const float scale,
        const bool apply_overlay,
        const size_t proposal_index,
        const bool full_execution,
        common_flydelta_counterfactual_trial & trial,
        common_flydelta_decision_margin & margin,
        common_flydelta_representation_diagnostics & geometry,
        bool & geometry_available,
        std::string & arm_id,
        std::string & error) {
    error.clear();
    trial = {};
    margin = {};
    geometry = {};
    geometry_available = false;
    arm_id.clear();
    common_flydelta_arm_batch_request batch;
    batch.wave_id = "adaptive-alpha";
    const std::string identity = job.id + "\n" + batch.wave_id + "\n" +
        std::to_string(proposal_index) + "\n" + std::to_string(scale) + "\n" +
        (apply_overlay ? "overlay" : "baseline");
    batch.batch_id = job.id + ":adaptive-alpha:" +
        hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
    common_flydelta_arm_request arm;
    arm.job_id = job.id;
    arm.wave_id = batch.wave_id;
    arm.proposal_index = proposal_index;
    arm.arm_id = batch.batch_id + ":0";
    arm.context_ref = job.seed.baseline_ref;
    arm.fixture_ref = fixture.id;
    arm.intervention_ref = job.seed.candidate_ref;
    arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
    arm.layer_indices = {layer_index};
    arm.coefficients = {apply_overlay ? 1.0f : 0.0f};
    arm.alpha = apply_overlay ? scale : 0.0f;
    arm.apply_overlay = apply_overlay;
    arm.fresh_context = true;
    arm.request_capture = true;
    arm.request_teacher_forced_margin = true;
    arm.request_generation = full_execution;
    arm.request_host_verification = full_execution;
    arm.max_capture_bytes = 4U * 1024U * 1024U;
    arm.max_generated_tokens = full_execution ? 64 : 0;
    if (!common_flydelta_arm_request_validate(arm, error)) return false;
    batch.arms.push_back(arm);
    common_flydelta_arm_batch_result results;
    if (!daemon_flydelta_execute_batch(provider, batch, results, error) ||
            results.arms.size() != 1 || !results.arms.front().executed) {
        if (error.empty()) error = "FlyDelta AdaptiveAlpha arm did not execute";
        return false;
    }
    const auto & result = results.arms.front();
    trial.executed = result.executed;
    trial.verifier_known = result.verifier_known;
    trial.passed = daemon_flydelta_arm_semantic_passed(result);
    trial.quality = result.quality;
    trial.overlay_applied = apply_overlay;
    trial.intervention_count = apply_overlay ? 1 : 0;
    trial.evidence_ref = full_execution ? result.generation_ref : result.capture_ref;
    margin = result.margin;
    arm_id = arm.arm_id;
    return common_flydelta_counterfactual_trial_validate(trial, error) &&
        common_flydelta_decision_margin_validate(margin, error);
}

bool daemon_flydelta_execute_batch(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_arm_batch_request & request,
        common_flydelta_arm_batch_result & result,
        std::string & error) {
    common_agent_server_flydelta_binding binding;
    binding.prepare_arm = [provider](const auto & arm, auto & generation, std::string & callback_error) {
        return daemon_flydelta_prepare_arm(provider, arm, generation, callback_error);
    };
    binding.finalize_arm = [provider](const auto & arm, const auto & generation,
            auto & arm_result, std::string & callback_error) {
        return daemon_flydelta_finalize_arm(provider, arm, generation, arm_result, callback_error);
    };
    binding.score_teacher_forced_margin_batch = [provider](
            const auto & arms, const auto & contexts, auto & scorer, auto & margins,
            std::string & callback_error) {
        return daemon_flydelta_score_teacher_forced_margin_batch(
            provider, arms, contexts, scorer, margins, callback_error);
    };
    const bool native_batch = provider->host->server().per_sequence_cvec_batch_enabled() &&
        provider->host->context_key().n_parallel > 1;
    return common_agent_server_context_host_run_flydelta_arm_batch(
        provider->host, binding, native_batch, request, result, error);
}



} // namespace agent_daemon_flydelta_internal
