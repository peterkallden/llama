#include "agent-daemon-adapter.h"
#include "../adaptation/flydelta-teaching-material-store.h"
#include "../adaptation/agent-learning-lifecycle-store.h"

#include "../cli/agent-cli-host-adapter.h"
#include "../cli/agent-cli-selection.h"
#include "../runtime/agent-inference-capacity-gate.h"
#include "../runtime/agent-model-loaders.h"
#include "../runtime/agent-server-context-host.h"
#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-bootstrap-zoom-state-store.h"
#include "agent/adaptation/flydelta/flydelta-alpha-response-search.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"
#include "agent/adaptation/flydelta/flydelta-deep-search.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-representation-augmentation-state-store.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"
#include "hash/hash.h"
#include "tools/server/server-context.h"
#include "llama.h"
#include "agent/agent-scope.h"
#include "tools/agent/cli/agent-cli-memory-tools.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "../data/agent-data-store-factory.h"
#ifdef LLAMA_MEMORY_USE_COZO
#include "memory/cozo/memory-cozo.h"
#endif
#ifdef LLAMA_PLAN_USE_COZO
#include "plan/cozo/plan-cozo.h"
#endif
#ifdef LLAMA_MEMORY_USE_SQLITE
#include "memory/sqlite/memory-sqlite.h"
#endif
#ifdef LLAMA_PLAN_USE_SQLITE
#include "plan/sqlite/plan-sqlite.h"
#endif

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>

namespace {

using json = nlohmann::ordered_json;

// V0 production provider.  The daemon owns opaque resource references; this
// provider resolves only explicitly supplied JSON resources and fails closed
// when any semantic material is absent.  It deliberately does not infer a
// prompt, repair, decision pair or verifier from a FlyDelta job.
struct daemon_flydelta_resource_provider {
    std::shared_ptr<common_agent_server_context_host> host;
    agent_resource_store * resources = nullptr;
    agent_resource_read_authority authority;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    int n_predict = 0;
    int n_threads = 0;
    size_t model_n_embd = 0;
    size_t model_n_layers = 0;
    // The lifecycle journal is the daemon-owned durable state seam.  Raw
    // activations remain ephemeral and are retained only for the bounded
    // arm wave that produced them.
    std::shared_ptr<common_learning_lifecycle_store> lifecycle_store;
    std::mutex capture_mutex;
    std::unordered_map<std::string, std::shared_ptr<const common_flydelta_hidden_state_capture>> captures;
    mutable std::mutex composed_direction_mutex;
    std::unordered_map<std::string, std::vector<common_flydelta_basis_direction>> composed_directions;
    std::mutex orchestration_mutex;
    std::unordered_map<std::string,
        std::pair<common_flydelta_experiment_plan, common_flydelta_utility_history>> orchestration_states;
    std::unordered_map<std::string, std::string> bootstrap_state_by_orchestration_ref;
    std::unordered_map<std::string, std::string> bootstrap_state_by_continuation;
};

bool daemon_flydelta_read_json(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        json & parsed,
        std::string & error) {
    error.clear();
    if (provider.resources == nullptr || reference.empty() || reference.size() > 512) {
        error = "FlyDelta resource provider received an invalid reference";
        return false;
    }
    std::string text;
    if (!provider.resources->read_text(reference, provider.authority, 1024U * 1024U, text, error)) {
        return false;
    }
    try {
        parsed = json::parse(text);
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta referenced resource is not valid JSON: ") + exception.what();
        return false;
    }
    if (!parsed.is_object()) {
        error = "FlyDelta referenced resource must be a JSON object";
        return false;
    }
    return true;
}

bool daemon_flydelta_parse_context(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        common_agent_generation_request & request,
        std::string & error) {
    json parsed;
    if (!daemon_flydelta_read_json(provider, reference, parsed, error)) return false;
    request = {};
    request.purpose = common_agent_generation_purpose::draft;
    request.options.n_predict = provider.n_predict;
    request.options.n_threads = provider.n_threads;
    if (parsed.contains("n_predict")) {
        if (!parsed["n_predict"].is_number_integer() || parsed["n_predict"].get<int>() <= 0) {
            error = "FlyDelta context n_predict must be positive";
            return false;
        }
        request.options.n_predict = std::min(provider.n_predict, parsed["n_predict"].get<int>());
    }
    try {
        if (parsed.contains("messages")) {
            if (!parsed["messages"].is_array() || parsed["messages"].empty()) {
                error = "FlyDelta context messages must be a non-empty array";
                return false;
            }
            request.messages = common_chat_msgs_parse_oaicompat(parsed["messages"]);
        } else if (parsed.contains("prompt") && parsed["prompt"].is_string() &&
                !parsed["prompt"].get<std::string>().empty()) {
            request.messages = {{"user", parsed["prompt"].get<std::string>()}};
        } else {
            error = "FlyDelta context requires messages or prompt";
            return false;
        }
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta context messages are invalid: ") + exception.what();
        return false;
    }
    return true;
}

bool daemon_flydelta_parse_directions(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        std::vector<common_flydelta_basis_direction> & directions,
        std::string & error) {
    {
        std::lock_guard<std::mutex> lock(provider.composed_direction_mutex);
        const auto composed = provider.composed_directions.find(reference);
        if (composed != provider.composed_directions.end()) {
            directions = composed->second;
            if (directions.empty() || directions.size() > 64) {
                error = "FlyDelta composed intervention is invalid";
                return false;
            }
            return true;
        }
    }
    json parsed;
    if (!daemon_flydelta_read_json(provider, reference, parsed, error)) return false;
    if (!parsed.contains("directions") || !parsed["directions"].is_array() ||
            parsed["directions"].empty() || parsed["directions"].size() > 64) {
        error = "FlyDelta intervention requires a bounded directions array";
        return false;
    }
    directions.clear();
    directions.reserve(parsed["directions"].size());
    std::set<uint32_t> layers;
    try {
        for (const auto & item : parsed["directions"]) {
            if (!item.is_object() || !item.contains("layer_index") ||
                    !item["layer_index"].is_number_unsigned() ||
                    (!item.contains("values") && !item.contains("sparse_values"))) {
                error = "FlyDelta intervention direction is malformed";
                return false;
            }
            common_flydelta_basis_direction direction;
            direction.layer_index = static_cast<int32_t>(item["layer_index"].get<uint32_t>());
            if (item.contains("values")) {
                if (!item["values"].is_array()) {
                    error = "FlyDelta intervention direction values must be an array";
                    return false;
                }
                direction.values = item["values"].get<std::vector<float>>();
            } else {
                if (!item["sparse_values"].is_array() || item["sparse_values"].empty() ||
                        item["sparse_values"].size() > provider.model_n_embd) {
                    error = "FlyDelta sparse intervention direction is malformed";
                    return false;
                }
                direction.values.assign(provider.model_n_embd, 0.0f);
                std::set<size_t> indices;
                for (const auto & entry : item["sparse_values"]) {
                    if (!entry.is_object() || !entry.contains("index") ||
                            !entry["index"].is_number_unsigned() || !entry.contains("value") ||
                            !entry["value"].is_number()) {
                        error = "FlyDelta sparse intervention entry is malformed";
                        return false;
                    }
                    const auto index = entry["index"].get<size_t>();
                    const auto value = entry["value"].get<float>();
                    if (index >= provider.model_n_embd || !std::isfinite(value) ||
                            !indices.insert(index).second) {
                        error = "FlyDelta sparse intervention entry is invalid";
                        return false;
                    }
                    direction.values[index] = value;
                }
            }
            if (direction.layer_index < 1 ||
                    static_cast<size_t>(direction.layer_index) >= provider.model_n_layers ||
                    direction.values.size() != provider.model_n_embd ||
                    !layers.insert(static_cast<uint32_t>(direction.layer_index)).second) {
                error = "FlyDelta intervention direction is incompatible with the resident model";
                return false;
            }
            double squared = 0.0;
            for (const float value : direction.values) {
                if (!std::isfinite(value)) {
                    error = "FlyDelta intervention direction contains a non-finite value";
                    return false;
                }
                squared += static_cast<double>(value) * value;
            }
            if (!std::isfinite(squared) || squared <= 0.0) {
                error = "FlyDelta intervention direction has zero norm";
                return false;
            }
            const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared));
            for (float & value : direction.values) value *= inverse_norm;
            directions.push_back(std::move(direction));
        }
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta intervention direction values are invalid: ") + exception.what();
        return false;
    }
    return true;
}

bool daemon_flydelta_parse_behavior_delta(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        common_flydelta_behavior_delta & delta,
        common_flydelta_intervention_credit & credit,
        std::string & error) {
    json value;
    if (!daemon_flydelta_read_json(provider, reference, value, error)) return false;
    try {
        if (value.value("kind", "") != "flydelta_behavior_delta" ||
                !value.contains("values") || !value["values"].is_array() ||
                !value.contains("credit") || !value["credit"].is_object()) {
            error = "FlyDelta behavior-delta resource is missing typed delta or credit material";
            return false;
        }
        delta = {};
        delta.id = value.value("id", reference);
        delta.behavior_key = value.value("behavior_key", "");
        delta.capture_manifest_id = value.value("capture_manifest_id", "");
        delta.host_evidence_ref = value.value("host_evidence_ref", "");
        delta.scope_fingerprint = value.value("scope_fingerprint", "");
        delta.tokenizer_fingerprint = value.value("tokenizer_fingerprint", "");
        delta.template_fingerprint = value.value("template_fingerprint", "");
        delta.generation_semantics_fingerprint = value.value("generation_semantics_fingerprint", "");
        delta.model_profile_fingerprint = value.value("model_profile_fingerprint", "");
        delta.execution_context_fingerprint = value.value("execution_context_fingerprint", "");
        delta.capture_layout_revision = value.value("capture_layout_revision", "");
        delta.layer_index = value.value("layer_index", -1);
        delta.values = value.at("values").get<std::vector<float>>();
        const auto source = value.value("source", "tool_repair");
        if (source == "tool_repair") delta.source = common_adaptation_evidence_source::tool_repair;
        else if (source == "procedure_blueprint") delta.source = common_adaptation_evidence_source::procedure_blueprint;
        else if (source == "user_correction") delta.source = common_adaptation_evidence_source::user_correction;
        else if (source == "user_taught_concept") delta.source = common_adaptation_evidence_source::user_taught_concept;
        else {
            error = "FlyDelta behavior-delta source is unsupported";
            return false;
        }
        credit = {};
        const auto & credit_json = value.at("credit");
        credit.experiment_id = credit_json.value("experiment_id", "");
        credit.candidate_id = credit_json.value("candidate_id", "");
        credit.fixture_id = credit_json.value("fixture_id", "");
        credit.quality_delta = credit_json.value("quality_delta", 0.0f);
        credit.eligible_for_learning = credit_json.value("eligible_for_learning", false);
        const auto outcome = credit_json.value("outcome", "unknown");
        if (outcome == "helped") credit.outcome = common_flydelta_counterfactual_outcome::helped;
        else if (outcome == "neutral") credit.outcome = common_flydelta_counterfactual_outcome::neutral;
        else if (outcome == "harmed") credit.outcome = common_flydelta_counterfactual_outcome::harmed;
        else if (outcome == "unknown") credit.outcome = common_flydelta_counterfactual_outcome::unknown;
        else {
            error = "FlyDelta behavior-delta credit outcome is invalid";
            return false;
        }
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta behavior-delta resource is malformed: ") + exception.what();
        return false;
    }
    return common_flydelta_behavior_delta_validate(
        delta, provider.model_n_embd, 4U * 1024U * 1024U, error) &&
        common_flydelta_intervention_credit_validate(credit, error);
}

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
        std::lock_guard<std::mutex> lock(provider->capture_mutex);
        provider->captures[arm.arm_id] = generation.flydelta_capture;
    }
    json fixture;
    if (!daemon_flydelta_read_json(*provider, arm.fixture_ref, fixture, error)) return false;
    if (arm.request_host_verification &&
            (fixture.contains("expected_contains") || fixture.contains("expected_tool"))) {
        std::string expected;
        if (fixture.contains("expected_tool")) {
            if (!fixture["expected_tool"].is_string()) {
                error = "FlyDelta fixture expected_tool must be a string";
                return false;
            }
            expected = fixture["expected_tool"].get<std::string>();
        } else if (!fixture["expected_contains"].is_string()) {
            error = "FlyDelta fixture expected_contains must be a string";
            return false;
        }
        result.host_evaluated = true;
        result.verifier_known = true;
        if (expected.empty()) expected = fixture["expected_contains"].get<std::string>();
        // An individual arm has no paired baseline in this callback. A match
        // is therefore retained as NEUTRAL experimental truth, never HELPED.
        result.host_outcome = generation.content.find(expected) != std::string::npos
            ? common_flydelta_counterfactual_outcome::neutral
            : common_flydelta_counterfactual_outcome::harmed;
        result.quality = result.host_outcome == common_flydelta_counterfactual_outcome::neutral
            ? 1.0f : 0.0f;
    }
    result.generation_ref = "flydelta://runtime/generation/" + arm.arm_id;
    return common_flydelta_arm_result_validate(result, error);
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

bool daemon_flydelta_fixture_from_job(
        const common_flydelta_experiment_job & job,
        common_flydelta_experiment_fixture & fixture,
        std::string & error) {
    fixture = {};
    fixture.id = job.seed.verifier_ref;
    fixture.task_fingerprint = job.seed.task_fingerprint;
    fixture.model_profile_fingerprint = job.seed.model_profile_fingerprint;
    fixture.tokenizer_fingerprint = job.seed.tokenizer_fingerprint;
    fixture.template_fingerprint = job.seed.template_fingerprint;
    fixture.execution_context_fingerprint = job.seed.execution_context_fingerprint;
    fixture.verifier_revision = job.seed.verifier_ref;
    return common_flydelta_experiment_fixture_validate(fixture, error);
}

bool daemon_flydelta_parse_depth(
        const std::string & value, common_flydelta_search_depth & depth) {
    if (value == "bootstrap") depth = common_flydelta_search_depth::bootstrap;
    else if (value == "shallow") depth = common_flydelta_search_depth::shallow;
    else if (value == "deep") depth = common_flydelta_search_depth::deep;
    else return false;
    return true;
}

bool daemon_flydelta_parse_phase(
        const std::string & value, common_flydelta_experiment_phase & phase) {
    if (value == "bootstrap") phase = common_flydelta_experiment_phase::bootstrap;
    else if (value == "shallow_controls") phase = common_flydelta_experiment_phase::shallow_controls;
    else if (value == "deep_controls") phase = common_flydelta_experiment_phase::deep_controls;
    else return false;
    return true;
}

bool daemon_flydelta_parse_refinement(
        const std::string & value, common_flydelta_bootstrap_refinement_kind & kind) {
    if (value == "bootstrap_zoom") kind = common_flydelta_bootstrap_refinement_kind::bootstrap_zoom;
    else if (value == "adaptive_alpha") kind = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
    else return false;
    return true;
}

json daemon_flydelta_orchestration_state_json(
        const common_flydelta_experiment_plan & plan,
        const common_flydelta_utility_history & history,
        const std::string & bootstrap_state_ref = {}) {
    const auto & region = plan.continuation.region;
    return {
        {"kind", "flydelta_daemon_orchestration_state"},
        {"continuation", {
            {"direction_index", plan.continuation.direction_index},
            {"region_trial_index", plan.continuation.region_trial_index},
            {"search_score", plan.continuation.search_score},
            {"host_helped", plan.continuation.host_helped},
            {"region", {
                {"layer_indices", region.layer_indices},
                {"anchor_layer_index", region.anchor_layer_index},
                {"total_scale", region.total_scale},
                {"per_layer_scale", region.per_layer_scale},
                {"source", common_flydelta_layer_search_candidate_source_name(region.source)},
            }},
        }},
        {"depth", common_flydelta_search_depth_name(plan.depth)},
        {"phase", common_flydelta_experiment_phase_name(plan.phase)},
        {"bootstrap_refinement", common_flydelta_bootstrap_refinement_kind_name(plan.bootstrap_refinement)},
        {"budget", {
            {"depth", common_flydelta_search_depth_name(plan.budget.depth)},
            {"max_region_trials", plan.budget.max_region_trials},
            {"max_coefficient_trials", plan.budget.max_coefficient_trials},
            {"full_generation_top_k", plan.budget.full_generation_top_k},
            {"build_aggregate_directions", plan.budget.build_aggregate_directions},
            {"require_decision_margin", plan.budget.require_decision_margin},
            {"allow_tfo_lite", plan.budget.allow_tfo_lite},
            {"include_opposite_control", plan.budget.include_opposite_control},
        }},
        {"required_compatible_directions", plan.required_compatible_directions},
        {"require_decision_margin", plan.require_decision_margin},
        {"run_rank_two_controls_first", plan.run_rank_two_controls_first},
        {"tfo_lite_permitted_by_evidence", plan.tfo_lite_permitted_by_evidence},
        {"tfo_lite_requires_utility_gate", plan.tfo_lite_requires_utility_gate},
        {"run_tfo_lite", plan.run_tfo_lite},
        {"run_orthogonal_search", plan.run_orthogonal_search},
        {"history", {
            {"qualifying_streak", history.qualifying_streak},
            {"nonqualifying_streak", history.nonqualifying_streak},
        }},
        {"bootstrap_state_ref", bootstrap_state_ref},
    };
}

std::string daemon_flydelta_continuation_key(const common_flydelta_experiment_plan & plan) {
    std::string identity = std::to_string(plan.continuation.region.anchor_layer_index);
    for (const auto layer : plan.continuation.region.layer_indices) {
        identity += "\n" + std::to_string(layer);
    }
    return hash_sha256_hex(identity.data(), identity.size());
}

std::string daemon_flydelta_bootstrap_key(const common_flydelta_bootstrap_zoom_state & state) {
    std::string identity = std::to_string(state.anchor_layer);
    for (const auto layer : state.local_layers) identity += "\n" + std::to_string(layer);
    return hash_sha256_hex(identity.data(), identity.size());
}

bool daemon_flydelta_orchestration_state_from_json(
        const std::string & text,
        common_flydelta_experiment_plan & plan,
        common_flydelta_utility_history & history,
        std::string & error) {
    try {
        const auto value = json::parse(text);
        if (!value.is_object() || value.value("kind", "") != "flydelta_daemon_orchestration_state") {
            error = "FlyDelta lifecycle record is not an orchestration state";
            return false;
        }
        plan = {};
        const auto continuation = value.at("continuation");
        const auto region = continuation.at("region");
        plan.continuation.direction_index = continuation.value("direction_index", size_t{0});
        plan.continuation.region_trial_index = continuation.value("region_trial_index", size_t{0});
        plan.continuation.search_score = continuation.value("search_score", 0.0f);
        plan.continuation.host_helped = continuation.value("host_helped", false);
        plan.continuation.region.layer_indices = region.at("layer_indices").get<std::vector<uint32_t>>();
        plan.continuation.region.anchor_layer_index = region.value("anchor_layer_index", 0U);
        plan.continuation.region.total_scale = region.value("total_scale", 0.0f);
        plan.continuation.region.per_layer_scale = region.value("per_layer_scale", 0.0f);
        const auto source = region.value("source", "diagnostic_singleton");
        if (source == "diagnostic_singleton") {
            plan.continuation.region.source = common_flydelta_layer_search_candidate_source::diagnostic_singleton;
        } else if (source == "neighborhood_expansion") {
            plan.continuation.region.source = common_flydelta_layer_search_candidate_source::neighborhood_expansion;
        } else {
            error = "FlyDelta orchestration region source is invalid";
            return false;
        }
        if (!daemon_flydelta_parse_depth(value.at("depth").get<std::string>(), plan.depth) ||
                !daemon_flydelta_parse_phase(value.at("phase").get<std::string>(), plan.phase) ||
                !daemon_flydelta_parse_refinement(value.at("bootstrap_refinement").get<std::string>(),
                    plan.bootstrap_refinement)) {
            error = "FlyDelta orchestration phase metadata is invalid";
            return false;
        }
        const auto budget = value.at("budget");
        if (!daemon_flydelta_parse_depth(budget.at("depth").get<std::string>(), plan.budget.depth)) {
            error = "FlyDelta orchestration budget depth is invalid";
            return false;
        }
        plan.budget.max_region_trials = budget.value("max_region_trials", size_t{0});
        plan.budget.max_coefficient_trials = budget.value("max_coefficient_trials", size_t{0});
        plan.budget.full_generation_top_k = budget.value("full_generation_top_k", size_t{0});
        plan.budget.build_aggregate_directions = budget.value("build_aggregate_directions", false);
        plan.budget.require_decision_margin = budget.value("require_decision_margin", false);
        plan.budget.allow_tfo_lite = budget.value("allow_tfo_lite", false);
        plan.budget.include_opposite_control = budget.value("include_opposite_control", false);
        plan.required_compatible_directions = value.value("required_compatible_directions", size_t{1});
        plan.require_decision_margin = value.value("require_decision_margin", false);
        plan.run_rank_two_controls_first = value.value("run_rank_two_controls_first", false);
        plan.tfo_lite_permitted_by_evidence = value.value("tfo_lite_permitted_by_evidence", false);
        plan.tfo_lite_requires_utility_gate = value.value("tfo_lite_requires_utility_gate", false);
        plan.run_tfo_lite = value.value("run_tfo_lite", false);
        plan.run_orthogonal_search = value.value("run_orthogonal_search", false);
        const auto state_history = value.at("history");
        history.qualifying_streak = state_history.value("qualifying_streak", size_t{0});
        history.nonqualifying_streak = state_history.value("nonqualifying_streak", size_t{0});
        std::string validation_error;
        if (!common_flydelta_search_budget_validate(plan.budget, validation_error)) {
            error = "FlyDelta orchestration state budget is invalid: " + validation_error;
            return false;
        }
        return true;
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta orchestration state JSON is invalid: ") + exception.what();
        return false;
    }
}

bool daemon_flydelta_capture_for_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & arm_id,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture) {
    std::lock_guard<std::mutex> lock(provider->capture_mutex);
    const auto it = provider->captures.find(arm_id);
    if (it == provider->captures.end()) return false;
    capture = it->second;
    return static_cast<bool>(capture);
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

bool daemon_flydelta_run_bootstrap_zoom_slice(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error) {
    error.clear();
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, directions, error)) return false;

    common_flydelta_bootstrap_zoom_config zoom_config;
    std::vector<common_flydelta_bootstrap_zoom_candidate> candidates;
    if (resume.phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom) {
        if (!common_flydelta_propose_bootstrap_alpha_zoom(
                resume.anchor_layer, resume.selected_scale, zoom_config, candidates, error)) return false;
    } else if (resume.phase == common_flydelta_bootstrap_zoom_phase::profile_zoom) {
        if (!common_flydelta_propose_bootstrap_profile_zoom(
                resume.local_layers, resume.anchor_layer, resume.selected_scale,
                zoom_config, candidates, error)) return false;
    } else {
        // Sign control is the final deterministic BootstrapZoom slice.  The
        // next orchestration decision, not this runner, decides whether a
        // later rank-one adaptive-alpha slice is warranted.
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = common_flydelta_bootstrap_zoom_phase::sign_control;
        candidate.layer_indices = {resume.anchor_layer};
        candidate.layer_weights = {1.0f};
        candidate.total_scale = resume.selected_scale;
        candidate.opposite_sign_control = true;
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty()) {
        error = "FlyDelta BootstrapZoom produced no bounded candidates";
        return false;
    }
    const size_t remaining = resume.extra_model_trials >= zoom_config.max_extra_model_trials
        ? 0 : zoom_config.max_extra_model_trials - resume.extra_model_trials;
    if (remaining == 0) {
        error = "FlyDelta BootstrapZoom budget is exhausted";
        return false;
    }
    if (candidates.size() > remaining) candidates.resize(remaining);

    common_flydelta_arm_batch_request batch;
    batch.batch_id = job.id + ":bootstrap-zoom:" + std::to_string(resume.next_candidate_index);
    batch.wave_id = common_flydelta_bootstrap_zoom_phase_name(resume.phase);
    common_flydelta_arm_request baseline;
    baseline.job_id = job.id;
    baseline.wave_id = batch.wave_id;
    baseline.proposal_index = 0;
    baseline.arm_id = batch.batch_id + ":baseline";
    baseline.context_ref = job.seed.baseline_ref;
    baseline.fixture_ref = fixture.id;
    baseline.intervention_ref = job.seed.candidate_ref;
    baseline.batch_compatibility_key = baseline.context_ref + "\n" + baseline.fixture_ref;
    baseline.fresh_context = true;
    baseline.request_capture = true;
    baseline.request_teacher_forced_margin = true;
    baseline.request_generation = true;
    baseline.request_host_verification = true;
    baseline.max_generated_tokens = 64;
    for (const auto & direction : directions) baseline.layer_indices.push_back(
        static_cast<uint32_t>(direction.layer_index));
    baseline.coefficients.assign(baseline.layer_indices.size(), 0.0f);
    if (!common_flydelta_arm_request_validate(baseline, error)) return false;
    batch.arms.push_back(baseline);
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto & candidate = candidates[index];
        common_flydelta_arm_request arm;
        arm.job_id = job.id;
        arm.wave_id = batch.wave_id;
        arm.proposal_index = index + 1;
        arm.arm_id = batch.batch_id + ":" + std::to_string(index);
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = fixture.id;
        arm.intervention_ref = job.seed.candidate_ref;
        arm.batch_compatibility_key = baseline.batch_compatibility_key;
        arm.apply_overlay = true;
        arm.fresh_context = true;
        arm.alpha = candidate.opposite_sign_control ? -candidate.total_scale : candidate.total_scale;
        arm.layer_indices = candidate.layer_indices;
        arm.coefficients = candidate.layer_weights;
        arm.request_capture = true;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = true;
        arm.request_host_verification = true;
        arm.max_generated_tokens = 64;
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result results;
    if (!daemon_flydelta_execute_batch(provider, batch, results, error) ||
            results.arms.size() != batch.arms.size()) {
        if (error.empty()) error = "FlyDelta BootstrapZoom batch returned incomplete arms";
        return false;
    }

    next = resume;
    std::vector<common_flydelta_bootstrap_zoom_trial> new_trials;
    common_flydelta_search_pipeline_direction_result direction_result;
    direction_result.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction_result.direction.layer_index = directions.front().layer_index;
    direction_result.direction.values = directions.front().values;
    direction_result.direction.origin = "runtime_resource";
    direction_result.direction.extraction_id = job.seed.candidate_ref;
    direction_result.direction.source_samples = 1;
    direction_result.direction.retained_samples = 1;
    direction_result.direction.experimental_only = true;
    const auto & baseline_result = results.arms.front();
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto & arm = batch.arms[index + 1];
        const auto & arm_result = results.arms[index + 1];
        common_flydelta_representation_diagnostics diagnostics;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, baseline.arm_id, arm,
                candidates[index].layer_indices.front(), diagnostics, error)) return false;
        common_flydelta_bootstrap_zoom_trial trial;
        trial.candidate = candidates[index];
        trial.outcome = arm_result.host_outcome;
        trial.host_evaluated = arm_result.host_evaluated;
        trial.verifier_known = arm_result.verifier_known;
        trial.margin_available = baseline_result.margin.available && arm_result.margin.available;
        trial.margin_delta = trial.margin_available
            ? arm_result.margin.normalized_delta() - baseline_result.margin.normalized_delta() : 0.0f;
        trial.diagnostics_available = diagnostics.layer_index != 0;
        trial.diagnostics = diagnostics;
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
        new_trials.push_back(trial);

        common_flydelta_intervention_region_trial region_trial;
        region_trial.candidate.layer_indices = candidates[index].layer_indices;
        region_trial.candidate.anchor_layer_index = candidates[index].layer_indices.front();
        region_trial.candidate.total_scale = candidates[index].total_scale;
        region_trial.candidate.per_layer_scale = candidates[index].total_scale /
            std::sqrt(static_cast<float>(candidates[index].layer_indices.size()));
        region_trial.requested_total_scale = candidates[index].total_scale;
        region_trial.executed_total_scale = candidates[index].total_scale;
        region_trial.outcome = arm_result.host_outcome;
        region_trial.quality_delta = arm_result.quality - baseline_result.quality;
        region_trial.margin = arm_result.margin;
        region_trial.margin_comparison.available = trial.margin_available;
        region_trial.margin_comparison.baseline = baseline_result.margin;
        region_trial.margin_comparison.candidate = arm_result.margin;
        region_trial.executed = arm_result.executed;
        region_trial.verifier_known = arm_result.verifier_known;
        region_trial.geometry_available = trial.diagnostics_available;
        region_trial.geometry = diagnostics;
        region_trial.safe_to_continue = trial.diagnostics_available &&
            diagnostics.leakage <= 1.0f && diagnostics.shift_norm <= 1.0f;
        region_trial.promising = trial.margin_available && trial.margin_delta > 0.0f;
        region_trial.search_score = trial.margin_delta;
        region_trial.evidence_ref = arm_result.generation_ref;
        direction_result.region_trials.push_back(std::move(region_trial));
    }
    next.completed_trials.insert(next.completed_trials.end(), new_trials.begin(), new_trials.end());
    next.extra_model_trials += new_trials.size();
    next.next_candidate_index += new_trials.size();
    if (!common_flydelta_select_bootstrap_zoom_trial(next.completed_trials, next.selection, error)) return false;
    if (next.selection.selected) {
        const auto & selected = next.completed_trials[next.selection.trial_index];
        next.selected_scale = selected.candidate.total_scale;
        next.best_margin_delta = selected.margin_delta;
        next.best_search_score = next.selection.search_score;
    }
    if (resume.phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom) {
        next.phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
    } else if (resume.phase == common_flydelta_bootstrap_zoom_phase::profile_zoom &&
            zoom_config.include_opposite_sign_control) {
        next.phase = common_flydelta_bootstrap_zoom_phase::sign_control;
    } else {
        next.phase = common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
    }
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.search_status = output.directions.front().region_trials.empty()
        ? common_flydelta_search_status::no_useful_utility
        : common_flydelta_search_status::candidate_available;
    return true;
}

bool daemon_flydelta_run_search_pipeline(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_search_pipeline_config & config,
        common_flydelta_search_pipeline_result & result,
        std::string & error) {
    common_flydelta_experiment_fixture fixture;
    fixture.id = job.seed.verifier_ref;
    fixture.task_fingerprint = job.seed.task_fingerprint;
    fixture.model_profile_fingerprint = job.seed.model_profile_fingerprint;
    fixture.tokenizer_fingerprint = job.seed.tokenizer_fingerprint;
    fixture.template_fingerprint = job.seed.template_fingerprint;
    fixture.execution_context_fingerprint = job.seed.execution_context_fingerprint;
    fixture.verifier_revision = job.seed.verifier_ref;
    if (!common_flydelta_experiment_fixture_validate(fixture, error)) return false;

    std::vector<common_flydelta_basis_direction> basis;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, basis, error)) return false;
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
    std::sort(input.layer_anchors.begin(), input.layer_anchors.end());
    input.layer_anchors.erase(std::unique(input.layer_anchors.begin(), input.layer_anchors.end()),
        input.layer_anchors.end());
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
        arm.request_generation = true;
        arm.request_host_verification = true;
        // Diagnostics are first-class search observations.  The production
        // host captures the small requested layer set and keeps it only for
        // this bounded comparison wave.
        arm.request_capture = true;
        arm.max_generated_tokens = 64;
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
                apply_overlay ? "region-scalar" : "baseline", 0, arm, runner_error)) return false;
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
        trial.verifier_known = arm_result.verifier_known;
        trial.passed = arm_result.host_outcome == common_flydelta_counterfactual_outcome::helped;
        trial.quality = arm_result.quality;
        trial.overlay_applied = apply_overlay;
        trial.intervention_count = arm.layer_indices.size();
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
                    ? 0U : layer->layer_indices.front();
                const auto direction = std::find_if(available.begin(), available.end(),
                    [&](const auto & value) { return value.layer_index == static_cast<int32_t>(diagnostic_layer); });
                if (direction != available.end()) {
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
                    common_flydelta_representation_diagnostics diagnostics;
                    if (!common_flydelta_representation_diagnostics_from_captures(
                            *baseline_capture, *overlay_capture, delta,
                            4U * 1024U * 1024U, diagnostics, runner_error)) return false;
                    geometry.available = true;
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
                    batch.wave_id, index, arm, runner_error)) return false;
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
            trial.verifier_known = arm.verifier_known;
            trial.passed = arm.host_outcome == common_flydelta_counterfactual_outcome::helped;
            trial.quality = arm.quality;
            trial.overlay_applied = true;
            trial.intervention_count = batch.arms[index].layer_indices.size();
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
                                candidates[index].anchor_layer_index);
                        });
                    if (direction != available.end()) {
                        common_flydelta_behavior_delta delta;
                        delta.id = arm.arm_id + ":diagnostic";
                        delta.behavior_key = job.seed.behavior_key;
                        delta.capture_manifest_id = batch.arms[index].context_ref;
                        delta.host_evidence_ref = job.seed.evidence_ref;
                        delta.model_profile_fingerprint = provider->model_profile_fingerprint;
                        delta.execution_context_fingerprint = job.seed.execution_context_fingerprint;
                        delta.capture_layout_revision = provider->capture_layout_revision;
                        delta.layer_index = direction->layer_index;
                        delta.values = direction->values;
                        common_flydelta_representation_diagnostics diagnostics;
                        if (!common_flydelta_representation_diagnostics_from_captures(
                                *baseline_capture, *overlay_capture, delta,
                                4U * 1024U * 1024U, diagnostics, runner_error)) return false;
                        geometry.available = true;
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
    return common_flydelta_run_search_pipeline_batched(
        fixture, config, {input}, runner, batch_runner, result, error);
}

bool daemon_flydelta_register_composed_directions(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & source_ref,
        std::vector<common_flydelta_basis_direction> directions,
        const std::string & derivation,
        std::string & reference,
        std::string & error) {
    error.clear();
    if (directions.empty() || directions.size() > 64 || derivation.empty()) {
        error = "FlyDelta composed direction artifact is invalid";
        return false;
    }
    json serialized = {{"directions", json::array()}};
    std::string identity = source_ref + "\n" + derivation;
    for (const auto & direction : directions) {
        if (direction.layer_index < 1 || direction.values.size() != provider->model_n_embd) {
            error = "FlyDelta composed direction artifact has an invalid layer or dimension";
            return false;
        }
        serialized["directions"].push_back({
            {"layer_index", static_cast<uint32_t>(direction.layer_index)},
            {"values", direction.values},
        });
        identity += "\n" + std::to_string(direction.layer_index);
        for (const float value : direction.values) identity += ":" + std::to_string(value);
    }
    reference = "flydelta://runtime/composed/" +
        hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
    if (provider->resources != nullptr) {
        agent_resource_put_request request;
        request.name = "flydelta-composed-" +
            hash_sha256_hex(identity.data(), identity.size()).substr(0, 16) + ".json";
        request.description = "Durable composed FlyDelta intervention direction";
        request.mime_type = "application/json";
        request.text = serialized.dump();
        request.scope = common_runtime_resource_scope::session;
        request.namespace_id = provider->authority.namespace_id;
        request.session_id = provider->authority.session_id;
        request.source_provider = "flydelta";
        request.source_tool = "compose-direction";
        request.created_at = std::time(nullptr);
        request.metadata.purpose = "flydelta_composed_direction";
        request.metadata.content_summary =
            "Durable normalized composed FlyDelta direction for a resumed bounded slice.";
        request.metadata.processing_cache_key = reference;
        request.lineage.parent_uri = source_ref;
        request.lineage.chunk_count = 1;
        request.lineage.chunk_index = 0;
        request.lineage.derivation = derivation;
        agent_resource_descriptor descriptor;
        if (!provider->resources->put_text(request, descriptor, error)) return false;
        if (descriptor.uri.empty()) {
            error = "FlyDelta composed direction resource has no URI";
            return false;
        }
        reference = descriptor.uri;
    }
    {
        std::lock_guard<std::mutex> lock(provider->composed_direction_mutex);
        provider->composed_directions[reference] = std::move(directions);
    }
    return true;
}

bool daemon_flydelta_register_composed_direction(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & source_ref,
        const common_flydelta_low_rank_basis & basis,
        const std::vector<float> & coefficients,
        std::string & reference,
        std::string & error) {
    error.clear();
    if (basis.vectors.empty() || basis.vectors.size() != coefficients.size() ||
            basis.layer_index < 1 || basis.dimension == 0) {
        error = "FlyDelta composed coefficient direction has invalid basis shape";
        return false;
    }
    std::vector<float> values(basis.dimension, 0.0f);
    double squared = 0.0;
    std::string identity = source_ref + "\n" + std::to_string(basis.layer_index);
    for (size_t index = 0; index < coefficients.size(); ++index) {
        if (!std::isfinite(coefficients[index]) ||
                basis.vectors[index].size() != basis.dimension) {
            error = "FlyDelta composed coefficient direction contains invalid values";
            return false;
        }
        identity += "\n" + std::to_string(coefficients[index]);
        for (size_t value = 0; value < basis.dimension; ++value) {
            values[value] += coefficients[index] * basis.vectors[index][value];
        }
    }
    for (const float value : values) squared += static_cast<double>(value) * value;
    if (!std::isfinite(squared) || squared <= 0.0) {
        error = "FlyDelta composed coefficient direction has zero norm";
        return false;
    }
    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared));
    for (float & value : values) value *= inverse_norm;
    common_flydelta_basis_direction direction;
    direction.layer_index = basis.layer_index;
    direction.values = std::move(values);
    return daemon_flydelta_register_composed_directions(
        provider, source_ref, {std::move(direction)},
        "flydelta:coefficient-composition", reference, error);
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
        std::string & error) {
    error.clear();
    if (coefficients.empty()) {
        error = "FlyDelta coefficient batch is empty";
        return false;
    }
    common_flydelta_arm_batch_request request;
    request.batch_id = job.id + ":coefficient";
    request.wave_id = apply_overlay ? "coefficient-controls" : "coefficient-baseline";
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
        arm.layer_indices = apply_overlay ? std::vector<uint32_t>{
            static_cast<uint32_t>(basis.layer_index)} : std::vector<uint32_t>{};
        arm.coefficients = apply_overlay ? std::vector<float>{1.0f} : std::vector<float>{};
        arm.alpha = apply_overlay ? strength : 0.0f;
        arm.apply_overlay = apply_overlay;
        arm.fresh_context = true;
        arm.request_capture = true;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = true;
        arm.request_host_verification = true;
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
        trial.passed = arm.host_outcome == common_flydelta_counterfactual_outcome::helped;
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
bool daemon_flydelta_resolve_evidence_direction_candidates(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const int32_t layer_index,
        const size_t required_count,
        std::vector<common_flydelta_direction_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    for (const auto & reference : job.behavior_delta_ids) {
        common_flydelta_behavior_delta delta;
        common_flydelta_intervention_credit credit;
        if (!daemon_flydelta_parse_behavior_delta(*provider, reference, delta, credit, error)) {
            return false;
        }
        // Natural Shallow/Deep capacity is based on verified positive
        // evidence. Experimental UNKNOWN/NEUTRAL material belongs to the
        // orthogonal/augmentation paths and must not raise this basis.
        if (credit.outcome != common_flydelta_counterfactual_outcome::helped ||
                !credit.eligible_for_learning || delta.layer_index != layer_index) {
            continue;
        }
        if (delta.behavior_key != job.seed.behavior_key ||
                delta.model_profile_fingerprint != job.seed.model_profile_fingerprint ||
                delta.execution_context_fingerprint != job.seed.execution_context_fingerprint ||
                delta.capture_layout_revision != provider->capture_layout_revision) {
            error = "FlyDelta post-Bootstrap evidence delta is incompatible with the resumed job";
            return false;
        }
        common_flydelta_direction_candidate candidate;
        candidate.kind = common_flydelta_direction_kind::raw_repair;
        candidate.layer_index = delta.layer_index;
        candidate.values = std::move(delta.values);
        candidate.origin = "runtime_verified_evidence";
        candidate.extraction_id = reference;
        candidate.source_samples = 1;
        candidate.retained_samples = 1;
        candidate.experimental_only = false;
        if (!common_flydelta_direction_candidate_validate(
                candidate, provider->model_n_embd, error)) return false;
        candidates.push_back(std::move(candidate));
    }
    if (candidates.size() < required_count) {
        error = "FlyDelta post-Bootstrap phase requires compatible HELPED evidence directions at the selected layer";
        return false;
    }
    return true;
}

bool daemon_flydelta_run_rank1_alpha_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_fixture & fixture,
        const uint32_t layer_index,
        const float scale,
        const bool apply_overlay,
        const size_t proposal_index,
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
    arm.request_generation = true;
    arm.request_host_verification = true;
    arm.max_capture_bytes = 4U * 1024U * 1024U;
    arm.max_generated_tokens = 64;
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
    trial.passed = result.host_outcome == common_flydelta_counterfactual_outcome::helped;
    trial.quality = result.quality;
    trial.overlay_applied = apply_overlay;
    trial.intervention_count = apply_overlay ? 1 : 0;
    trial.evidence_ref = result.generation_ref;
    margin = result.margin;
    arm_id = arm.arm_id;
    return common_flydelta_counterfactual_trial_validate(trial, error) &&
        common_flydelta_decision_margin_validate(margin, error);
}

bool daemon_flydelta_run_adaptive_alpha_slice(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error) {
    error.clear();
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, directions, error)) {
        return false;
    }
    const auto direction = std::find_if(directions.begin(), directions.end(), [&](const auto & value) {
        return value.layer_index == static_cast<int32_t>(resume.anchor_layer);
    });
    if (direction == directions.end()) {
        error = "FlyDelta AdaptiveAlpha anchor direction is absent from the intervention resource";
        return false;
    }
    common_flydelta_alpha_response_search_config config;
    const float prior_scale = resume.alpha_response_available && resume.alpha_response.selected
        ? resume.alpha_response.scale : resume.selected_scale;
    config.seed_scale = std::max(0.0001f, std::min(config.max_scale, prior_scale));
    std::string baseline_arm_id;
    size_t proposal_index = resume.next_candidate_index;
    const auto runner = [provider, &job, &fixture, layer = resume.anchor_layer,
            &baseline_arm_id, &proposal_index](const common_flydelta_experiment_fixture &,
            const float scale, const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & geometry_available, std::string & runner_error) mutable {
        std::string arm_id;
        if (!daemon_flydelta_run_rank1_alpha_arm(provider, job, fixture, layer, scale,
                apply_overlay, proposal_index++, trial, margin, geometry,
                geometry_available, arm_id, runner_error)) return false;
        if (!apply_overlay) {
            baseline_arm_id = std::move(arm_id);
            return true;
        }
        if (baseline_arm_id.empty()) {
            runner_error = "FlyDelta AdaptiveAlpha candidate ran before its baseline";
            return false;
        }
        common_flydelta_arm_request diagnostic_arm;
        diagnostic_arm.arm_id = arm_id;
        diagnostic_arm.context_ref = job.seed.baseline_ref;
        diagnostic_arm.intervention_ref = job.seed.candidate_ref;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, baseline_arm_id,
                diagnostic_arm, layer, geometry, runner_error)) return false;
        geometry_available = geometry.layer_index != 0;
        return true;
    };
    std::vector<common_flydelta_alpha_response_trial> trials;
    common_flydelta_alpha_response_selection selection;
    if (!common_flydelta_run_alpha_response_search(
            fixture, config, runner, trials, selection, error)) return false;

    next = resume;
    next.phase = common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
    next.refinement_kind = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
    next.alpha_response_available = true;
    next.alpha_response = selection;
    if (selection.selected) {
        next.selected_scale = selection.scale;
        next.best_search_score = selection.utility;
        if (selection.best_margin_available) {
            next.best_margin_delta = selection.best_margin_delta_normalized;
        }
    }
    next.next_candidate_index = proposal_index;

    common_flydelta_search_pipeline_direction_result direction_result;
    direction_result.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction_result.direction.layer_index = direction->layer_index;
    direction_result.direction.values = direction->values;
    direction_result.direction.origin = "runtime_adaptive_alpha";
    direction_result.direction.extraction_id = job.seed.candidate_ref;
    direction_result.direction.source_samples = 1;
    direction_result.direction.retained_samples = 1;
    direction_result.direction.experimental_only = true;
    for (const auto & alpha_trial : trials) {
        common_flydelta_intervention_region_trial region_trial;
        region_trial.candidate.layer_indices = {resume.anchor_layer};
        region_trial.candidate.anchor_layer_index = resume.anchor_layer;
        region_trial.candidate.total_scale = alpha_trial.scale;
        region_trial.candidate.per_layer_scale = alpha_trial.scale;
        region_trial.requested_total_scale = alpha_trial.requested_scale;
        region_trial.executed_total_scale = alpha_trial.scale;
        region_trial.outcome = alpha_trial.outcome;
        region_trial.quality_delta = alpha_trial.counterfactual.quality;
        region_trial.margin = alpha_trial.margin;
        region_trial.executed = alpha_trial.counterfactual.executed;
        region_trial.verifier_known = alpha_trial.counterfactual.verifier_known;
        region_trial.geometry_available = alpha_trial.geometry_available;
        region_trial.geometry = alpha_trial.geometry;
        region_trial.safe_to_continue = alpha_trial.safe_to_continue;
        region_trial.promising = alpha_trial.safe_to_continue &&
            alpha_trial.utility > config.utility_epsilon;
        region_trial.search_score = alpha_trial.utility;
        region_trial.evidence_ref = alpha_trial.counterfactual.evidence_ref;
        if (!common_flydelta_intervention_region_trial_validate(region_trial, error)) return false;
        direction_result.region_trials.push_back(std::move(region_trial));
    }
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.alpha_response_available = true;
    output.alpha_response = selection;
    output.search_status = selection.selected
        ? common_flydelta_search_status::candidate_available
        : common_flydelta_search_status::no_useful_utility;
    return true;
}

// Orthogonal remains a same-phase experimental escape.  The common layer
// derives the profile-space residual from persisted BootstrapZoom trials; the
// daemon owns only the fresh baseline/control wave that validates it.
bool daemon_flydelta_run_orthogonal_slice(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error) {
    error.clear();
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    common_flydelta_orthogonal_search_config config;
    common_flydelta_orthogonal_search_input input;
    if (!common_flydelta_prepare_orthogonal_search_input(config, resume, input, error)) {
        return false;
    }
    common_flydelta_orthogonal_search_result orthogonal;
    if (!common_flydelta_build_orthogonal_search_direction(
            config, input.rank1_intervention, input.arms, orthogonal, error)) return false;
    if (!orthogonal.available) {
        output = {};
        output.search_status = common_flydelta_search_status::no_useful_utility;
        next = resume;
        return true;
    }
    std::vector<common_flydelta_basis_direction> available;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, available, error)) {
        return false;
    }
    const auto normalize = [](std::vector<float> values) {
        double energy = 0.0;
        for (const float value : values) energy += static_cast<double>(value) * value;
        if (energy > 0.0) {
            const float inverse = static_cast<float>(1.0 / std::sqrt(energy));
            for (float & value : values) value *= inverse;
        }
        return values;
    };
    struct orthogonal_control {
        std::string label;
        std::vector<float> profile;
        bool opposite = false;
    };
    std::vector<orthogonal_control> controls;
    controls.push_back({"rank1", input.rank1_intervention, false});
    controls.push_back({"orthogonal", orthogonal.direction, false});
    std::vector<float> sum(input.rank1_intervention.size(), 0.0f);
    std::vector<float> difference(input.rank1_intervention.size(), 0.0f);
    for (size_t index = 0; index < sum.size(); ++index) {
        sum[index] = input.rank1_intervention[index] + orthogonal.direction[index];
        difference[index] = input.rank1_intervention[index] - orthogonal.direction[index];
    }
    controls.push_back({"sum", normalize(std::move(sum)), false});
    controls.push_back({"difference", normalize(std::move(difference)), true});

    common_flydelta_arm_batch_request batch;
    batch.batch_id = job.id + ":orthogonal";
    batch.wave_id = "orthogonal-controls";
    common_flydelta_arm_request baseline;
    baseline.job_id = job.id;
    baseline.wave_id = batch.wave_id;
    baseline.proposal_index = 0;
    baseline.context_ref = job.seed.baseline_ref;
    baseline.fixture_ref = fixture.id;
    baseline.intervention_ref = job.seed.candidate_ref;
    baseline.batch_compatibility_key = baseline.context_ref + "\n" + baseline.fixture_ref;
    baseline.layer_indices = input.local_layers;
    baseline.coefficients.assign(input.local_layers.size(), 0.0f);
    baseline.fresh_context = true;
    baseline.request_capture = true;
    baseline.request_teacher_forced_margin = true;
    baseline.max_capture_bytes = 4U * 1024U * 1024U;
    baseline.max_generated_tokens = 64;
    baseline.arm_id = "flydelta://runtime/" + job.id + "/orthogonal/baseline";
    if (!common_flydelta_arm_request_validate(baseline, error)) return false;
    batch.arms.push_back(baseline);

    std::vector<common_flydelta_bootstrap_zoom_candidate> candidates;
    candidates.reserve(controls.size());
    for (size_t control_index = 0; control_index < controls.size(); ++control_index) {
        const auto & control = controls[control_index];
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = control.opposite
            ? common_flydelta_bootstrap_zoom_phase::sign_control
            : common_flydelta_bootstrap_zoom_phase::profile_zoom;
        candidate.total_scale = std::max(0.0001f, resume.selected_scale);
        candidate.opposite_sign_control = control.opposite;
        for (size_t layer = 0; layer < input.local_layers.size(); ++layer) {
            if (std::fabs(control.profile[layer]) <= 0.000001f) continue;
            const auto direction = std::find_if(available.begin(), available.end(),
                [&](const auto & value) {
                    return value.layer_index == static_cast<int32_t>(input.local_layers[layer]);
                });
            if (direction == available.end()) {
                error = "FlyDelta orthogonal control lacks a compatible source direction";
                return false;
            }
            candidate.layer_indices.push_back(input.local_layers[layer]);
            candidate.layer_weights.push_back(control.profile[layer]);
        }
        candidate.layer_weights = normalize(std::move(candidate.layer_weights));
        if (!common_flydelta_bootstrap_zoom_candidate_validate(candidate, error)) return false;
        common_flydelta_arm_request arm;
        arm.job_id = job.id;
        arm.wave_id = batch.wave_id;
        arm.proposal_index = control_index + 1;
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = fixture.id;
        arm.intervention_ref = job.seed.candidate_ref;
        arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
        arm.layer_indices = candidate.layer_indices;
        arm.coefficients = candidate.layer_weights;
        arm.alpha = candidate.total_scale;
        arm.apply_overlay = true;
        arm.fresh_context = true;
        arm.request_capture = true;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = true;
        arm.request_host_verification = true;
        arm.max_capture_bytes = 4U * 1024U * 1024U;
        arm.max_generated_tokens = 64;
        const std::string identity = job.id + "\n" + batch.wave_id + "\n" + control.label;
        arm.arm_id = "flydelta://runtime/" + job.id + "/orthogonal/" +
            hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        candidates.push_back(std::move(candidate));
        batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result executed;
    if (!daemon_flydelta_execute_batch(provider, batch, executed, error) ||
            executed.arms.size() != batch.arms.size()) {
        if (error.empty()) error = "FlyDelta orthogonal control batch returned incomplete arms";
        return false;
    }
    const auto & baseline_result = executed.arms.front();
    if (!baseline_result.executed || baseline_result.arm_id != baseline.arm_id) {
        error = "FlyDelta orthogonal baseline arm is invalid";
        return false;
    }
    std::vector<common_flydelta_bootstrap_zoom_trial> trials;
    trials.reserve(candidates.size());
    const auto anchor_direction = std::find_if(available.begin(), available.end(),
        [&](const auto & value) {
            return value.layer_index == static_cast<int32_t>(resume.anchor_layer);
        });
    if (anchor_direction == available.end()) {
        error = "FlyDelta orthogonal surface lacks the rank-one anchor direction";
        return false;
    }
    common_flydelta_search_pipeline_direction_result direction_result;
    // A direction candidate is an embedding-space vector.  Orthogonal here
    // is a local layer-profile coordinate, so retain the compatible anchor
    // vector and record the new experimental surface exclusively in origin
    // and the persisted surface trials rather than fabricating a mismatched
    // embedding-space direction.
    direction_result.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction_result.direction.layer_index = anchor_direction->layer_index;
    direction_result.direction.values = anchor_direction->values;
    direction_result.direction.origin = "runtime_orthogonal_profile";
    direction_result.direction.extraction_id = job.seed.candidate_ref;
    direction_result.direction.source_samples = orthogonal.source_arm_count;
    direction_result.direction.retained_samples = orthogonal.source_arm_count;
    direction_result.direction.experimental_only = true;
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto & arm = executed.arms[index + 1];
        if (!arm.executed || arm.arm_id != batch.arms[index + 1].arm_id) {
            error = "FlyDelta orthogonal control arm is invalid";
            return false;
        }
        common_flydelta_bootstrap_zoom_trial trial;
        trial.candidate = candidates[index];
        trial.outcome = arm.host_outcome;
        trial.host_evaluated = arm.host_evaluated;
        trial.verifier_known = arm.verifier_known;
        trial.margin_available = arm.margin.available && baseline_result.margin.available;
        trial.margin_delta = trial.margin_available
            ? arm.margin.normalized_delta() - baseline_result.margin.normalized_delta() : 0.0f;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, baseline.arm_id,
                batch.arms[index + 1], trial.candidate.layer_indices.front(),
                trial.diagnostics, error)) return false;
        trial.diagnostics_available = trial.diagnostics.schema_version == 1 &&
            trial.diagnostics.layer_index != 0;
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
        trials.push_back(trial);
        common_flydelta_intervention_region_trial region_trial;
        region_trial.candidate.layer_indices = trial.candidate.layer_indices;
        region_trial.candidate.anchor_layer_index = resume.anchor_layer;
        region_trial.candidate.total_scale = trial.candidate.total_scale;
        region_trial.candidate.per_layer_scale = trial.candidate.total_scale /
            std::sqrt(static_cast<float>(std::max<size_t>(1, trial.candidate.layer_indices.size())));
        region_trial.requested_total_scale = trial.candidate.total_scale;
        region_trial.executed_total_scale = trial.candidate.total_scale;
        region_trial.outcome = trial.outcome;
        region_trial.quality_delta = arm.quality - baseline_result.quality;
        region_trial.margin = arm.margin;
        region_trial.executed = arm.executed;
        region_trial.verifier_known = arm.verifier_known;
        region_trial.geometry_available = trial.diagnostics_available;
        region_trial.geometry = trial.diagnostics;
        region_trial.safe_to_continue = trial.diagnostics_available &&
            trial.diagnostics.cosine >= config.minimum_cosine &&
            trial.diagnostics.progress > 0.0f &&
            trial.diagnostics.leakage <= config.maximum_leakage &&
            trial.diagnostics.shift_norm <= config.maximum_shift_norm;
        region_trial.promising = region_trial.safe_to_continue &&
            trial.margin_available && trial.margin_delta > 0.0f;
        region_trial.search_score = trial.margin_available ? trial.margin_delta : 0.0f;
        region_trial.evidence_ref = arm.generation_ref;
        if (!common_flydelta_intervention_region_trial_validate(region_trial, error)) return false;
        direction_result.region_trials.push_back(std::move(region_trial));
    }
    common_flydelta_bootstrap_zoom_selection selection;
    bool has_safe_trial = false;
    for (const auto & trial : trials) {
        if (trial.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
        const bool geometry_safe = !trial.diagnostics_available ||
            (trial.diagnostics.cosine >= config.minimum_cosine &&
             trial.diagnostics.progress > 0.0f &&
             trial.diagnostics.leakage <= config.maximum_leakage &&
             trial.diagnostics.shift_norm <= config.maximum_shift_norm);
        if (geometry_safe) {
            has_safe_trial = true;
            break;
        }
    }
    if (has_safe_trial && !common_flydelta_select_bootstrap_zoom_trial(
            trials, selection, error)) return false;
    direction_result.region_selection.selected = selection.selected;
    direction_result.region_selection.trial_index = selection.trial_index;
    direction_result.region_selection.score = selection.search_score;
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.search_status = selection.selected
        ? common_flydelta_search_status::candidate_available
        : common_flydelta_search_status::no_useful_utility;
    output.selection.selected = selection.selected;
    output.selection.intervention_region = selection.selected;
    output.selection.direction_index = 0;
    output.selection.region_trial_index = selection.trial_index;
    output.selection.score = selection.search_score;
    next = resume;
    next.parent_surface_revision = resume.surface_revision;
    next.surface_revision = resume.surface_revision + 1;
    next.search_rank = 2;
    next.surface_origin = "orthogonal_profile";
    next.parent_surface_ref = resume.state_ref;
    next.surface_trials = std::move(trials);
    return common_flydelta_bootstrap_zoom_state_validate(next, error);
}

bool daemon_flydelta_run_post_bootstrap_slice(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_plan & plan,
        common_flydelta_search_pipeline_result & output,
        std::string & error) {
    error.clear();
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    const int32_t layer_index = static_cast<int32_t>(plan.continuation.region.anchor_layer_index);
    std::vector<common_flydelta_direction_candidate> candidates;
    const size_t max_rank = plan.phase == common_flydelta_experiment_phase::shallow_controls
        ? 2 : 4;
    if (!daemon_flydelta_resolve_evidence_direction_candidates(
            provider, job, layer_index, 2, candidates, error)) return false;
    if (candidates.size() > max_rank) candidates.resize(max_rank);
    common_flydelta_low_rank_basis basis;
    if (!common_flydelta_build_low_rank_basis(
            candidates.front().values.size(), max_rank, candidates, basis, error)) return false;

    common_flydelta_coefficient_search_config coefficient_config;
    coefficient_config.max_candidates = plan.budget.max_coefficient_trials == 0
        ? 16 : std::min<size_t>(16, plan.budget.max_coefficient_trials);
    coefficient_config.strategy = plan.run_tfo_lite
        ? common_flydelta_coefficient_search_strategy::tfo_lite
        : common_flydelta_coefficient_search_strategy::coordinate;
    const auto run_single = [provider, &job, &fixture](
            const common_flydelta_experiment_fixture &,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<float> & coefficients,
            const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & has_geometry, std::string & runner_error) {
        std::vector<common_flydelta_counterfactual_trial> trials;
        std::vector<common_flydelta_decision_margin> margins;
        std::vector<common_flydelta_representation_diagnostics> geometries;
        std::vector<bool> geometry_flags;
        if (!daemon_flydelta_run_coefficient_arm_batch(
                provider, job, fixture, current_basis, {coefficients}, apply_overlay,
                trials, margins, geometries, geometry_flags, runner_error)) return false;
        trial = std::move(trials.front());
        margin = std::move(margins.front());
        geometry = std::move(geometries.front());
        has_geometry = geometry_flags.front();
        return true;
    };
    const auto run_batch = [provider, &job, &fixture] (
            const common_flydelta_experiment_fixture &,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_representation_diagnostics> & geometries,
            std::vector<bool> & geometry_flags, std::string & runner_error) {
        return daemon_flydelta_run_coefficient_arm_batch(
            provider, job, fixture, current_basis, coefficients, true,
            trials, margins, geometries, geometry_flags, runner_error);
    };
    std::vector<common_flydelta_coefficient_trial> trials;
    common_flydelta_coefficient_selection selection;
    if (plan.phase == common_flydelta_experiment_phase::deep_controls) {
        common_flydelta_deep_search_config deep_config;
        deep_config.max_rank = std::min<size_t>(max_rank, 4);
        deep_config.max_directions = std::min<size_t>(candidates.size(), 4);
        deep_config.full_generation_top_k = plan.budget.full_generation_top_k == 0
            ? 3 : std::min<size_t>(3, plan.budget.full_generation_top_k);
        deep_config.coefficients = coefficient_config;
        std::vector<common_flydelta_deep_search_direction> deep_directions;
        for (const auto & candidate : candidates) {
            common_flydelta_deep_search_direction value;
            value.direction = candidate;
            value.decision_score_available = true;
            value.decision_score = candidate.median_alignment;
            deep_directions.push_back(std::move(value));
        }
        common_flydelta_deep_search_result deep_result;
        if (!common_flydelta_run_deep_search_batched(
                fixture, deep_config, deep_directions, run_single, run_batch,
                run_single, run_batch, deep_result, error)) return false;
        basis = deep_result.basis;
        trials = deep_result.coefficient_trials;
        selection = deep_result.coefficient_selection;
    } else if (!common_flydelta_run_low_rank_coefficient_search_batched(
            fixture, basis, coefficient_config, run_single, run_batch,
            trials, selection, error)) {
        return false;
    }

    common_flydelta_search_pipeline_direction_result direction_result;
    direction_result.direction = candidates.front();
    direction_result.direction.values = basis.vectors.front();
    direction_result.direction.layer_index = basis.layer_index;
    direction_result.direction.origin = "runtime_post_bootstrap_basis";
    direction_result.region_trials.reserve(trials.size());
    for (const auto & coefficient_trial : trials) {
        common_flydelta_intervention_region_trial trial;
        trial.candidate.layer_indices = {static_cast<uint32_t>(basis.layer_index)};
        trial.candidate.anchor_layer_index = static_cast<uint32_t>(basis.layer_index);
        trial.candidate.total_scale = coefficient_trial.executed_strength;
        trial.candidate.per_layer_scale = coefficient_trial.executed_strength;
        trial.requested_total_scale = coefficient_trial.requested_strength;
        trial.executed_total_scale = coefficient_trial.executed_strength;
        trial.outcome = coefficient_trial.outcome;
        trial.quality_delta = coefficient_trial.quality_delta;
        trial.margin = coefficient_trial.margin;
        trial.executed = coefficient_trial.executed;
        trial.verifier_known = coefficient_trial.verifier_known;
        trial.geometry_available = coefficient_trial.geometry_available;
        trial.geometry = coefficient_trial.geometry;
        trial.search_score = coefficient_trial.search_fitness;
        trial.promising = coefficient_trial.search_fitness > 0.0f;
        trial.safe_to_continue = !coefficient_trial.geometry_available ||
            (coefficient_trial.geometry.shift_norm <= 1.0f &&
             coefficient_trial.geometry.leakage <= 1.0f);
        trial.evidence_ref = "flydelta://runtime/coefficient-trial/" + job.id;
        if (!common_flydelta_intervention_region_trial_validate(trial, error)) return false;
        direction_result.region_trials.push_back(std::move(trial));
    }
    direction_result.region_selection.selected = selection.selected;
    direction_result.region_selection.trial_index = selection.trial_index;
    direction_result.region_selection.score = selection.score;
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.search_status = trials.empty()
        ? common_flydelta_search_status::no_useful_utility
        : common_flydelta_search_status::candidate_available;
    output.selection.selected = selection.selected;
    output.selection.intervention_region = true;
    output.selection.direction_index = 0;
    output.selection.region_trial_index = selection.trial_index;
    output.selection.score = selection.score;
    return true;
}

std::function<bool(
        const std::shared_ptr<common_agent_server_context_host> &,
        common_agent_server_flydelta_binding &,
        std::string &)>
make_daemon_flydelta_resource_binding_factory(
        const daemon_options & options,
        agent_resource_store * resources) {
    return [options, resources](const std::shared_ptr<common_agent_server_context_host> & host,
            common_agent_server_flydelta_binding & binding, std::string & error) {
        error.clear();
        if (!host || resources == nullptr) {
            error = "FlyDelta resource provider requires a resident host and resource store";
            return false;
        }
        const auto * context = host->server().get_llama_context();
        const auto * model = context == nullptr ? nullptr : llama_get_model(context);
        if (model == nullptr || llama_model_n_embd(model) <= 0 || llama_model_n_layer(model) <= 2) {
            error = "FlyDelta resource provider requires a loaded compatible model";
            return false;
        }
        auto provider = std::make_shared<daemon_flydelta_resource_provider>();
        provider->host = host;
        provider->resources = resources;
        provider->authority.namespace_id = "default-namespace";
        provider->authority.session_id = "default-session";
        provider->model_profile_fingerprint = options.adaptation_flydelta_model_profile_fingerprint.empty()
            ? "model:" + std::filesystem::path(options.model).filename().string()
            : options.adaptation_flydelta_model_profile_fingerprint;
        provider->capture_layout_revision = options.adaptation_flydelta_capture_layout_revision;
        provider->n_predict = options.n_predict;
        provider->n_threads = options.n_threads;
        provider->model_n_embd = static_cast<size_t>(llama_model_n_embd(model));
        provider->model_n_layers = static_cast<size_t>(llama_model_n_layer(model));
        {
            auto lifecycle = make_agent_learning_lifecycle_store(
                options.adaptation_flydelta_lifecycle_backend,
                options.adaptation_flydelta_lifecycle_path, error);
            if (!lifecycle) return false;
            provider->lifecycle_store = std::shared_ptr<common_learning_lifecycle_store>(
                std::move(lifecycle));
        }

        binding.primitives.capture = true;
        binding.primitives.overlay = true;
        binding.primitives.generation = true;
        binding.primitives.teacher_forced_scoring = true;
        binding.primitives.host_verification = true;
        binding.prepare_arm = [provider](const auto & arm, auto & request, std::string & callback_error) {
            return daemon_flydelta_prepare_arm(provider, arm, request, callback_error);
        };
        binding.finalize_arm = [provider](const auto & arm, const auto & generation,
                auto & result, std::string & callback_error) {
            return daemon_flydelta_finalize_arm(provider, arm, generation, result, callback_error);
        };
        binding.score_teacher_forced_margin_batch = [provider](
                const auto & arms, const auto & contexts, auto & scorer, auto & margins,
                std::string & callback_error) {
            return daemon_flydelta_score_teacher_forced_margin_batch(
                provider, arms, contexts, scorer, margins, callback_error);
        };
        binding.register_evaluator = [provider](common_flydelta_evaluator_config & config,
                common_flydelta_evaluator_callbacks & callbacks, std::string & callback_error) {
            config = {};
            config.pipeline.dimension = provider->model_n_embd;
            config.pipeline.max_directions = 1;
            config.pipeline.whirlpool_max_rounds = 1;
            config.pipeline.whirlpool_probes_per_round = 4;
            config.pipeline.whirlpool_max_trials = 4;
            config.pipeline.region_max_trials = 4;
            config.direction.dimension = provider->model_n_embd;
            config.direction.layer_index = 1;
            config.direction.behavior_key = "runtime/resource";
            config.direction.model_profile_fingerprint = provider->model_profile_fingerprint;
            config.direction.execution_context_fingerprint = "runtime-resource-v0";
            config.direction.capture_layout_revision = provider->capture_layout_revision;
            callbacks = {};
            const auto pipeline = config.pipeline;
            callbacks.resolve_behavior_delta = [provider](const std::string & reference,
                    common_flydelta_behavior_delta & delta,
                    common_flydelta_intervention_credit & credit,
                    std::string & resolver_error) {
                return daemon_flydelta_parse_behavior_delta(
                    *provider, reference, delta, credit, resolver_error);
            };
            // ConceptSynthesis produces ordinary experimental direction
            // material. Persist it in the same immutable direction registry
            // used by the normal search path; the scheduler passes the
            // returned reference as the next job's candidate_ref. This seam
            // never approves, activates or promotes the concept direction.
            callbacks.persist_experimental_direction = [provider](
                    const common_flydelta_direction_candidate & candidate,
                    std::string & direction_ref,
                    std::string & persist_error) {
                if (!common_flydelta_direction_candidate_validate(
                        candidate, provider->model_n_embd, persist_error)) return false;
                common_flydelta_basis_direction direction;
                direction.layer_index = candidate.layer_index;
                direction.values = candidate.values;
                const std::string source_ref = candidate.extraction_id.empty()
                    ? "flydelta://concept-graft/anonymous"
                    : candidate.extraction_id;
                return daemon_flydelta_register_composed_directions(
                    provider, source_ref, {std::move(direction)},
                    "flydelta:concept-graft", direction_ref, persist_error);
            };
            callbacks.run_search_pipeline = [provider, pipeline](
                    const common_flydelta_experiment_job & job,
                    common_flydelta_search_pipeline_result & result,
                    std::string & runner_error) {
                return daemon_flydelta_run_search_pipeline(provider, job, pipeline, result, runner_error);
            };
            callbacks.run_search_pipeline_with_state = [provider, pipeline](
                    const common_flydelta_experiment_job & job,
                    const common_flydelta_bootstrap_zoom_state * resume,
                    common_flydelta_search_pipeline_result & result,
                    common_flydelta_bootstrap_zoom_state & next,
                    std::string & runner_error) {
                if (resume != nullptr) {
                    return daemon_flydelta_run_bootstrap_zoom_slice(
                        provider, job, *resume, result, next, runner_error);
                }
                if (!daemon_flydelta_run_search_pipeline(provider, job, pipeline, result, runner_error)) {
                    return false;
                }
                std::vector<common_flydelta_basis_direction> directions;
                if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref,
                        directions, runner_error)) return false;
                common_flydelta_search_continuation continuation;
                if (!common_flydelta_select_search_continuation(result, continuation, runner_error)) {
                    return false;
                }
                next = {};
                next.behavior_key = job.seed.behavior_key;
                next.model_profile_fingerprint = job.seed.model_profile_fingerprint;
                next.capture_layout_revision = provider->capture_layout_revision;
                next.anchor_layer = continuation.region.anchor_layer_index == 0
                    ? static_cast<uint32_t>(directions.front().layer_index)
                    : continuation.region.anchor_layer_index;
                next.selected_scale = continuation.region.total_scale > 0.0f
                    ? continuation.region.total_scale : pipeline.scale.initial_scale;
                next.best_search_score = continuation.search_score;
                next.best_margin_delta = continuation.search_score;
                next.local_layers = continuation.region.layer_indices;
                if (next.local_layers.empty()) next.local_layers = {
                    static_cast<uint32_t>(directions.front().layer_index)};
                next.surface_origin = "bootstrap_rank1";
                return common_flydelta_bootstrap_zoom_state_validate(next, runner_error);
            };
            // Reuse the append-only lifecycle journal already used by the
            // adaptation subsystem.  The worker receives opaque refs only;
            // model material stays in the daemon resource provider.
            common_flydelta_bootstrap_zoom_lifecycle_context bootstrap_context;
            bootstrap_context.namespace_id = provider->authority.namespace_id;
            bootstrap_context.session_id = provider->authority.session_id;
            bootstrap_context.source_id = "daemon-flydelta";
            bootstrap_context.created_at = "daemon-runtime-v1";
            if (!common_flydelta_configure_bootstrap_zoom_lifecycle_callbacks(
                    *provider->lifecycle_store, bootstrap_context, callbacks, callback_error)) {
                return false;
            }
            const auto lifecycle_persist_bootstrap_state = callbacks.persist_bootstrap_zoom_state;
            callbacks.persist_bootstrap_zoom_state = [provider, lifecycle_persist_bootstrap_state](
                    const common_flydelta_bootstrap_zoom_state & state,
                    std::string & state_ref,
                    std::string & state_error) {
                if (!lifecycle_persist_bootstrap_state(state, state_ref, state_error)) return false;
                std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                provider->bootstrap_state_by_continuation[
                    daemon_flydelta_bootstrap_key(state)] = state_ref;
                return true;
            };
            common_flydelta_lifecycle_event_context augmentation_context;
            augmentation_context.source_id = "daemon-flydelta";
            augmentation_context.scope.namespace_id = provider->authority.namespace_id;
            augmentation_context.scope.session_id = provider->authority.session_id;
            augmentation_context.created_at = "daemon-runtime-v1";
            if (!common_flydelta_configure_representation_augmentation_lifecycle_callbacks(
                    *provider->lifecycle_store, augmentation_context, callbacks, callback_error)) {
                return false;
            }
            const auto resolve_bootstrap_state = callbacks.resolve_bootstrap_zoom_state;
            callbacks.resolve_search_orchestration_state = [provider, resolve_bootstrap_state](
                    const std::string & state_ref,
                    common_flydelta_experiment_plan & plan,
                    common_flydelta_utility_history & history,
                    std::string & state_error) {
                {
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    const auto cached = provider->orchestration_states.find(state_ref);
                    if (cached != provider->orchestration_states.end()) {
                        plan = cached->second.first;
                        history = cached->second.second;
                        return true;
                    }
                }
                if (state_ref.rfind("flydelta://state/bootstrap-zoom/", 0) == 0) {
                    common_flydelta_bootstrap_zoom_state bootstrap;
                    if (!resolve_bootstrap_state || !resolve_bootstrap_state(
                            state_ref, bootstrap, state_error)) return false;
                    plan = {};
                    plan.continuation.region.layer_indices = bootstrap.local_layers;
                    plan.continuation.region.anchor_layer_index = bootstrap.anchor_layer;
                    plan.continuation.region.total_scale = bootstrap.selected_scale;
                    plan.continuation.region.per_layer_scale = bootstrap.selected_scale /
                        std::sqrt(static_cast<float>(std::max<size_t>(1, bootstrap.local_layers.size())));
                    plan.continuation.search_score = bootstrap.best_search_score;
                    plan.depth = common_flydelta_search_depth::bootstrap;
                    plan.phase = common_flydelta_experiment_phase::bootstrap;
                    plan.budget = common_flydelta_search_budget_for_depth(plan.depth);
                    plan.required_compatible_directions = 1;
                    history = {};
                    return common_flydelta_search_budget_validate(plan.budget, state_error);
                }
                const auto records = provider->lifecycle_store->list(state_error);
                if (!state_error.empty()) return false;
                for (auto it = records.rbegin(); it != records.rend(); ++it) {
                    if (it->kind != common_learning_lifecycle_kind::flydelta_experiment ||
                            it->subject_id != state_ref) continue;
                    if (!daemon_flydelta_orchestration_state_from_json(
                            it->payload_json, plan, history, state_error)) return false;
                    std::string bootstrap_ref;
                    try {
                        bootstrap_ref = json::parse(it->payload_json).value("bootstrap_state_ref", "");
                    } catch (...) {
                        state_error = "FlyDelta orchestration bootstrap reference is invalid";
                        return false;
                    }
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    provider->orchestration_states[state_ref] = {plan, history};
                    if (!bootstrap_ref.empty()) {
                        provider->bootstrap_state_by_orchestration_ref[state_ref] = bootstrap_ref;
                        provider->bootstrap_state_by_continuation[
                            daemon_flydelta_continuation_key(plan)] = bootstrap_ref;
                    }
                    return true;
                }
                state_error = "FlyDelta orchestration state was not found";
                return false;
            };
            callbacks.persist_search_orchestration_state = [provider](
                    const common_flydelta_experiment_plan & plan,
                    const common_flydelta_utility_history & history,
                    std::string & state_ref,
                    std::string & state_error) {
                std::string bootstrap_ref;
                {
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    const auto it = provider->bootstrap_state_by_continuation.find(
                        daemon_flydelta_continuation_key(plan));
                    if (it != provider->bootstrap_state_by_continuation.end()) bootstrap_ref = it->second;
                }
                const auto payload = daemon_flydelta_orchestration_state_json(
                    plan, history, bootstrap_ref).dump();
                const auto digest = hash_sha256_hex(payload.data(), payload.size()).substr(0, 32);
                state_ref = "flydelta://state/orchestration/" + digest;
                common_learning_lifecycle_record record;
                record.event_id = "flydelta://event/orchestration/" + digest;
                record.subject_id = state_ref;
                record.kind = common_learning_lifecycle_kind::flydelta_experiment;
                record.status = common_learning_lifecycle_status::running;
                record.idempotency_key = "flydelta/orchestration/" + digest;
                record.source_id = "daemon-flydelta";
                record.namespace_id = provider->authority.namespace_id;
                record.session_id = provider->authority.session_id;
                record.content_hash = "sha256:" + hash_sha256_hex(payload.data(), payload.size());
                record.created_at = "daemon-runtime-v1";
                record.payload_json = payload;
                if (!provider->lifecycle_store->append(record, state_error)) return false;
                std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                provider->orchestration_states[state_ref] = {plan, history};
                if (!bootstrap_ref.empty()) {
                    provider->bootstrap_state_by_orchestration_ref[state_ref] = bootstrap_ref;
                }
                return true;
            };
            const auto resolve_orchestration_state = callbacks.resolve_search_orchestration_state;
            const auto persist_bootstrap_state = callbacks.persist_bootstrap_zoom_state;
            callbacks.run_search_pipeline_with_search_state = [provider, resolve_bootstrap_state,
                    resolve_orchestration_state, persist_bootstrap_state](const common_flydelta_experiment_job & job,
                    const std::string & state_ref,
                    common_flydelta_search_pipeline_result & result,
                    std::string & next_state_ref,
                    std::string & state_error) {
                common_flydelta_experiment_plan plan;
                common_flydelta_utility_history history;
                if (!resolve_orchestration_state(state_ref, plan, history, state_error)) return false;
                if (plan.phase != common_flydelta_experiment_phase::bootstrap) {
                    if (plan.phase == common_flydelta_experiment_phase::shallow_controls ||
                            plan.phase == common_flydelta_experiment_phase::deep_controls) {
                        if (!daemon_flydelta_run_post_bootstrap_slice(
                                provider, job, plan, result, state_error)) return false;
                        next_state_ref = state_ref;
                        return true;
                    }
                    state_error = "FlyDelta daemon received an unsupported post-Bootstrap phase";
                    return false;
                }
                std::string bootstrap_ref = job.bootstrap_zoom_state_ref;
                if (bootstrap_ref.empty()) {
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    const auto it = provider->bootstrap_state_by_orchestration_ref.find(state_ref);
                    if (it != provider->bootstrap_state_by_orchestration_ref.end()) bootstrap_ref = it->second;
                }
                if (bootstrap_ref.empty()) {
                    state_error = "FlyDelta Bootstrap continuation requires a BootstrapZoom state ref";
                    return false;
                }
                common_flydelta_bootstrap_zoom_state bootstrap;
                if (!resolve_bootstrap_state(bootstrap_ref, bootstrap, state_error)) return false;
                if (plan.run_orthogonal_search) {
                    if (!daemon_flydelta_run_orthogonal_slice(
                            provider, job, bootstrap, result, bootstrap, state_error)) return false;
                } else if (bootstrap.phase == common_flydelta_bootstrap_zoom_phase::adaptive_alpha) {
                    if (!daemon_flydelta_run_adaptive_alpha_slice(
                            provider, job, bootstrap, result, bootstrap, state_error)) return false;
                } else if (!daemon_flydelta_run_bootstrap_zoom_slice(
                        provider, job, bootstrap, result, bootstrap, state_error)) {
                    return false;
                }
                std::string persisted_bootstrap_ref;
                if (!persist_bootstrap_state || !persist_bootstrap_state(
                        bootstrap, persisted_bootstrap_ref, state_error)) return false;
                {
                    std::lock_guard<std::mutex> lock(provider->orchestration_mutex);
                    provider->bootstrap_state_by_continuation[
                        daemon_flydelta_continuation_key(plan)] = persisted_bootstrap_ref;
                }
                // The typed Bootstrap state is persisted by the evaluator's
                // regular state callback.  Preserve the orchestration ref for
                // the pure policy decision that follows this slice.
                next_state_ref = state_ref;
                return true;
            };
            callback_error.clear();
            return true;
        };
        return true;
    };
}

bool flydelta_batch_mode_valid(const std::string & mode) {
    return mode == "disabled" || mode == "auto" || mode == "required";
}

bool flydelta_native_batch_requested(const daemon_options & options) {
    return options.adaptation_flydelta_enabled &&
        options.adaptation_flydelta_batch_mode != "disabled" &&
        options.n_gpu_layers > 0;
}

void apply_flydelta_auto_batch_capacity(
        const daemon_options & options,
        common_agent_model_catalog & catalog) {
    if (!flydelta_native_batch_requested(options)) return;
    const std::string profile_id = options.model_profile.empty()
        ? catalog.default_profile
        : options.model_profile;
    const auto profile = catalog.profiles.find(profile_id);
    if (profile == catalog.profiles.end()) return;
    const auto base = catalog.bases.find(profile->second.base_model_id);
    if (base == catalog.bases.end() || base->second.backend != "server-context") return;

    // A profile can request a larger capacity explicitly. The ordinary 1/1
    // compatibility default is upgraded only for the selected FlyDelta
    // profile; disabling the batch policy keeps a deliberately scalar profile.
    if (profile->second.n_parallel == 1 && profile->second.n_sequences == 1) {
        const auto capacity = static_cast<int>(options.adaptation_flydelta_batch_parallelism);
        profile->second.n_parallel = capacity;
        profile->second.n_sequences = capacity;
    }
}

bool open_daemon_model_residency(
        const daemon_options & options,
        std::shared_ptr<common_agent_runtime_model_residency> & residency,
        std::string & error) {
    residency.reset();
    if (options.model_catalog.profiles.empty()) {
        error.clear();
        return true;
    }
    if (!flydelta_batch_mode_valid(options.adaptation_flydelta_batch_mode)) {
        error = "runtime.adaptation.flydelta.batch_mode must be disabled, auto or required";
        return false;
    }
    if (options.adaptation_flydelta_batch_parallelism == 0 ||
            options.adaptation_flydelta_batch_parallelism > 8) {
        error = "runtime.adaptation.flydelta.batch_parallelism must be between 1 and 8";
        return false;
    }
    auto catalog = options.model_catalog;
    apply_flydelta_auto_batch_capacity(options, catalog);
    if (!common_agent_validate_model_catalog(catalog, error)) {
        error = "models: " + error;
        return false;
    }
    const common_agent_runtime_model_loader_config loader_config{
        options.n_gpu_layers,
        options.n_threads,
        true,
        flydelta_native_batch_requested(options),
        options.agent_trace,
        options.adaptation_flydelta_batch_mode != "required",
    };
    std::unordered_map<std::string,
        std::shared_ptr<common_agent_runtime_model_loader>> loaders;
    loaders.emplace("cli", std::make_shared<common_agent_runtime_cli_model_loader>(loader_config));
#ifndef LLAMA_AGENT_ANDROID_CLI_ONLY
    loaders.emplace(
        "server-context",
        std::make_shared<common_agent_runtime_server_context_model_loader>(loader_config));
#endif
    residency = std::make_shared<common_agent_runtime_model_residency>(
        std::move(catalog),
        std::move(loaders));
    error.clear();
    return true;
}

std::optional<common_memory_policy_pack> make_daemon_policy_pack(
        const daemon_options & options) {
    common_memory_policy_pack pack;
    pack.id = "daemon-session-policy";
    pack.purpose = "Run host-owned resident agent turns through the foreground daemon.";
    pack.goal = options.default_mode == "agent"
        ? "Plan, execute bounded steps, and answer without widening host authority."
        : "Answer through the resident runtime without widening host authority.";
    pack.constraints = {
        "Treat memory, tools, plans, and resources as host-owned authority rather than model choices.",
        "Keep tool use within the configured profile and bounded runtime policy.",
    };
    pack.decisions = {
        "Inference backend: server-context resident host.",
        std::string("Tool profile: ") + (options.tool_profile.empty() ? "none" : options.tool_profile),
    };
    if (options.default_mode == "agent") {
        pack.preferred_procedures.push_back("Prefer bounded planning plus evidence-backed synthesis before answering.");
    }
    return pack;
}

std::string resolve_memory_backend(
        const std::string & backend,
        const std::string & memory_db,
        std::string & error) {
    std::string resolved = backend;
    if (resolved == "auto") {
        resolved = memory_db.empty() ? "in-memory" : "cozo";
    }
    if (resolved == "in-memory" && !memory_db.empty()) {
        error = "--memory-db requires --backend cozo or the default auto backend";
        return {};
    }
    error.clear();
    return resolved;
}

std::string resolve_plan_backend(
        const std::string & backend,
        const std::string & plan_db,
        std::string & error) {
    std::string resolved = backend;
    if (resolved == "auto") {
        resolved = plan_db.empty() ? "in-memory" : "cozo";
    }
    if (resolved == "in-memory" && !plan_db.empty()) {
        error = "--plan-db requires --plan-backend cozo or the default auto backend";
        return {};
    }
    error.clear();
    return resolved;
}

class daemon_agent_embedding_provider final : public agent_embedding_provider {
public:
    daemon_agent_embedding_provider(std::string model_path, int n_gpu_layers)
        : model_path_(std::move(model_path)),
          n_gpu_layers_(n_gpu_layers) {}

    bool embed(
            const std::string & purpose,
            const std::string & text,
            std::vector<float> & embedding,
            std::string & error) override {
        return ensure_memory_cli_embedding_from_model(
            model_path_,
            n_gpu_layers_,
            text,
            embedding,
            purpose.c_str(),
            error);
    }

private:
    std::string model_path_;
    int n_gpu_layers_ = 0;
};

common_agent_runtime_policy make_daemon_runtime_policy(const daemon_options & options) {
    common_agent_runtime_policy policy;
    policy.agent_inference_backend = "server-context";
    policy.tool_profile = options.tool_profile;
    policy.memory_learn = options.memory_learn;
    policy.memory_learn_show_candidate = options.memory_learn_show_candidate;
    policy.plan_show_summary = options.plan_show_summary;
    policy.agent_trace = options.agent_trace;
    policy.enable_reflection = true;
    policy.max_iterations = 2;
    policy.max_reflection_rounds = 1;
    // A zero value in host/bootstrap configuration means "use the safe
    // default". Keep daemon turns consistent with the CLI/runtime policy;
    // callers can still select a smaller positive budget explicitly.
    policy.max_tool_rounds = options.max_tool_rounds > 0
        ? options.max_tool_rounds
        : 16;
    common_tool_profile_snapshot profile_snapshot;
    std::string profile_error;
    if (resolve_common_tool_profile_snapshot(
            options.tool_profile,
            options.tool_capabilities,
            options.tool_profiles,
            profile_snapshot,
            profile_error)) {
        policy.allow_policy_gated_tool_proposals =
            profile_snapshot.allow_policy_gated_writes.value_or(false);
    }
    std::string deliberation_error;
    common_agent_deliberation_policy deliberation_policy;
    if (resolve_common_agent_deliberation_policy(
            options.thinking_mode,
            options.max_reflection_rounds,
            options.max_plan_revisions,
            options.max_research_iterations,
            deliberation_policy,
            deliberation_error)) {
        deliberation_policy.max_tool_rounds = options.max_tool_rounds > 0
            ? static_cast<int>(options.max_tool_rounds)
            : deliberation_policy.max_tool_rounds;
        policy.deliberation_policy = deliberation_policy;
    }
    return policy;
}

common_agent_runtime_config make_daemon_runtime_config(
        const daemon_options & options,
        const common_agent_daemon_runtime & runtime) {
    common_agent_runtime_config config;
    config.generation_config.n_predict = options.n_predict;
    config.generation_config.n_threads = options.n_threads;
    config.generation_config.context_size_tokens = static_cast<size_t>(std::max(0, options.context_size));
    // Keep daemon turns on the same host-owned family preflight path as the
    // CLI.  The daemon already exposes agent_plan=auto, but previously only
    // stored that value in orchestration config; the runtime generation flag
    // remained disabled and went straight to the full planner.
    config.generation_config.enable_tool_family_routing = options.agent_plan == "auto";
    config.generation_config.context_budgets = options.context_budgets;
    config.context_budgets = options.context_budgets;
    config.max_continuations = options.max_continuations;
    config.enable_memory_learning = options.memory_learn == "post-turn";
    config.memory_learning_config.min_confidence = options.memory_learn_min_confidence;
    config.memory_learning_config.min_expected_reuse = options.memory_learn_min_reuse;
    config.enable_adaptation_capture = options.adaptation_capture;
    config.adaptation_transaction_backend = options.adaptation_transaction_backend;
    config.adaptation_transaction_path = options.adaptation_transaction_path;
    config.adaptation_config.collection_allowed = options.adaptation_collection_allowed;
    config.adaptation_config.max_evidence = options.adaptation_max_evidence;
    config.adaptation_config.cause_classifier.stable_model_facing_tools =
        options.adaptation_stable_model_facing_tools;
    config.adaptation_config.domain_policy = options.adaptation_domains;
    config.enable_flydelta_capture_candidates =
        options.adaptation_flydelta_capture_candidates;
    config.enable_flydelta_candidate_lifecycle =
        options.adaptation_flydelta_capture_candidates &&
        !options.adaptation_flydelta_lifecycle_path.empty();
    config.flydelta_lifecycle_backend = options.adaptation_flydelta_lifecycle_backend;
    config.flydelta_lifecycle_path = options.adaptation_flydelta_lifecycle_path;
    config.flydelta_model_profile_fingerprint =
        options.adaptation_flydelta_model_profile_fingerprint.empty()
            ? (options.model_profile.empty()
                ? "model:" + options.model
                : "profile:" + options.model_profile)
            : options.adaptation_flydelta_model_profile_fingerprint;
    config.flydelta_capture_layout_revision =
        options.adaptation_flydelta_capture_layout_revision;
    config.flydelta_max_capture_candidates =
        options.adaptation_flydelta_max_capture_candidates;
    // Forward host-owned learning ingress into the existing runtime assembly.
    // The daemon transports these callbacks but never interprets semantic
    // evidence or manufactures capture jobs.
    config.flydelta_capture_job_enqueue = runtime.flydelta_capture_job_enqueue;
    config.procedure_teaching_request_provider =
        runtime.procedure_teaching_request_provider;
    config.user_correction_teaching_request_provider =
        runtime.user_correction_teaching_request_provider;
    config.user_taught_concept_relation_provider =
        runtime.user_taught_concept_relation_provider;
    config.flydelta_teaching_material_observer =
        runtime.flydelta_teaching_material_observer;
    config.flydelta_teaching_material_runtime =
        runtime.flydelta_teaching_material_runtime;
    return config;
}

bool configure_daemon_flydelta_teaching_material(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    error.clear();
    if (!options.adaptation_capture) return true;
    // The existing transaction backend may intentionally be JSONL/SQLite.
    // Do not silently reinterpret those paths as teaching-material storage;
    // only the shared Cozo relation is wired in this V0.
    if (options.adaptation_transaction_backend != "auto" &&
            options.adaptation_transaction_backend != "cozo") return true;

    common_flydelta_teaching_material_identity identity;
    identity.model_profile_fingerprint =
        options.adaptation_flydelta_model_profile_fingerprint.empty()
            ? (options.model_profile.empty()
                ? "model:" + std::filesystem::path(options.model).filename().string()
                : "profile:" + options.model_profile)
            : options.adaptation_flydelta_model_profile_fingerprint;
    identity.tokenizer_fingerprint = "daemon:tokenizer-unspecified-v1";
    identity.template_fingerprint = "daemon:template-unspecified-v1";
    identity.capture_layout_revision = options.adaptation_flydelta_capture_layout_revision;
    identity.execution_context_fingerprint = "daemon:server-context-v1";
    identity.scope_fingerprint = "daemon:teaching-material-v1";
    identity.verifier_revision = "daemon:host-verifier-v1";

    // The adaptation transaction path is already the host's durable Cozo
    // location. Cozo stores teaching material in its own relation, so this
    // does not mix ledger rows with material rows. An empty path deliberately
    // keeps the existing in-memory behavior for ephemeral hosts/tests.
    auto material_runtime = make_agent_flydelta_teaching_material_runtime(
        options.adaptation_transaction_backend,
        options.adaptation_transaction_path,
        std::move(identity),
        error);
    if (!material_runtime) return false;
    runtime.flydelta_teaching_material_runtime = material_runtime;
    return true;
}

bool configure_daemon_flydelta_server_binding(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    error.clear();
    if (!options.adaptation_flydelta_enabled) return true;
    if (!runtime.flydelta_server_binding_factory &&
            runtime.flydelta_server_binding_callbacks.has_value()) {
        runtime.flydelta_server_binding_factory =
            common_agent_server_flydelta_binding_factory_from_callbacks(
                std::move(*runtime.flydelta_server_binding_callbacks));
        runtime.flydelta_server_binding_callbacks.reset();
    }
    // A generic daemon may be configured for FlyDelta before an embedding
    // host has supplied semantic callbacks. Keep that state explicitly idle;
    // never invent prompts, repairs or host outcomes here.
    if (!runtime.flydelta_server_binding_factory) return true;
    if (!runtime.model_residency) {
        error = "FlyDelta startup binding requires a model catalog/residency manager";
        return false;
    }

    const std::string profile_id = options.model_profile.empty()
        ? runtime.model_residency->catalog().default_profile
        : options.model_profile;
    common_agent_runtime_model_resident_handle handle;
    if (!runtime.model_residency->acquire(profile_id, handle, error)) return false;
    const auto loaded = common_agent_runtime_loaded_model_cast(handle.model);
    if (!loaded || !loaded->server_context_host) {
        std::string release_error;
        runtime.model_residency->release(handle, release_error);
        error = "FlyDelta startup binding requires a resident server-context model";
        return false;
    }

    common_agent_server_flydelta_binding binding;
    if (!runtime.flydelta_server_binding_factory(
            loaded->server_context_host, binding, error)) {
        std::string release_error;
        runtime.model_residency->release(handle, release_error);
        return false;
    }
    if (!binding.teaching_material_runtime) {
        binding.teaching_material_runtime = runtime.flydelta_teaching_material_runtime;
    }
    if (!common_agent_daemon_register_flydelta_server_binding(
            runtime, loaded->server_context_host, std::move(binding), error)) {
        std::string release_error;
        runtime.model_residency->release(handle, release_error);
        return false;
    }
    if (options.adaptation_flydelta_batch_mode == "required" &&
            (!runtime.flydelta_model_adapter ||
                !runtime.flydelta_model_adapter->capabilities.bounded_arm_batch)) {
        std::string release_error;
        runtime.model_residency->release(handle, release_error);
        runtime.flydelta_model_adapter.reset();
        runtime.flydelta_model_host.reset();
        runtime.flydelta_model_capabilities = {};
        error = "FlyDelta native batch mode is required but unavailable for the selected resident model";
        return false;
    }
    runtime.flydelta_model_handle = std::move(handle);
    return true;
}

common_agent_orchestration_config make_daemon_orchestration_config(const daemon_options & options) {
    common_agent_orchestration_config config;
    config.prompt = "";
    config.agent_plan = options.agent_plan;
    config.agent_blueprint = options.agent_blueprint;
    return config;
}

common_agent_runtime_resident_request_config make_resident_request_config(
        const daemon_options & options) {
    common_agent_runtime_resident_request_config config{
        "",
        "",
        "",
        "",
        make_daemon_policy_pack(options),
        options.model,
        options.n_predict,
        options.n_gpu_layers,
        false,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
        options.n_threads,
        static_cast<size_t>(std::max(0, options.context_size)),
    };
    config.mmproj = options.mmproj;
    return config;
}

common_memory_query make_daemon_memory_query(
        const common_agent_runtime_session_host_turn_request & request) {
    common_memory_query query;
    query.text = request.prompt;
    query.limit = 8;
    query.token_budget = 768;

    common_agent_scope scope;
    scope.namespace_id = request.namespace_id;
    scope.session_id = request.session_id;
    scope.project_id = request.project_id;
    scope.turn_id = request.turn_id;
    scope.memory_scope = request.memory_scope;
    scope.plan_scope = request.plan_scope;
    common_agent_scope_apply(scope, query);
    return query;
}

agent_host_tool_selection_request make_daemon_tool_request(
        const daemon_options & options,
        const common_agent_runtime_session_host_turn_request & request,
        std::optional<bool> allow_policy_gated_writes) {
    agent_host_tool_selection_request tool_request;
    tool_request.tool_context.request_id = "daemon";
    tool_request.tool_context.turn_id = request.turn_id;
    tool_request.tool_context.scope.namespace_id = request.namespace_id;
    tool_request.tool_context.scope.session_id = request.session_id;
    tool_request.tool_context.scope.project_id = request.project_id;
    tool_request.tool_context.scope.turn_id = request.turn_id;
    tool_request.tool_context.scope.memory_scope = request.memory_scope;
    tool_request.tool_context.scope.plan_scope = request.plan_scope;
    tool_request.tool_context.memory_scope = request.memory_scope;
    tool_request.tool_context.plan_scope = request.plan_scope;
    tool_request.tool_context.profile_id = options.tool_profile;
    tool_request.tool_context.allowed_exposed_tool_names = request.allowed_exposed_tool_names;
    tool_request.tool_context.repository_root = options.repository_root;
    tool_request.tool_context.allow_network =
        has_enabled_mcp_provider(options.mcp_providers) ||
        !options.mcp_tool_command.empty();
    bool policy_gated_writes = false;
    if (allow_policy_gated_writes.has_value()) {
        policy_gated_writes = *allow_policy_gated_writes;
    } else if (request.allow_policy_gated_writes.has_value()) {
        policy_gated_writes = *request.allow_policy_gated_writes;
    }
    agent_tool_context_apply_policy_gated_writes(
        tool_request.tool_context, policy_gated_writes);
    tool_request.tool_context.max_calls = options.max_tool_rounds > 0 ? options.max_tool_rounds : 1;
    tool_request.tool_context.execution_control = request.execution_control;
    tool_request.tool_context.default_timeout_ms =
        options.tool_timeout_ms > 0
            ? options.tool_timeout_ms
            : tool_request.tool_context.default_timeout_ms;
    tool_request.repository_root = options.repository_root.empty()
        ? std::string()
        : std::filesystem::weakly_canonical(options.repository_root).string();
    tool_request.diagnostics = options.diagnostics;
    tool_request.resource_store_config = {
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    };
    tool_request.data_store_config = {options.data_backend, options.data_db};
    tool_request.tool_capabilities = options.tool_capabilities;
    tool_request.tool_family_descriptions = options.tool_family_descriptions;
    tool_request.tool_profiles = options.tool_profiles;
    tool_request.sandbox = options.sandbox;
    tool_request.resource_processor_policies = options.resource_processor_policies;
    append_configured_stdio_mcp_providers(options.mcp_providers, tool_request.mcp_providers);
    tool_request.openapi_providers = options.openapi_providers;
    if (tool_request.mcp_providers.empty()) {
        append_legacy_stdio_mcp_provider(
            options.mcp_tool_command,
            options.mcp_tool_args,
            options.mcp_tool_server_name,
            options.mcp_tool_prefix,
            tool_request.mcp_providers);
    }
    return tool_request;
}

} // namespace

bool resolve_agent_daemon_tooling(
        const daemon_options & options,
        const common_agent_runtime_resident_runtime * runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        agent_resource_store * resource_store,
        common_agent_runtime_tooling & tooling,
        std::string & error,
        common_agent_data_store * data_store,
        std::optional<bool> allow_policy_gated_writes) {
    tooling = {};
    common_memory_query query = make_daemon_memory_query(request);
    auto tool_request = make_daemon_tool_request(
        options, request, allow_policy_gated_writes);
    tool_request.data_store = data_store;
        std::string * current_plan_id = runtime != nullptr
        ? const_cast<std::string *>(&runtime->current_plan_id())
        : nullptr;
    std::unique_ptr<agent_embedding_provider> embedding_provider;
    const std::string embedding_model = options.embedding_model;
    if (!embedding_model.empty()) {
        embedding_provider = std::make_unique<daemon_agent_embedding_provider>(
            embedding_model,
            options.n_gpu_layers);
    }

    common_agent_cli_tool_selection selection;
    if (!resolve_agent_host_tool_selection(
            memory_store,
            &plan_store,
            resource_store,
            current_plan_id,
            options.tool_profile,
            tool_request,
            query,
            embedding_provider.get(),
            selection,
            error)) {
        error = "daemon tool provider resolution failed: " + error;
        return false;
    }
    selection.embedding_provider = std::move(embedding_provider);

    tooling = std::move(selection.tooling);
    if (selection.embedding_provider) {
        auto shared_embedding = std::shared_ptr<agent_embedding_provider>(
            std::move(selection.embedding_provider));
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_embedding));
    }
    if (selection.owned_resource_store) {
        auto shared_resource_store = std::shared_ptr<agent_resource_store>(
            std::move(selection.owned_resource_store));
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_resource_store));
    }
    for (auto & mcp_client : selection.mcp_clients) {
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(
            std::shared_ptr<agent_mcp_tool_client>(std::move(mcp_client))));
    }
    for (auto & openapi_provider : selection.openapi_providers) {
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(
            std::shared_ptr<agent_tool_provider>(std::move(openapi_provider))));
    }
    if (selection.tool_view) {
        auto shared_view = std::shared_ptr<agent_tool_view>(std::move(selection.tool_view));
        tooling.tool_view = shared_view.get();
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_view));
    }
    error.clear();
    return true;
}

namespace {

common_agent_runtime_session_host_build_config make_session_host_build_config(
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        agent_resource_store & resource_store,
        common_agent_data_store * data_store,
        const daemon_options & options,
        const std::shared_ptr<common_agent_daemon_config_store> & config_store,
        const common_agent_daemon_runtime & runtime) {
    return {
        memory_store,
        plan_store,
        make_resident_request_config(options),
        make_daemon_runtime_policy(options),
        make_daemon_runtime_config(options, runtime),
        make_daemon_orchestration_config(options),
        common_memory_scope::session,
        true,
        {},
        {},
        [config_store, &memory_store, &plan_store, &resource_store, data_store](
                const common_agent_runtime_resident_runtime * runtime,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            const auto current_options = config_store->snapshot();
            return resolve_agent_daemon_tooling(
                *current_options,
                runtime,
                request,
                memory_store,
                plan_store,
                &resource_store,
                tooling,
                error,
                data_store);
        },
    };
}

bool open_daemon_memory_store(
        const daemon_options & options,
        std::unique_ptr<common_memory_store> & store,
        std::string & error) {
    const std::string backend = resolve_memory_backend(options.backend, options.memory_db, error);
    if (!error.empty()) {
        return false;
    }
    if (backend == "cozo") {
#ifdef LLAMA_MEMORY_USE_COZO
        if (options.memory_db.empty()) {
            error = "--backend cozo requires --memory-db PATH";
            return false;
        }
        store = std::make_unique<common_memory_cozo_store>();
#else
        error = "this binary was built without LLAMA_MEMORY_COZO";
        return false;
#endif
    } else if (backend == "sqlite") {
#ifdef LLAMA_MEMORY_USE_SQLITE
        if (options.memory_db.empty()) { error = "--backend sqlite requires --memory-db PATH"; return false; }
        store = std::make_unique<common_memory_sqlite_store>();
#else
        error = "this binary was built without LLAMA_AGENT_STORAGE_SQLITE";
        return false;
#endif
    } else if (backend == "in-memory") {
        store = std::make_unique<common_memory_in_memory_store>();
    } else {
        error = "unknown memory backend: " + backend;
        return false;
    }
    return store->open(options.memory_db, error);
}

bool open_daemon_plan_store(
        const daemon_options & options,
        std::unique_ptr<common_plan_store> & store,
        std::string & error) {
    const std::string backend = resolve_plan_backend(options.plan_backend, options.plan_db, error);
    if (!error.empty()) {
        return false;
    }
    if (backend == "cozo") {
#ifdef LLAMA_PLAN_USE_COZO
        if (options.plan_db.empty()) {
            error = "--plan-backend cozo requires --plan-db PATH";
            return false;
        }
        store = std::make_unique<common_plan_cozo_store>();
#else
        error = "this binary was built without LLAMA_PLAN_COZO";
        return false;
#endif
    } else if (backend == "sqlite") {
#ifdef LLAMA_PLAN_USE_SQLITE
        if (options.plan_db.empty()) { error = "--plan-backend sqlite requires --plan-db PATH"; return false; }
        store = std::make_unique<common_plan_sqlite_store>();
#else
        error = "this binary was built without LLAMA_AGENT_STORAGE_SQLITE";
        return false;
#endif
    } else if (backend == "in-memory") {
        store = std::make_unique<common_plan_in_memory_store>();
    } else {
        error = "unknown plan backend: " + backend;
        return false;
    }
    return store->open(options.plan_db, error);
}

bool open_daemon_resource_store(
        const daemon_options & options,
        std::unique_ptr<agent_resource_store> & store,
        std::string & error) {
    store = make_agent_resource_store({
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    }, error);
    return store != nullptr;
}

bool open_daemon_data_store(
        const daemon_options & options,
        std::unique_ptr<common_agent_data_store> & store,
        std::string & error) {
    store = make_agent_data_store({options.data_backend, options.data_db}, error);
    return error.empty();
}

} // namespace

bool initialize_agent_daemon_environment(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    if (!runtime.config_store) {
        runtime.config_store = std::make_shared<common_agent_daemon_config_store>(
            std::make_shared<const daemon_options>(options));
    }
    if (!open_daemon_memory_store(options, runtime.memory_store, error)) {
        return false;
    }
    if (!open_daemon_plan_store(options, runtime.plan_store, error)) {
        return false;
    }
    if (!open_daemon_data_store(options, runtime.data_store, error)) {
        return false;
    }
    if (!open_daemon_resource_store(options, runtime.resource_store, error)) {
        return false;
    }
    if (!open_daemon_model_residency(options, runtime.model_residency, error)) {
        return false;
    }
    if (options.adaptation_flydelta_enabled &&
            !runtime.flydelta_server_binding_factory &&
            !runtime.flydelta_server_binding_callbacks.has_value()) {
        // The default llama-agent provider resolves explicit, durable resource
        // refs.  It is a real production binding, but it remains fail-closed:
        // a job without a context, fixture or intervention resource cannot
        // reach the model.
        runtime.flydelta_server_binding_factory =
            make_daemon_flydelta_resource_binding_factory(
                options, runtime.resource_store.get());
    }
    if (!configure_daemon_flydelta_teaching_material(
            options, runtime, error)) {
        error = "FlyDelta teaching-material runtime initialization failed: " + error;
        return false;
    }
    if (!configure_daemon_flydelta_server_binding(options, runtime, error)) {
        error = "FlyDelta server binding initialization failed: " + error;
        return false;
    }
    if (!parse_mode(options.default_mode, runtime.default_mode)) {
        error = "unsupported default mode: " + options.default_mode;
        return false;
    }

    auto session_manager_build_config = make_session_host_build_config(
        *runtime.memory_store,
        *runtime.plan_store,
        *runtime.resource_store,
        runtime.data_store.get(),
        options,
        runtime.config_store,
        runtime);
    session_manager_build_config.runtime_config.flydelta_teaching_material_runtime =
        runtime.flydelta_teaching_material_runtime;
    runtime.inference_gate = std::make_shared<common_agent_inference_capacity_gate>(
        options.inference_max_active);
    runtime.host = std::make_unique<common_agent_runtime_session_manager>(
        make_agent_runtime_session_manager_config({
            session_manager_build_config.memory_store,
            session_manager_build_config.plan_store,
            std::move(session_manager_build_config.resident_request),
            std::move(session_manager_build_config.policy),
            std::move(session_manager_build_config.runtime_config),
            std::move(session_manager_build_config.orchestration_config),
            session_manager_build_config.memory_scope,
            session_manager_build_config.memory_enabled,
            std::move(session_manager_build_config.installed_blueprint_candidates),
            std::move(session_manager_build_config.tooling),
            std::move(session_manager_build_config.tooling_resolver),
            {},
            runtime.inference_gate,
            {},
            runtime.model_residency,
        }));

    common_memory_store * memory_store = runtime.memory_store.get();
    common_plan_store * plan_store = runtime.plan_store.get();
    agent_resource_store * resource_store = runtime.resource_store.get();
    common_agent_data_store * data_store = runtime.data_store.get();
    runtime.tool_executor = [config_store = runtime.config_store, memory_store, plan_store, resource_store, data_store](
            const common_agent_daemon_tool_payload & payload,
            agent_tool_result & result,
            std::string & callback_error) mutable {
        if (memory_store == nullptr || plan_store == nullptr || resource_store == nullptr) {
            callback_error = "daemon tool executor stores are not initialized";
            return false;
        }
        daemon_options call_options = *config_store->snapshot();
        if (!payload.tool_profile.empty()) {
            call_options.tool_profile = payload.tool_profile;
        }
        common_agent_runtime_session_host_turn_request request;
        request.mode = common_agent_runtime_host_mode::chat;
        request.session_id = payload.session.session_id;
        request.namespace_id = payload.session.namespace_id;
        request.project_id = payload.project_id;
        request.turn_id = "mcp-tool";
        request.memory_scope = common_memory_scope::session;
        request.plan_scope = common_plan_scope::turn;

        common_agent_runtime_tooling tooling;
        if (!resolve_agent_daemon_tooling(
                call_options,
                nullptr,
                request,
                *memory_store,
                *plan_store,
                resource_store,
                tooling,
                callback_error,
                data_store,
                payload.allow_policy_gated_writes)) {
            return false;
        }
        if (tooling.tool_view == nullptr) {
            callback_error = "daemon tool executor resolved no tool view";
            return false;
        }
        agent_tool_call call{"daemon-mcp-tool", payload.tool_name, payload.arguments_json};
        if (!tooling.tool_view->validate(call, callback_error)) {
            return false;
        }
        result = tooling.tool_view->call(call, callback_error);
        return result.ok;
    };
    configure_agent_daemon_provider_probe(options, runtime);
    error.clear();
    return true;
}

void configure_agent_daemon_provider_probe(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime) {
    common_memory_store * probe_memory_store = runtime.memory_store.get();
    common_plan_store * probe_plan_store = runtime.plan_store.get();
    agent_resource_store * probe_resource_store = runtime.resource_store.get();
    common_agent_data_store * probe_data_store = runtime.data_store.get();
    runtime.probe_mcp_providers = [
        probe_options = options,
        probe_memory_store,
        probe_plan_store,
        probe_resource_store,
        probe_data_store](
            common_agent_runtime_tooling & retained_tooling,
            std::vector<common_agent_daemon_provider_readiness> & provider_status,
            std::string & probe_error) {
        retained_tooling = {};
        provider_status.clear();
        bool all_required_ready = true;
        for (const auto & configured : probe_options.mcp_providers) {
            if (!configured.enabled) {
                continue;
            }

            common_agent_daemon_provider_readiness status;
            status.id = configured.id.empty() ? configured.server_name : configured.id;
            status.required = configured.required;

            daemon_options provider_options = probe_options;
            provider_options.mcp_providers = {configured};
            common_agent_runtime_session_host_turn_request probe_request;
            probe_request.mode = common_agent_runtime_host_mode::chat;
            probe_request.session_id = "daemon-provider-probe";
            probe_request.namespace_id = "local";
            probe_request.project_id = "llama-agent";
            probe_request.turn_id = "daemon-provider-probe";
            probe_request.memory_scope = common_memory_scope::session;
            probe_request.plan_scope = common_plan_scope::turn;

            common_agent_runtime_tooling provider_tooling;
            std::string provider_error;
            const bool provider_ready = resolve_agent_daemon_tooling(
                provider_options,
                nullptr,
                probe_request,
                *probe_memory_store,
                *probe_plan_store,
                probe_resource_store,
                provider_tooling,
                provider_error,
                probe_data_store);
            if (provider_ready) {
                status.status = "ready";
                for (auto & tool : provider_tooling.tools) {
                    retained_tooling.tools.push_back(std::move(tool));
                }
                for (auto & resource : provider_tooling.owned_resources) {
                    retained_tooling.owned_resources.push_back(std::move(resource));
                }
            } else {
                status.status = "degraded";
                status.warning = provider_error.empty()
                    ? "provider probe failed"
                    : provider_error;
                if (status.required) {
                    all_required_ready = false;
                }
            }
            provider_status.push_back(std::move(status));
        }
        probe_error.clear();
        return all_required_ready;
    };
}
