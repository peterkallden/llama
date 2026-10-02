#include "agent/adaptation/flydelta/flydelta-deployment.h"

#include <map>
#include <string>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_sideband_manifest make_manifest(
        const std::string & id, const std::string & binding, int value) {
    common_flydelta_sideband_manifest result;
    result.id = id;
    result.artifact_path = "sidebands/" + id.substr(id.find_last_of('/') + 1) + ".flyd";
    result.artifact_hash = "sha256:artifact-" + id.substr(id.find_last_of('/') + 1);
    result.compatibility.base_model_id = "model:qwen";
    result.compatibility.base_model_fingerprint = "sha256:model";
    result.compatibility.tokenizer_fingerprint = "sha256:tokenizer";
    result.compatibility.template_fingerprint = "sha256:template";
    result.compatibility.architecture = "qwen2";
    result.compatibility.inference_layout_revision = "layout:v1";
    result.applicability.behavior_key = "tool_use/openalex";
    result.applicability.scope_fingerprint = "scope:openalex";
    result.applicability.verifier_revision = "oracle:v1";
    result.binding_key = binding;
    result.model_n_embd = 2;
    result.model_n_layers = 3;
    result.il_end = 2;
    result.evaluation_passed = true;
    result.oracle_revision = "oracle:" + std::to_string(value);
    return result;
}

static common_agent_model_profile make_profile() {
    common_agent_model_profile result;
    result.id = "profile:qwen";
    result.base_model_id = "model:qwen";
    result.base_model_fingerprint = "sha256:model";
    result.tokenizer_fingerprint = "sha256:tokenizer";
    result.chat_template_fingerprint = "sha256:template";
    result.context_size_tokens = 4096;
    result.sidebands.push_back({"", 0.5, "binding:a"});
    result.sidebands.push_back({"", 0.25, "binding:b"});
    return result;
}

static void make_active(
        common_flydelta_sideband_registry & registry,
        const common_flydelta_sideband_manifest & manifest,
        std::string & error) {
    registry.admit(manifest, error);
    registry.stage_canary(manifest.id, "evaluation:" + manifest.id, error);
    registry.activate(manifest.id, error);
    registry.bind_revision(manifest.binding_key, manifest.id, "", error);
}

int main() {
    std::string error;
    common_flydelta_sideband_registry registry;
    const auto active_a = make_manifest("flydelta://sideband/a", "binding:a", 1);
    const auto active_b = make_manifest("flydelta://sideband/b", "binding:b", 2);
    make_active(registry, active_a, error);
    make_active(registry, active_b, error);
    CHECK(error.empty());

    common_flydelta_deployment_request request;
    request.profile = make_profile();
    request.compatibility = active_a.compatibility;
    request.applicability = active_a.applicability;
    request.authority = common_flydelta_runtime_authority::active_only;
    request.model_n_embd = 2;
    request.model_n_layers = 3;
    request.model_profile_fingerprint = "profile:qwen";
    request.capture_layout_revision = "layout:v1";
    request.now_epoch_ms = 1;
    request.gate_request.requested_scale = 0.5f;
    request.max_overlay_bytes = 1024;

    std::map<std::string, double> seen_scales;
    std::map<std::string, float> payloads{{active_a.id, 1.0f}, {active_b.id, 2.0f}};
    common_flydelta_deployment_factory_config config;
    config.registry = &registry;
    config.load_activation = [&](
            const common_flydelta_sideband_manifest & manifest,
            double scale,
            const common_flydelta_deployment_request &,
            common_flydelta_activation_result & activation,
        std::string &) {
        seen_scales[manifest.id] = scale;
        const float payload = payloads[manifest.id];
        activation.gate.apply = true;
        activation.gate.scale = 1.0f;
        activation.gate.reason = "test";
        activation.overlay.enabled = true;
        activation.overlay.artifact_id = manifest.id;
        activation.overlay.n_embd = 2;
        activation.overlay.il_start = 1;
        activation.overlay.il_end = 2;
        activation.overlay.data = {
            payload, 0.0f, 0.0f, payload,
        };
        activation.sparse_overlay.enabled = true;
        activation.sparse_overlay.artifact_id = manifest.id;
        activation.sparse_overlay.n_embd = 2;
        activation.sparse_overlay.il_start = 1;
        activation.sparse_overlay.il_end = 2;
        activation.sparse_overlay.layer_indices = {1, 2};
        activation.sparse_overlay.data = activation.overlay.data;
        return true;
    };

    common_flydelta_deployment_result baseline;
    CHECK(common_flydelta_resolve_deployment(config, request, baseline, error));
    CHECK(!baseline.canary_selected && baseline.baseline.size() == 2);
    CHECK(baseline.effective[0].revision_id == active_a.id &&
        baseline.effective[1].revision_id == active_b.id);
    CHECK(baseline.baseline_deployment_fingerprint ==
        baseline.candidate_deployment_fingerprint);
    CHECK(baseline.activation && baseline.activation->overlay.enabled);
    CHECK(baseline.activation->overlay.data[0] == 3.0f);

    auto canary = make_manifest("flydelta://sideband/b-canary", "binding:b", 9);
    canary.status = common_flydelta_sideband_status::candidate;
    CHECK(registry.admit(canary, error));
    CHECK(registry.stage_canary(canary.id, "evaluation:canary", error, true));
    payloads[canary.id] = 9.0f;
    common_flydelta_sideband_review review;
    review.event_id = "review:canary-b";
    review.action = common_flydelta_review_action::stage_canary;
    review.has_canary_envelope = true;
    review.canary_envelope.binding_key = "binding:b";
    review.canary_envelope.candidate_revision_id = canary.id;
    review.canary_envelope.behavior_key = request.applicability.behavior_key;
    review.canary_envelope.scope_fingerprint = request.applicability.scope_fingerprint;
    review.canary_envelope.traffic_basis_points = 10000;
    review.canary_envelope.expires_at_epoch_ms = 4102444800000ULL;
    review.canary_envelope.max_observations = 10;
    review.canary_envelope.max_scale = 0.125f;
    review.canary_envelope.compatibility = canary.compatibility;
    review.canary_envelope.oracle_revision = "oracle:v1";
    review.canary_envelope.policy_revision = "policy:v1";
    review.canary_envelope.baseline_deployment_fingerprint =
        baseline.baseline_deployment_fingerprint;
    review.canary_envelope.rollback_revision_id = active_b.id;
    config.open_canaries = [&](std::vector<common_flydelta_sideband_review> & reviews,
            std::string &) { reviews = {review}; return true; };

    request.authority = common_flydelta_runtime_authority::canary_evaluation;
    request.allocation_key = "session:stable";
    common_flydelta_deployment_result candidate;
    CHECK(common_flydelta_resolve_deployment(config, request, candidate, error));
    CHECK(candidate.canary_selected && !candidate.active_only);
    CHECK(candidate.effective.size() == 2);
    CHECK(candidate.effective[0].revision_id == active_a.id &&
        candidate.effective[1].revision_id == canary.id);
    CHECK(candidate.baseline_deployment_fingerprint !=
        candidate.candidate_deployment_fingerprint);
    CHECK(candidate.activation && candidate.activation->overlay.data[0] == 10.0f);
    CHECK(seen_scales[canary.id] == 0.125);

    request.allocation_key.clear();
    common_flydelta_deployment_result fallback;
    CHECK(common_flydelta_resolve_deployment(config, request, fallback, error));
    CHECK(!fallback.canary_selected && fallback.active_only && fallback.fallback_to_active);
    CHECK(fallback.effective[1].revision_id == active_b.id);

    common_flydelta_canary_policy policy;
    common_flydelta_canary_policy_input policy_input;
    common_flydelta_canary_policy_decision decision;
    policy.mode = common_flydelta_canary_mode::policy;
    policy.min_observations = 2;
    policy.min_unique_allocations = 1;
    policy.allow_scope_expansion = true;
    policy.allow_promotion = true;
    policy.scope_step_fingerprints = {"scope:openalex", "scope:openalex-expanded"};
    policy_input.completed_observations = 2;
    policy_input.unique_allocations = 1;
    policy_input.semantic_evidence_complete = true;
    policy_input.target_gain = 1.0f;
    policy_input.next_scope_available = true;
    policy_input.current_traffic_basis_points = 100;
    CHECK(common_flydelta_decide_canary_disposition(
        policy, policy_input, decision, error));
    CHECK(decision.disposition == common_flydelta_canary_disposition::expand_scope);
    CHECK(decision.next_traffic_basis_points == 500);
    policy_input.current_traffic_basis_points = 1000;
    policy_input.next_scope_available = false;
    CHECK(common_flydelta_decide_canary_disposition(
        policy, policy_input, decision, error));
    CHECK(decision.disposition == common_flydelta_canary_disposition::promote_active);
    policy_input.harmed_results = 1;
    CHECK(common_flydelta_decide_canary_disposition(
        policy, policy_input, decision, error));
    CHECK(decision.disposition == common_flydelta_canary_disposition::close);

    common_flydelta_observation_budget budget;
    CHECK(budget.try_reserve("binding:b", "review:canary-b", 1, error));
    CHECK(!budget.try_reserve("binding:b", "review:canary-b", 1, error));
    CHECK(budget.reserved("binding:b", "review:canary-b") == 1);
    return 0;
}
