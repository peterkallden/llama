#include "../agent-daemon-flydelta-internal.h"

#include "agent/agent-residual-patch.h"

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_run_decision_margin_challenger(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_direction_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    if (!provider || !provider->resolve_decision_pair ||
            !provider->resolve_output_head_row) {
        // A missing decision-pair seam is an ordinary no-challenger result,
        // not a failure of the existing concept portfolio.
        return true;
    }
    if (job.teaching_material_group_ref.empty() || job.seed.verifier_ref.empty() ||
            job.seed.behavior_key.empty()) {
        error = "FlyDelta decision-margin challenger requires host relation and fixture refs";
        return false;
    }
    common_flydelta_decision_pair_request request;
    request.source = job.seed.source;
    request.behavior_key = job.seed.behavior_key;
    request.relation_ref = job.teaching_material_group_ref;
    request.fixture_ref = job.seed.verifier_ref;
    request.oracle_ref = job.seed.verifier_ref;
    request.oracle_revision = job.seed.verifier_ref;
    request.model_profile_fingerprint = job.seed.model_profile_fingerprint;
    request.tokenizer_fingerprint = job.seed.tokenizer_fingerprint;
    request.template_fingerprint = job.seed.template_fingerprint;
    common_flydelta_direction_search_config config;
    config.dimension = provider->model_n_embd;
    daemon_flydelta_concept_target target;
    if (!daemon_flydelta_resolve_concept_target(provider, job, target, error) ||
            target.layer_index <= 0) {
        if (error.empty()) error = "FlyDelta decision-margin challenger has no localized layer";
        return false;
    }
    config.layer_index = target.layer_index;
    config.source = request.source;
    config.behavior_key = request.behavior_key;
    config.model_profile_fingerprint = request.model_profile_fingerprint;
    config.execution_context_fingerprint = job.seed.execution_context_fingerprint;
    config.capture_layout_revision = provider->capture_layout_revision;
    config.mode = common_flydelta_direction_search_mode::experimental;
    common_flydelta_direction_candidate candidate;
    if (!common_flydelta_build_decision_output_margin_candidate_from_provider(
            config, request, provider->resolve_decision_pair,
            provider->resolve_output_head_row, candidate, error)) return false;
    candidates.push_back(std::move(candidate));
    return true;
}

} // namespace agent_daemon_flydelta_internal
