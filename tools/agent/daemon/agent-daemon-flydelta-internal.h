#pragma once

#include "agent-daemon-flydelta.h"

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
#include "agent/adaptation/flydelta/flydelta-concept.h"
#include "agent/adaptation/flydelta/flydelta-concept-capture.h"
#include "agent/adaptation/flydelta/flydelta-deep-search.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-representation-augmentation-state-store.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"
#include "agent/adaptation/flydelta/flydelta-semantic-decision.h"
#include "agent/adaptation/flydelta/flydelta-sideband-review-store.h"
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
#include <array>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>


namespace agent_daemon_flydelta_internal {

using json = nlohmann::ordered_json;

struct daemon_flydelta_resource_provider {
    std::shared_ptr<common_agent_server_context_host> host;
    agent_resource_store * resources = nullptr;
    agent_resource_read_authority authority;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    std::shared_ptr<common_flydelta_teaching_material_runtime> teaching_material_runtime;
    int n_predict = 0;
    int n_threads = 0;
    size_t model_n_embd = 0;
    size_t model_n_layers = 0;
    // The lifecycle journal is the daemon-owned durable state seam. Captures
    // are kept in memory for the active bounded wave and mirrored to the
    // existing resource store when a later resume/materialization step needs
    // them; they are never embedded in queue state.
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
    // These are installed from the same durable lifecycle callbacks that
    // serve ordinary worker resumes. Concept capture uses them only to locate
    // WHERE; it never creates a second state store or search policy.
    std::function<bool(
            const std::string &, common_flydelta_bootstrap_zoom_state &,
            std::string &)> resolve_bootstrap_zoom_state;
    std::function<bool(
            const common_flydelta_experiment_job &, const std::string &,
            common_flydelta_bootstrap_zoom_state &, std::string &)>
        resolve_bootstrap_zoom_state_for_job;
    std::function<bool(
            const std::string &, common_flydelta_representation_augmentation_state &,
            std::string &)> resolve_representation_augmentation_state;
    std::function<bool(
            const common_flydelta_experiment_job &, const std::string &,
            common_flydelta_representation_augmentation_state &, std::string &)>
        resolve_representation_augmentation_state_for_job;
    std::function<bool(
            const std::string &, common_flydelta_experiment_plan &,
            common_flydelta_utility_history &, std::string &)> resolve_search_orchestration_state;
    std::function<bool(
            const common_flydelta_experiment_job &, const std::string &,
            common_flydelta_experiment_plan &, common_flydelta_utility_history &,
        std::string &)> resolve_search_orchestration_state_for_job;
};

struct daemon_flydelta_concept_relation_material {
    common_flydelta_teaching_relation relation;
    std::string semantic_anchor;
    int32_t layer_index = -1;
    std::string parent_surface_ref;
    uint64_t parent_surface_revision = 0;
    float parent_evidence_rank = 0.0f;
};

struct daemon_flydelta_concept_target {
    int32_t layer_index = -1;
    std::vector<uint32_t> local_layers;
    std::string parent_surface_ref;
    uint64_t parent_surface_revision = 0;
    float parent_evidence_rank = 0.0f;
};

struct daemon_flydelta_augmentation_donor_material {
    common_flydelta_representation_donor_candidate candidate;
    std::string target_context_ref;
    std::string donor_context_ref;
    std::string fixture_ref;
    int32_t layer_index = -1;
    common_flydelta_representation_donor_qualification qualification;
    std::string target_capture_ref;
    std::string donor_capture_ref;
    common_flydelta_representation_latent_delta latent;
};

struct daemon_flydelta_augmentation_material {
    std::string state_ref;
    std::string parent_direction_ref;
    std::vector<daemon_flydelta_augmentation_donor_material> donors;
    std::string selected_control_ref;
};

std::shared_ptr<daemon_flydelta_resource_provider>
daemon_flydelta_provider_for_scope(
        const std::shared_ptr<daemon_flydelta_resource_provider> & source,
        const common_agent_scope & scope);

bool daemon_flydelta_lifecycle_record_matches_scope(
        const common_learning_lifecycle_record & record,
        const common_agent_scope & scope);

bool daemon_flydelta_read_json(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        json & parsed,
        std::string & error);
bool daemon_flydelta_read_json_bounded(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        size_t max_bytes,
        json & parsed,
        std::string & error);

struct daemon_flydelta_concept_relation_material;

bool daemon_flydelta_read_json_bounded(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        size_t max_bytes,
        json & parsed,
        std::string & error);

struct daemon_flydelta_concept_relation_material;

bool daemon_flydelta_parse_teaching_origin(
        const std::string & value,
        common_flydelta_teaching_origin & origin);

json daemon_flydelta_teaching_relation_json(
        const common_flydelta_teaching_relation & relation);

bool daemon_flydelta_persist_teaching_relation(
        agent_resource_store * resources,
        const agent_resource_read_authority & authority,
        const common_flydelta_teaching_relation & relation,
        common_flydelta_teaching_relation & persisted_relation,
        std::string & error);

bool daemon_flydelta_parse_teaching_relation(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        daemon_flydelta_concept_relation_material & material,
        std::string & error);

bool daemon_flydelta_read_json(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        json & parsed,
        std::string & error);

bool daemon_flydelta_read_json_bounded(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        size_t max_bytes,
        json & parsed,
        std::string & error);

bool daemon_flydelta_parse_context(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        common_agent_generation_request & request,
        std::string & error);

bool daemon_flydelta_parse_directions(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        std::vector<common_flydelta_basis_direction> & directions,
        std::string & error);

bool daemon_flydelta_parse_behavior_delta(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        common_flydelta_behavior_delta & delta,
        common_flydelta_intervention_credit & credit,
        std::string & error);

bool daemon_flydelta_prepare_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_arm_request & arm,
        common_agent_generation_request & request,
        std::string & error);

bool daemon_flydelta_score_teacher_forced_margin_batch(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::vector<common_flydelta_arm_request> & arms,
        const std::vector<common_agent_generation_request> & contexts,
        common_agent_inference & scorer,
        std::vector<common_flydelta_decision_margin> & margins,
        std::string & error);

bool daemon_flydelta_put_json_resource(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & name,
        const std::string & purpose,
        const std::string & parent_uri,
        const std::string & text,
        std::string & reference,
        std::string & error);

json daemon_flydelta_capture_json(
        const common_flydelta_hidden_state_capture & capture);

bool daemon_flydelta_capture_from_json(
        const json & value,
        common_flydelta_hidden_state_capture & capture,
        std::string & error);

bool daemon_flydelta_load_persisted_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & arm_id,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture);

bool daemon_flydelta_capture_layer(
        const common_flydelta_hidden_state_capture & capture,
        int32_t layer_index,
        std::vector<float> & values,
        std::string & error);

bool daemon_flydelta_resolve_concept_target(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        daemon_flydelta_concept_target & target,
        std::string & error);

bool daemon_flydelta_persist_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & arm_id,
        const std::shared_ptr<const common_flydelta_hidden_state_capture> & capture,
        std::string & reference,
        std::string & error);

bool daemon_flydelta_persist_concept_trajectory(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const daemon_flydelta_concept_relation_material & material,
        const std::string & group_ref,
        const std::string & job_id,
        const std::string & baseline_capture_ref,
        const std::string & conditioned_capture_ref,
        const std::string & control_capture_ref,
        const common_flydelta_hidden_state_capture & baseline,
        const common_flydelta_hidden_state_capture & conditioned,
        const common_flydelta_hidden_state_capture & control,
        std::string & trajectory_ref,
        std::string & error);

bool daemon_flydelta_verify_generation(
        const json & fixture,
        const std::string & generated,
        bool & verifier_known,
        bool & passed,
        std::string & error);

bool daemon_flydelta_arm_semantic_passed(const common_flydelta_arm_result & result);

bool daemon_flydelta_finalize_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_arm_request & arm,
        const common_agent_generation_result & generation,
        common_flydelta_arm_result & result,
        std::string & error);

bool daemon_flydelta_execute_batch(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_arm_batch_request & request,
        common_flydelta_arm_batch_result & result,
        std::string & error);

bool daemon_flydelta_run_concept_capture(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        std::vector<std::string> & trajectory_refs,
        std::string & error);

bool daemon_flydelta_run_concept_synthesis(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_concept_candidate> & candidates,
        std::string & error);

bool daemon_flydelta_fixture_from_job(
        const common_flydelta_experiment_job & job,
        common_flydelta_experiment_fixture & fixture,
        std::string & error);

bool daemon_flydelta_parse_depth(
        const std::string & value, common_flydelta_search_depth & depth);

bool daemon_flydelta_parse_phase(
        const std::string & value, common_flydelta_experiment_phase & phase);

bool daemon_flydelta_parse_refinement(
        const std::string & value, common_flydelta_bootstrap_refinement_kind & kind);

json daemon_flydelta_orchestration_state_json(
        const common_flydelta_experiment_plan & plan,
        const common_flydelta_utility_history & history,
        const std::string & bootstrap_state_ref);

std::string daemon_flydelta_continuation_key(const common_flydelta_experiment_plan & plan);

std::string daemon_flydelta_bootstrap_key(const common_flydelta_bootstrap_zoom_state & state);

bool daemon_flydelta_orchestration_state_from_json(
        const std::string & text,
        common_flydelta_experiment_plan & plan,
        common_flydelta_utility_history & history,
        std::string & error);

bool daemon_flydelta_capture_for_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & arm_id,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture);

bool daemon_flydelta_diagnostics_for_arm(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const std::string & baseline_arm_id,
        const common_flydelta_arm_request & arm,
        uint32_t layer,
        common_flydelta_representation_diagnostics & diagnostics,
        std::string & error);

bool daemon_flydelta_run_bootstrap_zoom_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error);

bool daemon_flydelta_run_search_pipeline(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_search_pipeline_config & config,
        common_flydelta_search_pipeline_result & result,
        std::string & error);

bool daemon_flydelta_register_composed_directions(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & source_ref,
        std::vector<common_flydelta_basis_direction> directions,
        const std::string & derivation,
        std::string & reference,
        std::string & error);

bool daemon_flydelta_register_composed_direction(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & source_ref,
        const common_flydelta_low_rank_basis & basis,
        const std::vector<float> & coefficients,
        std::string & reference,
        std::string & error);

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
        const bool full_execution = true,
        const std::string & wave_suffix = {});

bool daemon_flydelta_resolve_evidence_direction_candidates(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const int32_t layer_index,
        const size_t required_count,
        std::vector<common_flydelta_direction_candidate> & candidates,
        std::string & error);

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
        std::string & error);

bool daemon_flydelta_run_adaptive_alpha_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error);

bool daemon_flydelta_run_orthogonal_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error);

bool daemon_flydelta_run_post_bootstrap_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_plan & plan,
        common_flydelta_search_pipeline_result & output,
        std::string & error);

bool daemon_flydelta_parse_augmentation_donor(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & reference,
        const common_flydelta_representation_augmentation_state & state,
        const common_flydelta_experiment_job & job,
        daemon_flydelta_augmentation_donor_material & material,
        std::string & error);

json daemon_flydelta_augmentation_material_json(
        const daemon_flydelta_augmentation_material & material);

bool daemon_flydelta_augmentation_material_from_json(
        const json & value,
        daemon_flydelta_augmentation_material & material,
        std::string & error);

bool daemon_flydelta_load_augmentation_material(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & state_ref,
        daemon_flydelta_augmentation_material & material,
        std::string & error);

bool daemon_flydelta_persist_augmentation_material(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const daemon_flydelta_augmentation_material & material,
        std::string & error);

bool daemon_flydelta_capture_from_reference(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & reference,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture,
        std::string & error);

bool daemon_flydelta_augmentation_seed_result(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_representation_augmentation_state & state,
        common_flydelta_search_pipeline_result & result,
        std::string & error);

bool daemon_flydelta_capture_augmentation_pair(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        daemon_flydelta_augmentation_donor_material & material,
        std::string & error);

bool daemon_flydelta_capture_from_reference(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & reference,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture,
        std::string & error);

bool daemon_flydelta_run_donor_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_capture_manifest> & manifests,
        std::string & error);

bool daemon_flydelta_run_donor_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_capture_manifest> & manifests,
        std::string & error);

bool daemon_flydelta_run_representation_augmentation(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_search_pipeline_config & pipeline_config,
        const common_flydelta_representation_augmentation_state * resume,
        common_flydelta_search_pipeline_result & result,
        common_flydelta_representation_augmentation_state & next,
        std::string & error);

bool daemon_flydelta_run_counterfactual(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        std::vector<common_flydelta_counterfactual_report> & reports,
        std::string & error);

bool daemon_flydelta_run_evaluation(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_job & job,
        common_flydelta_evaluation_report & report,
        std::vector<common_flydelta_evaluation_fixture_result> & fixture_results,
        std::string & error);

} // namespace agent_daemon_flydelta_internal
