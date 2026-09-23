#include "agent/adaptation/flydelta/flydelta-collection.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-teaching-material.h"
#include "agent/adaptation/flydelta/flydelta-worker.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

int fail(const std::string & message) {
    std::cerr << "flydelta_concept_capture_smoke=failed reason=" << message << '\n';
    return 1;
}

common_flydelta_teaching_relation relation(const std::string & id) {
    common_flydelta_teaching_relation value;
    value.id = id;
    value.teaching_key = "dataset.grouped_sum.v1";
    value.source = common_adaptation_evidence_source::procedure_blueprint;
    value.behavior_key = "tool_choice/dataset/grouped_sum";
    value.scope.namespace_id = "local";
    value.scope.session_id = "concept-capture-smoke";
    value.task_fingerprint = "task:" + id;
    value.baseline_ref = "execution:" + id + "/baseline";
    value.conditioned_ref = "execution:" + id + "/conditioned";
    value.control_ref = "execution:" + id + "/control";
    value.verifier_ref = "verifier:grouped-sum:v1";
    value.evidence_ref = "evidence:" + id;
    value.status = common_flydelta_teaching_relation_status::resolved;
    value.baseline_origin = common_flydelta_teaching_origin::host_derived;
    value.conditioned_origin = common_flydelta_teaching_origin::host_derived;
    value.control_origin = common_flydelta_teaching_origin::host_counterfactual;
    value.confidence = 1.0f;
    value.host_approved = true;
    return value;
}

common_flydelta_teaching_material_identity identity() {
    common_flydelta_teaching_material_identity value;
    value.model_profile_fingerprint = "model:concept-capture-smoke";
    value.tokenizer_fingerprint = "tokenizer:v1";
    value.template_fingerprint = "template:v1";
    value.capture_layout_revision = "capture:v1";
    value.execution_context_fingerprint = "context:v1";
    value.scope_fingerprint = "scope:v1";
    value.verifier_revision = "verifier:v1";
    return value;
}

common_flydelta_experiment_collection_request parent_request(
        const std::string & group_ref) {
    common_flydelta_experiment_collection_request value;
    value.enabled = true;
    value.kind = common_flydelta_experiment_job_kind::search_pipeline;
    value.evidence.id = "evidence://concept-capture-smoke";
    value.evidence.source = common_adaptation_evidence_source::procedure_blueprint;
    value.evidence.behavior_key = "tool_choice/dataset/grouped_sum";
    value.evidence.scope.namespace_id = "local";
    value.evidence.scope.project_id = "flydelta-smoke";
    value.evidence.scope.session_id = "concept-capture-smoke";
    value.evidence.scope.turn_id = "turn-1";
    value.evidence.task_fingerprint = "task:concept-capture-smoke";
    value.evidence.baseline_ref = "execution:baseline";
    value.evidence.candidate_ref = "execution:conditioned";
    value.evidence.verifier_ref = "verifier:grouped-sum:v1";
    value.evidence.transaction_ids = {"learning://concept-capture"};
    value.evidence.cause = common_learning_cause::host_contract;
    value.evidence.host_verified = true;
    value.behavior_key = value.evidence.behavior_key;
    value.model_profile_fingerprint = "model:concept-capture-smoke";
    value.tokenizer_fingerprint = "tokenizer:v1";
    value.template_fingerprint = "template:v1";
    value.execution_context_fingerprint = "context:v1";
    value.capture_manifest_ids = {"flydelta://capture/manifest/smoke"};
    value.behavior_delta_ids = {"flydelta://delta/smoke"};
    value.alpha_search.candidates = {0.05f, 0.1f};
    value.alpha_search.max_candidates = 2;
    value.code_revision = "concept-capture-smoke:v1";
    value.teaching_material_group_ref = group_ref;
    return value;
}

common_flydelta_concept_candidate candidate() {
    common_flydelta_concept_candidate value;
    value.concept_key = "dataset.grouped_sum.v1";
    value.extraction_id = "extraction:concept-capture-smoke";
    value.behavior_key = "tool_choice/dataset/grouped_sum";
    value.model_profile_fingerprint = "model:concept-capture-smoke";
    value.capture_layout_revision = "capture:v1";
    value.layer_index = 12;
    value.values = {0.26726124f, 0.53452248f, 0.80178374f};
    value.source_trajectories = 2;
    value.retained_trajectories = 2;
    value.median_alignment = 0.9f;
    value.control_residualized = true;
    return value;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    std::string error;

    common_flydelta_teaching_material_store material_store;
    const auto material_identity = identity();
    if (!material_store.observe(relation("relation:one"), material_identity, error) ||
            !material_store.observe(relation("relation:two"), material_identity, error)) {
        return fail("teaching relations were not persisted: " + error);
    }

    common_flydelta_teaching_material_group group;
    if (!material_store.group_ready(
            "dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
            material_identity, 2, group, error) || !group.relation_set_ready ||
            group.trajectory_material_ready) {
        return fail("relation-ready material group was not formed: " + error);
    }

    const auto root = fs::temp_directory_path() / "llama-agent-flydelta-concept-capture-smoke";
    std::error_code ignored;
    fs::remove_all(root, ignored);

    auto request = parent_request(group.group_ref);
    common_flydelta_experiment_collection_result collection_result;
    if (!common_flydelta_collect_experiment_job(
            root, {}, request, collection_result, error) ||
            collection_result != common_flydelta_experiment_collection_result::enqueued) {
        return fail("parent search job was not queued: " + error);
    }

    common_flydelta_claimed_experiment_job parent;
    if (!common_flydelta_experiment_queue_claim_next(root, {}, parent, error) ||
            parent.job.kind != common_flydelta_experiment_job_kind::search_pipeline) {
        return fail("parent search job was not claimable: " + error);
    }

    if (!common_flydelta_collect_next_action_job(
            root, {}, parent.job,
            common_flydelta_next_action::prepare_concept_material,
            {}, {}, {}, {}, collection_result, error) ||
            collection_result != common_flydelta_experiment_collection_result::enqueued) {
        return fail("concept capture follow-up was not queued: " + error);
    }

    common_flydelta_evaluator_config evaluator_config;
    evaluator_config.pipeline.dimension = 3;
    common_flydelta_evaluator_callbacks callbacks;
    callbacks.run_concept_capture = [](
            const common_flydelta_experiment_job &,
            std::vector<std::string> & trajectory_refs,
            std::string &) {
        trajectory_refs = {
            "trajectory://concept-capture-smoke/one",
            "trajectory://concept-capture-smoke/two",
        };
        return true;
    };

    common_flydelta_experiment_worker_report capture_report;
    if (!common_flydelta_experiment_worker_run_evaluator_once(
            root, {}, evaluator_config, callbacks, capture_report, error) ||
            capture_report.state != common_flydelta_experiment_queue_state::succeeded ||
            capture_report.completed_job.kind != common_flydelta_experiment_job_kind::concept_capture ||
            capture_report.completed_job.teaching_material_group_ref != group.group_ref ||
            capture_report.concept_trajectory_refs.size() != 2 ||
            !capture_report.has_next_action ||
            capture_report.next_action != common_flydelta_next_action::run_concept_synthesis) {
        return fail("bounded concept capture worker slice failed: " + error);
    }

    for (const auto & ref : capture_report.concept_trajectory_refs) {
        if (!material_store.observe_trajectory(group.group_ref, ref, material_identity, error)) {
            return fail("captured trajectory was not persisted: " + error);
        }
    }
    common_flydelta_teaching_material_group ready_group;
    if (!material_store.resolve_ready_group(group.group_ref, ready_group, error) ||
            !ready_group.trajectory_material_ready ||
            ready_group.trajectory_refs.size() != 2) {
        return fail("captured trajectories did not make material ready: " + error);
    }

    if (!common_flydelta_collect_next_action_job(
            root, {}, capture_report.completed_job,
            common_flydelta_next_action::run_concept_synthesis,
            {}, {}, {}, {}, collection_result, error) ||
            collection_result != common_flydelta_experiment_collection_result::enqueued) {
        return fail("synthesis follow-up was not queued: " + error);
    }

    callbacks = {};
    callbacks.run_concept_synthesis = [](
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_concept_candidate> & candidates,
            std::string &) {
        candidates.push_back(candidate());
        return true;
    };
    callbacks.persist_experimental_direction = [](
            const common_flydelta_direction_candidate & value,
            std::string & direction_ref,
            std::string &) {
        if (!value.experimental_only || value.values.size() != 3) return false;
        direction_ref = "flydelta://concept-direction/smoke";
        return true;
    };
    common_flydelta_experiment_worker_report synthesis_report;
    if (!common_flydelta_experiment_worker_run_evaluator_once(
            root, {}, evaluator_config, callbacks, synthesis_report, error) ||
            synthesis_report.state != common_flydelta_experiment_queue_state::succeeded ||
            synthesis_report.completed_job.kind != common_flydelta_experiment_job_kind::concept_synthesis ||
            synthesis_report.completed_job.teaching_material_group_ref != group.group_ref ||
            synthesis_report.concept_candidates.size() != 1 ||
            synthesis_report.graft_direction_ref != "flydelta://concept-direction/smoke" ||
            !synthesis_report.has_next_action ||
            synthesis_report.next_action != common_flydelta_next_action::run_bootstrap ||
            synthesis_report.concept_candidates.front().concept_key !=
                "dataset.grouped_sum.v1") {
        return fail("concept synthesis handoff failed: " + error);
    }

    if (!common_flydelta_collect_next_action_job(
            root, {}, synthesis_report.completed_job,
            common_flydelta_next_action::run_bootstrap,
            {}, {}, {}, synthesis_report.graft_direction_ref,
            collection_result, error) ||
            collection_result != common_flydelta_experiment_collection_result::enqueued) {
        return fail("concept graft search follow-up was not queued: " + error);
    }
    common_flydelta_claimed_experiment_job graft_search;
    if (!common_flydelta_experiment_queue_claim_next(root, {}, graft_search, error) ||
            graft_search.job.kind != common_flydelta_experiment_job_kind::search_pipeline ||
            graft_search.job.seed.candidate_ref != synthesis_report.graft_direction_ref) {
        return fail("concept graft did not resume the ordinary search pipeline: " + error);
    }

    fs::remove_all(root, ignored);
    std::cout << "flydelta_concept_capture_smoke=completed"
              << " relation_ready=yes"
              << " capture_refs=" << capture_report.concept_trajectory_refs.size()
              << " trajectory_material_ready=yes"
              << " synthesis_handoff=yes"
              << " graft_to_search=yes"
              << " candidate_count=" << synthesis_report.concept_candidates.size()
              << " learning_credit=none"
              << " promotion=false\n";
    return 0;
}
