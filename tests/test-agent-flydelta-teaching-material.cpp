#include "agent/adaptation/flydelta/flydelta-job.h"
#include "agent/adaptation/flydelta/flydelta-concept-capture.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-representation-augmentation.h"
#include "agent/adaptation/flydelta/flydelta-teaching-material.h"

#include <nlohmann/json.hpp>

#include <memory>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_teaching_relation relation(const std::string & id) {
    common_flydelta_teaching_relation value;
    value.id = id;
    value.teaching_key = "dataset.grouped_sum.v1";
    value.source = common_adaptation_evidence_source::procedure_blueprint;
    value.behavior_key = "tool_choice/dataset/grouped_sum";
    value.scope.namespace_id = "local";
    value.scope.session_id = "session";
    value.task_fingerprint = "task:" + id;
    value.baseline_ref = "capture:" + id + "/baseline";
    value.conditioned_ref = "capture:" + id + "/conditioned";
    value.control_ref = "capture:" + id + "/control";
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

static common_flydelta_teaching_material_identity identity() {
    common_flydelta_teaching_material_identity value;
    value.model_profile_fingerprint = "model:v1";
    value.tokenizer_fingerprint = "tokenizer:v1";
    value.template_fingerprint = "template:v1";
    value.capture_layout_revision = "capture:v1";
    value.execution_context_fingerprint = "context:v1";
    value.scope_fingerprint = "scope:v1";
    value.verifier_revision = "verifier:v1";
    return value;
}

int main() {
    std::string error;
    common_flydelta_teaching_material_store store;
    const auto material_identity = identity();
    CHECK(common_flydelta_teaching_material_identity_validate(material_identity, error));
    CHECK(store.observe(relation("relation:1"), material_identity, error));
    CHECK(store.observe(relation("relation:1"), material_identity, error));

    common_flydelta_teaching_material_group group;
    CHECK(store.group_ready("dataset.grouped_sum.v1",
        "tool_choice/dataset/grouped_sum", material_identity, 2, group, error));
    CHECK(!group.relation_set_ready);
    CHECK(store.observe(relation("relation:2"), material_identity, error));
    CHECK(store.group_ready("dataset.grouped_sum.v1",
        "tool_choice/dataset/grouped_sum", material_identity, 2, group, error));
    CHECK(group.relation_set_ready && !group.trajectory_material_ready &&
        group.relation_refs.size() == 2);
    common_flydelta_teaching_material_group resolved;
    CHECK(store.resolve_relation_set(group.group_ref, resolved, error));
    CHECK(resolved.relation_set_ready && !resolved.trajectory_material_ready &&
        resolved.group_ref == group.group_ref);
    CHECK(store.resolve_ready_group(group.group_ref, resolved, error));
    CHECK(!resolved.trajectory_material_ready);
    CHECK(store.observe_trajectory(group.group_ref, "trajectory://one", material_identity, error));
    CHECK(store.observe_trajectory(group.group_ref, "trajectory://two", material_identity, error));
    CHECK(store.resolve_ready_group(group.group_ref, resolved, error));
    CHECK(resolved.trajectory_material_ready && resolved.trajectory_refs.size() == 2);
    CHECK(store.resolve_ready_group("flydelta://teaching-material/missing", resolved, error));
    CHECK(!resolved.trajectory_material_ready);

    const auto snapshot = common_flydelta_teaching_material_store_to_json(store);
    common_flydelta_teaching_material_store restored;
    CHECK(common_flydelta_teaching_material_store_from_json(snapshot, restored, error));
    CHECK(restored.groups().size() == store.groups().size());
    CHECK(restored.groups().front().relation_refs.size() == 2);
    CHECK(restored.groups().front().trajectory_material_ready);
    auto legacy_snapshot = nlohmann::ordered_json::parse(snapshot);
    for (auto & item : legacy_snapshot["groups"]) {
        item.erase("relation_set_ready");
        item.erase("trajectory_material_ready");
        item.erase("trajectory_refs");
        item.erase("minimum_trajectories");
        item["ready"] = true;
    }
    common_flydelta_teaching_material_store legacy_restored;
    CHECK(common_flydelta_teaching_material_store_from_json(
        legacy_snapshot.dump(), legacy_restored, error));
    CHECK(legacy_restored.groups().front().relation_set_ready);
    CHECK(!legacy_restored.groups().front().trajectory_material_ready);

    auto other_identity = material_identity;
    other_identity.template_fingerprint = "template:v2";
    CHECK(store.observe(relation("relation:3"), other_identity, error));
    CHECK(store.groups().size() == 2);

    auto material_runtime = std::make_shared<common_flydelta_teaching_material_runtime>(
        material_identity);
    CHECK(material_runtime->observe_relation(relation("relation:runtime-1"), error));
    CHECK(material_runtime->observe_relation(relation("relation:runtime-2"), error));
    bool relation_ready = false;
    bool trajectory_ready = true;
    CHECK(material_runtime->inspect_group(
        material_runtime->store().groups().front().group_ref,
        relation_ready, trajectory_ready, error));
    CHECK(relation_ready && !trajectory_ready);
    common_flydelta_teaching_material_group family_group;
    bool family_available = false;
    CHECK(material_runtime->resolve_group_for_family(
        "dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    CHECK(family_available && family_group.group_ref == material_runtime->store().groups().front().group_ref);
    CHECK(material_runtime->resolve_group_for_family(
        "dataset.missing", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    CHECK(!family_available);

    common_flydelta_representation_augmentation_action action;
    CHECK(common_flydelta_select_concept_synthesis_escape(false, true, action, error));
    CHECK(action == common_flydelta_representation_augmentation_action::retain);
    CHECK(common_flydelta_select_concept_synthesis_escape(true, false, action, error));
    CHECK(action == common_flydelta_representation_augmentation_action::retain);
    CHECK(common_flydelta_select_concept_synthesis_escape(true, true, action, error));
    CHECK(action == common_flydelta_representation_augmentation_action::run_concept_synthesis);
    CHECK(common_flydelta_select_concept_material_escape(
        true, true, false, action, error));
    CHECK(action == common_flydelta_representation_augmentation_action::prepare_concept_material);
    CHECK(common_flydelta_select_concept_material_escape(
        true, true, true, action, error));
    CHECK(action == common_flydelta_representation_augmentation_action::run_concept_synthesis);

    common_flydelta_concept_capture_plan capture_plan;
    capture_plan.id = "plan:grouped-sum:relation-1";
    capture_plan.group_ref = group.group_ref;
    capture_plan.relation_ref = "relation:1";
    capture_plan.baseline_ref = "capture:relation:1/baseline";
    capture_plan.conditioned_ref = "capture:relation:1/conditioned";
    capture_plan.control_ref = "capture:relation:1/control";
    capture_plan.semantic_anchor = "tool-choice";
    capture_plan.layer_index = 12;
    capture_plan.identity = material_identity;
    CHECK(common_flydelta_concept_capture_plan_validate(capture_plan, error));
    const auto capture_plan_json = common_flydelta_concept_capture_plan_to_json(capture_plan);
    common_flydelta_concept_capture_plan decoded_capture_plan;
    CHECK(common_flydelta_concept_capture_plan_from_json(
        capture_plan_json, decoded_capture_plan, error));
    CHECK(decoded_capture_plan.group_ref == capture_plan.group_ref);

    common_flydelta_experiment_job job;
    job.schema_version = 1;
    job.id = "job:concept-synthesis";
    job.kind = common_flydelta_experiment_job_kind::concept_synthesis;
    job.seed.id = "seed:concept-synthesis";
    job.seed.behavior_key = "tool_choice/dataset/grouped_sum";
    job.seed.source = common_adaptation_evidence_source::procedure_blueprint;
    job.seed.split = common_flydelta_training_split::train;
    job.seed.scope.namespace_id = "local";
    job.seed.scope.session_id = "session";
    job.seed.task_fingerprint = "task:concept";
    job.seed.model_profile_fingerprint = "model:v1";
    job.seed.tokenizer_fingerprint = "tokenizer:v1";
    job.seed.template_fingerprint = "template:v1";
    job.seed.execution_context_fingerprint = "context:v1";
    job.seed.baseline_ref = "capture:relation:1/baseline";
    job.seed.candidate_ref = "capture:relation:1/conditioned";
    job.seed.verifier_ref = "verifier:grouped-sum:v1";
    job.seed.evidence_ref = "evidence:relation:1";
    job.teaching_material_group_ref = group.group_ref;
    job.code_revision = "test:v1";
    CHECK(common_flydelta_experiment_job_validate(job, 8, error));
    const auto encoded = common_flydelta_experiment_job_to_json(job);
    common_flydelta_experiment_job decoded;
    CHECK(common_flydelta_experiment_job_from_json(encoded, decoded, error));
    CHECK(decoded.kind == common_flydelta_experiment_job_kind::concept_synthesis);
    CHECK(decoded.teaching_material_group_ref == job.teaching_material_group_ref);

    common_flydelta_experiment_job capture_job = job;
    capture_job.id = "job:concept-capture";
    capture_job.kind = common_flydelta_experiment_job_kind::concept_capture;
    CHECK(common_flydelta_experiment_job_validate(capture_job, 8, error));

    common_flydelta_evaluator_config evaluator_config;
    evaluator_config.pipeline.dimension = 3;
    common_flydelta_evaluator_callbacks callbacks;
    callbacks.run_concept_synthesis = [](
            const common_flydelta_experiment_job &,
            std::vector<common_flydelta_concept_candidate> & candidates,
            std::string &) {
        common_flydelta_concept_candidate candidate;
        candidate.concept_key = "dataset.grouped_sum.v1";
        candidate.extraction_id = "extraction:grouped-sum:1";
        candidate.behavior_key = "tool_choice/dataset/grouped_sum";
        candidate.model_profile_fingerprint = "model:v1";
        candidate.capture_layout_revision = "capture:v1";
        candidate.layer_index = 12;
        candidate.values = {0.26726124f, 0.53452248f, 0.80178374f};
        candidate.source_trajectories = 2;
        candidate.retained_trajectories = 2;
        candidate.median_alignment = 0.9f;
        candidate.control_residualized = true;
        candidates.push_back(std::move(candidate));
        return true;
    };
    callbacks.persist_experimental_direction = [](
            const common_flydelta_direction_candidate & candidate,
            std::string & direction_ref,
            std::string &) {
        if (!candidate.experimental_only || candidate.values.size() != 3) return false;
        direction_ref = "flydelta://concept-direction/test";
        return true;
    };
    common_flydelta_evaluator_result evaluator_result;
    CHECK(common_flydelta_evaluate_job(
        job, evaluator_config, callbacks, evaluator_result, error));
    CHECK(evaluator_result.concept_candidates.size() == 1);
    CHECK(evaluator_result.direction_candidates.size() == 1);
    CHECK(evaluator_result.direction_candidates.front().experimental_only);
    CHECK(evaluator_result.direction_candidates.front().origin ==
        "host_taught_extracted");
    CHECK(evaluator_result.graft_direction_ref == "flydelta://concept-direction/test");
    CHECK(evaluator_result.next_action == common_flydelta_next_action::run_bootstrap);

    common_flydelta_evaluator_callbacks capture_callbacks;
    capture_callbacks.run_concept_capture = [](
            const common_flydelta_experiment_job &,
            std::vector<std::string> & refs,
            std::string &) {
        refs = {"trajectory://one", "trajectory://two"};
        return true;
    };
    common_flydelta_evaluator_result capture_result;
    CHECK(common_flydelta_evaluate_job(
        capture_job, evaluator_config, capture_callbacks, capture_result, error));
    CHECK(capture_result.concept_trajectory_refs.size() == 2);
    CHECK(capture_result.next_action == common_flydelta_next_action::run_concept_synthesis);
    return 0;
}
