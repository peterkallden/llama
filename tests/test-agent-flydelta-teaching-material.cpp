#include "agent/adaptation/flydelta/flydelta-job.h"
#include "agent/adaptation/flydelta/flydelta-concept-capture.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-representation-augmentation.h"
#include "agent/adaptation/flydelta/flydelta-teaching-material.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <filesystem>
#include <chrono>
#include <thread>

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
    CHECK(common_flydelta_teaching_material_execution_matches(material_identity,
        "model:v1", "tokenizer:v1", "template:v1", "context:v1", "capture:v1", error));
    CHECK(!common_flydelta_teaching_material_execution_matches(material_identity,
        "model:v1", "tokenizer:other", "template:v1", "context:v1", "capture:v1", error));
    CHECK(error.find("tokenizer fingerprint") != std::string::npos);
    CHECK(!common_flydelta_teaching_material_execution_matches(material_identity,
        "model:v1", "tokenizer:v1", "template:v1", "context:v1", "capture:other", error));
    CHECK(error.find("capture layout revision") != std::string::npos);
    auto incompatible_source = relation("relation:historical");
    incompatible_source.tokenizer_fingerprint = "tokenizer:historical";
    CHECK(!store.observe(incompatible_source, material_identity, error));
    CHECK(incompatible_source.tokenizer_fingerprint == "tokenizer:historical");
    CHECK(store.groups().empty());
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

    // Layer-local readiness must count distinct authorized relations, never
    // combine trajectories from different layers or alter the legacy group.
    CHECK(store.observe_layer_trajectory(group.group_ref, 12, "relation:1", "trajectory://12-1", material_identity, error));
    CHECK(store.observe_layer_trajectory(group.group_ref, 12, "relation:1", "trajectory://12-1", material_identity, error));
    CHECK(store.observe_layer_trajectory(group.group_ref, 13, "relation:2", "trajectory://13-2", material_identity, error));
    CHECK(store.resolve_layer_group(group.group_ref, 12, resolved, error));
    CHECK(!resolved.trajectory_material_ready);
    CHECK(!store.observe_layer_trajectory(group.group_ref, 12, "relation:1", "trajectory://conflict", material_identity, error));
    CHECK(!store.observe_layer_trajectory(group.group_ref, 12, "relation:missing", "trajectory://missing", material_identity, error));
    CHECK(!store.observe_layer_trajectory(group.group_ref, 13, "relation:1", "trajectory://12-1", material_identity, error));
    CHECK(store.observe_layer_trajectory(group.group_ref, 12, "relation:2", "trajectory://12-2", material_identity, error));
    CHECK(store.resolve_layer_group(group.group_ref, 12, resolved, error));
    CHECK(resolved.trajectory_material_ready && resolved.trajectory_refs.size() == 2);
    CHECK(store.resolve_layer_group(group.group_ref, 13, resolved, error));
    CHECK(!resolved.trajectory_material_ready);

    CHECK(store.observe_layer_trajectory(group.group_ref, 12, "relation:1", "trajectory://challenge-1", material_identity, error, "set:one"));
    CHECK(store.observe_layer_trajectory(group.group_ref, 12, "relation:2", "trajectory://challenge-2", material_identity, error, "set:two"));
    CHECK(store.resolve_layer_group(group.group_ref, 12, resolved, error, "set:one"));
    CHECK(!resolved.trajectory_material_ready);
    CHECK(store.resolve_layer_group(group.group_ref, 12, resolved, error, "set:two"));
    CHECK(!resolved.trajectory_material_ready);
    const auto snapshot = common_flydelta_teaching_material_store_to_json(store);
    common_flydelta_teaching_material_store restored;
    CHECK(common_flydelta_teaching_material_store_from_json(snapshot, restored, error));
    CHECK(restored.groups().size() == store.groups().size());
    CHECK(restored.groups().front().relation_refs.size() == 2);
    CHECK(restored.groups().front().trajectory_material_ready);
    CHECK(restored.resolve_layer_group(group.group_ref, 12, resolved, error));
    CHECK(resolved.trajectory_material_ready && resolved.trajectory_refs.size() == 2);
    CHECK(restored.resolve_layer_group(group.group_ref, 13, resolved, error));
    CHECK(!resolved.trajectory_material_ready);
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

    // Positive-only and contrast material are alternative representations of
    // the same family. The established contrast route must win regardless of
    // insertion order, while incomplete material must not mask a ready group.
    auto positive_relation = [](const std::string & id) {
        auto value = relation(id);
        value.schema_version = 2;
        value.positive_only = true;
        value.source = common_adaptation_evidence_source::user_taught_concept;
        value.baseline_ref.clear();
        value.control_ref.clear();
        value.baseline_origin = common_flydelta_teaching_origin::none;
        value.control_origin = common_flydelta_teaching_origin::none;
        value.conditioned_origin = common_flydelta_teaching_origin::user_supplied;
        return value;
    };
    for (bool positive_first : {false, true}) {
        common_flydelta_teaching_material_runtime mixed(material_identity);
        if (positive_first) {
            for (int i = 0; i < 3; ++i)
                CHECK(mixed.observe_relation(positive_relation("positive:" + std::to_string(i)), error));
        }
        CHECK(mixed.observe_relation(relation("contrast:1"), error));
        CHECK(mixed.observe_relation(relation("contrast:2"), error));
        if (!positive_first) {
            for (int i = 0; i < 3; ++i)
                CHECK(mixed.observe_relation(positive_relation("positive:" + std::to_string(i)), error));
        }
        CHECK(mixed.resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
            family_group, family_available, error));
        CHECK(family_available && !family_group.positive_only);
        CHECK(mixed.resolve_group_for_behavior("tool_choice/dataset/grouped_sum",
            family_group, family_available, error));
        CHECK(family_available && !family_group.positive_only);
        CHECK(mixed.resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
            family_group, family_available, error, common_flydelta_teaching_material_mode::positive_only));
        CHECK(family_available && family_group.positive_only);
        CHECK(mixed.store().group_ready("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
            material_identity, 2, family_group, error));
        CHECK(family_group.relation_set_ready && !family_group.positive_only);
    }
    // A rejected contrast must not mutate or replace a valid selected group.
    auto rejected_contrast = relation("contrast:rejected");
    rejected_contrast.host_approved = false;
    CHECK(!material_runtime->observe_relation(rejected_contrast, error));
    CHECK(material_runtime->resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    CHECK(family_available && !family_group.positive_only);

    common_flydelta_teaching_material_store partially_ready;
    CHECK(partially_ready.observe(relation("contrast:incomplete"), material_identity, error));
    for (int i = 0; i < 3; ++i)
        CHECK(partially_ready.observe(positive_relation("positive:ready:" + std::to_string(i)), material_identity, error));
    CHECK(partially_ready.group_ready("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        material_identity, 2, family_group, error));
    CHECK(family_group.relation_set_ready && family_group.positive_only);

    // Missing extraction control is incomplete material, not rejected contrast.
    common_flydelta_teaching_material_store missing_control;
    auto pending_contrast = relation("contrast:pending-control");
    pending_contrast.control_ref.clear();
    pending_contrast.control_origin = common_flydelta_teaching_origin::none;
    CHECK(missing_control.observe(pending_contrast, material_identity, error));
    CHECK(missing_control.groups().front().relation_refs.empty());
    CHECK(missing_control.groups().front().pending_relation_refs.size() == 1);
    for (int i = 0; i < 3; ++i)
        CHECK(missing_control.observe(positive_relation("positive:pending:" + std::to_string(i)), material_identity, error));
    common_flydelta_teaching_material_store restored_pending;
    CHECK(common_flydelta_teaching_material_store_from_json(
        common_flydelta_teaching_material_store_to_json(missing_control), restored_pending, error));
    CHECK(restored_pending.group_ready("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        material_identity, 2, family_group, error));
    CHECK(family_group.relation_set_ready && family_group.positive_only);
    CHECK(restored_pending.observe(relation("contrast:pending-control"), material_identity, error));
    CHECK(restored_pending.groups().front().pending_relation_refs.empty());
    CHECK(restored_pending.groups().front().relation_refs.size() == 1);

    // Rejection is a durable assessment, not absence of usable material.
    // Replay the real lifecycle codec after closing its JSONL writer.
    const auto assessment_path = std::filesystem::temp_directory_path() /
        ("flydelta-contrast-assessment-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".jsonl");
    auto journal = std::make_shared<common_learning_jsonl_lifecycle_store>();
    CHECK(journal->open(assessment_path, error));
    auto assessed = std::make_shared<common_flydelta_teaching_material_runtime>(material_identity);
    assessed->bind_assessment_journal(journal);
    CHECK(assessed->observe_relation(relation("assessed:contrast:1"), error));
    CHECK(assessed->observe_relation(relation("assessed:contrast:2"), error));
    for (int i = 0; i < 3; ++i)
        CHECK(assessed->observe_relation(positive_relation("assessed:positive:" + std::to_string(i)), error));
    CHECK(assessed->resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    CHECK(family_available && !family_group.positive_only);
    const auto rejected_group_ref = family_group.group_ref;
    CHECK(assessed->assess_contrast(rejected_group_ref, "verifier:assessment", {"evidence:rejected"}, true, error,
        common_flydelta_teaching_material_revision(assessed->store().groups().front()), "verifier:v1"));
    CHECK(!assessed->resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    CHECK(error.find("reassessment") != std::string::npos);
    CHECK(assessed->resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error, common_flydelta_teaching_material_mode::positive_only));
    CHECK(family_available && family_group.positive_only);
    // A rejected contrast need not have reached the accepted-material store.
    // The resolver must still replay it before falling back to positives.
    for (int i = 0; i < 3; ++i) {
        auto positive = positive_relation("standalone:positive:" + std::to_string(i));
        positive.teaching_key = "concept:standalone";
        CHECK(assessed->observe_relation(positive, error));
    }
    auto rejected_source = relation("standalone:rejected");
    rejected_source.teaching_key = "concept:standalone";
    rejected_source.status = common_flydelta_teaching_relation_status::rejected;
    rejected_source.baseline_ref.clear();
    rejected_source.control_ref.clear();
    const auto accepted_groups = assessed->store().groups().size();
    CHECK(assessed->observe_relation(rejected_source, error));
    CHECK(assessed->store().groups().size() == accepted_groups);
    CHECK(!assessed->resolve_group_for_family("concept:standalone", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    const auto assessed_snapshot = common_flydelta_teaching_material_store_to_json(assessed->store());
    assessed.reset();
    journal.reset();
    journal = std::make_shared<common_learning_jsonl_lifecycle_store>();
    CHECK(journal->open(assessment_path, error));
    assessed = std::make_shared<common_flydelta_teaching_material_runtime>(material_identity);
    CHECK(common_flydelta_teaching_material_store_from_json(assessed_snapshot, assessed->store(), error));
    assessed->bind_assessment_journal(journal);
    CHECK(!assessed->resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    CHECK(!assessed->resolve_group_for_family("concept:standalone", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    CHECK(assessed->resolve_group_for_family("concept:standalone", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error, common_flydelta_teaching_material_mode::positive_only));
    CHECK(family_available && family_group.positive_only);
    CHECK(assessed->observe_relation(relation("assessed:contrast:3"), error));
    CHECK(!assessed->resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    CHECK(!assessed->assess_contrast(rejected_group_ref, "verifier:assessment", {"evidence:revalidated"}, false, error,
        "sha256:stale-material", "verifier:v1"));
    CHECK(assessed->assess_contrast(rejected_group_ref, "verifier:assessment", {"evidence:revalidated"}, false, error,
        common_flydelta_teaching_material_revision(assessed->store().groups().front()), "verifier:v1"));
    CHECK(assessed->resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    CHECK(family_available && !family_group.positive_only);
    // A later rejection must not be swallowed by an earlier idempotency key.
    CHECK(assessed->assess_contrast(rejected_group_ref, "verifier:assessment", {"evidence:rejected"}, true, error,
        common_flydelta_teaching_material_revision(assessed->store().groups().front()), "verifier:v1"));
    CHECK(!assessed->resolve_group_for_family("dataset.grouped_sum.v1", "tool_choice/dataset/grouped_sum",
        family_group, family_available, error));
    assessed.reset();
    journal.reset();
    std::error_code assessment_cleanup_error;
    std::filesystem::remove(assessment_path, assessment_cleanup_error);

    // Concurrent workers update the same durable index through the runtime
    // facade. Neither completed relation may be lost from the saved snapshot.
    std::string concurrent_snapshot;
    common_flydelta_teaching_material_persistence concurrent_persistence;
    concurrent_persistence.save = [&](const std::string & snapshot, std::string &) {
        concurrent_snapshot = snapshot;
        return true;
    };
    common_flydelta_teaching_material_runtime concurrent_runtime(material_identity, concurrent_persistence);
    CHECK(concurrent_runtime.observe_relation(relation("relation:parallel-1"), error));
    CHECK(concurrent_runtime.observe_relation(relation("relation:parallel-2"), error));
    const auto concurrent_group = concurrent_runtime.store().groups().front().group_ref;
    bool first_saved = false, second_saved = false;
    std::string first_error, second_error;
    std::thread first([&] { first_saved = concurrent_runtime.observe_layer_trajectory(
        concurrent_group, 12, "relation:parallel-1", "trajectory:parallel-1", first_error, "set:parallel"); });
    std::thread second([&] { second_saved = concurrent_runtime.observe_layer_trajectory(
        concurrent_group, 12, "relation:parallel-2", "trajectory:parallel-2", second_error, "set:parallel"); });
    first.join();
    second.join();
    CHECK(first_saved && second_saved);
    common_flydelta_teaching_material_store concurrent_restored;
    CHECK(common_flydelta_teaching_material_store_from_json(concurrent_snapshot, concurrent_restored, error));
    CHECK(concurrent_restored.resolve_layer_group(concurrent_group, 12, resolved, error, "set:parallel"));
    CHECK(resolved.trajectory_material_ready && resolved.trajectory_refs.size() == 2);

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
        candidate.control_trajectories = 2;
        candidate.retained_control_trajectories = 2;
        candidate.median_alignment = 0.9f;
        candidate.control_residualized = true;
        candidates.push_back(std::move(candidate));
        common_flydelta_concept_candidate prototype;
        prototype.synthesis_semantics =
            common_flydelta_concept_synthesis_semantics::positive_prototype;
        prototype.concept_key = "dataset.grouped_sum.v1";
        prototype.extraction_id = "extraction:grouped-sum:1";
        prototype.behavior_key = "tool_choice/dataset/grouped_sum";
        prototype.origin = "host_taught_positive_prototype";
        prototype.model_profile_fingerprint = "model:v1";
        prototype.capture_layout_revision = "capture:v1";
        prototype.layer_index = 12;
        prototype.values = {0.80178374f, 0.53452248f, 0.26726124f};
        prototype.source_trajectories = 2;
        prototype.retained_trajectories = 2;
        prototype.control_trajectories = 2;
        prototype.retained_control_trajectories = 2;
        prototype.median_alignment = 0.9f;
        prototype.control_residualized = false;
        candidates.push_back(std::move(prototype));
        return true;
    };
    callbacks.persist_experimental_direction = [](
            const common_flydelta_direction_candidate & candidate,
            std::string & direction_ref,
            std::string &) {
        if (!candidate.experimental_only || candidate.values.size() != 3) return false;
        direction_ref = candidate.kind == common_flydelta_direction_kind::positive_prototype
            ? "flydelta://concept-direction/prototype"
            : "flydelta://concept-direction/control";
        return true;
    };
    common_flydelta_evaluator_result evaluator_result;
    CHECK(common_flydelta_evaluate_job(
        job, evaluator_config, callbacks, evaluator_result, error));
    CHECK(evaluator_result.concept_candidates.size() == 2);
    CHECK(evaluator_result.direction_candidates.size() == 2);
    CHECK(evaluator_result.direction_candidates.front().experimental_only);
    CHECK(evaluator_result.direction_candidates.front().origin ==
        "host_taught_extracted");
    CHECK(evaluator_result.graft_direction_refs.size() == 2);
    CHECK(evaluator_result.graft_direction_ref ==
        "flydelta://concept-direction/control");
    CHECK(evaluator_result.graft_direction_refs[1] ==
        "flydelta://concept-direction/prototype");
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
    // The challenge freezes its inputs across queue serialization and uses
    // its own layer-family persister, never the ordinary paired/local graft.
    auto challenge = job;
    challenge.concept_localization.capture_set_ref = "challenge:one";
    challenge.concept_localization.origin_state_ref = "state:origin";
    challenge.concept_localization.origin_direction_ref = challenge.seed.candidate_ref;
    challenge.concept_localization.layers = {12, 14};
    challenge.concept_localization.max_capture_model_calls = 12;
    CHECK(common_flydelta_experiment_job_from_json(common_flydelta_experiment_job_to_json(challenge), decoded, error));
    CHECK(decoded.concept_localization.layers == challenge.concept_localization.layers);
    CHECK(common_flydelta_concept_localization_material_key(decoded) == common_flydelta_concept_localization_material_key(challenge));
    decoded.seed.baseline_ref = "baseline:other";
    CHECK(common_flydelta_concept_localization_material_key(decoded) != common_flydelta_concept_localization_material_key(challenge));
    auto invalid_challenge = challenge;
    invalid_challenge.concept_localization.layers = {12, 12};
    CHECK(!common_flydelta_experiment_job_validate(invalid_challenge, 8, error));
    callbacks.run_concept_localization_synthesis = [&](const auto & request,
            std::vector<common_flydelta_concept_candidate> & output, std::string & assessment_ref, std::string & callback_error) {
        if (!callbacks.run_concept_synthesis(request, output, callback_error)) return false;
        const auto original = output;
        for (auto candidate : original) {
            candidate.layer_index = 14;
            candidate.extraction_id += ":layer14";
            candidate.values = {0, 1, 0};
            output.push_back(std::move(candidate));
        }
        assessment_ref = "scan:one";
        return true;
    };
    size_t persisted_families = 0;
    callbacks.persist_experimental_layer_family_for_job = [&](const auto &, const auto & directions,
            const std::string & assessment_ref, std::string & reference, std::string &) {
        if (assessment_ref != "scan:one" || directions.size() != 2 ||
                directions[0].layer_index != 12 || directions[1].layer_index != 14 ||
                directions[0].values == directions[1].values) return false;
        reference = "family:" + std::to_string(++persisted_families);
        return true;
    };
    CHECK(common_flydelta_evaluate_job(challenge, evaluator_config, callbacks, evaluator_result, error));
    CHECK(persisted_families == 2 && evaluator_result.graft_direction_refs.size() == 2);
    CHECK(evaluator_result.concept_localization_assessment_ref == "scan:one");
    callbacks.run_concept_localization_synthesis = [](const auto &, auto & output, std::string & ref, std::string &) {
        output.clear();
        ref = "scan:no-signal";
        return true;
    };
    CHECK(common_flydelta_evaluate_job(challenge, evaluator_config, callbacks, evaluator_result, error));
    CHECK(evaluator_result.next_action == common_flydelta_next_action::retain && evaluator_result.graft_direction_refs.empty());
    // An explicit v2 representation-only relation has no fake contrast.
    auto positive_sample = relation("positive:1");
    positive_sample.schema_version = 2;
    positive_sample.positive_only = true;
    positive_sample.source = common_adaptation_evidence_source::user_taught_concept;
    positive_sample.baseline_ref.clear(); positive_sample.control_ref.clear();
    positive_sample.baseline_origin = common_flydelta_teaching_origin::none;
    positive_sample.control_origin = common_flydelta_teaching_origin::none;
    CHECK(common_flydelta_teaching_relation_validate(positive_sample, error));
    common_learning_transaction positive_transaction;
    positive_transaction.id = "transaction:positive";
    common_adaptation_evidence_relation behavioral_relation;
    CHECK(!common_agent_teaching_relation_to_evidence_relation(
        positive_sample, positive_transaction, behavioral_relation, error));
    auto wrong_mode = positive_sample;
    wrong_mode.schema_version = 1;
    CHECK(!common_flydelta_teaching_relation_validate(wrong_mode, error));
    CHECK(store.observe(positive_sample, material_identity, error));
    const auto positive_group_ref = store.groups().back().group_ref;
    CHECK(positive_group_ref != group.group_ref && store.groups().back().positive_only);
    CHECK(!store.groups().back().relation_set_ready);
    for (int i = 2; i <= 3; ++i) {
        auto next = positive_sample;
        next.id += ":" + std::to_string(i);
        next.task_fingerprint += ":" + std::to_string(i);
        next.conditioned_ref += ":" + std::to_string(i);
        CHECK(store.observe(next, material_identity, error));
    }
    CHECK(store.groups().back().relation_set_ready && !store.groups().back().trajectory_material_ready);
    for (int i = 1; i <= 3; ++i) CHECK(store.observe_trajectory(positive_group_ref,
        "positive-trajectory:" + std::to_string(i), material_identity, error));
    common_flydelta_teaching_material_store positive_restored;
    CHECK(common_flydelta_teaching_material_store_from_json(
        common_flydelta_teaching_material_store_to_json(store), positive_restored, error));
    CHECK(positive_restored.groups().back().positive_only &&
        positive_restored.groups().back().trajectory_material_ready);
    auto positive_plan = capture_plan;
    positive_plan.schema_version = 2; positive_plan.positive_only = true;
    positive_plan.baseline_ref.clear(); positive_plan.control_ref.clear();
    CHECK(common_flydelta_concept_capture_plan_validate(positive_plan, error));
    CHECK(common_flydelta_concept_capture_plan_from_json(
        common_flydelta_concept_capture_plan_to_json(positive_plan), decoded_capture_plan, error));
    CHECK(decoded_capture_plan.positive_only && decoded_capture_plan.baseline_ref.empty());
    positive_plan.schema_version = 1;
    CHECK(!common_flydelta_concept_capture_plan_validate(positive_plan, error));

    return 0;
}
