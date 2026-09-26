#include "agent/adaptation/flydelta/flydelta-evidence.h"
#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"

#include <array>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_learning_transaction transaction(
        const std::string & id,
        common_learning_signal_type signal_type) {
    common_learning_transaction value;
    value.id = id;
    value.created_at = "2026-09-13T00:00:00Z";
    value.observation.id = id;
    value.observation.scope.namespace_id = "local";
    value.observation.scope.project_id = "project";
    value.observation.scope.session_id = "session";
    value.observation.source_turn_id = id + ":turn";
    value.observation.source_plan_id = id + ":plan";
    value.observation.signals.push_back({signal_type, value.observation.source_plan_id,
        "step", "dataset.inspect", id + ":evidence", "verified", "diagnostics", "native"});
    value.observation.evidence_ids = {id + ":evidence"};
    value.observation.cause = common_learning_cause::model_behavior;
    value.observation.verification = common_learning_verification::host_verified;
    value.observation.idempotency_key = id + ":key";
    value.observation.content_hash = "sha256:" + id;
    value.observation.collection_allowed = true;
    return value;
}

int main() {
    std::string error;
    auto failed = transaction("learning://transaction/failed", common_learning_signal_type::tool_failure);
    auto repaired = transaction("learning://transaction/repaired", common_learning_signal_type::successful_recovery);
    common_flydelta_behavior_transition transition;
    CHECK(common_flydelta_tool_repair_transition_from_transactions(
        failed, repaired, "sha256:task", "tool_use/diagnostics/missing-argument",
        "execution:failed", "execution:repaired",
        "verifier:v1", transition, error));

    common_flydelta_contrast_set contrast_set;
    CHECK(common_flydelta_contrast_set_from_transitions(
        "flydelta://contrast/1", "missing-required-tool-argument", {transition}, 8,
        contrast_set, error));
    CHECK(contrast_set.positive_transaction_ids[0] == repaired.id);
    CHECK(contrast_set.negative_transaction_ids[0] == failed.id);

    common_flydelta_experiment_fixture fixture;
    fixture.id = "flydelta://fixture/1";
    fixture.task_fingerprint = "sha256:task";
    fixture.model_profile_fingerprint = "sha256:model";
    fixture.tokenizer_fingerprint = "sha256:tokenizer";
    fixture.template_fingerprint = "sha256:template";
    fixture.execution_context_fingerprint = "sha256:execution-context";
    fixture.verifier_revision = "verifier:v1";
    common_flydelta_counterfactual_report report;
    report.experiment_id = "flydelta://experiment/1";
    report.fixture_id = fixture.id;
    report.candidate_id = "flydelta://candidate/1";
    report.baseline_profile_id = "base";
    report.candidate_profile_id = "overlay";
    report.baseline.executed = true;
    report.baseline.verifier_known = true;
    report.baseline.passed = false;
    report.baseline.evidence_ref = "evidence:failed";
    report.candidate.executed = true;
    report.candidate.verifier_known = true;
    report.candidate.passed = true;
    report.candidate.overlay_applied = true;
    report.candidate.quality = 1.0f;
    report.candidate.evidence_ref = "evidence:passed";
    report.outcome = common_flydelta_counterfactual_outcome::helped;
    report.quality_delta = 1.0f;
    common_flydelta_intervention_credit credit;
    CHECK(common_flydelta_intervention_credit_from_report(report, credit, error));
    CHECK(credit.eligible_for_learning);

    report.outcome = common_flydelta_counterfactual_outcome::unknown;
    credit.outcome = common_flydelta_counterfactual_outcome::unknown;
    credit.eligible_for_learning = true;
    CHECK(!common_flydelta_intervention_credit_validate(credit, error));

    common_adaptation_evidence source;
    source.id = "evidence://tool-repair/1";
    source.source = common_adaptation_evidence_source::tool_repair;
    source.behavior_key = "tool_use/diagnostics/missing-argument";
    source.scope.namespace_id = "local";
    source.scope.project_id = "project";
    source.scope.session_id = "session";
    source.task_fingerprint = "sha256:task";
    source.baseline_ref = "execution:failed";
    source.candidate_ref = "execution:repaired";
    source.verifier_ref = "verifier:v1";
    source.transaction_ids = {failed.id, repaired.id};
    source.host_verified = true;
    common_flydelta_experiment_seed seed;
    CHECK(common_flydelta_experiment_seed_from_evidence(
        source, "tool_use/diagnostics/missing-argument", "sha256:model",
        "sha256:tokenizer", "sha256:template", "sha256:tool-resource-context",
        common_flydelta_training_split::holdout, seed, error));

    common_flydelta_behavior_transition generic_transition;
    CHECK(common_flydelta_behavior_transition_from_evidence(
        source, failed.id, repaired.id, generic_transition, error));
    CHECK(generic_transition.source == common_adaptation_evidence_source::tool_repair);
    CHECK(generic_transition.behavior_key == source.behavior_key);
    CHECK(generic_transition.baseline_transaction_id == failed.id);
    CHECK(generic_transition.candidate_transaction_id == repaired.id);

    common_flydelta_teaching_relation teaching_relation;
    CHECK(common_flydelta_teaching_relation_from_evidence(
        source, teaching_relation, error));
    CHECK(teaching_relation.status == common_flydelta_teaching_relation_status::resolved);
    CHECK(teaching_relation.source == common_adaptation_evidence_source::tool_repair);
    CHECK(teaching_relation.baseline_origin == common_flydelta_teaching_origin::observed);
    CHECK(teaching_relation.conditioned_origin == common_flydelta_teaching_origin::observed);
    CHECK(teaching_relation.host_approved);
    common_flydelta_behavior_transition teaching_transition;
    CHECK(common_flydelta_teaching_relation_to_transition(
        teaching_relation, source, failed.id, repaired.id, teaching_transition, error));
    CHECK(teaching_transition.id == source.id + "/flydelta/behavior");

    auto procedure_evidence = source;
    procedure_evidence.id = "evidence://procedure/1";
    procedure_evidence.source = common_adaptation_evidence_source::procedure_blueprint;
    CHECK(common_flydelta_procedure_blueprint_teaching_relation_from_evidence(
        procedure_evidence, teaching_relation, error));
    CHECK(teaching_relation.source == common_adaptation_evidence_source::procedure_blueprint);

    auto user_evidence = source;
    user_evidence.id = "evidence://user-correction/1";
    user_evidence.source = common_adaptation_evidence_source::user_correction;
    CHECK(common_flydelta_teaching_relation_from_evidence(
        user_evidence, teaching_relation, error));
    CHECK(teaching_relation.source == common_adaptation_evidence_source::user_correction);

    auto host_relation = common_adaptation_evidence_relation{};
    host_relation.id = "adaptation://procedure/1";
    host_relation.source = common_adaptation_evidence_source::procedure_blueprint;
    host_relation.scope = source.scope;
    host_relation.behavior_key = "procedure/grouped-aggregation";
    host_relation.task_fingerprint = "sha256:procedure-task";
    host_relation.baseline_ref = "execution:procedure-baseline";
    host_relation.candidate_ref = "execution:procedure-conditioned";
    host_relation.verifier_ref = "verifier:procedure-v1";
    common_flydelta_teaching_relation explicit_relation;
    CHECK(common_flydelta_teaching_relation_from_host_relation(
        host_relation, "evidence://procedure/explicit", "execution:procedure-control",
        common_flydelta_teaching_relation_status::resolved,
        common_flydelta_teaching_origin::host_derived,
        common_flydelta_teaching_origin::host_derived,
        common_flydelta_teaching_origin::host_counterfactual,
        0.9f, true, explicit_relation, error));
    CHECK(explicit_relation.control_ref == "execution:procedure-control");

    common_agent_procedure_teaching_request procedure_request;
    procedure_request.relation_id = "relation://procedure/grouped-sum/1";
    procedure_request.teaching_key = "dataset.grouped_sum.v1";
    procedure_request.procedure_ref = "procedure://dataset/grouped-sum/v1";
    procedure_request.blueprint_ref = "blueprint://dataset/grouped-sum/v1";
    procedure_request.scope = source.scope;
    procedure_request.behavior_key = "tool_choice/dataset/grouped_sum";
    procedure_request.task_fingerprint = "sha256:procedure-task";
    procedure_request.baseline_ref = "execution:procedure-baseline";
    procedure_request.conditioned_ref = "execution:procedure-conditioned";
    procedure_request.control_ref = "execution:procedure-control";
    procedure_request.verifier_ref = "verifier:procedure-v1";
    procedure_request.evidence_ref = "evidence://procedure/host-relation";
    procedure_request.confidence = 0.9f;
    procedure_request.host_scope_admitted = true;
    procedure_request.host_verified = true;
    procedure_request.reusable = true;
    procedure_request.require_control = true;
    const auto procedure_result = common_agent_build_procedure_teaching_relation(procedure_request);
    CHECK(procedure_result.status == common_agent_teaching_build_status::resolved);
    CHECK(procedure_result.relation.has_value());
    CHECK(procedure_result.relation->teaching_key == "dataset.grouped_sum.v1");
    CHECK(procedure_result.relation->procedure_ref == procedure_request.procedure_ref);
    CHECK(procedure_result.relation->blueprint_ref == procedure_request.blueprint_ref);
    CHECK(procedure_result.relation->host_approved);

    auto no_contrast_request = procedure_request;
    no_contrast_request.conditioned_ref = no_contrast_request.baseline_ref;
    const auto no_contrast_result = common_agent_build_procedure_teaching_relation(no_contrast_request);
    CHECK(no_contrast_result.status == common_agent_teaching_build_status::no_contrast);
    CHECK(!no_contrast_result.relation.has_value());

    auto missing_control_request = procedure_request;
    missing_control_request.control_ref.reset();
    const auto missing_control_result = common_agent_build_procedure_teaching_relation(missing_control_request);
    CHECK(missing_control_result.status == common_agent_teaching_build_status::incompatible_control);

    auto unverified_request = procedure_request;
    unverified_request.host_verified = false;
    const auto unverified_result = common_agent_build_procedure_teaching_relation(unverified_request);
    CHECK(unverified_result.status == common_agent_teaching_build_status::not_host_verified);

    common_agent_teaching_contrast_spec contrast;
    contrast.id = "contrast://user-correction/grouped-sum/1";
    contrast.positive_ref = "execution:user-correction-conditioned";
    contrast.negative_ref = "execution:user-correction-observed";
    contrast.control_ref = "execution:user-correction-control";
    contrast.changed_dimension = "dataset.operation.intent";
    contrast.invariant_dimensions = {"dataset", "requested_fields", "output_contract"};
    contrast.positive_origin = common_flydelta_teaching_origin::host_derived;
    contrast.negative_origin = common_flydelta_teaching_origin::observed;
    contrast.control_origin = common_flydelta_teaching_origin::host_counterfactual;
    contrast.host_verified = true;
    CHECK(common_agent_teaching_contrast_spec_validate(contrast, error));

    common_agent_user_correction_teaching_request correction_request;
    correction_request.relation_id = "relation://user-correction/grouped-sum/1";
    correction_request.teaching_key = "dataset.grouped_sum.v1";
    correction_request.source_turn_ref = "turn://user-correction/1";
    correction_request.observed_model_execution_ref = contrast.negative_ref;
    correction_request.correction_ref = "feedback://user-correction/1";
    correction_request.scope = source.scope;
    correction_request.behavior_key = "tool_choice/dataset/grouped_sum";
    correction_request.task_fingerprint = "sha256:user-correction-task";
    correction_request.baseline_ref = contrast.negative_ref;
    correction_request.conditioned_ref = contrast.positive_ref;
    correction_request.control_ref = contrast.control_ref;
    correction_request.verifier_ref = "verifier://dataset/v2";
    correction_request.evidence_ref = "evidence://user-correction/1";
    correction_request.contrast_ref = contrast.id;
    correction_request.control_origin = contrast.control_origin;
    correction_request.confidence = 0.9f;
    correction_request.host_scope_admitted = true;
    correction_request.host_verified = true;
    correction_request.reusable = true;
    correction_request.require_control = true;
    const auto correction_result = common_agent_build_user_correction_teaching_relation(correction_request);
    CHECK(correction_result.status == common_agent_teaching_build_status::resolved);
    CHECK(correction_result.relation.has_value());
    CHECK(correction_result.relation->source == common_adaptation_evidence_source::user_correction);
    CHECK(correction_result.relation->contrast_ref == contrast.id);

    auto incorrect_baseline = correction_request;
    incorrect_baseline.baseline_ref = "execution:unrelated";
    CHECK(common_agent_build_user_correction_teaching_relation(incorrect_baseline).status ==
        common_agent_teaching_build_status::no_contrast);
    auto missing_correction_control = correction_request;
    missing_correction_control.control_ref.reset();
    CHECK(common_agent_build_user_correction_teaching_relation(missing_correction_control).status ==
        common_agent_teaching_build_status::incompatible_control);

    common_learning_transaction correction_transaction = transaction(
        "learning://user-correction/1", common_learning_signal_type::user_correction);
    common_adaptation_evidence_relation correction_evidence_relation;
    CHECK(common_agent_teaching_relation_to_evidence_relation(
        *correction_result.relation, correction_transaction, correction_evidence_relation, error));
    common_agent_request correction_host_request;
    correction_host_request.turn_id = source.scope.turn_id;
    correction_host_request.session_id = source.scope.session_id;
    correction_host_request.project_id = source.scope.project_id;
    common_plan_state correction_host_plan;
    correction_host_plan.id = "user-correction-plan";
    common_agent_result correction_host_result;
    correction_host_result.learning_signals.push_back({common_learning_signal_type::user_correction,
        correction_host_plan.id, {}, {}, correction_request.evidence_ref, "explicit correction"});
    common_adaptation_evidence correction_evidence;
    CHECK(common_adaptation_evidence_from_turn(correction_host_request, correction_host_plan,
        correction_host_result, correction_evidence_relation, correction_evidence, error));
    CHECK(correction_evidence.source == common_adaptation_evidence_source::user_correction);
    CHECK(correction_evidence.host_verified);

    auto unresolved = teaching_relation;
    unresolved.status = common_flydelta_teaching_relation_status::no_contrast;
    unresolved.host_approved = false;
    CHECK(common_flydelta_teaching_relation_validate(unresolved, error));

    auto incomplete = teaching_relation;
    incomplete.status = common_flydelta_teaching_relation_status::resolved;
    incomplete.host_approved = false;
    CHECK(!common_flydelta_teaching_relation_validate(incomplete, error));

    const std::array generic_sources = {
        common_adaptation_evidence_source::reflection_alternative,
        common_adaptation_evidence_source::planning_revision,
        common_adaptation_evidence_source::research_alternative,
        common_adaptation_evidence_source::dataset_resource,
        common_adaptation_evidence_source::workflow_code,
        common_adaptation_evidence_source::procedure_blueprint,
        common_adaptation_evidence_source::user_correction,
        common_adaptation_evidence_source::user_taught_concept,
    };
    for (const auto generic_source : generic_sources) {
        auto generic_evidence = source;
        generic_evidence.id = std::string("evidence://generic/") +
            common_adaptation_evidence_source_name(generic_source);
        generic_evidence.source = generic_source;
        generic_evidence.behavior_key = std::string("behavior/") +
            common_adaptation_evidence_source_name(generic_source);
        CHECK(common_flydelta_behavior_transition_from_evidence(
            generic_evidence, failed.id, repaired.id, generic_transition, error));
        CHECK(generic_transition.source == generic_source);
        CHECK(generic_transition.behavior_key == generic_evidence.behavior_key);
    }
    CHECK(seed.split == common_flydelta_training_split::holdout);
    common_flydelta_experiment_fixture seed_fixture;
    CHECK(common_flydelta_experiment_fixture_from_seed(seed, seed_fixture, error));
    CHECK(seed_fixture.verifier_revision == source.verifier_ref);
    source.host_verified = false;
    CHECK(!common_flydelta_experiment_seed_from_evidence(
        source, "tool_use/diagnostics/missing-argument", "sha256:model",
        "sha256:tokenizer", "sha256:template", "sha256:tool-resource-context",
        common_flydelta_training_split::train, seed, error));
    return 0;
}
