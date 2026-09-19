#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"

#include <iostream>
#include <string>

namespace {

common_agent_procedure_teaching_request grouped_sum_request() {
    common_agent_procedure_teaching_request request;
    request.relation_id = "relation://smoke/procedure/grouped-sum";
    request.teaching_key = "dataset.grouped_sum.v1";
    request.procedure_ref = "procedure://smoke/grouped-sum/v1";
    request.blueprint_ref = "blueprint://smoke/grouped-sum/v1";
    request.scope.namespace_id = "local";
    request.scope.project_id = "flydelta-smoke";
    request.scope.session_id = "procedure-session";
    request.scope.turn_id = "procedure-turn";
    request.behavior_key = "tool_choice/dataset/grouped_sum";
    request.task_fingerprint = "sha256:smoke-grouped-sum-task";
    request.baseline_ref = "execution://smoke/model-describe";
    request.conditioned_ref = "execution://smoke/host-aggregate";
    request.control_ref = "execution://smoke/model-describe-column";
    request.verifier_ref = "verifier://smoke/dataset-tool/v1";
    request.evidence_ref = "evidence://smoke/procedure/grouped-sum";
    request.confidence = 1.0f;
    request.host_scope_admitted = true;
    request.host_verified = true;
    request.reusable = true;
    request.require_control = true;
    return request;
}

int fail(const std::string & message) {
    std::cerr << "flydelta_procedure_teaching_smoke=failed reason=" << message << '\n';
    return 1;
}

} // namespace

int main() {
    const auto request = grouped_sum_request();
    const auto resolved = common_agent_build_procedure_teaching_relation(request);
    if (resolved.status != common_agent_teaching_build_status::resolved ||
            !resolved.relation || !resolved.relation->host_approved ||
            resolved.relation->teaching_key != "dataset.grouped_sum.v1" ||
            resolved.relation->source != common_adaptation_evidence_source::procedure_blueprint) {
        return fail("valid host relation was not resolved");
    }

    common_learning_transaction transaction;
    transaction.id = "learning://procedure-smoke/transaction-1";
    transaction.observation.scope = request.scope;
    transaction.observation.cause = common_learning_cause::host_contract;
    transaction.observation.content_hash = "identity:smoke-procedure";
    transaction.created_at = "2026-01-01T00:00:00Z";
    common_adaptation_evidence_relation evidence_relation;
    std::string error;
    if (!common_agent_procedure_teaching_relation_to_evidence_relation(
            *resolved.relation, transaction, evidence_relation, error)) {
        return fail("resolved relation was not mapped to generic evidence: " + error);
    }
    common_agent_request host_request;
    host_request.turn_id = request.scope.turn_id;
    host_request.session_id = request.scope.session_id;
    host_request.project_id = request.scope.project_id;
    common_plan_state host_plan;
    host_plan.id = "procedure-plan-1";
    common_agent_result host_result;
    host_result.learning_signals.push_back({common_learning_signal_type::procedure_verification,
        host_plan.id, {}, {}, request.evidence_ref, "host verified procedure"});
    common_adaptation_evidence evidence;
    if (!common_adaptation_evidence_from_turn(
            host_request, host_plan, host_result, evidence_relation, evidence, error)) {
        return fail("generic evidence relation was not materialized: " + error);
    }
    common_flydelta_capture_candidate_collector collector("model-smoke", "layout-v1");
    if (!collector.observe_verified_relation(
            evidence_relation, evidence, transaction, error) || collector.candidates().size() != 1) {
        return fail("resolved procedure relation did not reach the existing capture queue: " + error);
    }

    auto no_contrast = request;
    no_contrast.conditioned_ref = no_contrast.baseline_ref;
    const auto no_contrast_result = common_agent_build_procedure_teaching_relation(no_contrast);
    if (no_contrast_result.status != common_agent_teaching_build_status::no_contrast ||
            no_contrast_result.relation) {
        return fail("missing contrast was admitted");
    }

    auto not_reusable = request;
    not_reusable.reusable = false;
    const auto not_reusable_result = common_agent_build_procedure_teaching_relation(not_reusable);
    if (not_reusable_result.status != common_agent_teaching_build_status::not_reusable) {
        return fail("non-reusable procedure was admitted");
    }

    std::cout << "flydelta_procedure_teaching_smoke=completed"
              << " source=procedure_blueprint"
              << " teaching_key=" << resolved.relation->teaching_key
              << " host_approved=" << (resolved.relation->host_approved ? "yes" : "no")
              << " capture_candidates=" << collector.candidates().size()
              << " learning_credit=none"
              << " promotion=false"
              << " unresolved_no_contrast=" << common_agent_teaching_build_status_name(
                    no_contrast_result.status)
              << '\n';
    return 0;
}
