#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"

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
              << " learning_credit=none"
              << " promotion=false"
              << " unresolved_no_contrast=" << common_agent_teaching_build_status_name(
                    no_contrast_result.status)
              << '\n';
    return 0;
}
