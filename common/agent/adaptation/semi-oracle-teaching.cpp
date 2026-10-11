#include "agent/adaptation/semi-oracle-teaching.h"

#include <sstream>

bool common_semi_oracle_make_teaching_relation(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_result & rejected,
        const common_semi_oracle_result & accepted,
        const common_agent_scope & scope,
        const std::string & relation_id,
        const std::string & rejected_execution_ref,
        const std::string & accepted_execution_ref,
        const std::string & evidence_ref,
        common_flydelta_teaching_relation & relation,
        std::string & error) {
    error.clear();
    if (!common_semi_oracle_learning_authorized(ruleset, rejected) ||
            !common_semi_oracle_learning_authorized(ruleset, accepted) ||
            rejected.verdict != common_semi_oracle_verdict::violated ||
            accepted.verdict != common_semi_oracle_verdict::satisfied ||
            rejected.evaluator_ref != accepted.evaluator_ref ||
            rejected.evaluator_revision != accepted.evaluator_revision) {
        error = "semi-oracle teaching requires a verified rejected/accepted contrast from one evaluator and RuleSet revision";
        return false;
    }
    if (relation_id.empty() || rejected_execution_ref.empty() ||
            accepted_execution_ref.empty() || evidence_ref.empty() ||
            rejected_execution_ref == accepted_execution_ref ||
            scope.namespace_id.empty() || scope.session_id.empty()) {
        error = "semi-oracle teaching relation is missing host execution or scope references";
        return false;
    }

    std::ostringstream identity;
    identity << "semi-oracle://ruleset/" << ruleset.id << "@" << ruleset.revision
             << "/action/" << ruleset.action_contract_fingerprint;
    const auto key = identity.str();

    relation = {};
    relation.id = relation_id;
    relation.teaching_key = key;
    relation.source = common_adaptation_evidence_source::user_taught_concept;
    relation.behavior_key = key;
    relation.scope = scope;
    relation.task_fingerprint = key;
    relation.baseline_ref = rejected_execution_ref;
    relation.conditioned_ref = accepted_execution_ref;
    relation.verifier_ref = accepted.evaluator_ref;
    relation.oracle_revision = accepted.evaluator_revision + ":" +
        ruleset.semantic_fingerprint + ":" + ruleset.action_contract_fingerprint;
    relation.evidence_ref = evidence_ref;
    relation.contrast_ref = rejected_execution_ref + "<>" + accepted_execution_ref;
    relation.status = common_flydelta_teaching_relation_status::resolved;
    relation.baseline_origin = common_flydelta_teaching_origin::observed;
    relation.conditioned_origin = common_flydelta_teaching_origin::host_derived;
    relation.confidence = 1.0f;
    relation.host_approved = true;
    return common_flydelta_teaching_relation_validate(relation, error);
}
