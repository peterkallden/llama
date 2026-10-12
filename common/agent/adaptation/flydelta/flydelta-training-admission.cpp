#include "agent/adaptation/flydelta/flydelta-training-admission.h"

#include <algorithm>
#include <set>

namespace {

bool same_scope(const common_agent_scope & left, const common_agent_scope & right) {
    return left.namespace_id == right.namespace_id &&
        left.project_id == right.project_id && left.session_id == right.session_id;
}

bool contains_id(const std::vector<std::string> & values, const std::string & id) {
    return std::find(values.begin(), values.end(), id) != values.end();
}

} // namespace

bool common_flydelta_admit_training_cases(
        const std::vector<common_flydelta_training_case_source> & sources,
        const common_flydelta_training_admission_policy & policy,
        std::vector<common_training_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    if (policy.minimum_independent_relations < 3 ||
            policy.max_relations < policy.minimum_independent_relations ||
            sources.size() < policy.minimum_independent_relations ||
            sources.size() > policy.max_relations) {
        error = "FlyDelta training admission relation count is outside policy bounds";
        return false;
    }

    std::set<std::string> task_fingerprints;
    std::set<std::string> transaction_ids;
    std::string teaching_key;
    std::string behavior_key;
    std::string model_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string oracle_revision;
    std::string hypothesis;
    float minimum_confidence = 1.0f;
    for (const auto & source : sources) {
        const auto & relation = source.relation;
        const auto & evidence = source.evidence;
        const auto & learning_case = source.learning_case;
        const auto & transaction = source.transaction;

        if (!common_flydelta_teaching_relation_validate(relation, error) ||
                !common_adaptation_evidence_validate(evidence, 64, error) ||
                !common_learning_case_validate(learning_case, 64, 2048, error) ||
                !common_learning_transaction_validate(transaction, 64, error)) return false;
        if (relation.status != common_flydelta_teaching_relation_status::resolved ||
                !relation.host_approved || !evidence.host_verified ||
                evidence.id != relation.evidence_ref ||
                evidence.source != relation.source ||
                evidence.behavior_key != relation.behavior_key ||
                evidence.task_fingerprint != relation.task_fingerprint ||
                evidence.baseline_ref != relation.baseline_ref ||
                evidence.candidate_ref != relation.conditioned_ref ||
                evidence.verifier_ref != relation.verifier_ref ||
                (!learning_case.teaching_key.empty() && learning_case.teaching_key != relation.teaching_key) ||
                (!learning_case.behavior_key.empty() && learning_case.behavior_key != relation.behavior_key) ||
                (!learning_case.contrast_ref.empty() && learning_case.contrast_ref != relation.contrast_ref) ||
                (!learning_case.verifier_ref.empty() && learning_case.verifier_ref != relation.verifier_ref) ||
                (!learning_case.verifier_revision.empty() && learning_case.verifier_revision != relation.oracle_revision) ||
                learning_case.status != common_learning_case_status::approved ||
                learning_case.observation_id != transaction.id ||
                learning_case.source != relation.source ||
                transaction.observation.cause != common_learning_cause::model_behavior ||
                transaction.observation.verification != common_learning_verification::host_verified ||
                !same_scope(learning_case.scope, transaction.observation.scope) ||
                !same_scope(relation.scope, transaction.observation.scope) ||
                !contains_id(evidence.transaction_ids, transaction.id)) {
            error = "FlyDelta relation, evidence, approved case and transaction do not form one host-verified example";
            return false;
        }
        for (const auto & evidence_id : learning_case.evidence_ids) {
            if (!contains_id(transaction.observation.evidence_ids, evidence_id)) {
                error = "FlyDelta learning case references evidence outside its transaction";
                return false;
            }
        }
        if (relation.model_profile_fingerprint.empty() || relation.tokenizer_fingerprint.empty() ||
                relation.template_fingerprint.empty() || relation.oracle_revision.empty() ||
                relation.contrast_ref.empty()) {
            error = "FlyDelta training relation lacks model, template, contrast or verifier revision provenance";
            return false;
        }

        if (teaching_key.empty()) {
            teaching_key = relation.teaching_key;
            behavior_key = relation.behavior_key;
            model_fingerprint = relation.model_profile_fingerprint;
            tokenizer_fingerprint = relation.tokenizer_fingerprint;
            template_fingerprint = relation.template_fingerprint;
            oracle_revision = relation.oracle_revision;
            hypothesis = "Verified behavior: " + relation.behavior_key;
        } else if (teaching_key != relation.teaching_key || behavior_key != relation.behavior_key ||
                model_fingerprint != relation.model_profile_fingerprint ||
                tokenizer_fingerprint != relation.tokenizer_fingerprint ||
                template_fingerprint != relation.template_fingerprint ||
                oracle_revision != relation.oracle_revision) {
            error = "FlyDelta training group mixes behavior or incompatible model/verifier revisions";
            return false;
        }
        if (!task_fingerprints.insert(relation.task_fingerprint).second ||
                !transaction_ids.insert(transaction.id).second) {
            error = "FlyDelta training group reuses a task or transaction as independent evidence";
            return false;
        }
        minimum_confidence = std::min(minimum_confidence, relation.confidence);
    }

    if (task_fingerprints.size() < policy.minimum_independent_relations) {
        error = "FlyDelta training group has too few independent task fingerprints";
        return false;
    }
    for (const auto & source : sources) {
        common_training_candidate_promotion promotion;
        promotion.hypothesis = hypothesis;
        promotion.observed_occurrences = task_fingerprints.size();
        promotion.verified_recoveries = task_fingerprints.size();
        promotion.confidence = minimum_confidence;
        common_training_candidate candidate;
        if (!common_training_candidate_from_approved_case(
                source.learning_case, source.transaction, promotion, candidate, error)) return false;
        candidate.teaching_key = teaching_key;
        candidate.behavior_key = behavior_key;
        candidate.learning_case_id = source.learning_case.id;
        candidate.teaching_relation_id = source.relation.id;
        candidate.evidence_ref = source.evidence.id;
        candidate.model_profile_fingerprint = source.relation.model_profile_fingerprint;
        candidate.tokenizer_fingerprint = source.relation.tokenizer_fingerprint;
        candidate.template_fingerprint = source.relation.template_fingerprint;
        candidate.contrast_ref = source.relation.contrast_ref;
        candidate.verifier_ref = source.relation.verifier_ref;
        candidate.verifier_revision = source.relation.oracle_revision;
        if (!common_training_candidate_qualifies(candidate, {}, error)) return false;
        candidates.push_back(std::move(candidate));
    }
    return true;
}
