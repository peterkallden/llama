#include "agent/adaptation/training-candidate.h"

#include <algorithm>

static bool valid_score(float value) { return value >= 0.0f && value <= 1.0f; }

const char * common_learning_destination_name(common_learning_destination destination) {
    switch (destination) {
        case common_learning_destination::retain: return "retain";
        case common_learning_destination::memory: return "memory";
        case common_learning_destination::procedure: return "procedure";
        case common_learning_destination::training_candidate: return "training_candidate";
        case common_learning_destination::reject: return "reject";
    }
    return "reject";
}

const char * common_training_candidate_status_name(common_training_candidate_status status) {
    switch (status) {
        case common_training_candidate_status::observed: return "observed";
        case common_training_candidate_status::eligible: return "eligible";
        case common_training_candidate_status::approved: return "approved";
        case common_training_candidate_status::rejected: return "rejected";
        case common_training_candidate_status::revoked: return "revoked";
    }
    return "rejected";
}

common_learning_destination common_learning_destination_for(
        const common_learning_observation & observation) {
    if (observation.cause == common_learning_cause::project_knowledge) return common_learning_destination::memory;
    if (observation.cause == common_learning_cause::host_contract || observation.cause == common_learning_cause::policy) {
        return common_learning_destination::retain;
    }
    if (observation.cause != common_learning_cause::model_behavior) return common_learning_destination::retain;
    if (observation.verification == common_learning_verification::unverified) return common_learning_destination::retain;
    return common_learning_destination::training_candidate;
}

bool common_training_candidate_qualifies(
        const common_training_candidate & candidate,
        const common_training_candidate_policy & policy,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1) { error = "unsupported training candidate schema"; return false; }
    if (candidate.id.empty()) { error = "training candidate requires id"; return false; }
    if (candidate.transaction_ids.empty() || candidate.transaction_ids.size() > policy.max_transaction_ids) {
        error = "training candidate transaction bound is invalid"; return false;
    }
    if (candidate.hypothesis.empty() || candidate.hypothesis.size() > policy.max_text_size ||
            candidate.approved_prompt.empty() || candidate.approved_prompt.size() > policy.max_text_size ||
            candidate.approved_target.empty() || candidate.approved_target.size() > policy.max_text_size) {
        error = "training candidate contains invalid bounded text"; return false;
    }
    if (candidate.cause != common_learning_cause::model_behavior) {
        error = "training candidate cause is not model behavior"; return false;
    }
    if (candidate.observed_occurrences < policy.min_occurrences) { error = "training candidate has insufficient occurrences"; return false; }
    if (candidate.verified_recoveries < policy.min_verified_recoveries) { error = "training candidate has insufficient verified recoveries"; return false; }
    if (candidate.contradictions != 0) { error = "training candidate has unresolved contradictions"; return false; }
    if (!valid_score(candidate.confidence) || candidate.confidence < policy.min_confidence) { error = "training candidate confidence is below threshold"; return false; }
    if (candidate.status != common_training_candidate_status::approved) {
        error = "training candidate requires explicit approval";
        return false;
    }
    return true;
}

bool common_training_candidate_from_approved_case(
        const common_learning_case & learning_case,
        const common_learning_transaction & transaction,
        const common_training_candidate_promotion & promotion,
        common_training_candidate & candidate,
        std::string & error) {
    error.clear();
    if (learning_case.status != common_learning_case_status::approved) {
        error = "training candidate promotion requires an approved learning case";
        return false;
    }
    if (!common_learning_case_validate(learning_case, 64, 2048, error)) return false;
    if (!common_learning_transaction_validate(transaction, 64, error)) return false;
    if (learning_case.observation_id != transaction.observation.id) {
        error = "learning case and transaction observation ids do not match";
        return false;
    }
    if (common_learning_destination_for(transaction.observation) != common_learning_destination::training_candidate) {
        error = "observation is not classified for training-candidate promotion";
        return false;
    }
    if (promotion.hypothesis.empty() || promotion.hypothesis.size() > 2048 ||
            !valid_score(promotion.confidence)) {
        error = "training candidate promotion contains invalid bounded evidence";
        return false;
    }
    if (promotion.verified_recoveries > promotion.observed_occurrences || promotion.contradictions > promotion.observed_occurrences) {
        error = "training candidate promotion counts are inconsistent";
        return false;
    }
    for (const auto & evidence_id : learning_case.evidence_ids) {
        if (std::find(transaction.observation.evidence_ids.begin(), transaction.observation.evidence_ids.end(), evidence_id) ==
                transaction.observation.evidence_ids.end()) {
            error = "learning case evidence is not present in the source transaction";
            return false;
        }
    }

    candidate = {};
    candidate.id = "learning://candidate/" + learning_case.id;
    candidate.transaction_ids = {transaction.id};
    candidate.cause = transaction.observation.cause;
    candidate.hypothesis = promotion.hypothesis;
    candidate.approved_prompt = learning_case.input;
    candidate.approved_target = learning_case.preferred_action;
    candidate.observed_occurrences = promotion.observed_occurrences;
    candidate.verified_recoveries = promotion.verified_recoveries;
    candidate.contradictions = promotion.contradictions;
    candidate.confidence = promotion.confidence;
    candidate.redaction_policy_id = learning_case.redaction_policy_id;
    candidate.redaction_method = learning_case.redaction_method;
    candidate.redaction_status = learning_case.redaction_status;
    candidate.status = common_training_candidate_status::approved;
    candidate.learning_domain = learning_case.learning_domain;
    candidate.tool_family = learning_case.tool_family;
    candidate.provider_kind = learning_case.provider_kind;
    return true;
}
